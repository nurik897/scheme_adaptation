"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import math
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
from torchvision.transforms.functional import to_pil_image, rotate

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import pytesseract
import pickle

from craft import CRAFT

from collections import OrderedDict

def recognizeText(text_image_list):
    tessdata_dir_config = r'--tessdata-dir "./tessdata" --oem 3 --psm 8'
    recognized_array = []
    for i in text_image_list:
        rec_string = pytesseract.image_to_string(i[0], lang='rus', config=tessdata_dir_config)
        recognized_array += [[rec_string], i[1]]
    return recognized_array

def recognizeLines(filename):
    img = cv2.imread(f'./result/{filename}/res_{filename}_stripped.jpg', 0)
    lsd = cv2.createLineSegmentDetector(_scale=0.7, _sigma_scale=0.9)
    lines = lsd.detect(img, 1)[0]
    emptyimg = np.zeros((img.shape), dtype=np.uint8)
    if lines is None:
        print(f'lines not detected in {filename}')
        return 0
    else:
        drawn_img = lsd.drawSegments(emptyimg, lines)
        cv2.imwrite(f"./result/{filename}/{filename}_lsd.png", drawn_img)
        with open(f'./result/{filename}/{filename}_lines.txt', 'w') as f:
            for i in lines:
                f.write(i.astype(np.int).tolist().__repr__() + '\n')
    return 0

def regionRecognize(k, image_path):
    print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda,
                                         args.poly, refine_net)

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/" + filename + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=f'{result_folder}{filename}/')

    stripped_image = image
    background_color = []
    for color_plane in image.transpose(2, 0, 1):
        flat = color_plane.flatten()
        hist, bin_edges = np.histogram(flat, bins=256)
        background_color += [bin_edges[hist.argmax()]]
    background_color = np.array([round(x) for x in background_color])
    text_image_list = []

    for ii, poly in enumerate(polys):
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly

        l1 = pow(sum([pow(x0 - x1, 2) + pow(y0 - y1, 2)]), 0.5)
        l2 = pow(sum([pow(x1 - x2, 2) + pow(y1 - y2, 2)]), 0.5)

        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = [(x1, y1), (x2, y2), (x3, y3), (x0, y0)] if l2 > l1 else [(x0, y0),
                                                                                                           (x1, y1),
                                                                                                           (x2, y2),
                                                                                                           (x3, y3)]
        x_min = min([x0, x1, x2, x3])
        x_max = max([x0, x1, x2, x3])
        y_min = min([y0, y1, y2, y3])
        y_max = max([y0, y1, y2, y3])
        x_min, x_max, y_min, y_max = [int(round(temp)) for temp in [x_min, x_max, y_min, y_max]]

        dx0 = x1 - x0
        dy0 = y1 - y0
        dx1 = x3 - x2
        dy1 = y3 - y2

        angle = math.atan((dx0 - dx1) / (dy0 - dy1 + 1e-8))

        offset = 3
        crop = (slice(y_min, y_max), slice(x_min, x_max))
        crop1 = (slice(max(y_min - offset, 0), min(y_max + offset, image.shape[0])), slice(max(x_min - offset, 0), min(x_max + offset, image.shape[1])))

        cropped = to_pil_image(image[crop1])

        background_polygon = np.ones((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
        background_polygon = background_polygon * background_color.reshape((1, 1, 3))
        stripped_image[crop] = background_polygon

        f = lambda x, y: -1 if x >= y else 1
        rotated = rotate(cropped,
                         (90 - angle * 180 / math.pi),
                         expand=True)
        text_image_list += [[rotated, ((x0, y0), (x1, y1), (x2, y2), (x3, y3))]]
        # os.mkdir(f"./result/crops_{filename}") if not (os.path.exists(f"./result/crops_{filename}")) else 0
        os.mkdir(f"./result/{filename}") if not (os.path.exists(f"./result/{filename}")) else 0
        os.mkdir(f"./result/{filename}/crops_{filename}") if not (os.path.exists(f"./result/{filename}/crops_{filename}")) else 0
        rotated.save(f"./result/{filename}/crops_{filename}/rotated_{ii}.png")

    to_pil_image(stripped_image).save(result_folder + "/" + filename + "/res_" + filename + '_stripped.jpg')
    return text_image_list, stripped_image, filename


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='./craft_mlt_25k.pth', type=str, help='pretrained model, format .pth')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()
    print(len(image_list))
    # load data
    for k, image_path in enumerate(image_list):
        text_image_list, scheme, filename = regionRecognize(k, image_path)
        recognized_out = recognizeText(text_image_list)
        with open(f'./result/{filename}/{filename}_text.txt', 'w') as f:
            for i in recognized_out:
                f.write(i.__repr__() + '\n')

        recognizeLines(filename)


        # print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        # image = imgproc.loadImage(image_path)
        #
        # bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        #
        # # save score text
        # filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, score_text)
        #
        #
        # stripped_image = image
        # background_color = []
        # for color_plane in image.transpose(2, 0, 1):
        #     flat = color_plane.flatten()
        #     hist, bin_edges = np.histogram(flat, bins=256)
        #     background_color += [bin_edges[hist.argmax()]]
        # background_color = np.array([round(x) for x in background_color])
        # text_image_list = []
        #
        # for ii, poly in enumerate(polys):
        #
        #     (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
        #
        #     l1 = pow(sum([pow(x0 - x1, 2) + pow(y0 - y1, 2)]), 0.5)
        #     l2 = pow(sum([pow(x1 - x2, 2) + pow(y1 - y2, 2)]), 0.5)
        #
        #     (x0, y0), (x1, y1), (x2, y2), (x3, y3) = [(x1, y1), (x2, y2), (x3, y3), (x0, y0)] if l2 > l1 else [(x0, y0),
        #                                                                                                        (x1, y1),
        #                                                                                                        (x2, y2),
        #                                                                                                        (x3, y3)]
        #     x_min = min([x0, x1, x2, x3])
        #     x_max = max([x0, x1, x2, x3])
        #     y_min = min([y0, y1, y2, y3])
        #     y_max = max([y0, y1, y2, y3])
        #     x_min, x_max, y_min, y_max = [int(round(temp)) for temp in [x_min, x_max, y_min, y_max]]
        #
        #     dx0 = x1 - x0
        #     dy0 = y1 - y0
        #     dx1 = x3 - x2
        #     dy1 = y3 - y2
        #
        #     angle = math.atan((dx0 - dx1) / (dy0 - dy1 + 1e-8))
        #
        #     crop = (slice(y_min, y_max), slice(x_min, x_max))
        #
        #     cropped = to_pil_image(image[crop])
        #
        #     background_polygon = np.ones((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
        #     background_polygon = background_polygon * background_color.reshape((1, 1, 3))
        #     stripped_image[crop] = background_polygon
        #
        #     f = lambda x, y: -1 if x >= y else 1
        #     rotated = rotate(cropped,
        #                      (90 - angle * 180 / math.pi),
        #                      expand=True)
        #     text_image_list += [[rotated, ((x0, y0), (x1, y1), (x2, y2), (x3, y3))]]
        #     os.mkdir(f"./result/crops_{filename}") if not (os.path.exists(f"./result/crops_{filename}")) else 0
        #     rotated.save(f"./result/crops_{filename}/rotated_{ii}.png")
        #
        # to_pil_image(stripped_image).save(result_folder + "/res_" + filename + '_stripped.jpg')

    print("elapsed time : {}s".format(time.time() - t))

