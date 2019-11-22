import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import os
import constants
import glob
import numpy as np
import cv2
import math
from point import Point
from bounding_box import BoundingBox
from YOLOv3 import yolo_inference
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import copy
# import point.Point as Point


def printBoundingBox(bbox):
    print("=> x1: %.5f, y1: %.5f, x2: %.5f, y2: %.5f" % (bbox.pmin.x, bbox.pmin.y, bbox.pmax.x, bbox.pmax.y))

def overlappedArea(bbox1, bbox2): 
    dx = min(bbox1.pmax.x, bbox2.pmax.x) - max(bbox1.pmin.x, bbox2.pmin.x)
    dy = min(bbox1.pmax.y, bbox2.pmax.y) - max(bbox1.pmin.y, bbox2.pmin.y)
    if (dx>=0) and (dy>=0):
        return dx * dy
    else: return 0

def percentajeOverlappedArea(realArea, overArea):
    per = overArea * 100 / realArea
    return per

def joinBBoxes(bbox1, bbox2):
    xmin = min(bbox1.pmin.x, bbox2.pmin.x)
    ymin = min(bbox1.pmin.y, bbox2.pmin.y)
    xmax = max(bbox1.pmax.x, bbox2.pmax.x)
    ymax = max(bbox1.pmax.y, bbox2.pmax.y)
    bbox = BoundingBox(Point(xmin, ymin), Point(xmax, ymax))
    return bbox

def joinOverlappedBBoxes_recursively(bbox_list, threshold=48):
    print('================bbox_list', len(bbox_list))
    if len(bbox_list) == 1:
        return bbox_list
    else:
        print('000000000f0f0f0f0f0f')
        for i in range(1,len(bbox_list)):
            overlap_area = overlappedArea(bbox_list[0], bbox_list[i])
            if overlap_area > 0:
                area1 = bbox_list[0].area()
                area2 = bbox_list[i].area()
                percentaje = .0
                if area1 <= area2:
                    percentaje = percentajeOverlappedArea(area1, overlap_area)
                else:
                    percentaje = percentajeOverlappedArea(area2, overlap_area)
                if percentaje >= threshold:
                    joinBox = joinBBoxes(bbox_list[0], bbox_list[i])
                    bbox_list[i] = joinBox
                    del bbox_list[0]
                    # first = bbox_list[0]
                    break
        first = bbox_list[0]
        if len(bbox_list) > 1:
            sub_bbox_list = bbox_list[1:]
            print('sub_bbox_list: ', sub_bbox_list, len(sub_bbox_list))
            sub_bbox_list = joinOverlappedBBoxes_recursively(sub_bbox_list, threshold)
            # print('sub_bbox_list: ', sub_bbox_list, len(sub_bbox_list))
            sub_bbox_list.insert(0, first)
            bbox_list = sub_bbox_list
        return bbox_list
        

def joinOverlappedBBoxes(bboxes, threshold=48):
    bboxes_r = copy.deepcopy(bboxes)
    for i in range(len(bboxes) - 1):
        for idx in range(i+1, len(bboxes)):
            # print(i, idx)
            overlap_area = overlappedArea(bboxes[i], bboxes[idx])
            if overlap_area > 0:
                area1 = bboxes[i].area()
                area2 = bboxes[idx].area()
                percentaje = .0
                # print('area1, area2: ', area1, area2)
                if area1 <= area2:
                    percentaje = percentajeOverlappedArea(area1, overlap_area)
                else:
                    percentaje = percentajeOverlappedArea(area2, overlap_area)
                if percentaje >= threshold:
                    joinBox = joinBBoxes(bboxes[i], bboxes[idx])
                    bboxes_r[idx] = joinBox
                    del bboxes_r[i]
    return bboxes_r

def personDetectionInSegment(video_name, frames_segment, yolo_model, img_size, conf_thres, nms_thres, classes, num_frames):
    bbox_persons_in_segment = []
    if num_frames == 'all':
        for frame_info in frames_segment:
            frame_name = str(frame_info[0][0])
            frame_path = os.path.join(video_name, frame_name)
            bbox_persons_in_frame = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frame_path)
            bbox_persons_in_segment.append(bbox_persons_in_frame)
    elif num_frames == 'first':
        frame_info = frames_segment[0]
        frame_name = str(frame_info[0][0])
        frame_path = os.path.join(video_name, frame_name)
        bbox_persons_in_frame = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frame_path)
        bbox_persons_in_segment.append(bbox_persons_in_frame)
    elif  num_frames == 'extremes':
        frame_info_first = frames_segment[0]
        frame_name_first = str(frame_info_first[0][0])
        frame_path_first = os.path.join(video_name, frame_name_first)
        bbox_persons_in_frame_first = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frame_path_first)

        frame_info_end = frames_segment[len(frames_segment)-1]
        frame_name_end = str(frame_info_end[0][0])
        frame_path_end = os.path.join(video_name, frame_name_end)
        bbox_persons_in_frame_end = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frame_path_end)

        bbox_persons_in_segment.append(bbox_persons_in_frame_first)
        bbox_persons_in_segment.append(bbox_persons_in_frame_end)
    return bbox_persons_in_segment

def personDetectionInFrame(model, img_size, conf_thres, nms_thres, classes, frame_path, plot = False):
    
    print('='*20+' YOLOv3 - ', frame_path)
    img = yolo_inference.preProcessImage(frame_path, img_size)
    detections = yolo_inference.inference(model, img, conf_thres, nms_thres)
    image = np.array(Image.open(frame_path))
    # print('image type: ', type(image), image.dtype, image.shape)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(image)
    bbox_persons = []
    if detections is not None:
        # print('detectios rescale: ', type(detections), detections.size())
        detections = yolo_inference.rescale_boxes(detections, 416, image.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if classes[int(cls_pred)] == 'person':
                pmin = Point(x1, y1)
                pmax = Point(x2,y2)
                bbox_persons.append(BoundingBox(pmin,pmax))
            # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            if plot:
                box_w = x2 - x1
                box_h = y2 - y1
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='g', facecolor="none")
                ax.add_patch(bbox)
                plt.text( x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top", bbox={"color": 'r', "pad": 0}, )
    return bbox_persons

def distance(p1, p2):
    distance = math.sqrt(((p1.x - p2.x)** 2) + ((p1.y - p2.y)** 2))
    return distance

def computeBoundingBoxFromMask(mask):
    """
    *** mask: rgb numpy image
    """

    mask = thresholding_cv2(mask)
    # mask = mask.astype('uint8')
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # print('ccccc: ', mask.shape)
    img = process_mask(mask)
    img, contours = findContours(img, remove_fathers=True)
    
    bboxes = bboxes_from_contours(img, contours)
    # print(len(bboxes),bboxes)
    return bboxes

def process_mask(img):
    kernel_exp = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_exp)
    kernel_dil = np.ones((7, 7), np.uint8)
    img = cv2.dilate(img, kernel_dil, iterations=1)
    kernel_clo = np.ones((11, 11), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_clo)
    return img

def findContours(img, remove_fathers = True):
    # Detect edges using Canny
    # canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    # color = cv2.Scalar(0, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if remove_fathers:
        removed = []
        for idx,contour in enumerate(contours):
            if hierarchy[0, idx, 3] == -1:
                removed.append(contour)
        contours = removed
    # print('contours: ', len(contours))
    
    for i in range(len(contours)):
        cv2.drawContours(drawing, contours, i, (0, 255, 0), 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    # cv2.imshow('Contours', drawing)
    return drawing, contours

def plotBBoxesOnImage(ax, bboxes, color):
    # shape = img.shape
    # if shape[2] == 1:
    #     img = np.squeeze(img,2)
    #     img = gray2rgbRepeat(img)

    # fig, ax = plt.subplots()
    # ax.imshow(img)
    for i, box in enumerate(bboxes):
        # Create a Rectangle patch
        h = box.pmax.y - box.pmin.y
        w = box.pmax.x - box.pmin.x
        rect = patches.Rectangle((box.pmin.x, box.pmin.y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # cv2.rectangle(img,(box.pmin.x,box.pmin.y),(box.pmax.x,box.pmax.y),color,2)
        # cv2.rectangle(img, (int(bboxes[i][0]),int(bboxes[i][1])),
        #                 (int(bboxes[i][0] + bboxes[i][2]),int(bboxes[i][1] + bboxes[i][3])),
        #                 color, 2)
    # plt.show()
    return ax

def bboxes_from_contours(img, contours):
    contours_poly = [None]*len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    # drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    bboxes = []
    for i, rect in enumerate(boundRect):
        # print('REct: ', rect)
        bb = cvRect2BoundingBox(rect)
        bboxes.append(bb)
    #     color_red = (0,0,255)
    #     # cv2.drawContours(drawing, contours_poly, i, color)
    #     cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color_red, 2)
    return bboxes

def cvRect2BoundingBox(cvRect):
    pmin = Point(cvRect[0], cvRect[1])
    pmax = Point(cvRect[0]+cvRect[2], cvRect[1]+cvRect[3])
    bb = BoundingBox(pmin,pmax)
    # print('bbbbbbbbxxxxx: ', pmin.x, bb.center.x)
    return bb
    


def thresholding_cv2(x):
        x = 255*x #between 0-255
        x = x.astype('uint8')
        # th = cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
        # Otsu's thresholding
        x = cv2.GaussianBlur(x,(5,5),0)
        # print('x numpy: ', x.shape, x.dtype)
        ret2, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return th

def normalize_tensor(self, img):
        # print("normalize:", img.size())
        _min = torch.min(img)
        _max = torch.max(img)
        # print("min:", _min.item(), ", max:", _max.item())
        return (img - _min) / (_max - _min)

def get_anomalous_video(video_test_name, reduced_dataset = True):
    """ get anomalous video """
    label = video_test_name[:-3]
    path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED, video_test_name) if reduced_dataset else os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES, video_test_name)
    
    list_frames = os.listdir(path) 
    list_frames.sort()
    num_frames = len(glob.glob1(path, "*.jpg"))
    bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video_test_name+'.txt')
    data = []
    with open(bdx_file_path, 'r') as file:
        for row in file:
            data.append(row.split())
    data = np.array(data)
    bbox_infos_frames = []
    for frame in list_frames:
        num_frame = int(frame[len(frame) - 7:-4])
        if num_frame != int(data[num_frame, 5]):
            sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
            # print('Houston we have a problem: index frame does not equal to the bbox file!!!')
        flac = int(data[num_frame,6]) # 1 if is occluded: no plot the bbox
        xmin = int(data[num_frame, 1])
        ymin= int(data[num_frame, 2])
        xmax = int(data[num_frame, 3])
        ymax = int(data[num_frame, 4])
        info_frame = [frame, flac, xmin, ymin, xmax, ymax]
        bbox_infos_frames.append(info_frame)
    
    return path, label, bbox_infos_frames, num_frames

def tensor2numpy(x):
    x = x / 2 + 0.5
    x = x.numpy()
    x = np.transpose(x, (1, 2, 0))
    # print('x: ', type(x), x.shape)
    return x

def rgb2grayUnrepeat(x):
    x = x[:,:, 0]
    return x

def gray2rgbRepeat(x):
    x = np.stack([x, x, x], axis=2)
    return x