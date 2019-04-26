import cv2
import numpy as np
import math
import os
from PIL import Image 

import bbox


classes_dir = 'yolo-params/yolov3.txt'
weights     = 'yolo-params/yolov3.weights'
config      = 'yolo-params/yolov3.cfg'


def get_output_layers(net):
    ''' helper to detect_objects '''
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers





def detect_objects(img_read):
    ''' returns list of classes detected and corresponding bounding boxes '''

    image = cv2.imread(img_read)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classes_dir, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    class_list = []
    bbox_list = []

    for class_indx in class_ids:
        class_list.append( str(classes[class_indx]) )

    for box in boxes:
        bbox_list.append(bbox.BBox(box[0], box[1], box[2], box[3]))


    return class_list, confidences, bbox_list





def cleanup(class_list, confidences, bboxes, wth, hgt):
    ''' deletes redundancies in class_list '''

    new_class_list  = []
    new_confidences = []
    new_bboxes      = []

    if(len(class_list) == 0):
        return new_class_list, new_confidences, new_bboxes

    new_class_list.append(class_list[0])
    new_confidences.append(confidences[0])
    new_bboxes.append(bboxes[0])

    return new_class_list, new_confidences, new_bboxes


