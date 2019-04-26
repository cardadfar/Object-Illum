import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class BBox:
    '''     x: starting x position
            y: starting y position
          wth: width of box
          hgt: height of box
    '''

    def __init__(self, x, y, wth, hgt):
        global global_id_indx

        self.x = int(x)
        self.y = int(y)
        self.wth = int(wth)
        self.hgt = int(hgt)


def are_simmilar(bbox1, bbox2, img_wth, img_hgt):
    ''' returns true if bboxes are too simmilar, false else '''

    area_threshold = 0.95
    width_threshold = img_wth * 0.05
    height_threshold = img_hgt * 0.05

    x1_min = bbox1.x
    y1_min = bbox1.y
    x1_max = bbox1.x + bbox1.wth
    y1_max = bbox1.y + bbox1.hgt

    x2_min = bbox2.x
    y2_min = bbox2.y
    x2_max = bbox2.x + bbox2.wth
    y2_max = bbox2.y + bbox2.hgt


    x_min = max(x1_min, x2_min)
    x_max = min(x1_max, x2_max)
    y_min = max(y1_min, y2_min)
    y_max = min(y1_max, y2_max)

    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    area_min = min(bbox1_area, bbox2_area)

    shared_area = (x_max - x_min) * (y_max - y_min)

    area_intersection = shared_area / area_min
    width_simmilarity = abs(bbox1.wth - bbox2.wth)
    height_simmilarity = abs(bbox1.hgt - bbox2.hgt)

    if((area_intersection > area_threshold) and (width_simmilarity < width_threshold) and (height_simmilarity < height_threshold)):
        return True

    return False



