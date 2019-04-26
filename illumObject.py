import numpy as np
from PIL import Image

import luminance


global_id_indx = 0
illumObj_list = []

class IllumObject:
    ''' className: name of object
               id: index of illumObject in illumObj list
              wth: width of illumObject sampling
              hgt: height of illumObject sampling
            illum: storage for average illuminations of objects 
           access: number of frames sampled for object [used in averaging samples] 
    '''

    def __init__(self, className, wth, hgt):
        global global_id_indx

        self.className = className
        self.id = global_id_indx
        self.wth = wth
        self.hgt = hgt
        self.illum = np.zeros([hgt,wth], float)
        self.access = 0

        global_id_indx += 1

        illumObj_list.append(self)


    def reweight(self, img, bbox):
        ''' calculate local illumination average for specific object '''

        img_pixels = np.array(img)

        x_min = bbox.x
        y_min = bbox.y
        wth   = bbox.wth
        hgt   = bbox.hgt

        sampleRateX = (wth) / self.wth
        sampleRateY = (hgt) / self.hgt

        self.access += 1
        weight = 1 / self.access
        
        for j in range(self.hgt):
            for i in range(self.wth):
                x = int(x_min + i * sampleRateX)
                y = int(y_min + j * sampleRateY)

                hls = luminance.rgb_to_hls(img_pixels[y,x]);

                self.illum[j,i] = ((1 - weight) * self.illum[j,i]) + (weight * hls[1])




def clear():
    ''' returns global list of illumObjects '''

    illumObj_list = []


def get_illumObj_list():
    ''' returns global list of illumObjects '''

    return illumObj_list




def in_illumObj_list(class_name):
    ''' returns true if class_name is in global illumObj_list '''

    for illumObj in illumObj_list:
        if(illumObj.className == class_name):
            return True

    return False




def find_illumObj_with_class(class_name):
    ''' returns first index of where class_name is in global illumObj_list or false otherwise'''

    for illumObj in illumObj_list:
        if(illumObj.className == class_name):
            return illumObj

    return False
