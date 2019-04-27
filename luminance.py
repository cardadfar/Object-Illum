import numpy as np
from PIL import Image
import colorsys



def rgb_to_hls(pixel):
    ''' converts pixel rgb to hls '''

    pixel = pixel.astype(float)
    pixel[0] /= 255
    pixel[1] /= 255
    pixel[2] /= 255
    k = colorsys.rgb_to_hls(pixel[0], pixel[1], pixel[2])
    return k


def hls_to_rgb(pixel):
    ''' converts pixel hls to rgb '''

    res = colorsys.hls_to_rgb(pixel[0], pixel[1], pixel[2])
    res = list(res)
    res[0] *= 255
    res[1] *= 255
    res[2] *= 255
    return res



def pixel_luminance(pixel):
    ''' calculates illumination for each pixel '''

    pixel = pixel.astype(float)
    pixel /= 255

    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

def pixel_luminance_weight(pixel, del_lum):
    ''' calculates illumination for each pixel '''


    cur_lum = pixel_luminance(pixel)
    new_lum = cur_lum - del_lum
    lum_change = new_lum / cur_lum

    pixel = pixel.astype(float)
    pixel *= lum_change

    return pixel




def img_luminance(pixels):
    ''' calculates illumination array for entire image '''

    hgt, wth, col = pixels.shape
    lum = np.zeros([hgt, wth], float)

    for y in range(hgt):
        for x in range(wth):
            lum[y,x] = luminance(pixels[y,x])

    return lum



def global_avg(files):
    ''' globally calculate illumination average '''

    wth, hgt = Image.open(files[0]).size
    avg = np.zeros([hgt,wth,3], float)

    for file in files:
        img = Image.open(file)
        pixels = np.array(img)
        avg += pixels

    avg = avg / len(files) 
    res_pixels = np.zeros( (hgt, wth, 3), float)

    for y in range(hgt):
        for x in range(wth):
            res_pixels[y,x] = [int(avg[y,x,0]), int(avg[y,x,1]), int(avg[y,x,2])]

    return res_pixels