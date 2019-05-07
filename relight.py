import os
import math
import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

import videoConv
import illumObject
import globalIllum
import yolo_opencv
import luminance

    


def relighting_weights(img, illumObj, lum_weight, bbox):
    ''' get luminance difference between image and illumObj avergae
               img: PIL image
          illumObj: Local Illumination Object 
        lum_weight: 2D array storing illumination weights
              bbox: Bounding box of detected object on img
    '''

    img_pixels = np.array(img)
    img_wth, img_hgt = img.size

    x_init  = bbox.x;
    y_init  = bbox.y;
    box_wth = bbox.wth;
    box_hgt = bbox.hgt;

    for dy in range(box_hgt):
        for dx in range(box_wth):
            x = x_init + dx
            y = y_init + dy

            xf = (dx / box_wth) * illumObj.wth
            yf = (dy / box_hgt) * illumObj.hgt

            xs = math.floor(xf)
            ys = math.floor(yf)

            xf -= xs
            yf -= ys

            xss = xs + 1
            yss = ys + 1

            if( xss >= illumObj.wth ): 
                xss = illumObj.wth - 1
            if( yss >= illumObj.hgt ): 
                yss = illumObj.hgt - 1

            bot_lft = illumObj.illum[ys , xs ]
            bot_rgt = illumObj.illum[ys , xss]
            top_lft = illumObj.illum[yss, xs ]
            top_rgt = illumObj.illum[yss, xss]

            bot = bot_lft * (1 - xf) + bot_rgt * (xf)
            top = top_lft * (1 - xf) + top_rgt * (xf)
            cen = bot * (1 - yf) + top * (yf)


            if(img_wth > x) and (img_hgt > y):
                hls = luminance.rgb_to_hls(img_pixels[y,x]);

                lum_weight[y,x] = (hls[1] - cen)
    
    return lum_weight


def relighting_global_weights(img, lum_weight, globalIllum):
    ''' get luminance difference between image and global image avergae
                img: PIL image
         lum_weight: 2D array storing illumination weights
        globalIllum: Global Illumination Object
    '''

    img_pixels = np.array(img)
    hgt_g, wth_g, c = globalIllum.shape
    pixel_dif = img_pixels - globalIllum

    wth, hgt = img.size

    for y in range(hgt):
        for x in range(wth):

            if((lum_weight[y,x] == 0)):
                lum_weight[y,x] = luminance.pixel_luminance(pixel_dif[y,x])
    
    return lum_weight


def smoothen(lum_weight, itters):
    ''' smooths out images using [3x3] gaussian smoothening 
        lum_weight: 2D array storing illumination weights
            itters: Number of iterations to rerun smoothening operation on
    '''

    lum_weight_new = lum_weight

    smooth = np.array([[ 6/100,  10/100,   6/100],
                       [10/100,  36/100,  10/100],
                       [ 6/100,  10/100,   6/100]])

    for i in range(itters):
        lum_weight_new = signal.convolve2d(lum_weight_new, smooth, boundary='symm', mode='same')

    return lum_weight_new




def relight(img, hue_weight, lum_weight, sat_weight, step):
    ''' relights image in hsl space according to assigned weights and step-size
               img: PIL image
        hue_weight: 2D array storing hue weights
        lum_weight: 2D array storing illumination weights
        sat_weight: 2D array storing saturation weights
              step: [3x1] set of weights for [hue, lum, sat] relighting
    '''


    img_pixels  = np.array(img)
    hgt, wth, c = img_pixels.shape
    img_pix_new = np.zeros((hgt,wth, c), float)
    
    for y in range(hgt):
        for x in range(wth):

            hls = luminance.rgb_to_hls(img_pixels[y,x])
            hls = np.asarray(hls)
            hls[0] -= step[0] * hue_weight[y,x]
            hls[1] -= step[1] * lum_weight[y,x]
            hls[2] -= step[2] * sat_weight[y,x]

            hls[hls > 1] = 1
            hls[hls < 0] = 0

            img_pix_new[y,x] = luminance.hls_to_rgb(hls)




    
    img_new = Image.fromarray(np.uint8(img_pix_new))

    return img_new



def greyscale(img):
    ''' returns image with luminace values only
        img: PIL image
    '''

    img_pixels  = np.array(img)
    hgt, wth, c = img_pixels.shape
    img_grey    = np.zeros((hgt,wth, c), float)
    
    for y in range(hgt):
        for x in range(wth):
            hls = luminance.rgb_to_hls(img_pixels[y,x])
            l = hls[1]
            l *= 255
            img_grey[y,x] = [l,l,l]

    img_new = Image.fromarray(np.uint8(img_grey))

    return img_new



def difference(img1, img2):
    ''' returns difference between img1 and img2 pixels
        img1: PIL image
        img2: PIL image
    '''

    img1_pixels  = np.array(img1)
    img2_pixels  = np.array(img2)

    img_dif = abs(img1_pixels - img2_pixels)
    dif = Image.fromarray(np.uint8(img_dif))

    return dif


def load_files(input_dir, file_types):
    ''' loads filenames into array
        input_dir: Directory to search into
        file_types: file extensions to accept
    '''

    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = [os.path.join(input_dir, f) for f in input_files]
    added_files = []

    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in file_types:
            added_files.append(file)

    added_files.sort()
    return added_files







def main(input_dir, output_dir, file_types, iters, num_hues, num_lums, num_sats, hls_steps) :
    ''' main routine that computes image relighting
         input_dir: Directory to search into
        output_dir: Directory to save images to
        file_types: file extensions to accept
        iters: number of iterations to rerun algorithm on
        num_hues: number of hue buckets to create [global relighting]
        num_lums: number of luminance buckets to create [global relighting]
        num_sats: number of saturation buckets to create [global relighting]
        hls_steps: timesteps for [hue, luminance, saturation] relighting
    '''


    kmeans_iter = 100

    files = load_files(input_dir, file_types)
    if len(files) == 0:
        print("Detected No Images. Please Check File Extensions.")
        return

    images = []


    for file in files:
        images.append(Image.open(file))

    wth, hgt = Image.open(files[0]).size

    start = time.time()
    print("Clustering Data | {0:1d} Hues for {1:1d} Itterations".format(num_hues, kmeans_iter))
    bounds = globalIllum.cluster(Image.open(files[0]), num_hues, kmeans_iter)
    print("Clustering Data. Took {0:.2f}s. ".format(time.time() - start))

    for i in range(num_hues):
        for j in range(num_lums):
            for k in range(num_sats):
                hl = (i    ) / num_hues
                hu = (i + 1) / num_hues
                ll = (j    ) / num_lums
                lu = (j + 1) / num_lums
                sl = (k    ) / num_sats
                su = (k + 1) / num_sats
                if hl == 0:
                    hl += 0.0001
                globalIllum.globalIllum('color_' + str(i) + '_' + str(j) + '_' + str(k), hl, hu, ll, lu, sl, su)

    print("Detected {0:1d} Images. Starting Now...".format(len(images)))
    for n in range(iters):

        globalIllum.clear()
        illumObject.clear()
        print("-----------------------------")
        print("Beginning Itteration %d of %d" % ((n+1), iters))


        for i in range(len(images)):
            img = images[i]
            file = files[i]
            wth, hgt = img.size
            print("Loading Global Illumination [%d/%d]" % ((i+1), len(images)))
            globalIllum.reweight(img)

            print("Loading Local Illumination  [%d/%d]" % ((i+1), len(images)))
            class_list, confidences, bboxes = yolo_opencv.detect_objects(file)
            class_list, confidences, bboxes = yolo_opencv.cleanup(class_list, confidences, bboxes, wth, hgt)

            for j in range(len(class_list)):
                class_name = class_list[j]
                bbox       = bboxes[j]

                if not illumObject.in_illumObj_list(class_name):
                    div = 1
                    illumObj = illumObject.IllumObject(class_name, int(bbox.wth/div), int(bbox.hgt/div))
                    illumObj.reweight(img, bbox)
                else:
                    illumObj = illumObject.find_illumObj_with_class(class_name)
                    illumObj.reweight(img, bbox)


        for i in range(len(images)):
    
            start = time.time()

            img = images[i]
            file = files[i]
            wth, hgt = img.size
            print("Weighting Image [%d/%d]" % ((i+1), len(images)))

            hue_weight = np.zeros([hgt, wth], float)
            lum_weight = np.zeros([hgt, wth], float)
            sat_weight = np.zeros([hgt, wth], float)
            hue_weight, lum_weight, sat_weight = globalIllum.relight(img, hue_weight, lum_weight, sat_weight)

            class_list, confidences, bboxes = yolo_opencv.detect_objects(file)
            class_list, confidences, bboxes = yolo_opencv.cleanup(class_list, confidences, bboxes, wth, hgt)
            
            for j in range(len(class_list)):
                class_name = class_list[j]
                bbox       = bboxes[j]
                illumObj   = illumObject.find_illumObj_with_class(class_name)
                lum_weight = relighting_weights(img, illumObj, lum_weight, bbox)
            



            smooth_factor = 250
            hue_weight = smoothen(hue_weight, smooth_factor)
            lum_weight = smoothen(lum_weight, smooth_factor)
            sat_weight = smoothen(sat_weight, smooth_factor)


            res = relight(img, hue_weight, lum_weight, sat_weight, hls_steps)

            print("Weighting Done. Took {0:.2f}s. ".format(time.time() - start))

            show_plots = False
            if show_plots:
                f, axarr = plt.subplots(3, 1)
                axarr[0].imshow(hue_weight)
                axarr[1].imshow(lum_weight)
                axarr[2].imshow(sat_weight)
                axarr[0].get_xaxis().set_visible(False)
                axarr[0].get_yaxis().set_visible(False)
                axarr[1].get_xaxis().set_visible(False)
                axarr[1].get_yaxis().set_visible(False)
                axarr[2].get_xaxis().set_visible(False)
                axarr[2].get_yaxis().set_visible(False)
                plt.show()


            # returns segmentation plot of global hue classes
            show_class = False
            if show_class:
                img_seg = globalIllum.class_segment(img)
                plt.imshow(img_seg)
                plt.show()


            # returns segmentation plot of global hue class averages
            show_seg = False
            if show_seg:
                img_seg = globalIllum.segment(img)
                yaxis = math.floor(math.sqrt(len(images)))
                xaxis = math.ceil(len(images)/yaxis)

                xs = i % xaxis
                ys = int(i / xaxis)
                if(i == 0):
                    new_im = Image.new('RGB', (wth * xaxis, hgt * yaxis))
                new_im.paste(img_seg, (xs*wth,ys*hgt))
                if(i == len(images) - 1):
                    new_im.show()


            # returns comparison chart of global hues before and after relighting
            show_charts = False
            if show_charts:
                globalIllum.plot_chart_multi(img, res)

        

            images[i] = res

            # saves image to output directory
            save = True
            if save:
                res.save(output_dir + 'bottle-image-%d.jpg' % (i))


    for i in range(len(images)):
        file = files[i]
        print(file)


input_dir = "inputs/"
output_dir = "outputs/"
file_types = [".jpg"]
iters = 1
num_hues = 10
num_lums = 1
num_sats = 1
hls_steps = [1.0, 1.0, 1.0]

main(input_dir, output_dir, file_types, iters, num_hues, num_lums, num_sats, hls_steps)
