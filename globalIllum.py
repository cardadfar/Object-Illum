import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

import luminance


globalIllum_list = []

class globalIllum:
    '''      name: name of object
               hl: lower hue limit
               hu: upper hue limit
               ll: lower luminance limit
               lu: upper luminance limit
               sl: lower saturation limit
               su: upper saturation limit
            h_avg: average hue value
            l_avg: average luminance value
            s_avg: average saturation value
       num_pixels: number of pixels detected in group
    '''

    def __init__(self, name, hl, hu, ll, lu, sl, su):

        self.name = name
        self.hl = hl
        self.hu = hu
        self.ll = ll
        self.lu = lu
        self.sl = sl
        self.su = su
        self.h_avg = 0
        self.l_avg = 0
        self.s_avg = 0
        self.num_pixels = 0

        globalIllum_list.append(self)


    def add(self, avg_h, avg_l, avg_s, num_pixels):
        ''' adds avg hls values from image to global average bucket
                avg_h: average hue value from image range
                avg_l: average luminance value from image range
                avg_s: average saturation value from image range
            num_pixels: number of pixels detected in group
        '''

        total_pixels = self.num_pixels + num_pixels

        if(total_pixels != 0): 
            self.h_avg = (self.num_pixels / total_pixels) * self.h_avg + (num_pixels / total_pixels) * avg_h
            self.l_avg = (self.num_pixels / total_pixels) * self.l_avg + (num_pixels / total_pixels) * avg_l
            self.s_avg = (self.num_pixels / total_pixels) * self.s_avg + (num_pixels / total_pixels) * avg_s
        self.num_pixels = total_pixels


def print_globals():
    ''' prints global parameters (debug mode) '''

    for glo in globalIllum_list:
        print( str(glo.name) + ' | global hues [{0:.2f}, {1:.2f}] : {2:.2f}'.format(glo.hl, glo.hu, glo.h_avg))
        print( str(glo.name) + ' | global lums: [{0:.2f}, {1:.2f}] : {2:.2f}'.format(glo.ll, glo.lu, glo.l_avg))
        print( str(glo.name) + ' | global sats: [{0:.2f}, {1:.2f}] : {2:.2f}'.format(glo.sl, glo.su, glo.s_avg))


def clear():
    ''' clears global parameters '''

    for glo in globalIllum_list:
        glo.h_avg = 0
        glo.l_avg = 0
        glo.s_avg = 0
        glo.num_pixels = 0


def cluster(img, num_hues, iters):
    ''' finds optimal partitioning of hues from img using k-means
             img: image to sample pixels from
        num_hues: number of hue buckets to generate
           iters: iterations to run k-means for
    '''

    img_pixels = np.array(img)
    hgt, wth, c = img_pixels.shape
    img_hls  = np.zeros((hgt, wth, c), float)
    img_seg  = np.zeros((hgt, wth, c), float)

    for y in range(hgt):
        for x in range(wth):
            img_hls[y,x] = luminance.rgb_to_hls(img_pixels[y,x])

    img_temp = img_hls.transpose()
    img_hue_temp = img_temp[0]
    img_lum_temp = img_temp[1]
    img_sat_temp = img_temp[2]
    img_hues = img_hue_temp.transpose()
    img_lums = img_lum_temp.transpose()
    img_sats = img_sat_temp.transpose()

    img_hues = img_hues.flatten()
    img_hues = np.reshape(img_hues, (1, len(img_hues)))
    img_hues = np.tile(img_hues.T, (1,2))

    kmeans = KMeans(n_clusters=num_hues, init='k-means++', max_iter=iters).fit(img_hues)
    centers = kmeans.cluster_centers_
    centers = (centers.T)[0]
    centers = np.sort(centers)

    bounds = np.zeros((num_hues,2), float)

    for i in range(num_hues):
        if (i == 0):
            bounds[i,0] = 0.0001
            bounds[i,1] = (centers[i  ] + centers[i+1]) / 2.0
        elif (i == num_hues-1):
            bounds[i,0] = (centers[i-1] + centers[i  ]) / 2.0
            bounds[i,1] = 1.0
        else:
            bounds[i,0] = (centers[i-1] + centers[i  ]) / 2.0
            bounds[i,1] = (centers[i  ] + centers[i+1]) / 2.0


    return bounds



def segment(img):
    ''' creates segmentation plot of global hue class averages
             img: PIL image
    '''

    img_pixels = np.array(img)
    hgt, wth, c = img_pixels.shape
    img_hls  = np.zeros((hgt, wth, c), float)
    img_seg  = np.zeros((hgt, wth, c), float)

    for y in range(hgt):
        for x in range(wth):
            img_hls[y,x] = luminance.rgb_to_hls(img_pixels[y,x])

    img_temp = img_hls.transpose()
    img_hue_temp = img_temp[0]
    img_lum_temp = img_temp[1]
    img_sat_temp = img_temp[2]
    img_hues = img_hue_temp.transpose()
    img_lums = img_lum_temp.transpose()
    img_sats = img_sat_temp.transpose()

    for glo in globalIllum_list:

        hl = glo.hl
        hu = glo.hu
        ll = glo.ll
        lu = glo.lu
        sl = glo.sl
        su = glo.su

        global_hue = glo.h_avg
        global_lum = glo.l_avg
        global_sat = glo.s_avg

        glo_rgb = luminance.hls_to_rgb([global_hue, global_lum, global_sat])

        hues_thres = np.copy(img_hues)
        lums_thres = np.copy(img_lums)
        sats_thres = np.copy(img_sats)

        hues_thres[hues_thres < hl] = 0
        hues_thres[hues_thres > hu] = 0
        hues_thres[lums_thres < ll] = 0
        hues_thres[lums_thres > lu] = 0
        hues_thres[sats_thres < sl] = 0
        hues_thres[sats_thres > su] = 0

        hues_indices = (hues_thres != 0)
        img_seg[hues_indices] = glo_rgb


    seg = Image.fromarray(np.uint8(img_seg))
    return seg



def class_segment(img):
    ''' creates segmentation plot of global hue classes
             img: PIL image
    '''

    img_pixels = np.array(img)
    hgt, wth, c = img_pixels.shape
    img_hls  = np.zeros((hgt, wth, c), float)
    img_seg  = np.zeros((hgt, wth   ), float)

    for y in range(hgt):
        for x in range(wth):
            img_hls[y,x] = luminance.rgb_to_hls(img_pixels[y,x])

    img_temp = img_hls.transpose()
    img_hue_temp = img_temp[0]
    img_lum_temp = img_temp[1]
    img_sat_temp = img_temp[2]
    img_hues = img_hue_temp.transpose()
    img_lums = img_lum_temp.transpose()
    img_sats = img_sat_temp.transpose()

    for i in range(len(globalIllum_list)):

        glo = globalIllum_list[i]

        hl = glo.hl
        hu = glo.hu
        ll = glo.ll
        lu = glo.lu
        sl = glo.sl
        su = glo.su

        global_hue = glo.h_avg
        global_lum = glo.l_avg
        global_sat = glo.s_avg

        glo_rgb = luminance.hls_to_rgb([global_hue, global_lum, global_sat])

        hues_thres = np.copy(img_hues)
        lums_thres = np.copy(img_lums)
        sats_thres = np.copy(img_sats)

        hues_thres[hues_thres < hl] = 0
        hues_thres[hues_thres > hu] = 0
        hues_thres[lums_thres < ll] = 0
        hues_thres[lums_thres > lu] = 0
        hues_thres[sats_thres < sl] = 0
        hues_thres[sats_thres > su] = 0

        hues_indices = (hues_thres != 0)
        img_seg[hues_indices] = i

    return img_seg


def plot_chart(img):
    ''' comparison chart of global hues of img
             img: PIL image
    '''

    img_pixels = np.array(img)
    hgt, wth, c = img_pixels.shape
    img_hls  = np.zeros((hgt, wth, c), float)

    for y in range(hgt):
        for x in range(wth):
            img_hls[y,x] = luminance.rgb_to_hls(img_pixels[y,x])

    img_temp = img_hls.transpose()
    img_hue_temp = img_temp[0]
    img_lum_temp = img_temp[1]
    img_sat_temp = img_temp[2]
    img_hues = img_hue_temp.transpose()
    img_lums = img_lum_temp.transpose()
    img_sats = img_sat_temp.transpose()


    offset = 15
    patchSize = 100
    patchHalfsize = patchSize / 2
    width = offset + (patchSize + offset) * 6
    height = offset + (patchSize + offset) * 4
    im = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    for i in range(len(globalIllum_list)):

        glo = globalIllum_list[i]

        hl = glo.hl
        hu = glo.hu
        ll = glo.ll
        lu = glo.lu
        sl = glo.sl
        su = glo.su

        global_hue = glo.h_avg
        global_lum = glo.l_avg
        global_sat = glo.s_avg

        hues_thres = np.copy(img_hues)
        lums_thres = np.copy(img_lums)
        sats_thres = np.copy(img_sats)

        hues_thres[hues_thres < hl] = 0
        hues_thres[hues_thres > hu] = 0
        hues_thres[lums_thres < ll] = 0
        hues_thres[lums_thres > lu] = 0
        hues_thres[sats_thres < sl] = 0
        hues_thres[sats_thres > su] = 0

        lums_thres[hues_thres < hl] = 0
        lums_thres[hues_thres > hu] = 0
        lums_thres[lums_thres < ll] = 0
        lums_thres[lums_thres > lu] = 0
        lums_thres[sats_thres < sl] = 0
        lums_thres[sats_thres > su] = 0

        sats_thres[hues_thres < hl] = 0
        sats_thres[hues_thres > hu] = 0
        sats_thres[lums_thres < ll] = 0
        sats_thres[lums_thres > lu] = 0
        sats_thres[sats_thres < sl] = 0
        sats_thres[sats_thres > su] = 0

        num_pixels = np.count_nonzero(hues_thres.flatten())
        avg_hue    = np.sum(hues_thres.flatten())
        avg_lum    = np.sum(lums_thres.flatten())
        avg_sat    = np.sum(sats_thres.flatten())


        if(num_pixels != 0):
            avg_hue /= num_pixels
            avg_lum /= num_pixels
            avg_sat /= num_pixels


        avg_rgb = luminance.hls_to_rgb([avg_hue, avg_lum, avg_sat])
        glo_rgb = luminance.hls_to_rgb([global_hue, global_lum, global_sat])

        ix = i % 6
        iy = int(i / 6)
        rx = offset + (patchSize + offset) * ix
        ry = offset + (patchSize + offset) * iy

        draw.rectangle((rx, ry, rx + patchSize, ry + patchHalfsize),
                       fill=(int(avg_rgb[0]),
                             int(avg_rgb[1]),
                             int(avg_rgb[2])))
        draw.rectangle((rx, ry + patchHalfsize, rx + patchSize, ry + patchSize),
                       fill=(int(glo_rgb[0]),
                             int(glo_rgb[1]),
                             int(glo_rgb[2])))
        draw.multiline_text((rx + patchHalfsize - 20, ry + 2 + patchSize),
                            '[{0:.2f}, {1:.2f}]'.format(hl, hu),
                            fill=(0, 0, 0))


    im.show()
    return im


def plot_chart_multi(img1, img2):
    ''' comparison chart of global hues between img1 and img2
             img1: PIL image
             img2: PIL image
    '''

    img1_pixels = np.array(img1)
    img2_pixels = np.array(img2)
    hgt, wth, c = img1_pixels.shape
    img1_hls  = np.zeros((hgt, wth, c), float)
    img2_hls  = np.zeros((hgt, wth, c), float)

    for y in range(hgt):
        for x in range(wth):
            img1_hls[y,x] = luminance.rgb_to_hls(img1_pixels[y,x])
            img2_hls[y,x] = luminance.rgb_to_hls(img2_pixels[y,x])

    img1_temp = img1_hls.transpose()
    img1_hue_temp = img1_temp[0]
    img1_lum_temp = img1_temp[1]
    img1_sat_temp = img1_temp[2]
    img1_hues = img1_hue_temp.transpose()
    img1_lums = img1_lum_temp.transpose()
    img1_sats = img1_sat_temp.transpose()

    img2_temp = img2_hls.transpose()
    img2_hue_temp = img2_temp[0]
    img2_lum_temp = img2_temp[1]
    img2_sat_temp = img2_temp[2]
    img2_hues = img2_hue_temp.transpose()
    img2_lums = img2_lum_temp.transpose()
    img2_sats = img2_sat_temp.transpose()


    offset = 15
    patchSize = 100
    patchHalfsize = patchSize / 2
    width = offset + (patchSize + offset) * 6
    height = offset + (patchSize + offset) * 5
    im = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    for i in range(len(globalIllum_list)):

        glo = globalIllum_list[i]

        hl = glo.hl
        hu = glo.hu
        ll = glo.ll
        lu = glo.lu
        sl = glo.sl
        su = glo.su

        global_hue = glo.h_avg
        global_lum = glo.l_avg
        global_sat = glo.s_avg


        hues1_thres = np.copy(img1_hues)
        lums1_thres = np.copy(img1_lums)
        sats1_thres = np.copy(img1_sats)

        hues2_thres = np.copy(img2_hues)
        lums2_thres = np.copy(img2_lums)
        sats2_thres = np.copy(img2_sats)

        avg1_hue = 0
        avg1_lum = 0
        avg1_sat = 0
        num1_pixels = 0
        avg2_hue = 0
        avg2_lum = 0
        avg2_sat = 0
        num2_pixels = 0

        hues1_thres[hues1_thres < hl] = 0
        hues1_thres[hues1_thres > hu] = 0
        hues1_thres[lums1_thres < ll] = 0
        hues1_thres[lums1_thres > lu] = 0
        hues1_thres[sats1_thres < sl] = 0
        hues1_thres[sats1_thres > su] = 0

        lums1_thres[hues1_thres < hl] = 0
        lums1_thres[hues1_thres > hu] = 0
        lums1_thres[lums1_thres < ll] = 0
        lums1_thres[lums1_thres > lu] = 0
        lums1_thres[sats1_thres < sl] = 0
        lums1_thres[sats1_thres > su] = 0

        sats1_thres[hues1_thres < hl] = 0
        sats1_thres[hues1_thres > hu] = 0
        sats1_thres[lums1_thres < ll] = 0
        sats1_thres[lums1_thres > lu] = 0
        sats1_thres[sats1_thres < sl] = 0
        sats1_thres[sats1_thres > su] = 0

        hues2_thres[hues2_thres < hl] = 0
        hues2_thres[hues2_thres > hu] = 0
        hues2_thres[lums2_thres < ll] = 0
        hues2_thres[lums2_thres > lu] = 0
        hues2_thres[sats2_thres < sl] = 0
        hues2_thres[sats2_thres > su] = 0

        lums2_thres[hues2_thres < hl] = 0
        lums2_thres[hues2_thres > hu] = 0
        lums2_thres[lums2_thres < ll] = 0
        lums2_thres[lums2_thres > lu] = 0
        lums2_thres[sats2_thres < sl] = 0
        lums2_thres[sats2_thres > su] = 0

        sats2_thres[hues2_thres < hl] = 0
        sats2_thres[hues2_thres > hu] = 0
        sats2_thres[lums2_thres < ll] = 0
        sats2_thres[lums2_thres > lu] = 0
        sats2_thres[sats2_thres < sl] = 0
        sats2_thres[sats2_thres > su] = 0

        num1_pixels = np.count_nonzero(hues1_thres.flatten())
        avg1_hue    = np.sum(hues1_thres.flatten())
        avg1_lum    = np.sum(lums1_thres.flatten())
        avg1_sat    = np.sum(sats1_thres.flatten())

        num2_pixels = np.count_nonzero(hues2_thres.flatten())
        avg2_hue    = np.sum(hues2_thres.flatten())
        avg2_lum    = np.sum(lums2_thres.flatten())
        avg2_sat    = np.sum(sats2_thres.flatten())


        if(num1_pixels != 0):
            avg1_hue /= num1_pixels
            avg1_lum /= num1_pixels
            avg1_sat /= num1_pixels


        if(num2_pixels != 0):
            avg2_hue /= num2_pixels
            avg2_lum /= num2_pixels
            avg2_sat /= num2_pixels


        avg1_rgb = luminance.hls_to_rgb([avg1_hue, avg1_lum, avg1_sat])
        avg2_rgb = luminance.hls_to_rgb([avg2_hue, avg2_lum, avg2_sat])
        glo_rgb = luminance.hls_to_rgb([global_hue, global_lum, global_sat])

        ix = i % 6
        iy = int(i / 6)
        rx = offset + (patchSize + offset) * ix
        ry = offset + (patchHalfsize + patchSize + offset) * iy

        draw.rectangle((rx, ry, rx + patchSize, ry + patchHalfsize),
                       fill=(int(avg1_rgb[0]),
                             int(avg1_rgb[1]),
                             int(avg1_rgb[2])))

        
        draw.rectangle((rx, ry + patchHalfsize, rx + patchSize, ry + patchSize),
                       fill=(int(avg2_rgb[0]),
                             int(avg2_rgb[1]),
                             int(avg2_rgb[2])))

        draw.rectangle((rx, ry + patchSize, rx + patchSize, ry + patchSize + patchHalfsize),
                       fill=(int(glo_rgb[0]),
                             int(glo_rgb[1]),
                             int(glo_rgb[2])))

        draw.multiline_text((rx + patchHalfsize - 20, ry + 2 + patchSize + patchHalfsize),
                            '[{0:.2f}, {1:.2f}]'.format(hl, hu),
                            fill=(0, 0, 0))


    im.show()
    return im




def reweight(img):
    ''' reweight global averages ber hue range
            img: PIL image
    '''

    img_pixels = np.array(img)
    hgt, wth, c = img_pixels.shape
    img_hls  = np.zeros((hgt, wth, c), float)
    img_hues = np.zeros((hgt, wth   ), float)
    img_lums = np.zeros((hgt, wth   ), float)
    img_sats = np.zeros((hgt, wth   ), float)

    for y in range(hgt):
        for x in range(wth):
            img_hls[y,x] = luminance.rgb_to_hls(img_pixels[y,x])

    img_temp = img_hls.transpose()
    img_hue_temp = img_temp[0]
    img_lum_temp = img_temp[1]
    img_sat_temp = img_temp[2]
    img_hues = img_hue_temp.transpose()
    img_lums = img_lum_temp.transpose()
    img_sats = img_sat_temp.transpose()

    for i in range(len(globalIllum_list)):

        glo = globalIllum_list[i]

        hl = glo.hl
        hu = glo.hu
        ll = glo.ll
        lu = glo.lu
        sl = glo.sl
        su = glo.su

        hues_thres = np.copy(img_hues)
        lums_thres = np.copy(img_lums)
        sats_thres = np.copy(img_sats)

        
        avg_hue = 0
        avg_lum = 0
        avg_sat = 0
        num_pixels = 0

        if (True):

            hues_thres = hues_thres.flatten()
            lums_thres = lums_thres.flatten()
            sats_thres = sats_thres.flatten()

            hues_thres[hues_thres < hl] = 0
            hues_thres[hues_thres > hu] = 0
            hues_thres[lums_thres < ll] = 0
            hues_thres[lums_thres > lu] = 0
            hues_thres[sats_thres < sl] = 0
            hues_thres[sats_thres > su] = 0

            lums_thres[hues_thres < hl] = 0
            lums_thres[hues_thres > hu] = 0
            lums_thres[lums_thres < ll] = 0
            lums_thres[lums_thres > lu] = 0
            lums_thres[sats_thres < sl] = 0
            lums_thres[sats_thres > su] = 0

            sats_thres[hues_thres < hl] = 0
            sats_thres[hues_thres > hu] = 0
            sats_thres[lums_thres < ll] = 0
            sats_thres[lums_thres > lu] = 0
            sats_thres[sats_thres < sl] = 0
            sats_thres[sats_thres > su] = 0
            

            num_pixels = np.count_nonzero(hues_thres)
            avg_hue    = np.sum(hues_thres)
            avg_lum    = np.sum(lums_thres)
            avg_sat    = np.sum(sats_thres)

            

            if(num_pixels != 0):
                avg_hue /= num_pixels
                avg_lum /= num_pixels
                avg_sat /= num_pixels
                glo.add(avg_hue, avg_lum, avg_sat, num_pixels)

        else:

            for j in range(hgt):
                for i in range(wth):
                    if (hues_thres[j,i] < hu):
                        if (lums_thres[j,i] > ll) and (lums_thres[j,i] < lu):
                            if (sats_thres[j,i] > sl) and (sats_thres[j,i] < su):
                                avg_hue += hues_thres[j,i]
                                avg_lum += lums_thres[j,i]
                                avg_sat += sats_thres[j,i]
                                num_pixels += 1

            if(num_pixels != 0):
                avg_hue /= num_pixels
                avg_lum /= num_pixels
                avg_sat /= num_pixels
                glo.add(avg_hue, avg_lum, avg_sat, num_pixels)
        


def relight(img, hue_weight, lum_weight, sat_weight):
    ''' calculate relighting values for hue, lum, and sat and store them in corresponding weights
               img: PIL image
        hue_weight: 2D array storing hue relighting weights
        lum_weight: 2D array storing luminance relighting weights
        sat_weight: 2D array storing saturation relighting weights
    '''

    img_pixels = np.array(img)
    hgt, wth, c  = img_pixels.shape
    img_hls  = np.zeros((hgt, wth, c), float)
    img_hues = np.zeros((hgt, wth   ), float)
    img_lums = np.zeros((hgt, wth   ), float)
    img_sats = np.zeros((hgt, wth   ), float)

    for y in range(hgt):
        for x in range(wth):
            img_hls[y,x] = luminance.rgb_to_hls(img_pixels[y,x])

    img_temp = img_hls.transpose()
    img_hue_temp = img_temp[0]
    img_lum_temp = img_temp[1]
    img_sat_temp = img_temp[2]
    img_hues = img_hue_temp.transpose()
    img_lums = img_lum_temp.transpose()
    img_sats = img_sat_temp.transpose()

    for i in range(len(globalIllum_list)):

        glo = globalIllum_list[i]

        hl = glo.hl
        hu = glo.hu
        ll = glo.ll
        lu = glo.lu
        sl = glo.sl
        su = glo.su

        global_hue = glo.h_avg
        global_lum = glo.l_avg
        global_sat = glo.s_avg

        avg_hue = 0
        avg_lum = 0
        avg_sat = 0
        num_pixels = 0

        hues_thres = np.copy(img_hues)
        lums_thres = np.copy(img_lums)
        sats_thres = np.copy(img_sats)

        if (True):

            hues_thres[hues_thres < hl] = 0
            hues_thres[hues_thres > hu] = 0
            hues_thres[lums_thres < ll] = 0
            hues_thres[lums_thres > lu] = 0
            hues_thres[sats_thres < sl] = 0
            hues_thres[sats_thres > su] = 0

            lums_thres[hues_thres < hl] = 0
            lums_thres[hues_thres > hu] = 0
            lums_thres[lums_thres < ll] = 0
            lums_thres[lums_thres > lu] = 0
            lums_thres[sats_thres < sl] = 0
            lums_thres[sats_thres > su] = 0

            sats_thres[hues_thres < hl] = 0
            sats_thres[hues_thres > hu] = 0
            sats_thres[lums_thres < ll] = 0
            sats_thres[lums_thres > lu] = 0
            sats_thres[sats_thres < sl] = 0
            sats_thres[sats_thres > su] = 0

            num_pixels = np.count_nonzero(hues_thres.flatten())
            avg_hue    = np.sum(hues_thres.flatten())
            avg_lum    = np.sum(lums_thres.flatten())
            avg_sat    = np.sum(sats_thres.flatten())


            if(num_pixels != 0):
                avg_hue /= num_pixels
                avg_lum /= num_pixels
                avg_sat /= num_pixels
                hues_indices = (hues_thres != 0)
                hue_weight[hues_indices] = avg_hue - global_hue
                lum_weight[hues_indices] = avg_lum - global_lum
                sat_weight[hues_indices] = avg_sat - global_sat

        else:
            
            for j in range(hgt):
                for i in range(wth):
                    if (hues_thres[j,i] < hu):
                        if (lums_thres[j,i] > ll) and (lums_thres[j,i] < lu):
                            if (sats_thres[j,i] > sl) and (sats_thres[j,i] < su):
                                avg_hue += hues_thres[j,i]
                                avg_lum += lums_thres[j,i]
                                avg_sat += sats_thres[j,i]
                                num_pixels += 1

            if(num_pixels != 0):
                avg_hue /= num_pixels
                avg_lum /= num_pixels
                avg_sat /= num_pixels
                hues_indices = (hues_thres != 0)
                hue_weight[hues_indices] = avg_hue - global_hue
                lum_weight[hues_indices] = avg_lum - global_lum
                sat_weight[hues_indices] = avg_sat - global_sat

    return hue_weight, lum_weight, sat_weight

