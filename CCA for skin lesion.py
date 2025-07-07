import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
import PIL
from PIL import Image
import props as props
import matplotlib.image as mpimg
import skimage
from skimage.measure import label,regionprops
from openpyxl.styles.builtins import output
img=cv2.imread('C:/Users/dell/PycharmProjects/img processing/50.bmp', 0)

def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255

    def numberobjects(img2):
        size = np.shape(img2)
        named_1 = np.zeros(size, dtype=np.int)
        label = 1
        for i in range(1, size[0]):
            for j in range(1, size[1]):
                if (img2[i, j] == 255):
                    upperlabel = named_1[i - 1, j]
                    prevlabel = named_1[i, j - 1]
        thislabel = upperlabel
        if (prevlabel != 0):
            thislabel = prevlabel
        if (thislabel == 0):
            thislabel = label
        label = label + 1
        named_1[i, j] = thislabel
        return named_1

    def listinlistcopy(lista, dstloc, srcloc):
        while (lista[srcloc].__len__() > 0):
            lista[dstloc].append(lista[srcloc].pop())
            lista.__delitem__(srcloc)

    def findindex(lista, value):
        r = -1
        for k in range(lista.__len__()):
            if (lista[k].count(value) > 0):
                r = k
            break
        return r

    def ifexists(lista, value):
        for k in range(lista.__len__()):
            if (lista[k].count(value) > 0):
                return 1

    def secondpass(img2):
        list_objects = list([[0]])
        size = np.shape(img2)
        for i in range(1, size[0]):
            for j in range(1, size[1]):
                currvalue = img2[i, j]
        if (currvalue != 0):
            uppervalue = img2[i - 1, j]
            leftvalue = img2[i, j - 1]
        k = findindex(list_objects, currvalue)
        if (k == -1):
            list_objects.append([currvalue])
        elif (uppervalue != 0 and leftvalue != 0 and uppervalue != leftvalue):
            upperindex = findindex(list_objects, uppervalue)
            leftindex = findindex(list_objects, leftvalue)
            if (upperindex != leftindex):
                listinlistcopy(list_objects, upperindex, leftindex)
        return list_objects

    def colorobjects(img2, list_objects):
        size = np.shape(img2)
        newimg = np.ones((size), dtype=np.uint8) * 255
        for i in range(size[0]):
            for j in range(size[1]):
                if (img2[i, j] != 0):
                    res = findindex(list_objects, img2[i, j])
                    if (res != -1):
                        newimg[i, j] = res * 40
        return newimg

    def circularity(img2):
        img1 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        newImg = cv2.resize(img1, (0, 0), fx=0.50, fy=0.50)
        gray = cv2.cvtColor(newImg, cv2.COLOR_RGB2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        (contours, hierarchy) = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        newImg2 = cv2.drawContours(newImg, contours, -1, (0, 255, 0), 2)
        height = np.size(newImg2, 0)
        width = np.size(newImg2, 1)
        list_of_area = []
        label_img = skimage.measure.label(newImg2)
        regions = skimage.measure.regionprops(label_img)
        for props in regions:
            a = props.area
            list_of_area.append(a)
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        l = []
        list_of_perimeter = []
        square1 = []
        list_of_area2 = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            k = len(approx)
            l.append(k)
        # list of edges
        size_of_list = len(l)
        for i in l:
            mul = size_of_list * l[i]
            list_of_perimeter.append(mul)
        for i in range(0, len(list_of_perimeter)):
            perimeter1 = list_of_perimeter[i] ^ 2
            # having p2 value
            square1.append(perimeter1)
        for i in range(0, len(list_of_area)):
            p1 = 4 * 3.14 * list_of_area[i]
            # having p1 value
            list_of_area2.append(p1)
        cir = [i / j for i, j in zip(square1, list_of_area2)]
        count = 0
        Empty1 = []
        for i in range(len(cir)):
            count = count + list_of_area2[i]
            Empty1.append(count)
        fig, ax = plt.subplots()
        ax.scatter(Empty1, cir,  alpha=0.6)
        ax.set_title('Scatter plot of x-y pairs')
        ax.set_xlabel('Area value')
        ax.set_ylabel('Circularity value')
        plt.show()
    afterPass1 = numberobjects(img)
    listobjects = secondpass(afterPass1)
    print('number of objects')
    print(len(listobjects) - 1)
    showimage = (colorobjects(afterPass1, listobjects))
    plt.imshow(img2)
    plt.show()
    circularity(img2)
    return img2
removeSmallComponents(img,250)