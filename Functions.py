import numpy as np
import skimage
import skimage.util
from skimage import measure
from skimage import exposure, img_as_ubyte
from skimage import filters, draw
from skimage import data, io, util
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.draw import polygon
from image_registration import cross_correlation_shifts
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.ndimage import shift
from skimage.segmentation import active_contour
from statistics import mean
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d
from scipy.spatial import distance
from scipy.signal import convolve2d
from scipy.optimize import minimize
import cv2
import math
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageFont, ImageDraw, ImageTk, ImageFilter
from matplotlib import cm
import multiprocessing
import os, sys
import tkinter as tk
from scipy import interpolate
import time
from shapely.geometry import Polygon
from distancemap import distance_map_from_binary_matrix
from tkinter import filedialog
from tkinter import messagebox
import functools
from functools import partial
from scipy.optimize import curve_fit
from scipy.integrate import quad
from matplotlib.widgets import Slider




def MatToUint8(Image):
    #Convert an image in the Uint8 Type, the input necessary to the cv2 library functions
    Image = img_as_ubyte(exposure.rescale_intensity(Image))
    return Image

# python function replica of matlab's mat2gray
def Mat2gray(A):
    norm = cv2.normalize(A, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm

def Bwareafilt(input_mask):
    #Keeps the largest white area in a mask
    labels_mask = measure.label(input_mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
       for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    mask = labels_mask
    return mask


def Imadjust (Im):
    #Imadjust saturates the bottom 1% and the top 1% of all pixel values.
    #This operation increases the contrast of the output image

    p1, p99 = np.percentile(Im, (1, 99))
    Im_adjusted=skimage.exposure.rescale_intensity(Im, in_range=(p1, p99))

    #fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Istogramma dell'immagine originale
    #axs[0].hist(Im.ravel(), bins=256, color='blue', alpha=0.7)
    #axs[0].axvline(x=p1, color='red', linestyle='--', alpha=0.5,label='1° percentile')
    #axs[0].axvline(x=p99, color='red', linestyle='--', alpha=0.5, label='99° percentile')
    #axs[0].set_title('Original histogram')
    #axs[0].legend()
    #axs[0].set_xlim(-0.01, 1)  # Imposta i limiti dell'asse X tra 0 e 1
    #axs[0].set_ylim(0, 27000)
    #axs[0].set_xlabel('Intensity')
    #axs[0].set_ylabel('Frequency')


    # Istogramma dell'immagine dopo il riscalamento
    #axs[1].hist(Im_adjusted.ravel(), bins=256, color='green', alpha=0.7)
    #axs[1].set_title('Stretched histogram')
    #axs[1].set_xlim(-0.01, 1)  # Imposta i limiti dell'asse X tra 0 e 1
    #axs[1].set_ylim(0, 27000)
    #axs[1].set_xlabel('Intensity')
    #axs[1].set_ylabel('Frequency')

    #plt.show()

    return Im_adjusted

def intersect_mtlb(a, b):
    c = []
    index_a = []
    index_b_temp = []
    i=0
    while i < len(a):
        if a[i] in b:
            c.append(a[i])
            index_a.append(i)
            index_b_temp.append(b.index(a[i]))
        i=i+1

    c.sort()
    index_b=[]
    for point in index_b_temp:
        if point not in index_b:
            index_b.append(point)

    return c,index_a,index_b






def Border_control (Im, Th):
    # Function that identifies the columns on the right and left sides of image not belonging to chest (arms)

    # Input:
    # -Im = Grey_scale_image
    # -Th = manual, threshold

    # Output:
    # BWt: binary image after manual threshold application
    # i1mask: vector containing the number of columns on the right side not belonging to chest
    # iendmask: vector containing the number of columns on the left side not belonging to chest

    #Manual  threshold for background
    BW = Im > Th
    seed = skimage.morphology.rectangle(10,1)
    BWt = skimage.morphology.closing(BW, seed)
    BWt = ndimage.binary_fill_holes(BWt)
    Number_of_pixels = np.count_nonzero(BWt)

    # if pixel number is >= 20000 (large chest image: it valuates only the first and last 20 columns)
    # if pixel number is < 20000 (small chest image: it valuates the first 30 and the last 25 columns)

    if Number_of_pixels >= 20000:
        a1 = 20
        a2 = 20
    else:
        a1 = 30
        a2 = 25

    # Right side control
    i = 0
    right_pixels = []
    while i < a1:
        column = BWt[:, i]
        right_pixels.append(np.count_nonzero(column))
        i = i + 1

    right_pixels_diff = np.diff(right_pixels)
    positive_r = np.array(right_pixels_diff >= 0, int)
    i_right_pixels_diff = np.concatenate(([0], positive_r))
    # it finds the index corresponding yo the last column whose length is lower than previous ones
    i1mask = np.argwhere(i_right_pixels_diff == 0).max()


    # Left side control (same process)
    i = 0
    left_pixels = []
    while i < a2:
        column = BWt[:, -1-i]
        left_pixels.append(np.count_nonzero(column))
        i = i + 1

    left_pixels_diff = np.diff(left_pixels)
    positive_l = np.array(left_pixels_diff >= 0, int)
    i_left_pixels_diff = np.concatenate(([0], positive_l))
    iendmask = np.argwhere(i_left_pixels_diff == 0).max()

    return BWt, i1mask, iendmask



def maskborder(Im , Bw , i1 , iend):
     # function for the definition and application of mask that deletes pixel on the image borders not belonging to the chest

     #inputs:
     # -Im: grey-scale image
     #- BW: binary image after manual thresholding
     #- i1: vector containing the number of columns on the right side not belonging to chest
     #- iend: vector containing the number of columns on the left side not belonging to chest

     #outputs:
     #- Im_filt: grey-scale image after mask application
     #- BW_filt: binary image after mask application
     #- yhalf: y position of the point located in the half of the image (horizontal line)
     #- xhalf: x position of the point located in the half of the image (vertical line)

     # mean of number of columns(found for each slice) that have to be deleted
     m_1_mask=  int(np.floor(np.mean(i1)))
     m_end_mask = int(np.floor(np.mean(iend)))

     # mask definition
     x_dim = np.size(Im,0)
     y_dim = np.size(Im,1)
     mask = np.ones((x_dim, y_dim))
     mask[:,0:m_1_mask]=0
     mask[:,y_dim-m_end_mask:y_dim]=0


     BW_filt = cv2.bitwise_and(MatToUint8(Bw),MatToUint8(Bw), mask=MatToUint8(mask))
     BW_filt = Bwareafilt(BW_filt)


     #Coordinates of the half point of an image
     ones_coordinates = np.nonzero(BW_filt)
     y = ones_coordinates[0]
     x = ones_coordinates[1]

     x_half= np.round((np.max(x)+np.min(x))/2)
     y_half= np.round((np.max(y)+np.min(y))/2)

     #Application of the mask to gray-scale image
     BW_filt = BW_filt.astype(np.uint16)
     Im_filt= cv2.bitwise_and(MatToUint8(Im),MatToUint8(Im),mask = MatToUint8(BW_filt))
     Im_filt=Mat2gray(Im_filt)

     # Disegna il primo subplot
     #plt.figure()
     #plt.imshow(MatToUint8(Im),'gray')
     #plt.axis('off')
     #plt.draw()

     # Disegna il secondo subplot
     #plt.figure()
     #plt.imshow(MatToUint8(Bw),'gray')
     #plt.axis('off')
     #plt.draw()


     # Disegna il terzo subplot
     #plt.figure()
     #plt.imshow(MatToUint8(mask),'gray')
     #plt.axis('off')
     #plt.draw()


     # Disegna il quarto subplot
     #plt.figure()
     #plt.imshow(MatToUint8(BW_filt),'gray')
     #plt.axis('off')
     #plt.draw()

     # Mostra il subplot


     #plt.figure()
     #plt.imshow(Im_filt,'gray')
     #plt.axis('off')
     #plt.show()

     return Im_filt, BW_filt, x_half, y_half


def smooth (y):
   # The function calculate the moving average, with a window of 5, useful for histogram signal
   yy=[]
   yy.append(y[0])
   yy.append((y[0]+y[1]+y[2])/3)
   i=2
   while i <= len(y)-3:
       yy.append((y[i-2]+y[i-1]+y[i]+y[i+1]+y[i+2])/5)
       i=i+1
   yy.append((y[-2]+y[-3]+y[-1])/3)
   yy.append(y[-1])
   return yy




def Hist_threshold (c,binlocations,greyvaluemin,greyvaluemax):
  # The function identifies the concavity surrounding the two main
  # peaks of the histogram by looking for the position of maximum divergence
  # between the histogram and a Guassian fitting.
  # The function can be applied both to eliminate background pixels and
  # than to identify the threshold for lung segmentation, depending on
  # of the greyvaluemax and greyvaluemin values entered in input

  #Inputs:
  # counts: counts resulting from image histogram
  # binlocations: grey values resulting from image histogram
  # greyvaluemin: minimum grey value considered by function
  # greyvaluemax: maximum grey value considered by function

  #Outputs:
  # l: threshold resulting from histogram partitioning method
  # iltot: index in binilocation matrix corresponding to threshold value



  #Counts preparation
  c = np.array(smooth(c))

  #Elimination of the element 0
  a=np.where(c == 0)[0]
  c = np.delete(c, a)
  binlocations=np.array(binlocations)
  binlocations = np.delete(binlocations, a)

  iselected = np.where((binlocations >= greyvaluemin) & (binlocations <= greyvaluemax))[0]
  counts1 = c[iselected]
  binlocations1 = binlocations[iselected]
  counts1 = np.array(counts1)
  binlocations1 = np.array(binlocations1)

  #definition of a Gaussian curve in the same range of histogram grey value
  #Mean value
  countstot=np.sum(counts1)
  mmatrix = np.multiply(binlocations1,counts1)
  m = np.sum(mmatrix) / countstot
  #Variance
  variancematrix= np.multiply(np.multiply(binlocations1-m,binlocations1-m),counts1)
  variance = np.sum(variancematrix) / countstot
  std = math.sqrt(variance)

  x1lim = m - std
  x2lim = m + std



  #Gaussian equation. Gaussian has the same area as the histogram
  y = norm.pdf(binlocations1, m, std)
  sumy = np.sum(y)
  y1 = (countstot / sumy) * y



  #plt.figure()
  #plt.plot(binlocations1,counts1,color='blue')
  #plt.axvline(x=x1lim,color='orange', linestyle='--')
  #plt.axvline(x=x2lim,color='orange', linestyle='--')
  #plt.axvline(x=m,color='black', linestyle='--')
  #plt.xlim(greyvaluemin,greyvaluemax)
  #plt.plot(binlocations1,y1,color='red')
  #plt.xlabel('gray intensity')  # Modifica l'etichetta dell'asse x
  #plt.ylabel('counts')



  # indices of values included around the mean value area
  n = np.where((binlocations1 >= x1lim) & (binlocations1 <= x2lim))[0]

  # gaussian curve has the same area under the curve as the one under the histogram
  d = y1[n] - counts1[n]
  il = np.argmax(d)
  iltot = il + n[0] - 1

  #Threshold value
  l = binlocations1[iltot]

  #plt.axvline(x=l, color='green')
  #plt.show()

  return l,iltot


def Bwareafilt_N (Image, n_objects):
    binary_image = np.array(Image > 0.5,bool)
    region_props = measure.regionprops(MatToUint8(binary_image))
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    top_n_regions = sorted_regions[:n_objects]
    mask=np.zeros_like(binary_image)
    for region in top_n_regions:
        rr, cc = polygon(region.coords[:, 0], region.coords[:, 1])
        mask[rr, cc] = 1
    Filtered=cv2.bitwise_and(MatToUint8(Image), MatToUint8(Image), mask=MatToUint8(mask))
    Filtered=Mat2gray(Filtered)
    return Filtered



def Bwareafilt_N_Range(Image,min_area,max_area):
    labels = measure.label(Image, connectivity=1)

    # Calcola l'area di ciascun oggetto
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]

    # Filtra gli oggetti in base all'area desiderata
    filtered_labels = np.zeros_like(labels)
    for label in np.unique(labels):
        if min_area <= areas[label - 1] <= max_area:
            filtered_labels[labels == label] = label

    # Ottiene l'immagine finale mantenendo solo gli oggetti filtrati
    Image_new = (filtered_labels > 0).astype(np.uint8)
    return Image_new



def keep_n_largest_objects(binary_image, n_objects):
    labeled_image, num_labels = ndimage.label(binary_image)
    regions = measure.regionprops(labeled_image)

    # Ordina le regioni in base all'area in modo decrescente
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)

    # Seleziona solo i primi N oggetti più grandi
    selected_regions = sorted_regions[:n_objects]

    # Crea un array vuoto delle stesse dimensioni dell'immagine binaria
    filtered_image = np.zeros_like(binary_image)

    # Assegna 1 alle posizioni corrispondenti agli oggetti selezionati
    for region in selected_regions:
        filtered_image[labeled_image == region.label] = 1

    return filtered_image



def lung_segmentation (I_imadjust,threshold,mask,xhalf,BWt):
    # inputs:
    # I_imadjust: grey-scale image
    # threshold: threshold found with histogram partitioning method
    # mask: binary image representin the chest area used as mask
    # xhalf:x position of the point located in the half of the image
    # BWt: binary image of chest

    # outputs:
    # BWlung: binary image resulting from lung segmentation
    # lung1: matrix containing the coordinates relating to right lung
    # lung2: matrix containing the coordinates relating to left lung
    # lung_fraction: lung percentage


    # Threshold application
    BWIt = I_imadjust > threshold

    #plt.figure()
    #plt.imshow(BWIt,'gray')
    #plt.axis('off')
    #plt.draw()


    # complementary image
    BWlungC = cv2.bitwise_not(MatToUint8(BWIt))

    #plt.figure()
    #plt.imshow(BWlungC, 'gray')
    #plt.axis('off')
    #plt.draw()

    # application of binary image of chest as mask in order to delete white border pixels
    BWlungmask = Mat2gray(BWlungC)

    # operation of erosion to strict the mask
    seI = skimage.morphology.rectangle(11, 1)
    maskr = skimage.morphology.erosion(mask, seI)

    BWlungmask = cv2.bitwise_and(MatToUint8(BWlungmask), MatToUint8(BWlungmask), mask=MatToUint8(maskr))

    #plt.figure()
    #plt.imshow(BWlungmask, 'gray')
    #plt.axis('off')
    #plt.draw()


    # operation of erosion applied on lung segmentation result
    se = skimage.morphology.rectangle(2, 1)
    BWlung1 = skimage.morphology.erosion(BWlungmask, se)


    # operation that only keeps the two largest elements
    BWlung2 = skimage.morphology.remove_small_objects(np.array(BWlung1,bool), min_size=300)
    #BWlung2 = Bwareafilt_N(BWlung2, 2)
    BWlung2 = keep_n_largest_objects(BWlung2,2)


    # Operation of closing
    se2 = pd.read_excel('disk7.xlsx')
    se2 = np.array(se2)
    BWlung4 = skimage.morphology.closing(BWlung2, se2)
    BWlung = ndimage.binary_fill_holes(BWlung4)

    #plt.figure()
    #plt.imshow(BWlung, 'gray')
    #plt.axis('off')
    #plt.show()


    ones = np.argwhere(BWlung == 1)
    row1 = [point[0] for point in ones]
    column1 = [point[1] for point in ones]
    lungs = [[point[1],point[0]] for point in ones]


    # Right half image (right lung)
    lungs = np.array(lungs)
    ilung1 = np.where(column1 <= xhalf)[0]
    lung1 = lungs[ilung1]
    lung1 = np.array(lung1).tolist()
    ilung2 = np.where(column1 > xhalf)[0]
    lung2 = lungs[ilung2]
    lung2 = np.array(lung2).tolist()


    #plt.figure()
    #plt.imshow(I_imadjust,'gray')
    #plt.scatter([x[0] for x in lung1],[x[1] for x in lung1],color='red',s=2,alpha=0.5)
    #plt.scatter([x[0] for x in lung2],[x[1] for x in lung2],color='red',s=2,alpha=0.5)
    #plt.axis('off')
    #plt.show()


    # Lung percentage
    lung_size = np.count_nonzero(BWlung)
    thorax_size = np.count_nonzero(BWt)
    lung_fraction = np.round(np.divide(lung_size,thorax_size), 2)

    return BWlung, lung1, lung2, lung_fraction

def flat_contours(contours):
    #Creo questa funzione per evitare di avere delle liste annidate tra loro
    flat_contours = []
    for cnt in contours:
        for pt in cnt:
            flat_contours.append(list(pt[0]))
    return flat_contours


def outer_contour(BWt, Xhalf, Yhalf):
     #Rilevamento coordinate bordo esterno prima di operazioni morfologiche

     Xhalf = int(Xhalf)
     Yhalf = int(Yhalf)

     contours_pre, hierarchy = cv2.findContours(MatToUint8(BWt), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     contours_pre = flat_contours(contours_pre)
     contours_pre.reverse()

     #contours_pre = measure.find_contours(BWt)

     x_contours_pre = np.array([x[0] for x in contours_pre])
     y_contours_pre = np.array([y[1] for y in contours_pre])

     xmin = np.min(x_contours_pre)
     ixmin = np.where(x_contours_pre == xmin)[0].min()
     contours_pre = np.roll ( contours_pre , -ixmin , axis=0)

     x_contours_pre = np.array([x[0] for x in contours_pre])
     y_contours_pre = np.array([y[1] for y in contours_pre])

     #Valutare caso disco (20 su matlab)
     se = pd.read_excel('disk5.xlsx')
     se = np.array(se)
     BWtorax = skimage.morphology.closing(BWt, se)
     se1 = skimage.morphology.rectangle(1, 20)
     BWtorax = skimage.morphology.closing(BWtorax, se1)
     s2 = skimage.morphology.rectangle(1, 3)
     BWtorax = skimage.morphology.erosion(BWtorax, s2)


     image = MatToUint8(BWtorax)
     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )


     #DISEGNO, ELIMINARE QUESTA PARTE
     #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
     #cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
     #cv2.imshow('None approximation', image)
     #cv2.waitKey(0)

     contours = flat_contours(contours)
     contours.reverse() #COSì I CONTORNI VENGONO TROVATI IN SENSO ORARIO

     x_contours = np.array([x[0] for x in contours])
     y_contours = np.array([y[1] for y in contours])

     xmin = np.min(x_contours)
     ixmin = np.where(x_contours == xmin)[0].min()
     contours = np.roll(contours, -ixmin, axis=0) #TRASLO I CONTORNI

     x_contours = np.array([x[0] for x in contours])
     y_contours = np.array([y[1] for y in contours])



     d = np.diff(y_contours)
     imax1 = np.where(d==1)[0]#First max index
     imax1 = imax1[0]
     max1 = contours[int(imax1)]

     imin = np.where(d==-1)[0]
     imin = [x for x in imin if x > imax1]
     imin = imin[0]


     imax2 = np.where(d==1)[0]
     imax2=[x for x in imax2 if x > imin]



     if len(imax2) != 0:
         max2=contours[imax2[0]]
     else:
         imax2=imax1
         max2 =contours[int(imax2)]

     #plt.figure()
     #plt.plot(x_contours,y_contours)
     #plt.scatter(max1[0], max1[1], color='red', s=30,zorder=2)
     #plt.scatter(max2[0], max2[1], color='red', s=30,zorder=2)
     #plt.xlabel('X coordinate')
     #plt.ylabel('Y coordinate')
     #plt.ylim(0,Yhalf)
     #plt.gca().invert_yaxis()
     #plt.draw()

     #plt.figure()
     #plt.imshow(image,'gray')
     #plt.scatter(max1[0],max1[1],color='red',s=30,zorder=2)
     #plt.scatter(max2[0],max2[1],color='red',s=30,zorder=2)
     #plt.axis('off')
     #plt.draw()




     min1 = contours[imin]
     minmatrix = [point for point in contours_pre if point[1]<= Yhalf]
     minmatrix_x=np.array([element[0] for element in minmatrix])
     minmatrix_y=np.array([element[1] for element in minmatrix])




     interval = (minmatrix_x > max1[0]) & (minmatrix_x < max2[0])

     minmatrix = [element for element,condizione in zip(minmatrix,interval) if condizione == True]
     minmatrix_x = np.array([element[0] for element in minmatrix])
     minmatrix_y = np.array([element[1] for element in minmatrix])

     if len(minmatrix_y) > 0:
         mincy = np.max(minmatrix_y)
         iminclast = np.where(minmatrix_y == mincy)[0].max()
         pmin = minmatrix[iminclast]
     else:
         pmin = np.array([0,0])





     #EVENTUALI CORREZIONI
     if max1[0] > Xhalf:
         ixhalfall = np.where(x_contours == Xhalf)[0]
         minhalf = np.min(np.array([element for element,index in zip(y_contours,ixhalfall) if index in ixhalfall]))
         pmin[0]=Xhalf
         pmin[1]=minhalf
         ixhalf = np.where(np.array(x_contours)<Xhalf)[0]
         interval = np.arange(ixhalf[0],ixhalf[-1]+1)
         imax1 = np.min(np.array([element for element, index in zip(y_contours,interval) if index in interval]))
         max1 = contours[imax1 + ixhalf[0]]

     elif max2[0] < Xhalf:
         ixhalfall = np.where(x_contours == Xhalf)[0]
         minhalf = np.min(np.array([element for element, index in zip(y_contours, ixhalfall) if index in ixhalfall]))
         pmin[0] = Xhalf
         pmin[1] = minhalf
         ixhalf = np.where(x_contours > Xhalf)[0]
         interval = np.arange(ixhalf[0], ixhalf[-1] + 1)
         imax2 = np.min(np.array([element for element, index in zip(y_contours, interval) if index in interval]))
         max2 = contours[imax2 + ixhalf[0]]

     if y_contours[0] < Yhalf:
         i1 = np.where(y_contours == Yhalf)[0].max()
         firstcontour = contours[i1:-1].tolist()
         contours_n = contours.copy()
         interval = np.arange(i1+1,len(contours_n),1)
         contours_n = [x for index,x in enumerate(contours_n) if index not in interval]
         contours_n = [arr.tolist() for arr in contours_n]
         contours_n = firstcontour + contours_n
     else:
         contours_n = contours.copy()

     #plt.figure()
     #plt.plot(x_contours_pre,y_contours_pre)
     #plt.axvline(x=max1[0], linestyle='--', color='red')
     #plt.axvline(x=max2[0], linestyle='--', color='red')
     #plt.scatter(pmin[0],pmin[1],color='red',s=30,zorder=2)
     #plt.xlabel('X coordinate')
     #plt.ylabel('Y coordinate')
     #plt.ylim(0, Yhalf)
     #plt.gca().invert_yaxis()
     #plt.draw()

     #plt.figure()
     #plt.imshow(BWt,'gray')
     #plt.scatter(pmin[0],pmin[1],color='red',s=30)
     #plt.axis('off')
     #plt.show()




     max1=max1.tolist()
     max2=max2.tolist()
     pmin=pmin.tolist()





     #DISEGNO PUNTI
     #color = (0, 255, 0)
     #markerType = cv2.MARKER_CROSS
     #markerSize = 15
     #thickness = 5
     #cv2.drawMarker(image, max1, color, markerType, markerSize, thickness)
     #cv2.imshow('Image', image)
     #cv2.waitKey(0)
     #cv2.destroyAllWindows()

     #color = (255, 0, 0)
     #markerType = cv2.MARKER_CROSS
     #markerSize = 15
     #thickness = 5
     #cv2.drawMarker(image, [int(pmin[0]),int(pmin[1])] ,color, markerType, markerSize, thickness)
     #cv2.imshow('Image', image)
     #cv2.waitKey(0)
     #cv2.destroyAllWindows()

     #color = (150, 150, 0)
     #markerType = cv2.MARKER_CROSS
     #markerSize = 15
     #thickness = 5
     #cv2.drawMarker(image, max2, color, markerType, markerSize, thickness)
     #cv2.imshow('Image', image)
     #cv2.waitKey(0)
     #cv2.destroyAllWindows()

     return max1, max2, pmin, BWtorax, contours_n


def visualize_select(images, numtot, sex):

    #User selection of slices

    #input:
    #images: all images belonging to the patient
    #numtot: number of slices
    #sex: patient's sex from file dicom

    #output:
    # n1: first slice for depression quantification
    # ndend: last slice for depression quantification
    # ns: slice selected for indices computation
    # s1: first slice for inner contour analysis
    # send: last slice for inner contour analysis
    # numimages: number of images analyzed (both for depression quantification and inner contour analysis)
    # deprange: number of images for depression quantification
    # srange: number of images for inner contour analysis
    # gender: patient's gender
    nums = str(numtot)
    numrow = int(nums[0])

    position = [np.round((np.size(images[0],0))/2) , 10]
    i=0
    fnt=ImageFont.truetype("arial.ttf",25)
    d = []
    while i<numtot:
        a = Image.fromarray(np.uint8(cm.gray(images[i])*255))
        a = a.convert('L')
        draw = ImageDraw.Draw(a)
        draw.text((position[0], position[1]), str(i+1), font=fnt, fill=255)
        d.append(a)
        i=i+1


    new_images = []
    for image in d:
        new_images.append (np.array(image))

    Montage_Matlab(new_images,'lol')
    montage = skimage.util.montage(new_images, grid_shape=(numrow, round(numtot/numrow)+1) , fill=0 , padding_width=10)

    def graph():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(montage, 'gray')
        ax.axis('off')
        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.show()


    def visualize_select_2():
            try:
                n1 = int(label1_input.get())
                ndend = int(label2_input.get())
                ns = int(label3_input.get())
                gender = clicked.get()
                if n1 > ndend:
                    message.config(text=' Error in slice selection')
                elif gender not in ("M", "F"):
                    message.config(text=' Error in sex selection')
                else:
                    return window.quit()
            except ValueError:
                message.config(text=' Error in slice selection')



    window = tk.Tk()
    window.geometry("400x450")
    window.title("Slices selection")
    window.configure(background="cyan")
    window.wm_resizable(False,False)


    label1 = tk.Label(window, text=" First slice:", pady=10)
    label1.grid(row=0, column=1, sticky="N", padx=20, pady=10)
    label1_input = tk.Entry()
    label1_input.grid(row=0, column=2, sticky="WE", padx=10, pady=10)
    label1_input.focus()

    label2 = tk.Label(window, text=" Last slice:", pady=10)
    label2.grid(row=1, column=1, sticky="N", padx=20, pady=10)
    label2_input = tk.Entry()
    label2_input.grid(row=1, column=2, sticky="WE", padx=10, pady=10)

    label3 = tk.Label(window, text=" Selected slice:", pady=10)
    label3.grid(row=2, column=1, sticky="N", padx=20, pady=10)
    label3_input = tk.Entry()
    label3_input.grid(row=2, column=2, sticky="WE", padx=10, pady=10)

    label4 = tk.Label(window, text=" Gender:", pady=10)
    label4.grid(row=3, column=1, sticky="N", padx=20, pady=10)
    clicked = tk.StringVar()
    if ("M" in str(sex)):
        clicked.set("M")
    elif ("F" in str(sex)):
        clicked.set("F")
    else:
        clicked.set("")
    label4_input = tk.OptionMenu(window, clicked, "", "M", "F")
    label4_input.grid(row=3, column=2, sticky="WE", padx=10, pady=10)

    message = tk.Label(window, text="", pady=10)
    message.grid(row=5, column=1, sticky="N", padx=20, pady=10)


    Button1 = tk.Button(window, text="Show Images" , command=graph)
    Button1.grid(row=4, column=1, sticky="WE", padx=50, pady=10)


    Button2 = tk.Button(window, text="Ok")
    Button2.grid(row=4, column=2, sticky="WE", padx=50, pady=10)
    Button2.config(command=visualize_select_2)

    window.mainloop()

    n1 = int(label1_input.get())
    ndend = int(label2_input.get())
    ns = int(label3_input.get())
    gender = clicked.get()
    nend = ns+14

    window.destroy()
    plt.close()

    if n1<1:
        n1=1

    if nend>numtot:
        nend = numtot

    if nend<ndend:
        nend = ndend

    #slices analyzed
    numimages = nend - n1 + 1

    #slices for depression quantification
    deprange = ndend - n1 + 1

    #first slice for inner contour analysis
    s1 = ns - n1 + 1

    #last slice for inner contour analysis
    send = numimages

    #number of slices for inner contour analysis
    srange = send - s1 + 1

    #Correction for Python (in python, indexes start from 0)
    n1=n1-1
    nend=nend-1
    ndend=ndend-1
    ns=ns-1
    s1=s1-1
    send=send-1

    return n1,ndend,ns,s1,send,numimages,deprange,srange,gender



def visualize_select_heart(images):

    numtot = len(images)
    nums = str(numtot)
    numrow = int(nums[0])

    position = [np.round((np.size(images[0],0))/2) , 10]
    i=0
    fnt=ImageFont.truetype("arial.ttf",25)
    d = []
    while i<numtot:
        a = Image.fromarray(np.uint8(cm.gray(images[i])*255))
        a = a.convert('L')
        draw = ImageDraw.Draw(a)
        draw.text((position[0], position[1]), str(i+1), font=fnt, fill=255)
        d.append(a)
        i=i+1


    new_images = []
    for image in d:
        new_images.append (np.array(image))

    montage = skimage.util.montage(new_images, grid_shape=(numrow, round(numtot/numrow)+1) , fill=0 , padding_width=10)

    def graph():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(montage, 'gray')
        ax.axis('off')
        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.show()


    def visualize_select_2():
            try:
                n1 = int(label1_input.get())
                n2 = int(label2_input.get())
                ns = int(label3_input.get())
                if n1 > n2:
                    message.config(text=' Error in slice selection')
                else:
                    return window.quit()
            except ValueError:
                message.config(text=' Error in slice selection')



    window = tk.Tk()
    window.geometry("400x450")
    window.title("Slices selection")
    window.configure(background="cyan")
    window.wm_resizable(False,False)


    label1 = tk.Label(window, text=" First slice:", pady=10)
    label1.grid(row=0, column=1, sticky="N", padx=20, pady=10)
    label1_input = tk.Entry()
    label1_input.grid(row=0, column=2, sticky="WE", padx=10, pady=10)
    label1_input.focus()

    label2 = tk.Label(window, text=" Last slice:", pady=10)
    label2.grid(row=1, column=1, sticky="N", padx=20, pady=10)
    label2_input = tk.Entry()
    label2_input.grid(row=1, column=2, sticky="WE", padx=10, pady=10)

    label3 = tk.Label(window, text=" Slice selected:", pady=10)
    label3.grid(row=2, column=1, sticky="N", padx=20, pady=10)
    label3_input = tk.Entry()
    label3_input.grid(row=2, column=2, sticky="WE", padx=10, pady=10)

    message = tk.Label(window, text="", pady=10)
    message.grid(row=5, column=1, sticky="N", padx=20, pady=10)


    Button1 = tk.Button(window, text="Show Images" , command=graph)
    Button1.grid(row=4, column=1, sticky="WE", padx=50, pady=10)


    Button2 = tk.Button(window, text="Ok")
    Button2.grid(row=4, column=2, sticky="WE", padx=50, pady=10)
    Button2.config(command=visualize_select_2)

    window.mainloop()

    n1 = int(label1_input.get())
    n2 = int(label2_input.get())
    ns = int(label3_input.get())

    window.destroy()
    plt.close()

    if n1<1:
        n1=1

    if n2>numtot:
        n2 = numtot

    if ns < n1:
        ns = n1

    if ns > n2:
        ns = n2



    #Correction for Python (in python, indexes start from 0)
    n1=n1-1
    n2=n2-1
    ns=ns-1


    return n1,n2,ns









def visualize_test(Images,numtot):
    #Function made for test, only to visualize result, not to be used in the final program
    nums = str(numtot)
    numrow = int(nums[0])
    new_images = []
    for image in Images:
        new_images.append(MatToUint8(image))

    montage = skimage.util.montage(new_images, grid_shape=(numrow, round(numtot / numrow)+1), fill=0, padding_width=10)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(montage, 'gray')
    ax.axis('off')
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.show()

def sub2ind(sz, row, col):
    n_rows = sz[0]
    return [n_rows * (c-1) + r for r, c in zip(row, col)]


def depression_eval(pmax1,pmax2,BWt,I_imadjust,yhalf,pixel_distance):
    # inputs:
    # pmax1: coordinates of first maximum point of outer chest contour
    # pmax2: coordinates of second maximum point of outer chest contour
    # BWt: binary image after pre-processing
    # I_imadjust: grey-scale image after pre-processing
    # yhalf: y position of the half point of image
    # pixel_distance: vector containing vertical and horizontal distances between pixels(mm)

    # outputs:
    # Bfill: binary image of chest after depression correction with elliptical curve between the two maximum points
    # I: grey-scale image with elliptical curve
    # depression: binary image of depression
    # depression_area: depression area
    # corrthorax_area: thorax area after correction

    image = Mat2gray(I_imadjust)
    pmax1_temp=np.array(pmax1)
    pmax2_temp=np.array(pmax2)


    points = np.argwhere(BWt==1)
    mat = [[xx[1], xx[0]] for xx in points]
    cols = []
    rows = []
    for point in  mat:
        cols.append(point[0])
        rows.append(point[1])

    cols=np.array(cols)
    rows=np.array(rows)

    #points at the same y coordinate as the first maximum point
    ix1all = np.where(rows==pmax1[1])[0]

    #Point with minimum x coordinate
    px1=np.min(np.take(cols,ix1all))
    pmax1_temp[0]=px1

    #points at the same y coordinate as the second maximum point
    ix2all = np.where(rows==pmax2[1])[0]

    #Point with minimum x coordinate
    px2=np.max(np.take(cols,ix2all))
    pmax2_temp[0]=px2


    #ellipse major axis extreme points
    x1=pmax1_temp[0]
    x2=pmax2_temp[0]
    y1=pmax1_temp[1]
    y2=pmax2_temp[1]
    xm=[x1,x2]
    ym=[y1,y2]

    #plt.scatter(xm[0],ym[0],color='blue',s=30)
    #plt.scatter(xm[1],ym[1],color='blue',s=30)
    #plt.axis('off')
    #plt.draw()

    #ellipse eccentricity
    eccentricity = 0.99

    #number of points between the 2 maximum points
    numPoints=    np.max(  [np.abs(np.diff(xm)) ,  np.abs(np.diff(ym))]   ) + 1

    #ellipse equation
    a = 0.5 * np.sqrt (np.add(   np.square(np.subtract(x2,x1)),   np.square(np.subtract(y2,y1))))
    b = math.sqrt(1-eccentricity*eccentricity) * a

    #angle varies between 0 and pi (half ellipse)
    t = np.linspace(0 , -1*math.pi , int(numPoints))
    X = a * np.cos(t)
    Y = b * np.sin(t)

    #ellipse rotation angle respect to horizontal line
    angles = np.arctan2(np.subtract(y2,y1) , np.subtract(x2,x1))

    #ellipse coordinates
    x =   np.subtract( np.add(   np.add(x1,x2)/2    ,   np.multiply(X,np.cos(angles)))  , np.multiply(Y,np.sin(angles)))
    y =   np.add(      np.add(   np.add(y1,y2)/2    ,   np.multiply(X,np.sin(angles)))  , np.multiply(Y,np.cos(angles)))


    #indices corrresponding to x and y coordinates belonging to ellipse
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    index=list(zip(y,x))


    #pixel corrisponding to ellipse added to binary and grey-scale images
    BWline = BWt.copy()
    I1 = image.copy()
    for point in index:
        BWline[point]=1
        I1[point]=1

    #plt.figure()
    #plt.imshow(BWline,'gray')
    #plt.scatter(xm[0],ym[0],color='blue',s=30)
    #plt.scatter(xm[1],ym[1],color='blue',s=30)
    #plt.axis('off')
    #plt.draw()

    #depression correction with morpholigical operation
    se = pd.read_excel('disk20.xlsx')
    se = np.array(se)

    Bfill = ndimage.binary_closing(BWline,se)
    Bfill = ndimage.binary_fill_holes(Bfill)

    #plt.figure()
    #plt.imshow(Bfill,'gray')
    #plt.axis('off')
    #plt.draw()

    #plt.figure()
    #plt.imshow(BWt,'gray')
    #plt.axis('off')
    #plt.draw()


    #depression area found after image subtraction
    depression=cv2.subtract(MatToUint8(Bfill),MatToUint8(BWt))
    threshold_value = filters.threshold_otsu(depression)
    depression = depression > threshold_value

    #plt.figure()
    #plt.imshow(depression, 'gray')
    #plt.axis('off')
    #plt.draw()

    #operation that only keeps the largest element and the upper half of image
    depression[int(yhalf):,:]=0
    depression = Bwareafilt(depression)

    #plt.figure()
    #plt.imshow(depression, 'gray')
    #plt.axis('off')
    #plt.draw()



    #depression correction on grey-scale image
    I=image.copy()
    depression = np.array(depression,bool)
    I[depression]=1

    #plt.figure()
    #plt.imshow(I, 'gray')
    #plt.axis('off')
    #plt.show()

    #number of nonzero elements in matrix representing depression area
    depression_area_p = np.count_nonzero(depression)

    #single pixel area in mm
    pixel_area= pixel_distance[0] * pixel_distance[1]

    #depression area in mm
    depression_area = depression_area_p * pixel_area

    ##correct chest area computation
    #number of nonzero elements in matrix representing correct chest
    corrthorax_area_p = np.count_nonzero(Bfill)
    #correct chest area in mm
    corrthorax_area = corrthorax_area_p * pixel_area

    return Bfill,I,depression,depression_area,corrthorax_area


def inner_contours_seg(I_imadjust,BWlung,contour,xhalf,lung1,lung2):
    #the algorithm goes through the outer curvature in clockwise direction
    #until the start point is found again. While going through the outer curvature
    #the number of steps is counted. Every 12 steps the actual point and the point
    #12 steps before are connected and a perpendicular line in the mid-point
    #is generated. Then the algorithm has to find the intersection point between
    #the perpendicular line and the first point crossed by it on the two lungs.
    #If the perpendicular does not meet the lungs the point is located at the
    #same distance as the previous point.

    # Input:
    # I_imadjust: grey-scale image
    # BWlung: binary image representing segmented lungs
    # contour: coordinates of outer chest contour
    # xhalf: x position of the point located in the half of the image
    # lung1: coordinates of pixels releted to the right lung
    # lung2: coordinates of pixels releted to the left lung

    # Output:
    # inters: points representing the inner chest contour

    #plt.figure()
    #plt.plot([x[0] for x in contour], [x[1] for x in contour],color='yellow')
    #plt.imshow(BWlung,'gray')

    nsteps = 12
    #number of intersection points to find (calculated from the number of outer chest contour points  and step number
    nrow = int(np.round((len(contour)-nsteps)/nsteps))+1

    # initialization
    # middle point
    midX = np.zeros((nrow,1))
    midY = np.zeros((nrow,1))

    #Intersection points
    Inters = np.zeros((nrow, 2))

    #distance between mean point and intersection point
    distancemidint = np.zeros((nrow,1))

    #mean distances
    distancem = np.zeros((nrow,1))

    #varianza delle distanze
    stdm  = np.zeros((nrow,1))

    #Counter
    a = -1

    #partition of the image in 4 quarters: image partitioned in a right and
    #left half (indices: ic2 and ic4)
    x_contour = [x[0] for x in contour]
    y_contour = [y[1] for y in contour]
    icx = np.where( x_contour == xhalf)[0]
    ic2 = icx[0]
    ic4 = icx[1]




    #image partitioned in a upper and lower half (indices: ic1 and ic3)
    #index on outer contour corresponding to the first point:
    ic1 = 0
    #indices on outer contour at the same y as the first point:
    ic3x = list(np.where(y_contour == y_contour[0])[0])
    #elimination of consecutive points at the same y
    ic3delete = np.where (np.diff(ic3x) == 1)[0]
    if len(ic3delete) != 0:
        ic3delete = ic3delete +1
        ic3x = [x for index,x in enumerate(ic3x) if index not in ic3delete]
    #index corresponding to the point with greater x:
    ic3 = ic3x[1]

    #algorithm goes through the entire outer chest contour
    i=0
    while i<len(contour)-nsteps:

        #starter and end point of each segment
        p1=np.array(contour[i],int)
        p2=np.array(contour[i+nsteps],int)

        xpoints=[p1[0],p2[0]]
        ypoints=[p1[1],p2[1]]



        #middle point of segment

        a = a + 1


        midX[a] = np.round(mean([p1[0],p2[0]]))
        midY[a] = np.round(mean([p1[1],p2[1]]))

        #if the 2 points are at the same y, perpendicular line is: x=midX
        if p2[1]==p1[1]:
            # Vector with the same pixel number of the image
            y = np.linspace(0,np.size(I_imadjust,0)-1,np.size(I_imadjust,1))
            # Perpendicular line
            x = midX[a] * np.ones((1, np.size(I_imadjust,1)))[0]
            # Slope assigned to inf
            slope = 10000000000000000 #Sarebbe infinito


        #if the 2 points are at the same x, perpendicular line is: y=midY
        elif p2[0]==p1[0]:
            x = np.linspace(0, np.size(I_imadjust, 0)-1, np.size(I_imadjust, 1))
            y = midY[a] * np.ones((1, np.size(I_imadjust, 1)))[0]
            # Slope assigned to 0
            slope = 0




        #otherwise segment slope is computed
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            #perpendicular line slope
            slope = (-1) / slope
            #x vector with the same pixel number of the image
            x = np.linspace(0, np.size(I_imadjust, 0)-1, np.size(I_imadjust, 1))
            #y corresponding to x coordinates calculated from equation
            #(perpendicular line passing through middle point of the
            #segment)
            y = np.round(slope * (x - midX[a]) + midY[a])
            y = np.array(y,dtype=int)
            # Elimination of y values outside the image (and corresponding x values)
            idelete = np.where((y <= 0) | (y >= np.size(I_imadjust, 0)-1))[0]
            y = np.delete(y,idelete)
            x = np.delete(x,idelete)

        #plt.plot(x,y,color='blue')


        # Matrix containing x and y coordinates of perpendicular line points
        k=0
        mat=[]
        x=list(x)
        y=list(y)
        while k < len(x):
            mat.append([int(x[k]), int(y[k])])
            k=k+1



        #application of algorithm with different conditions based on the quadrant analyzed
        #first quadrant: right lung considered (outer contour upper border)
        #if the slope of the perpendicular increases (> 0) the intersection point
        #is the first one(that with the smallest y), if the slope is negative the point is
        #the last one(that with the lowest y), if the slope is 0 (parallel to the y axis)
        #and if the slope does not exist (parallel to the x axis) the point of
        #intersection is the first one.

        if (i >= ic1) & (i <= ic2):
            #common points between perpendicular line and right lung region
            v,_,ib = intersect(lung1,mat)

            #%if there are common points
            if len(ib) != 0:
                # if the slope is negative, the y of the intersection points
                # decrease (y (end) <y (2)) so the order of the indices is
                # reversed because I want the point with lower y coordinate
                # that is the last one
                v_first = v[0]
                v_last = v[-1]
                if v_first[1] > v_last[1]:
                    ib = np.sort(ib)[::-1]
                    #ib=ib[::-1]
                # Otherwise the intersection point is the first one
            if len(ib)!=0:
                Inters[a] = mat[ib[0]]



        #Second quadrant: left lung considered (outer contour upper border):
        #same method as for the first quadrant except in the case of
        #slope=0 (y = cost) where I want the last point
        elif (i>=ic2) & (i<=ic3):

            v,_,ib = intersect(lung2,mat)
            if len(ib) != 0:
                # if slope<0 or y=cost the order of the indices is
                # reversed because I want the point with lower y coordinate
                # that is the last one
                v_first = v[0]
                v_last = v[-1]
                if v_first[1] > v_last[1]:
                    ib = np.sort(ib)[::-1]
                    #ib = ib[::-1]
            #Otherwise the intersection point is the first one
            if len(ib) != 0:
                Inters[a] = mat[ib[0]]




        #third quarter: left lung considered (outer countour lower border):
        #if the slope of the perpendicular increases (> 0) the intersection point
        #is the last one (that with the greatest y), if the slope is negative (<0)
        #the point is the first one (that with the greatest y), if the slope is 0
        #(parallel to the y axis) and if the slope does not exist (parallel to the x axis)
        #the intersection point is the last one.
        elif (i>=ic3) & (i<=ic4):

            v,_,ib = intersect(lung2,mat)

            if len(ib) != 0:
                # if slope<0 or y=cost the order of the indices is
                # reversed because I want the point with lower y coordinate
                # that is the last one
                v_first = v[0]
                v_last = v[-1]
                if v_first[1] > v_last[1]:
                    ib = np.sort(ib)[::-1]
                    #ib = ib[::-1]
            #Otherwise the intersection point is the last one
            if len(ib) != 0:
                Inters[a] = mat[ib[-1]]



        #fourth quadrant: rightlung considered (outer contour lower border):
        #same method as for the third quadrant except in the case of
        #slope=0 (y = cost) where I want the first point

        else:
            v,_,ib = intersect(lung1,mat)
            if len(ib) != 0:
                # if slope<0 or y=cost the order of the indices is reversed
                # because I want the first point
                v_first = v[0]
                v_last = v[-1]
                if v_first[1] > v_last[1]:
                    ib = np.sort(ib)[::-1]
                    #ib = ib[::-1]

            #intersection point is the last one
            if len(ib) != 0:
                Inters[a] = mat[ib[-1]]


        IntersX=[]
        IntersY=[]
        for Point in Inters:
            IntersX.append(Point[0])
            IntersY.append(Point[1])


        # cases where the perpendicular line doesn't cross the lungs
        # distance between middle point and intersection point
        distancemidint[a] = math.sqrt((midX[a]-IntersX[a])*(midX[a]-IntersX[a])+(midY[a]-IntersY[a])*(midY[a]-IntersY[a]))
        # mean value
        distancem[a] = np.mean(distancemidint[0:a+1])
        # standard deviation
        stdm[a] = np.std(distancemidint[0:a+1])


        # Analysis beging to the second intersection point
        if a > 0:
            # if the distance between the midpoint and the intersection point is
            # greater than 2*mean distance-standard deviation (threshold)
            if (distancemidint[a]) >= ((2 * np.floor(distancem[a])) - np.floor(stdm[a])):
                # Distance replaced by the previous one
                distancemidint[a] = distancemidint[a - 1]
                # Mean value recomputed
                distancem[a] = np.mean(distancemidint[0:a + 1])
                # distance between the midpoint and perpendicular line points
                matX = [x[0] for x in mat]
                matY = [y[1] for y in mat]
                distmatmid = np.round( np.sqrt(((midX[a] - matX) * (midX[a] - matX)) + ((midY[a] - matY) * (midY[a] - matY))))
                distmatmid = distmatmid.tolist()
                slope = int(np.round(slope))
                # Midpoint index
                distmatmid = np.array(distmatmid,dtype=int)
                imid = int(np.where(distmatmid == 0)[0])
                # Analysis diversified based on the quadrant
                # First quadrant
                if (i >= ic1) & (i <= ic2):
                    if slope < 0:
                        # elimination of the points after the mid point
                        # (the last ones: y decreases)
                        distmatmid = np.delete(distmatmid, np.arange(imid+1, len(distmatmid)))
                        mat[imid + 1::] = []
                        # if slope > 0, x=cost(slope=a) and y=cost(slope=0)
                    else:
                        # Elimination of the points before the mid point
                        # (the first ones: y increases)
                            distmatmid = np.delete(distmatmid, np.arange(imid))
                            mat[0:imid] = []
                # second quadrant
                elif (i > ic2) & (i <= ic3):
                    if slope <= 0:
                        # elimination of the points after the mid point
                        # (the last ones: y decreases)
                            distmatmid = np.delete(distmatmid, np.arange(imid+1, len(distmatmid)))
                            mat[imid + 1::] = []
                        # if slope > 0, x=cost(slope=a)
                    else:
                        # Elimination of the points before the mid point
                        # (the first ones: y increases)
                            distmatmid = np.delete(distmatmid, np.arange(imid))
                            mat[0:imid] = []


                # Third quadrant
                elif (i > ic3) & (i <= ic4):
                    if slope < 0:
                        # elimination of the points after the mid point
                        # (the last ones: y decreases)
                            distmatmid = np.delete(distmatmid, np.arange(imid))
                            mat[0:imid] = []
                        # if slope > 0, x=cost(slope=a)
                    else:
                        # Elimination of the points before the mid point
                        # (the first ones: y increases)
                            distmatmid = np.delete(distmatmid, np.arange(imid+1, len(distmatmid)))
                            mat[imid + 1::] = []


                # Fourth quadrant
                else:
                    if slope <= 0:
                        # elimination of the points after the mid point
                        # (the last ones: y decreases)
                            distmatmid = np.delete(distmatmid, np.arange(imid))
                            mat[0:imid] = []
                        # if slope > 0, x=cost(slope=a)
                    else:
                        # Elimination of the points before the mid point
                        # (the first ones: y increases)
                            distmatmid = np.delete(distmatmid, np.arange(imid+1, len(distmatmid)))
                            mat[imid + 1::] = []

                # Index of the point closer to the wanted distance
                distmatmid = np.array(distmatmid)
                iinters = np.argmin(np.abs(distmatmid - distancemidint[a]))
                if iinters != None:
                    Inters[a] = mat[iinters]

            # if intersection point is zero: error in the computation

            if (Inters[a][0] == 0) & (Inters[a][1] == 0):
                # Point replaced by a point at the same distance as the
                # previous one
                distancemidint[a] = distancemidint[a - 1]
                distancem[a] = distancem[a - 1]



        i=i+nsteps
    #plt.scatter([x[0] for x in Inters], [x[1] for x in Inters], color='red' , s=20,zorder=2)
    #plt.axis('off')
    #plt.show()
    return Inters


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    polygon = [(col, row) for row, col in zip(vertex_row_coords, vertex_col_coords)]
    image = Image.new('L', shape, 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(polygon, outline=1, fill=1)
    mask = np.array(image.getdata()).reshape(shape[::-1])
    return mask.astype(bool)


def contourinterpolation(pmin,pmax1,pmax2,pint,I_imadjust,contour):
    #the innercontour_seg function depends on the correct segmentation of the
    #lungs, which often does not occur due to similarity in the grey values
    #between lungs and thoracic tissue. This function has the goal of
    #check the points found, delete the invalid ones and then
    #do an interpolation of the correct points to obtain the inner contour
    #of the thorax with the vertebral body include

    #Inputs:
    #pmin: coordinates of minimum point of outer chest contour
    #pmax1: coordinates of first maximum point of outer chest contour
    #pmax2: coordinates of second maximum point of outer chest contour
    #pint: intersection points found by innercontour_seg function
    #I_imadjust: grey-scale image
    #contour: coordinates of outer contour points

    #Outputs:
    #pcontour: coordinates of inner contour points
    #contourmask: binary image of mask representng the inner chest portion


    #preparation of points for interpolation
    #deletion of points outside the outer chest contour (errors in the previous function)

    contour_x = [x[0] for x in contour]
    contour_y = [y[1] for y in contour]


    contestmax = np.max(contour_x)
    contestmin = np.min(contour_x)



    pint = pint.tolist()
    #pint = list(filter(lambda x: x is not None, pint))
    pint_x = np.array([x[0] for x in pint],dtype=int)
    pint_y = np.array([y[1] for y in pint],dtype=int)


    icpintdelete1 = np.where(pint_x >= contestmax)[0]
    icpintdelete2 = np.where(pint_x <= contestmin)[0]


    icpintdelete = np.sort(np.concatenate((icpintdelete1,icpintdelete2)))
    pint = [point for index, point in enumerate(pint) if index not in icpintdelete]
    pint_x = np.array([x[0] for x in pint], dtype=int)
    pint_y = np.array([y[1] for y in pint], dtype=int)

    #Riordino, così voglio che il primo elemento di pint sia quello con la x più piccola
    minima_x = np.argmin(pint_x)
    pint=np.roll(pint, -minima_x, axis=0).tolist()
    pint_x = np.array([x[0] for x in pint], dtype=int)
    pint_y = np.array([y[1] for y in pint], dtype=int)



    #End point of upper contour: point with maximum x coordinate
    pmax =np.max(pint_x)
    ixmax = np.argmax(pint_x)
    #deletion of points at the same x coordinate as the end point
    isamexmax = np.where(pint_x==pmax)[0]
    isamexmax = [i for i in isamexmax if i > ixmax]


    if len(isamexmax)!=0:
        idelete = np.sort(np.concatenate((isamexmax, np.array([ixmax],dtype=int))))
    else:
        idelete = np.array([ixmax],dtype=int)


    pint = [point for index, point in enumerate(pint) if index not in idelete]
    pint_x = np.array([x[0] for x in pint], dtype=int)
    pint_y = np.array([y[1] for y in pint], dtype=int)




    #partition in 2 groups of points: upper and lower contour
    pint1 = pint[0:ixmax+1]
    pint2 = pint[ixmax:len(pint)+1]

    #x coordinates sorted in ascending order (upper contour)
    pint1_x = np.array([x[0] for x in pint1], dtype=int)
    pint1_y = np.array([y[1] for y in pint1], dtype=int)

    ip1 = np.argsort(pint1_x)
    pint1x = np.sort(pint1_x)

    pint2_x = np.array([x[0] for x in pint2], dtype=int)
    pint2_y = np.array([y[1] for y in pint2], dtype=int)

    ip2 = np.argsort(pint2_x)[::-1]
    pint2x = pint2_x[ip2]

    #y coordinates corresponding to x coordinates
    pint1 = [ [a,b]  for a,b in zip(pint1x,pint1_y[ip1])]
    pint2 = [ [a,b]  for a,b in zip(pint2x,pint2_y[ip2])]

    pint1_x = np.array([x[0] for x in pint1], dtype=int)
    pint1_y = np.array([y[1] for y in pint1], dtype=int)
    pint2_x = np.array([x[0] for x in pint2], dtype=int)
    pint2_y = np.array([y[1] for y in pint2], dtype=int)




    #deletion of consecutive points with same x coordinate (previous point)


    xdelete1 =  np.append( ~(np.diff(pint1_x).astype(bool)) , False )
    pint1 = [punto for punto,condizione in zip(pint1,xdelete1) if condizione==False]
    pint1_x = np.array([x[0] for x in pint1], dtype=int)
    pint1_y = np.array([y[1] for y in pint1], dtype=int)



    xdelete2 = np.append(  ~(np.diff(pint2_x).astype(bool))  , False )
    pint2 = [punto for punto,condizione in zip(pint2,xdelete2) if condizione==False]
    pint2_x = np.array([x[0] for x in pint2], dtype=int)
    pint2_y = np.array([y[1] for y in pint2], dtype=int)



    #Starting point of upper contour: the point with minimum x coordinate
    ixmin = np.argmin(pint1_x)

    #deletion of points with a grater y coordinate as the one with maximum x
    ip1control = np.where(pint1_y>pint_y[ixmax])[0]
    if len(ip1control)!=0:
        pint1 = [point for index, point in enumerate(pint1) if index not in ip1control]
        pint1_x = np.array([x[0] for x in pint1], dtype=int)
        pint1_y = np.array([y[1] for y in pint1], dtype=int)


    #deletion of points in pint2 with a lower x coordiante than the first point of pint1
    ixincorrect = np.where(pint2_x <= pint1_x[ixmin])[0]
    if len(ixincorrect)!=0:
        pint2 = [point for index, point in enumerate(pint2) if index not in ixincorrect]
        pint2_x = np.array([x[0] for x in pint2], dtype=int)
        pint2_y = np.array([y[1] for y in pint2], dtype=int)


    #first point of pint1 added as the end point of pint2
    pint2.append(pint1[ixmin])
    pint2_x = np.array([x[0] for x in pint2], dtype=int)
    pint2_y = np.array([y[1] for y in pint2], dtype=int)

    ##selection of different range of contour points
    #x coordinates of outer contour maximum and minimum points taken as reference points

    #intersection points < first maximum point (upper contour)
    i1max1 = np.where(pint1_x < pmax1[0])[0]

    #intersection points between first maximum and minimum
    imax1min = np.where((pint1_x >= pmax1[0]) & (pint1_x <= pmin[0]))[0]

    #there are few points in this range: if there is only one point, the difference isn't computed
    if len(imax1min) > 1 :
        # Point with minimum y coordinate
        imax1minf = np.argmin(pint1_y[imax1min])
        imax1minf = imax1minf + imax1min[0]
        # deletion of previous points (points with greater y)
        index_delete = np.arange(imax1min[0],imax1minf)
        pint1 = [point for index, point in enumerate(pint1) if index not in index_delete]
        pint1_x = np.array([x[0] for x in pint1], dtype=int)
        pint1_y = np.array([y[1] for y in pint1], dtype=int)
        # indices recomputed
        imax1min = np.where((pint1_x >= pmax1[0]) & (pint1_x <= pmin[0]))[0]



    # Intersection points between the minimum and the second maximum point (upper contour)
    iminmax2 = np.where((pint1_x >= pmin[0]) & (pint1_x <= pmax2[0]))[0]
    # Intersection points > second maximum
    imax2end = np.where(pint1_x > pmax2[0])[0]
    # Point with minimum y coordinate
    if len(imax2end)!=0:
        imax2endf = np.argmin(pint1_y[imax2end])
        imax2endf = imax2endf + imax2end[0] - 1
        index_delete = np.arange(imax2end[0], imax2endf)
        # Deletion of previous points (points with greater y)
        pint1 = [point for index, point in enumerate(pint1) if index not in index_delete]
        pint1_x = np.array([x[0] for x in pint1], dtype=int)
        pint1_y = np.array([y[1] for y in pint1], dtype=int)


    #%indices recomputed
    imax2end = np.where(pint1_x > pmax2[0])[0]
    # Points > second maximum point (lower contour)
    i1max2 = np.where(pint2_x>pmax2[0])[0]

    #difference between consecutive y coordinates
    dpint1 = np.diff(pint1_y)
    dpint2 = np.diff(pint2_y)

    ##control on upper contour points

    #for points < max1, y must decrease: deletion of y that increase (positive diff)
    if len(i1max1)>1:
        ipint1delete1 = np.where(dpint1[i1max1[0]:i1max1[-2]+1] >= 0 )[0]
        #addition of 1 because I want the second point
        ipint1delete1 = ipint1delete1 + 1
    else:
        ipint1delete1 = []


    #for points ranging between max1 and min, y must increase: deletion of y that decrease (negative diff)
    if len(imax1min)>1:
        ipint1delete2=np.where(dpint1[imax1min[0]:imax1min[-2]+1] < 0 )[0]
        ipint1delete2 = ipint1delete2 + imax1min[0]
    else:
        ipint1delete2 = []


    #For points ranging between min and max2, y must decrease: deletion of y that increase (positive diff)
    if len(iminmax2)>1:
        ipint1delete3=np.where(dpint1[iminmax2[0]:iminmax2[-2]+1] >= 0 )[0]
        ipint1delete3 = ipint1delete3 + iminmax2[0]
    else:
        ipint1delete3 = []

    # For points >max2, y must increase: deletion of y that decrease (negative diff)
    if len(imax2end)>1:
        ipint1delete4 = np.where(dpint1[imax2end[0]:imax2end[-2] + 1] < 0)[0]
        ipint1delete4 = ipint1delete4 + imax2end[0]
    else:
        ipint1delete4=[]



    ipint1delete = np.array(np.sort(np.concatenate([ipint1delete1,ipint1delete2,ipint1delete3,ipint1delete4])),dtype=int)

    # Control on lower contour points
    # For points >max2, y must increase: deletion of y that decrease (negative diff)
    if len(i1max2)>1:
        ipint2delete1 = np.where(dpint2[i1max2[0]:i1max2[-2] + 1] < 0)[0]
        ipint2delete1 = np.array(ipint2delete1 + 1, dtype=int)
        pint2 = [point for index, point in enumerate(pint2) if index not in ipint2delete1]
        pint2_x = np.array([x[0] for x in pint2], dtype=int)
        pint2_y = np.array([y[1] for y in pint2], dtype=int)



    # Elimination of incorrect points
    pint1 = [point for index, point in enumerate(pint1) if index not in ipint1delete]
    pint1_x = np.array([x[0] for x in pint1], dtype=int)
    pint1_y = np.array([y[1] for y in pint1], dtype=int)






    # Intersection points between the first and second maximum points (lower
    # contour): y must remain relatively constant (vertebral body included)
    imax1max2 = np.where((pint2_x > pmax1[0]) & (pint2_x<=pmax2[0]))[0]

    # First point of the interval
    pfirst = pint2_y[imax1max2[0]]
    # Last point of the interval
    plast = pint2_y[imax1max2[0]-1]

    # If first point is lower than the last one (greater y)
    if pfirst >= plast:
        # Difference from to the first point
        diffdelete = pint2_y[imax1max2] - pfirst
    else:
        diffdelete = pint2_y[imax1max2] - plast

    # Deletion of invalid points
    ipint2delete2 = np.where((diffdelete>5)|(diffdelete<-2))[0]
    ipint2delete2 = np.array(ipint2delete2 + imax1max2[0] - 1,dtype=int)


    pint2=[point for index, point in enumerate(pint2) if index not in ipint2delete2]
    pint2_x = np.array([x[0] for x in pint2], dtype=int)
    pint2_y = np.array([y[1] for y in pint2], dtype=int)

    #intersection points < first maximum point (lower contour)
    imax1end = np.where( pint2_x <= pmax1[0])[0]

    # Point with the maximum y
    if len(pint2_y[imax1end]) != 0:
        imax1endf = np.argmax(pint2_y[imax1end])
        imax1endf = imax1endf + imax1end[0] - 1
        # Deletion of previous points (with lower y)
        pint2[imax1end[0]:imax1endf] = []
        pint2_x = np.array([x[0] for x in pint2], dtype=int)
        pint2_y = np.array([y[1] for y in pint2], dtype=int)

    # Recomputed indices
    imax1end = np.where (pint2_x <= pmax1[0])[0]
    # Difference recomputed after incorrect point deletion
    if len(imax1end)>1:
        dpint2 = np.diff(pint2_y)
        # nei punti <max1, y deve diminuire elimino le y che aumentano
        ipint2delete3 = np.where(dpint2[imax1end[0]:imax1end[-2] + 1] > 0)[0]
        # Point deletion
        ipint2delete3 = ipint2delete3 + imax1end[0]
        pint2 = [point for index, point in enumerate(pint2) if index not in ipint2delete3]
        pint2_x = np.array([x[0] for x in pint2], dtype=int)
        pint2_y = np.array([y[1] for y in pint2], dtype=int)


    #interpolation
    #upper contour points

    x1=pint1_x
    y1=pint1_y
    xq1 = np.arange(pint1_x[0], pint1_x[-1]+0.05, 0.05)


    #lower contour points
    x2 = pint2_x[::-1]
    y2 = pint2_y[::-1]
    xq2 = np.arange(pint2_x[-1], pint2_x[0]+0.05, 0.05)

    f1 = interpolate.PchipInterpolator(x1,y1)
    sp = f1(xq1)

    f2 = interpolate.PchipInterpolator(x2,y2)
    sp2 = f2(xq2)


    # mask creation for inner chest portion
    # inner contour points after interpolation
    pcontour1 = np.concatenate((xq1,xq2[-2::-1]))
    pcontour2 = np.concatenate((sp,sp2[-2::-1]))
    pcontour = [ [x,y] for x,y in zip(pcontour1,pcontour2) ]



    se = skimage.morphology.disk(5)
    contourmask = poly2mask(pcontour2,pcontour1,[I_imadjust.shape[0] , I_imadjust.shape[1]])
    contourmask = skimage.morphology.closing(contourmask,se)

    #plt.figure()
    #plt.imshow(contourmask,'gray')
    #plt.axis('off')
    #plt.draw()

    #sed = pd.read_excel('disk5.xlsx')
    #sed = np.array(sed)
    #contourmask = skimage.morphology.dilation(contourmask,sed)
    #prova
    contourmask = skimage.morphology.convex_hull_image(contourmask)

    #plt.figure()
    #plt.imshow(contourmask, 'gray')
    #plt.axis('off')
    #plt.show()


    #plt.figure()
    #plt.imshow(I_imadjust,'gray')
    #plt.scatter(x1,y1,color='red',s=20,zorder=2)
    #plt.scatter(x2,y2,color='red',s=20,zorder=2)
    #plt.plot([x[0] for x in pcontour],[x[1] for x in pcontour],color='green',zorder=1)
    #plt.axis('off')
    #plt.show()


    return pcontour,contourmask


def remove_regions_with_x_equals_zero(image):
    # Etichetta tutte le regioni connesse nell'immagine
    labeled_image = measure.label(image)
    # Trova tutte le proprietà delle regioni
    props = measure.regionprops(labeled_image)
    # Identifica le regioni che contengono almeno un punto con x=0
    regions_to_remove = []
    for prop in props:
        for point in prop.coords:
            if point[1] == 0:  # Se y=0
                regions_to_remove.append(prop.label)
                break
    # Rimuovi le regioni trovate dall'immagine
    new_image = image.copy()
    for label in regions_to_remove:
        new_image[labeled_image == label] = 0
    return new_image

def remove_regions_with_x_equals_max(image):
    # Etichetta tutte le regioni connesse nell'immagine
    labeled_image = measure.label(image)
    # Trova tutte le proprietà delle regioni
    props = measure.regionprops(labeled_image)

    # Identifica le regioni che contengono almeno un punto con x=0
    regions_to_remove = []
    for prop in props:
        for point in prop.coords:
            if point[1] == image.shape[0]-1:  # Se y=max
                regions_to_remove.append(prop.label)
                break

    # Rimuovi le regioni trovate dall'immagine
    new_image = image.copy()
    for label in regions_to_remove:
        new_image[labeled_image == label] = 0
    return new_image



def innermask_seg(Im,contourmask,maskest,tl,th):
    #elimination of the vertebral body, using the mask that isolates the inner portion of the thorax

    #inputs:
    #I: grey-scale image
    #contourmask: mask that isolates inner chest portion, resulting from contourinterpolation function
    #maskest: binary image of the chest used as mask
    #tl: threshold value for lung segmentation resulting from hist_thereshold function
    #th: threshold value for heart segmentation resulting from hist_thereshold function

    #Outputs:
    #Ic: binary image of inner chest portion (vertebral body excluded)
    #contpoint: coordinates of Ic boundary points

    #------------------------------------------
    #heart and cardiac structures segmentation
    #application of mask to isolate inner chest portion
    Inew = Mat2gray(Im.copy())
    I = Mat2gray(Im.copy())
    mask = np.logical_not(contourmask)
    Inew[mask] = 0


    #cardiac threshold applied to the new image
    Iheartpre = np.array(Inew > th,dtype=bool)



    #isolation of pixels corresponding to vessels (low area elements)
    Ivessel = np.array(Bwareafilt_N_Range(Iheartpre,1,30),dtype=bool)

    #pixels related to vessels assigned to 0
    I[Ivessel] = 0



    #isolation of pixels corresponding to the heart
    Iheart = skimage.morphology.remove_small_objects(Iheartpre, min_size=400)






    #%morpological operation of dilation
    se1h = skimage.morphology.diamond(2)
    Iheart = skimage.morphology.dilation(Iheart,se1h)
    # pixels related to vessels assigned to 0
    Iheart = np.array(Iheart,dtype=bool)

    I[Iheart] = 0



    # thresholding for inner chest portion
    # application of lung threshold

    Ithpre = I > tl



    # complementary image
    Icompl = np.invert(Ithpre)



    #application of the binary mask (after erosion operation) to delete elements outside the chest
    seI = skimage.morphology.rectangle(20, 1)
    seI = skimage.transform.rotate(seI,90)
    maskr = skimage.morphology.erosion(maskest,seI)
    mask = np.logical_not(maskr)
    Icompl[mask]=0


    #elimination of small elements
    Itho = skimage.morphology.remove_small_objects(Icompl, min_size=500)
    Itho = remove_regions_with_x_equals_zero(Itho)
    Itho = remove_regions_with_x_equals_max(Itho)



    #application of morphological operations
    Ifill = ndimage.binary_fill_holes(Itho)
    ser = skimage.morphology.diamond(2)
    Ie = skimage.morphology.opening(Ifill,ser)




    Ie = skimage.morphology.remove_small_objects(Ie, min_size=200)


    sec = pd.read_excel('disk7.xlsx')
    sec = np.array(sec)
    Ic =  skimage.morphology.closing(Ie,sec)
    Ic = ndimage.binary_fill_holes(Ic)



    #plt.figure()
    #plt.imshow(Ic, 'gray')
    #plt.draw()

    #count of inner mask elements
    L = skimage.measure.label(Ic)
    numObjects = len(skimage.measure.regionprops(L))
    #if innner mask is composed by more than 1 element
    if numObjects > 1:
        Ic = keep_n_largest_objects(Ic,1)

        #inner mask is the one resulting from previous function (with
        #vertebral body included)
        #secme = pd.read_excel('disk3.xlsx')
        #secme = np.array(secme)
        #contourmaske = skimage.morphology.erosion(contourmask,secme)
        #Ic = contourmaske.copy()

        #plt.figure()
        #plt.imshow(Ic, 'gray')
        #plt.show()




    #coordinates of inner chest contour
    contpoint, _ = cv2.findContours(MatToUint8(Ic), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    contpoint = flat_contours(contpoint)
    contpoint.reverse()
    x_contours = [element[0] for element in contpoint]
    y_contours = [element[1] for element in contpoint]
    if len(x_contours)!=0:
        xmin = np.min(x_contours)
        ixmin = np.where(x_contours == xmin)[0].min()
        contpoint = np.roll(contpoint, -ixmin, axis=0)  # TRASLO I CONTORNI


    #plt.figure()
    #plt.imshow(Im,'gray')
    #plt.scatter([x[0] for x in contpoint],[x[1] for x in contpoint],color='red',s=25)
    #plt.show()

    return Ic, contpoint


def innermask_select(Imask):

    #input:
    #Imask: binary images of inner chest region

    #Output:
    #first_slice= number of slice from which the inner contour correction
    nums = str(len(Imask))
    numrow = int(nums[0])

    position = [np.round((np.size(Imask[0],0))/2) , 10]
    i=0
    fnt=ImageFont.truetype("arial.ttf",25)
    d = []
    while i<len(Imask):
        a = Imask[i].copy()
        image_data = MatToUint8(a)
        image = Image.fromarray(image_data, mode='L')
        draw = ImageDraw.Draw(image)
        draw.text((position[0], position[1]), str(i+1), font=fnt, fill=255)
        d.append(image)
        i=i+1


    new_images = []
    for image in d:
        new_images.append (np.array(image))


    montage = skimage.util.montage(new_images, grid_shape=(numrow, round(len(Imask)/numrow)+1) , fill=0 , padding_width=10)

    def graph():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(montage, 'gray')
        ax.axis('off')
        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.show()

    def visualize_select_3():
            try:
                n = int(label1_input.get())
                if n > len(Imask):
                    message.config(text=' Error in slice selection')
                else:
                    return window.quit()
            except ValueError:
                message.config(text=' Error in slice selection')

    window = tk.Tk()
    window.geometry("400x450")
    window.title("User Selection")
    window.configure(background="cyan")
    window.wm_resizable(False,False)


    label1 = tk.Label(window, text=" Select correct inner mask:", pady=10)
    label1.grid(row=0, column=1, sticky="N", padx=20, pady=10)
    label1_input = tk.Entry()
    label1_input.grid(row=0, column=2, sticky="WE", padx=10, pady=10)
    label1_input.focus()

    message = tk.Label(window, text="", pady=10)
    message.grid(row=2, column=1, sticky="N", padx=20, pady=10)


    Button1 = tk.Button(window, text="Show Images" , command=graph)
    Button1.grid(row=1, column=1, sticky="WE", padx=50, pady=10)

    Button2 = tk.Button(window, text="Ok")
    Button2.grid(row=1, column=2, sticky="WE", padx=50, pady=10)
    Button2.config(command=visualize_select_3)

    window.mainloop()

    first_slice = int(label1_input.get())-1

    window.destroy()
    plt.close()
    return first_slice




def inner_analysis (c_corr,I_imadjust):
    #function for the segmentation of inner chest area and of lungs after correction

    #inputs:
    #c_corr: coordinates of inner chest contour points after correction process
    #I_imadjust: grey-scale image

    #outputs:
    #innermaskn: binary image of inner chest area
    #Iinner: grey-scale image of inner chest are
    #Ilung: binary image of lungs after correct segmentation
    c_corr_x = [x[0] for x in c_corr]
    c_corr_y = [x[1] for x in c_corr]

    innermask = poly2mask(c_corr_y,c_corr_x,[I_imadjust.shape[0] , I_imadjust.shape[1]])

    #morphological operators applied to inner mask
    ser = skimage.morphology.rectangle(10,10)
    innermaskn = skimage.morphology.closing(innermask,ser)
    innermaskn = ndimage.binary_fill_holes(innermaskn)


    #inner mask applied to grey-scale image
    Iinner = Mat2gray(I_imadjust)
    Iinner[np.logical_not(innermaskn)] = 0

    #background pixel assigned to a (white)
    background = np.logical_not(np.array(Iinner,dtype=bool))
    Iinner[background]=1

    #manual threshold for blacking out the vessel
    I = Mat2gray(Iinner)
    Iheart = I > 0.3
    Ivessel = Bwareafilt_N_Range(Iheart,1,50)
    I[Ivessel]=0

    #manual threshold for lung segmentation
    Ilungpre = I < 0.3

    #application of morphological operators
    Ilung1 = ndimage.binary_fill_holes(Ilungpre)
    ser = skimage.morphology.rectangle(1,3)
    Ilung2 = skimage.morphology.erosion(Ilung1,ser)
    Ilung3 = skimage.morphology.remove_small_objects(Ilung2, min_size=30)
    sec = pd.read_excel('disk5.xlsx')
    sec = np.array(sec)
    Ilung = skimage.morphology.closing(Ilung3,sec)
    Ilung = ndimage.binary_fill_holes(Ilung)
    Ilung = keep_n_largest_objects(Ilung,2)


    return innermaskn,Iinner,Ilung


def inner_contour_correction(c1,c2,yhalf,im):

    # function for inner chest contour correction based on comparison between consecutive slices

    # inputs:
    # c1: coordinates of inner contour points belonging to the correct curve
    # c2: coordinates  of inner contour points belonging to the curve that need to be corrected
    # yhalf: y position of the point located in the half of the image

    # output:
    # c2corr: coordinates of inner contour points belonging to the incorrect curve after correction

    #----------------------------
    ##lower contour points
    #yhalf is taken as reference to divide the contour in 2 half (lower half is the one that need to be corrected)

    #plt.figure()
    #plt.imshow(im,'gray')
    #plt.plot([x[0] for x in c2],[x[1] for x in c2],color='green',zorder=2,linewidth=2)




    i1 = np.where(np.array([x[1] for x in c1]) >= int(yhalf))[0]
    i2 = np.where(np.array([x[1] for x in c2]) >= int(yhalf))[0]


    c1inf = [element for index,element in enumerate(c1) if index in i1]
    c2infpre = [element for index, element in enumerate(c2) if index in i2]

    #x sorted in descending order
    ic2infx = np.argsort(np.array([x[0] for x in c2infpre]))[::-1]
    c2infpre = sorted(c2infpre, key=lambda x:x[0] , reverse=True)

    #difference between consective x: deletion of points at the same x (diff=0)
    dxc2inf = np.diff([x[0] for x in c2infpre])
    #vector has to have the same length as c2infpre (addition of 1 in the last row)
    dxc2infn = np.append(dxc2inf,1)
    #difference vector transformed in indices ('0' correspond to the points to delete, while '1' to the correct ones)
    ixdelete = dxc2infn != 0
    #deletion of points found
    c2inf = [x for x,condition in zip(c2infpre,ixdelete) if condition == True]

    ##Upper contour points
    c1suppre = [element for index,element in enumerate(c1) if index not in i1]
    c2suppre = [element for index,element in enumerate(c2) if index not in i2]

    c2suppre_x=np.array([x[0] for x in c2suppre])
    c2supxmax = np.max(c2suppre_x)
    #it takes the point with the greater y coordinate (last)
    ic2supxmax = np.where(c2suppre_x == c2supxmax)[0].max()
    interval = np.arange(0,ic2supxmax+1)
    #difference between consecutive x coordinates
    c2sup = [x for index,x in enumerate(c2suppre) if index in interval]
    c2sup_x = np.array([x[0] for x in c2sup])
    dxc2sup = np.diff(c2sup_x)
    ideletec2sup = np.concatenate((np.array([False]),dxc2sup<=0))
    c2sup = [x for x, condition in zip(c2sup, ideletec2sup) if condition == False]

    #Same analysis on c1
    c1suppre_x = np.array([x[0] for x in c1suppre])
    c1supxmax = np.max(c1suppre_x)
    # it takes the point with the grater y coordinate (last)
    ic1supxmax = np.where(c1suppre_x == c1supxmax)[0].max()
    interval = np.arange(0, ic1supxmax + 1)
    # difference between consecutive x coordinates
    c1sup = [x for index, x in enumerate(c1suppre) if index in interval]
    c1sup_x = np.array([x[0] for x in c1sup])
    dxc1sup = np.diff(c1sup_x)
    ideletec1sup = np.concatenate((np.array([False]), dxc1sup <= 0))
    c1sup = [x for x, condition in zip(c1sup, ideletec1sup) if condition == False]


    ## computation of distance between the 2 contours
    #Initialization
    dc1c2 =  [None] * len(c2inf)
    dc1c2min = [None] * len(c2inf)
    idc1c2min = [None] * len(c2inf)


    # for each point of c2inf it computes the distance between each point of c2inf and all the points of c1infper ogni punto di c2inf calcolo la distanza tra quest'ultimo e i punti di
    # (c2inf point is hold still and a vector of distances between this point and c1 inf points is generated)
    u=0
    while u < len(c2inf):
        c1inf_x = np.array([x[0] for x in c1inf])
        c1inf_y = np.array([x[1] for x in c1inf])
        c2inf_x = np.array([x[0] for x in c2inf])
        c2inf_y = np.array([x[1] for x in c2inf])

        dc1c2[u] = np.sqrt(((c2inf_x[u]-c1inf_x)**2) + ((c2inf_y[u]-c1inf_y)**2))

        #it creates a vector of distances where it inserts the minimum
        #distance of distances previuos computed and the corresponding index
        #(taken as reference for c1inf points). Thus it maintains
        #the distance between the each c2inf points and the closest point of c1inf
        dc1c2min[u] = np.min(dc1c2[u])
        idc1c2min[u] = np.argmin(dc1c2[u])
        u=u+1

    # deletion of first points of dc1dc2min vector with distance>4
    # they don't have to be corrected: lack of points in the initial part
    ifirstdelete = np.where(np.array(dc1c2min)<4)[0]
    if len(ifirstdelete)!=0:
        ifirstdelete = ifirstdelete.min()
        for i in np.arange(0, ifirstdelete+1):
            dc1c2min[i] = 0


    ilastdelete = np.where(np.array(dc1c2min)<4)[0]
    if len(ilastdelete)!=0:
        ilastdelete = ilastdelete.max()
        for i in np.arange(ilastdelete, len(dc1c2min)):
            dc1c2min[i] = 0




    ##  threshold for correction of contour points
    #Maximum distance value
    maxdc1c2 = np.floor(np.max(dc1c2min))
    #Mean distance value
    meandc1c2 = np.ceil(np.mean(dc1c2min))
    #standard deviation of distance values
    stddc1c2 = np.std(dc1c2min)


    #if standard deviation is greater than 1.8, correction is needed:
    if stddc1c2 > 1.8:
        #indices corresponding to distances > threshold chosen for correction
        iincorrect = np.where(dc1c2min >= (2 * meandc1c2))[0]
        #if there are consecutive indices, it takes only 1 index (the first one of the interval)
        #difference between consecutive indices:
        iincorrectdiff = np.diff(iincorrect) < 2
        iincorrectdiff = np.concatenate((np.array([False]),iincorrectdiff))

        #It takes indices equal to 0 (i.e. where difference isn't 1)
        irange = [x for x,condition in zip(iincorrect,iincorrectdiff) if condition == False]



        ##correction process (c2inf): done for each irange
        #Initializing
        ic2p1 = [None] * len(irange)
        ic2p2 = [None] * len(irange)
        ic1p1 = [None] * len(irange)
        ic1p2 = [None] * len(irange)
        ip = [None] * len(irange)
        ip4 = [None] * len(irange)

        p1corr = [None] * len(irange)
        p2corr = [None] * len(irange)
        p3corr = [None] * len(irange)
        p4corr = [None] * len(irange)
        p5corr = [None] * len(irange)
        p6corr = [None] * len(irange)
        p7corr = [None] * len(irange)

        x = [None] * len(irange)
        y = [None] * len(irange)

        xq = [None] * len(irange)
        yq = [None] * len(irange)

        c2infcorrtot = [None] * len(irange)

        k=0
        while k < len(irange):
            #first extreme point for interpolation: point corresponding to
            #a difference between c1 and c2 lower than 3 (points from irange to 1)

            dc1c2min_int1=[]
            z=int(irange[k])
            while z>=0:
                dc1c2min_int1.append(dc1c2min[z])
                z=z-1

            ic2p1k = np.where(np.array(dc1c2min_int1)<3)[0]
            if len (ic2p1k)!=0:
                ic2p1k = np.min(ic2p1k)
            else:
                ic2p1k = [None]

            if ic2p1k != [None]:
                ic2p1[k] = int(irange[k] - ic2p1k )
            else:
                ic2p1[k] = 0
            # second extreme point for interpolation: point corresponding to
            # a difference between c1 and c2 lower than 3 (points from irange to end)

            interval = np.arange(irange[k],len(dc1c2min),1)
            z = int(irange[k])
            dc1c2min_int2 = []
            while z < len(dc1c2min):
                dc1c2min_int2.append(dc1c2min[z])
                z=z+1
            ic2p2k = np.where(np.array(dc1c2min_int2)<3)[0]

            if len(ic2p2k):
                ic2p2k = np.min(ic2p2k)
            else:
                ic2p2k = [None]


            if ic2p2k != [None]:
                ic2p2[k] = int(irange[k] + ic2p2k )
            else:
                ic2p2[k] = int(len(dc1c2min)-1)



            ##  correction with c1inf points
            ic1p1[k] = idc1c2min[int(ic2p1[k])]
            ic1p2[k] = idc1c2min[int(ic2p2[k])]



            # 5 middle points (in total 7 points with the 2 extreme ones):
            # found by selecting 5 equally space points of c1
            ip[k] = int(np.round(np.abs(ic1p2[k] - ic1p1[k]) / 7))
            ip4[k] = int(np.round(((ic1p1[k]+2*ip[k])+(ic1p2[k]-2*ip[k]))/2))



            #interpolation points
            p1corr[k]=c2inf[ic2p1[k]]
            p2corr[k]=c2inf[ic2p2[k]]
            if 0 <= ic1p1[k] + ip[k] < len(c1inf):
                p3corr[k] = c1inf[ic1p1[k] + ip[k]]
            else:
                p3corr[k] = p2corr[k]
            if 0 <= ic1p1[k] + (2 * ip[k]) < len(c1inf):
                p4corr[k] = c1inf[ic1p1[k] + (2 * ip[k])]
            else:
                p4corr[k] = p3corr[k]
            p5corr[k]=c1inf[ip4[k]]
            p6corr[k]=c1inf[ic1p2[k] - ip[k]]
            p7corr[k]=c1inf[ic1p2[k] - (2 * ip[k])]

            #plt.scatter(p1corr[k][0], p1corr[k][1], marker='x', color='black',s=50)
            #plt.scatter(p2corr[k][0], p2corr[k][1], marker='x', color='black',s=50)
            #plt.scatter(p3corr[k][0], p3corr[k][1], marker='x', color='black',s=50)
            #plt.scatter(p4corr[k][0], p4corr[k][1], marker='x', color='black',s=50)
            #plt.scatter(p5corr[k][0], p5corr[k][1], marker='x', color='black',s=50)
            #plt.scatter(p6corr[k][0], p6corr[k][1], marker='x', color='black',s=50)
            #plt.scatter(p7corr[k][0], p7corr[k][1], marker='x', color='black',s=50)






            #a constant is added to each x coordinates in order to prevent errors
            #in interp1 function (where x values can't be unique) ]
            x[k]=[p1corr[k][0],p3corr[k][0] + 0.01,p4corr[k][0] + 0.02,p5corr[k][0] + 0.03 ,p6corr[k][0] + 0.04, p7corr[k][0] + 0.05 ,p2corr[k][0] + 0.06]
            index_sorted = np.argsort(x[k]).astype(int)
            x[k] =np.array(x[k])[index_sorted]
            y[k]=[p1corr[k][1],p3corr[k][1],p4corr[k][1],p5corr[k][1],p6corr[k][1], p7corr[k][1],p2corr[k][1]]
            y[k]=np.array(y[k])[index_sorted]



            #interpolation interval has the same number as the number of points that are deleted

            xq[k] =np.linspace(p1corr[k][0], p2corr[k][0], ic2p2[k] - ic2p1[k] + 1)
            f = interpolate.PchipInterpolator(x[k],y[k])
            yq[k] = f (xq[k])
            yq[k][-1] = p2corr[k][-1]




            #correction with points resulting from interpolation
            if k == 0:
                c2infcorrtot[k] = c2inf
            else:
                c2infcorrtot[k] = c2infcorrtot[k - 1]



            c2infcorrtot_x=[x[0] for x in c2infcorrtot[k]]
            c2infcorrtot_y=[x[1] for x in c2infcorrtot[k]]
            interval = np.arange(ic2p1[k],ic2p2[k]+1)

            a=0
            for i in interval:
                c2infcorrtot_x[i] = round(xq[k][a])
                c2infcorrtot_y[i] = round(yq[k][a])
                a=a+1

            c2infcorrtot[k] = [[x,y] for x,y in zip(c2infcorrtot_x,c2infcorrtot_y)]
            k=k+1

        if len(irange)!=0:
            c2infcorr = c2infcorrtot[-1]  # QUI è -1
            c2infcorr_y = np.array([x[1] for x in c2infcorr])
            pdelete = np.where((c2infcorr_y > 224) | (c2infcorr_y < 10))[0]
            c2infcorr = [x for index, x in enumerate(c2infcorr) if index not in pdelete]

    else:
        #Correction not necessary, c2inf is maintained as the input one
        c2infcorr = c2infpre


    c1corr = c1sup + c1inf + [c1sup[0]]
    c2corr = c2sup + c2infcorr + [c2sup[0]]

    #plt.plot([x[0] for x in c2corr], [x[1] for x in c2corr], color='blue', linewidth=1 ,zorder=1, linestyle='dashed')
    #plt.gca().invert_yaxis()
    #plt.show()



    return c2corr


def contcorrinterpolation(contcorr,I_imadjust,pmin,pmax1,yhalf):

    # function for the preparation of the inner contour belonging to the slice where indices calculation is performed
    # inputs:
    # contcorr: coordinates of inner chest contour points
    # I_imadjust: grey-scale image
    # pmin: coordinates of outer chest minimum point
    # pmax1: coordinates of outer chest first maximum point
    # yhalf: y position of the point located in the half of the image

    #outputs:
    # pcontour: coordinates of inner chest contour points after interpolation
    # max1in: coordinates of first inner chest maximum point
    # max2in: coordinates of second inner chest maximum point
    # ncontrol: if ncontrol is equal to 1, it means that inner chest contour doesn't present errors in the upper half,
    # otherwise ncontrol is equal to 2 and thus the algorithm pass to analyze the inner chest contour of the consecutive slice

    ## preparation of points for interpolation
    # elimination of points at the same x as the point with maximum x coordinate

    contcorr_x = np.array([x[0] for x in contcorr])
    contcorr_y = np.array([x[1] for x in contcorr])


    xmax = np.max(contcorr_x)
    ixmax = np.argmax(contcorr_x)
    isamexmax = np.where(contcorr_x[ixmax+1::] == xmax)[0]
    if len(isamexmax)!=0:
        index_del = isamexmax + ixmax
    else:
        index_del = []
    cnew = contcorr.copy()
    cnew = [x for index,x in enumerate(cnew) if index not in index_del]


    #partition of contour in two parts: upper and lower part
    pint1 = cnew[0:ixmax+1]
    pint2 = cnew[ixmax::]


    #x sorted in ascendent order (superior contour)
    pint1_x = np.array([x[0] for x in pint1])
    pint1_y = np.array([x[1] for x in pint1])
    ip1 = np.argsort(pint1_x)
    pint1x = np.sort(pint1_x)


    #x sorted in descendent order (inferior contour)
    pint2_x = np.array([x[0] for x in pint2])
    pint2_y = np.array([x[1] for x in pint2])
    ip2 = np.argsort(pint2_x)[::-1]
    pint2x = np.sort(pint2_x)[::-1]



    #trovo le y corrispondenti alle x ordinate
    pint1y = np.array([x for x,index in zip(pint1_y,ip1)])
    pint2y = np.array([x for x,index in zip(pint2_y,ip2)])


    pint1= [[x,y] for x,y in zip(pint1x,pint1y)]
    pint2= [[x,y] for x,y in zip(pint2x,pint2y)]



    #elimination of consecutive points with same x coordinate (previous point in the difference)
    pint1_x = np.array([x[0] for x in pint1])
    pint1_y = np.array([x[1] for x in pint1])
    pint2_x = np.array([x[0] for x in pint2])
    pint2_y = np.array([x[1] for x in pint2])


    xdelete1 = np.logical_not(np.diff(pint1_x))
    xdelete1=np.append(xdelete1,False)
    pint1_new = [x for condition,x in zip(xdelete1,pint1) if condition==False]
    pint1_x = np.array([x[0] for x in pint1_new])
    pint1_y = np.array([x[1] for x in pint1_new])


    xdelete2 = np.logical_not(np.diff(pint2_x))
    xdelete2=np.append(xdelete2,False)
    pint2_new = [x for condition,x in zip(xdelete2,pint2) if condition==False]
    pint2_x = np.array([x[0] for x in pint2_new])
    pint2_y = np.array([x[1] for x in pint2_new])


    #starting point of contour upper half: the one with minimum x coordinate
    ixmin = np.argmin(pint1_x)

    #points in pint2 with x coordinates lower than the first point of pint1 (minimum point) are deleted
    ixincorrect = np.where(pint2_x <= pint1_x[ixmin])[0]
    pint2_new = [x for index, x in enumerate(pint2_new) if index not in ixincorrect]
    pint2_x = np.array([x[0] for x in pint2_new])
    pint2_y = np.array([x[1] for x in pint2_new])


    #starting point of upper half added to lower half as the end point
    pint2_new.append(pint1_new[ixmin])
    pint2_x = np.array([x[0] for x in pint2_new])
    pint2_y = np.array([x[1] for x in pint2_new])


    ## Control on upper half points
    #indices of upper half points within the x position of outer chest
    #minimum point and first outer chest maximum point

    pmin_x = pmin[0]
    pmin_y = pmin[1]

    pmax1_x = pmax1[0]
    pmax1_y = pmax1[1]



    ix = np.where((pint1_x < (pmin_x - 5)) & (pint1_x > (pmax1_x - 15)))[0]
    #% points among the one found with a distance >4 from yhalf (errors on inner contour upper half)
    pinner = int(yhalf.copy())
    iy = np.abs(pint1_y[ix[0]:ix[-1]+1] - pinner) < 4


    #if there are points that soddisfy this condition, the inner contour
    #analyzed can't be used for indices computation and the function stops
    if any(iy):
        ncontrol = 2
        pcontour = []
        max1in = []
        max2in = []
        return pcontour,max1in,max2in,ncontrol
        #otherwise the function continues with interpolation
    else:
        ncontrol = 1

    #interpolation
    #inner contour upper half points
    x1 = pint1_x.copy()
    y1 = pint1_y.copy()
    xq1 = np.arange(pint1_x[0],pint1_x[-1]+0.5,0.5)

    #inner contour lower half
    x2 = pint2_x[::-1]
    y2 = pint2_y[::-1]
    xq2 = np.arange( pint2_x[-1], pint2_x[0]+0.5, 0.5)

    #application of interpolation
    f1 = interpolate.pchip(x1,y1)
    sp = f1(xq1)

    f2 = interpolate.pchip(x2,y2)
    sp2 = f2(xq2)

    pcontour1 = np.concatenate((xq1, xq2[-2::-1]))
    pcontour2 = np.concatenate((sp, sp2[-2::-1]))

    #Result of interpolation
    pcontour = [[x,y] for x,y in zip(pcontour1,pcontour2)]


    ##Calculation of inner contour maximum points
    #points with x coordinate lower than the outer contour minimum point(first maximum point)
    inner1 = np.where(pint1_x < pmin_x)[0]
    pinner1 = [x for index,x in enumerate(pint1_new) if index in inner1]
    pinner1_y = [x[1] for x in pinner1]

    #points with x coordinate greater than the outer contour minimum point(first maximum point)
    inner2 = np.where(pint1_x > pmin_x)[0]
    pinner2 = [x for index, x in enumerate(pint1_new) if index in inner2]
    pinner2_y = [x[1] for x in pinner2]

    #first inner contour maximum point
    imax1in = np.argmin(pinner1_y)
    max1in = pinner1[imax1in]

    #second inner contour maximum point
    imax2in = np.argmin(pinner2_y)
    max2in = pinner2[imax2in]

    #plt.figure()
    #plt.imshow(I_imadjust,'gray')
    #plt.scatter([x[0] for x in contcorr],[x[1] for x in contcorr],color='blue',s=30,zorder=1)
    #plt.plot([x[0] for x in pcontour],[x[1] for x in pcontour],color='red',zorder=2)
    #plt.show()
    
    return pcontour, max1in, max2in, ncontrol



def inner_index(pcontour,pmax1,pmax2,Isel,pixel_distance,c):

    ## function for computation of inner thoracic distances and indices
    #inputs:
    #pcontour: coordinates of inner contour points
    #pmax1: coordinates of first outer contour maximum point
    #pmax2: coordinates of second outer contour maximum point
    #Isel: slice for index computation picked by algorithm
    #pixel_distance: vector containing vertical and horizontal distances between pixels(mm)
    #c: if it is equal to 1, the index computation is performed on the same
    #slice selected by user, otherwise if the slices are different c is equal to 2

    #outputs:
    #transversed: transversed diameter
    #emitdxd: right hemithorax antero-posterior diameter (APd)
    #emitsxd: left hemithorax antero-posterior diameter (APd)
    #iAsymetry: asymmetry index
    #iFlateness: flatness index
    #minsternum: sternum position
    #maxAPd: maximum antero-posterior diameter
    #minAPd: minimum antero-posterior diameter
    #Haller_ind: Haller index
    #Correction_ind: correction index
    #depression_ind: depression index
    #pvertebral body: vertebral body position
    #--------------------------------------------

    ## Transversed diameter

    #minimum x coordinate of inner contour
    pcontour_x=[x[0] for x in pcontour]
    pcontour_y=[x[1] for x in pcontour]

    #Minimum x coordinate of inner contour
    xmin = np.min(pcontour_x)
    idmin = np.argmin(pcontour_x)
    #Maximum x coordinate of inner contour
    xmax = np.max(pcontour_x)
    idmax = np.argmin(pcontour_x)

    #y coordinate of points with maximum and minimum x coordinate
    y_xmin = pcontour[idmin][1]
    y_xmax = pcontour[idmax][1]

    #extreme points of transversed diamter
    p1contour = [xmin, y_xmin]
    p2contour = [xmax, y_xmax]
    #transversed diameter computation: distance between the point with minimum
    #x coordinate and the point at the same y coordinate on the opposite side
    transversed_p = math.sqrt( (p2contour[0] - p1contour[0])**2 + (p2contour[1] - p1contour[1])**2 )
    transversed = (transversed_p * pixel_distance[0]) / 10

    ## Sternum position

    # point corresponding to max1 point on the inner contour (upper half of contour)
    imax1 = np.where(pcontour_x==np.array(pmax1[0]).astype(np.float64))[0].min()
    # point corresponding to max2 point on the inner contour (upper half of contour)
    imax2 = np.where(pcontour_x==np.array(pmax2[0]).astype(np.float64))[0].min()


    #minimum point of inner contour: sternum position (point with the maximum y
    #coordinate among the points found above)
    minsternumv = np.max(pcontour_y[imax1:imax2+1])
    iminsternuma = np.where(pcontour_y[imax1:imax2+1] == minsternumv)[0]
    # Among the points with the same y coordinate it is selected the middle one
    iminsternum = np.abs((iminsternuma[-1] + iminsternuma[0])/2) + imax1 - 1
    minsternum = pcontour[int(iminsternum)]


    # right and left hemithorax antero-posterior distances

    # points on the inner contour at the same x coordinate of the two outer contour maximum points



    iemitsx = np.where(pcontour_x == np.array(pmax2[0]).astype(np.float64))[0]
    iemitdx = np.where(pcontour_x == np.array(pmax1[0]).astype(np.float64))[0]


    #Antero-posterior distances computation
    emitsxd_p = np.sqrt((pcontour_x[iemitsx[0]] - pcontour_x[iemitsx[1]])**2 +  (pcontour_y[iemitsx[0]] - pcontour_y[iemitsx[1]])**2)
    emitsxd = (emitsxd_p * pixel_distance[0])/10

    emitdxd_p = np.sqrt((pcontour_x[iemitdx[0]] - pcontour_x[iemitdx[1]])**2 +  (pcontour_y[iemitdx[0]] - pcontour_y[iemitdx[1]])**2)
    emitdxd = (emitdxd_p * pixel_distance[0])/10

    # Asymmetry index computation
    iAsymmetry = emitdxd / emitsxd

    # Flatness index computation
    iFlatness = transversed / (np.max([emitsxd, emitdxd]))

    ## maxAPd and minAPd computation

    #vertebral body position: maximum point on the lower half on inner contour
    #point corresponding to max1 point on the inner contour (lower half of contour)
    i2max1 = np.where(pcontour_x == np.array(pmax1[0]).astype(np.float64))[0].max()
    #point corresponding to max2 point on the inner contour (lower half of contour)
    i2max2 = np.where(pcontour_x == np.array(pmax2[0]).astype(np.float64))[0].max()




    #vertebral body position: point with minimum y coordinate among the x
    #position of the two outer contour maximum points (the starting point is max2
    #because the x coordinates of lower half are in descending)
    ivertebralPoint = np.argmin (pcontour_y[i2max2:i2max1+1])
    ivertebralPoint = ivertebralPoint + i2max2 - 1
    vertebralPoint = pcontour[ivertebralPoint]



    #if c=2 the user is asked to selected the sternum position and the
    #vertebral body position
    if c==2:

        def onclick(event, ax, points):
            if event.inaxes == ax:
                if len(points) < 2:
                    ax.plot(event.xdata, event.ydata, 'ro' if len(points) == 0 else 'bo')
                    points.append((event.xdata, event.ydata))
                    plt.draw()

                    if len(points) == 2:
                        plt.pause(2)  # Mostra i punti per 2 secondi
                        plt.close()

        points = []  # Lista per memorizzare i punti inseriti dall'utente

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=None,
                            hspace=None)  # Regola i margini della figura
        ax.imshow(Isel, cmap='gray')
        ax.axis('off')

        # Regola la posizione e le dimensioni dell'asse
        ax.set_position([0.1, 0.1, 0.8, 0.8])

        # Testi posizionati nella parte inferiore della finestra
        fig.text(0.1, 0.03, "1. Insert sternum position", fontsize=15, ha='left')
        fig.text(0.1, 0.01, "2. Insert vertebral body position", fontsize=15, ha='left')

        # Connessione dell'evento di clic del mouse
        onclick_partial = partial(onclick, ax=ax, points=points)
        cid = fig.canvas.mpl_connect('button_press_event', onclick_partial)

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

        plt.show()

        minsternum = [round(x) for x in points[0]]
        vertebralPoint = [round(x) for x in points[1]]



    #extreme points on inner contour with the same y coordinate as the vertebral body position punti sul
    #contorno interno alla stessa y del punto sulla colonna
    #(points with minimum and maximum x coordinate)
    p1vertebraline = [xmin, vertebralPoint[1]]
    p2vertebraline = [xmax, vertebralPoint[1]]

    # Maximum left hemithorax point
    pmaxemitsx = pcontour[iemitsx[0]]
    # Maximum right hemithorax point
    pmaxemitdx = pcontour[iemitdx[0]]



    #point at the same x coordinate of the maximum left/right hemithorax point and at
    #the vertebral body position
    pvlinemitsx = [pmaxemitsx[0], p1vertebraline[1]]
    pvlinemitdx = [pmaxemitdx[0], p1vertebraline[1]]

    # max APd computation: distance between the maximum right/left hemithorax point
    # and at the vertebral body y position
    maxAPd_psx = math.sqrt( (pvlinemitsx[0] - pmaxemitsx[0])**2 + (pvlinemitsx[1] - pmaxemitsx[1])**2 )
    maxAPd_pdx = math.sqrt( (pvlinemitdx[0] - pmaxemitdx[0])**2 + (pvlinemitdx[1] - pmaxemitdx[1])**2 )

    # Maximum between left and right max APd distances
    maxAPd_p = np.max([maxAPd_psx, maxAPd_pdx])
    maxAPd = (maxAPd_p * pixel_distance[0]) / 10

    #point at the same x coordinate of sternum and at the the vertebral body y position
    pvertebralbody = [minsternum[0], p1vertebraline[1]]

    #Min APd calculation: minimum distance between sternum and vertebral body
    minAPd_p = math.sqrt((pvertebralbody[0] - minsternum[0])**2 + (pvertebralbody[1] - minsternum[1])**2)
    minAPd = (minAPd_p * pixel_distance[0])/10

    # Haller index
    Haller_ind = transversed / minAPd

    # correction index
    Correction_ind = ((maxAPd - minAPd) / maxAPd) * 100

    # Depression index
    depression_ind = emitsxd / minAPd

    ## image with inner thoracic distances

    plt.figure()
    plt.imshow(Isel,'gray')

    # point at sternum position
    plt.scatter(minsternum[0], minsternum[1], color='red', s=15, zorder=2)
    # extreme points of left APd
    plt.scatter([pcontour[iemitsx[0]][0], pcontour[iemitsx[1]][0]], [pcontour[iemitsx[0]][1], pcontour[iemitsx[1]][1]],
                color='red', s=15,zorder=2)
    # extreme points of right APd
    #plt.scatter([pcontour[iemitdx[0]][0], pcontour[iemitdx[1]][0]], [pcontour[iemitdx[0]][1], pcontour[iemitdx[1]][1]],
                #color='red', s=15,zorder=2)
    # point at vertebral body position
    #plt.scatter(p1contour[0],p1contour[1],color='red',s=15,zorder=2)
    #plt.scatter(p2contour[0],p2contour[1], color='red', s=15, zorder=2)
    plt.scatter(vertebralPoint[0], vertebralPoint[1], color='red', s=15,zorder=2)
    #plt.scatter( pcontour[iemitdx[0]][0] , pcontour[iemitdx[0]][1],color='red',s=15,zorder=2)
    #plt.scatter( pcontour[iemitdx[0]][0] , vertebralPoint[1],color='red',s=15,zorder=2)

    # transversed diameter
    #plt.plot([p2contour[0], p1contour[0]], [p2contour[1], p1contour[1]], color='red', linewidth=1.5,zorder=1)
    # right and left hemithorax APd
    plt.plot([pcontour[iemitsx[0]][0],pcontour[iemitsx[1]][0]],[pcontour[iemitsx[0]][1],pcontour[iemitsx[1]][1]], color='blue', linewidth=1.5,zorder=1)
    #plt.plot([pcontour[iemitdx[0]][0],pcontour[iemitdx[1]][0]],[pcontour[iemitdx[0]][1], pcontour[iemitdx[1]][1]], color='blue', linewidth=1.5,zorder=1)
    #plt.scatter([pcontour[iemitdx[0]][0], pcontour[iemitdx[1]][0]], [pcontour[iemitdx[0]][1], pcontour[iemitdx[1]][1]], color='red', s=15, zorder=2)
    # min Apd
    plt.plot([pvertebralbody[0], minsternum[0]], [pvertebralbody[1], minsternum[1]], color='yellow', linewidth=1.5,zorder=1)
    # horizontal line at the position of vertebral body
    plt.plot([p1vertebraline[0], p2vertebraline[0]], [p1vertebraline[1], p2vertebraline[1]], color='cyan', linewidth=1.5,zorder=1)
    #max APd: maximum distance between left (index imaxAPd=1) and right (index imaxAPd=2)one
    if maxAPd_p == maxAPd_psx:
        #plt.plot([ pcontour[iemitsx[0]][0], pvlinemitsx[0]] , [ pcontour[iemitsx[0]][1], pvlinemitsx[1]],color='green',linewidth=1.5,zorder=1)
        #plt.plot(pvlinemitsx[0], pvlinemitsx[1], color='red',markersize=6,zorder=1)
        print('-')
    else:
        #plt.plot([ pcontour[iemitdx[0]][0], pvlinemitdx[0]] , [ pcontour[iemitdx[0]][1], pvlinemitdx[1]],color='green', linewidth=1.5,zorder=1)
        #plt.plot(pvlinemitdx[0], pvlinemitdx[1], color='red', markersize=6,zorder=1)
        print('o')

    plt.show()


    return transversed,emitsxd,emitdxd,iAsymmetry,iFlatness,minsternum,maxAPd,minAPd,Haller_ind,Correction_ind,depression_ind,pvertebralbody



def Montage_Matlab (images,title):
    images_montage=[]
    for image in images:
        images_montage.append(util.img_as_ubyte(MatToUint8(image)))
    montage = util.montage(images_montage,padding_width=5)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)
    ax = plt.gca()
    ax.axis('off')
    io.imshow(montage,cmap='gray')
    io.show()


def Montage_Matlab_draw (images,title):
    images_montage=[]
    for image in images:
        images_montage.append(util.img_as_ubyte(image))
    montage = util.montage(images_montage,padding_width=5)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)
    ax = plt.gca()
    ax.axis('off')
    plt.imshow(montage,cmap='gray')
    plt.draw()


def Bw_area_n(image, num):
    labeled_image = skimage.measure.label(image)
    regions = skimage.measure.regionprops(labeled_image)
    regions.sort(key=lambda x: x.area, reverse=True)
    num_regions_to_keep = min(num, len(regions))
    largest_elements = regions[:num_regions_to_keep]
    image_new = np.zeros_like(image)
    for element in largest_elements:
        image_new[element.coords[:, 0], element.coords[:, 1]] = 1
    return image_new


def K_means(image,centroids,iterations=10):
    n_pixels = image.shape[0] * image.shape[1]
    pixels = np.reshape(image, (n_pixels, 1))

    # Applica k-means con 2 cluster
    n_clusters = centroids
    n_init = iterations  # Puoi scegliere il numero di inizializzazioni che preferisci
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_init)
    kmeans.fit(pixels)

    # Prendi i centroidi dei cluster
    centroids = kmeans.cluster_centers_

    # Ordina i centroidi in ordine crescente
    sorted_centroids = np.sort(centroids, axis=0)
    return sorted_centroids


def Heart_segmentation(Image,Inner_mask,Polm_mask):
    se = pd.read_excel('disk5.xlsx')
    se = np.array(se)

    Heart_mask =   Mat2gray(cv2.subtract(MatToUint8(Inner_mask),MatToUint8(Polm_mask)))
    Heart_mask =   np.array(skimage.morphology.binary_opening(Bwareafilt(Heart_mask),se),dtype=bool)
    Heart_mask =   ndimage.binary_fill_holes(Heart_mask)
    Heart_mask =   skimage.morphology.closing(Heart_mask,se)
    Heart_mask =   keep_n_largest_objects(Heart_mask,1)
    Heart_image =  cv2.bitwise_and(  MatToUint8(Image) ,  MatToUint8(Image) , mask=MatToUint8(Heart_mask))


    return Heart_mask,Heart_image




def heart_index(image,heart_mask,pixel_distance,pcontour,pmax1,pmax2):

    #Seeing the image instograms, we can separate our data in 4 groups (Background, Lungs, Body, Heart+Vertebral Column)
    #centroids_k = K_means(image,4)

    #We select a threshold between the body and the Heart+Column
    #threshold = np.mean([centroids_k[2],centroids_k[3]])
    #image_thresholded = image > threshold

    #Operations to Fill the vertebral column, delete vessels, and select the area with the centroid that has the lowest y
    #image_thresholded = ndimage.binary_fill_holes(image_thresholded)
    #se = pd.read_excel('disk3.xlsx')
    #se = np.array(se)
    #image_thresholded = skimage.morphology.binary_opening(image_thresholded,se)
    #image_thresholded = MatToUint8(image_thresholded)

    #labeled_image = measure.label(image_thresholded)
    #regions = measure.regionprops(labeled_image)

    #lowest_centroid = None
    #lowest_y = -float('inf')
    #for region in regions:
       # centroid_y, centroid_x = region.centroid

        #if centroid_y > lowest_y:
         #   lowest_y = centroid_y
          #  lowest_centroid = (centroid_x, centroid_y)

    #result_image = np.zeros_like(image)
    #if lowest_centroid is not None:
       # region_label = labeled_image[int(lowest_centroid[1]), int(lowest_centroid[0])]
        #result_image[labeled_image == region_label] = 255

    #result_image = Mat2gray(result_image)

    x_max = np.argmax(np.array([x[0] for x in pcontour]))

    pcontour_x = [x[0] for x in pcontour]
    pcontour_x = pcontour_x[0:x_max + 1]

    pcontour_y = [x[1] for x in pcontour]
    pcontour_y = pcontour_y[0:x_max + 1]

    ## Sternum position
    # point corresponding to max1 point on the inner contour (upper half of contour)
    imax1 = np.where(pcontour_x == np.array(pmax1[0]).astype(np.float64))[0]
    if len(imax1) != 0:
        imax1 = imax1.min()
    else:
        imax1 = np.abs(pcontour_x - np.array(pmax1[0]).astype(np.float64)).argmin()
    # point corresponding to max2 point on the inner contour (upper half of contour)
    imax2 = np.where(pcontour_x == np.array(pmax2[0]).astype(np.float64))[0]
    if len(imax2) != 0:
        imax2 = imax2.min()
    else:
        imax2 = np.abs(pcontour_x - np.array(pmax2[0]).astype(np.float64)).argmin()

    # minimum point of inner contour: sternum position (point with the maximum y
    # coordinate among the points found above)

    minsternumv = np.max(pcontour_y[imax1:imax2 + 1])
    iminsternuma = np.where(pcontour_y[imax1:imax2 + 1] == minsternumv)[0]
    # Among the points with the same y coordinate it is selected the middle one
    iminsternum = np.abs((iminsternuma[-1] + iminsternuma[0]) / 2) + imax1 - 1
    min_coord = pcontour[int(iminsternum)]
    min_coord = [min_coord[0], min_coord[1]]




    heart_contours,_  = cv2.findContours(MatToUint8(heart_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    heart_contours = flat_contours(heart_contours)
    #x_contours = np.array([x[0] for x in heart_contours])

    #Left cardiac lateral shift
    #imin = np.argmin(x_contours)
    #imax = np.argmax(x_contours)

    #pright = heart_contours[imin]
    #pleft = heart_contours[imax]

    #Right_lateral_distance = (lowest_centroid[0] - pright[0]) * pixel_distance[0]/10
    #Left_lateral_distance = (pleft[0] - lowest_centroid[0]) * pixel_distance[0]/10
    #Cardiac_left_lateral_shift = (Left_lateral_distance / (Left_lateral_distance + Right_lateral_distance)) * 100

    #Asimmetry index
    massimo_diametro_verticale = 0
    psup_max=heart_contours[0]
    pinf_max=heart_contours[0]
    for punto1 in heart_contours:
        for punto2 in heart_contours:
            if punto1 != punto2:
                distanza_verticale = abs(punto1[1] - punto2[1])
                if (distanza_verticale > massimo_diametro_verticale) & (punto1[0]==punto2[0]):
                    massimo_diametro_verticale = distanza_verticale
                    psup_max=punto1
                    pinf_max=punto2

    widest_paramedian_antero_posterior_diameter = massimo_diametro_verticale * pixel_distance[0]/10

    white_points =np.argwhere(heart_mask!=0)
    center_y = np.mean(white_points[:, 0])


    psup_min=heart_contours[0]
    pinf_min=heart_contours[0]
    min_dist = float('inf')
    for punto in heart_contours:
        dist = np.abs( punto[0] - min_coord[0])
        if (dist < min_dist) & (punto[1] < center_y):
            psup_min = punto
            min_dist = dist


    min_dist = float('inf')
    for punto in heart_contours:
        dist = np.abs(punto[0] - min_coord[0])
        if (dist < min_dist) & (punto[1] > center_y):
            pinf_min = punto
            min_dist = dist

    print(pinf_min)


    minimo_diametro_verticale = np.abs(np.linalg.norm(np.array(psup_min)-np.array(pinf_min)))

    narrowest_diameter_of_the_heart_at_level_of_the_xiphoid_process = minimo_diametro_verticale * pixel_distance[0]/10
    asymmetry_index = (widest_paramedian_antero_posterior_diameter / narrowest_diameter_of_the_heart_at_level_of_the_xiphoid_process)

    #Cardiac compression index
    massimo_diametro_trasversale = 0
    p_sx = heart_contours[0]
    p_dx = heart_contours[0]
    for punto1 in heart_contours:
        for punto2 in heart_contours:
            if punto1 != punto2:
                distanza_orizzontale = abs(punto1[0] - punto2[0])
                if (distanza_orizzontale > massimo_diametro_trasversale) & (punto1[1] == punto2[1]):
                    massimo_diametro_trasversale = distanza_orizzontale
                    p_sx = punto1
                    p_dx = punto2

    transverse_diameter = massimo_diametro_trasversale * pixel_distance[0]/10

    cardiac_compression = (transverse_diameter/narrowest_diameter_of_the_heart_at_level_of_the_xiphoid_process)

    plt.figure()
    plt.imshow(image,'gray')
    #plt.scatter(lowest_centroid[0],lowest_centroid[1],color='red',zorder=2)
    #plt.scatter(pright[0],pright[1],color='red',zorder=4)
    #plt.scatter(pleft[0],pleft[1],color='red',zorder=2)
    plt.scatter(psup_max[0],psup_max[1],color='red',zorder=2)
    plt.scatter(pinf_max[0],pinf_max[1],color='red',zorder=2)
    plt.scatter(p_sx[0], p_sx[1], color='red', zorder=2)
    plt.scatter(p_dx[0], p_dx[1], color='red', zorder=2)
    plt.scatter(psup_min[0], psup_min[1], color='red', zorder=2)
    plt.scatter(pinf_min[0], pinf_min[1], color='red', zorder=2)
    #plt.plot([lowest_centroid[0],lowest_centroid[0]],[0,image.shape[0]-1], color='red',zorder=5)
    #plt.plot([pleft[0], lowest_centroid[0]], [pleft[1], pleft[1]],linewidth=2, color='blue', zorder=1)
    #plt.plot([pright[0], lowest_centroid[0]], [pright[1], pright[1]],linewidth=2, color='green', zorder=3)
    plt.plot([pinf_max[0], psup_max[0]], [pinf_max[1], psup_max[1]], linewidth=2, color='orange', zorder=1)
    plt.plot([p_sx[0], p_dx[0]], [p_sx[1], p_dx[1]], linewidth=2, color='black', zorder=4)
    plt.plot([pinf_min[0], psup_min[0]], [pinf_min[1], psup_min[1]], linewidth=2, color='cyan', zorder=1)
    plt.show()

    #return Right_lateral_distance, Left_lateral_distance, widest_paramedian_antero_posterior_diameter, narrowest_diameter_of_the_heart_at_level_of_the_xiphoid_process, transverse_diameter, Cardiac_left_lateral_shift, asymmetry_index, cardiac_compression
    return widest_paramedian_antero_posterior_diameter, narrowest_diameter_of_the_heart_at_level_of_the_xiphoid_process, transverse_diameter, asymmetry_index, cardiac_compression




def delete_select(heart_images):

        # Funzione chiamata quando l'utente fa clic su un'immagine
        def toggle_selezione(index):
            selezioni[index] = not selezioni[index]   #Questo cambia le selezioni
            update_immagini()

        # Funzione chiamata quando l'utente fa clic su "OK"
        def conferma_selezioni():
            selezionate = [i for i, selezionata in enumerate(selezioni) if selezionata]
            messagebox.showinfo("Selezione confermata", f"Hai selezionato le immagini {selezionate}.")
            root.destroy()

        # Funzione per caricare un'immagine
        def carica_immagini():
            for index in range(len(heart_images)):
                image = heart_images[index].copy()  # Usa l'immagine dalla lista
                image = Image.fromarray(image)
                image_tk = image.resize((200, 200), Image.LANCZOS)
                images[index] = ImageTk.PhotoImage(image_tk)
            update_immagini()

        def crea_button(index):
            row = index // num_colonne
            column = index % num_colonne
            button = tk.Button(frameses[index], image=images[index], bg="white", borderwidth=0, relief="flat",
                               command=lambda idx=index: toggle_selezione(idx))
            button.grid(row=row, column=column, sticky="nsew")
            return button

        # Funzione per aggiornare le immagini visualizzate
        def update_immagini():
            for i in range(len(images)):
                if selezioni[i]:
                    # Cambia il colore di sfondo del frame a verde quando l'immagine è selezionata
                    frameses[i].config(bg="green",highlightbackground="green", highlightcolor="green", highlightthickness=6)
                else:
                    # Rimuovi il colore di sfondo verde quando l'immagine non è selezionata
                    frameses[i].config(bg="white",highlightbackground="white", highlightcolor="white", highlightthickness=0)
                    # Aggiungi l'immagine all'interno del frame

                # Rimuovi tutti i widget esistenti all'interno del frame
                for widget in frameses[i].winfo_children():
                    if widget != images[i]:
                        widget.destroy()

                # Crea un nuovo Button con l'immagine e la funzione di gestione degli eventi
                button = crea_button(i)


        # Creazione della finestra principale
        root = tk.Tk()
        root.title("Selezione di Immagini")

        numtot = len(heart_images)
        num_righe = int(math.sqrt(numtot))
        num_colonne = (numtot + num_righe - 1) // num_righe

        # Lista per tenere traccia delle immagini selezionate
        selezioni = [False] * len(heart_images)

        # Lista per immagazzinare le immagini caricate
        images = [None] * len(heart_images)

        # Etichette per visualizzare le immagini
        frameses = []

        for i in range(len(images)):
                frame = tk.Frame(root, width=200, height=200, bg="white", borderwidth=2, relief="solid")
                row = i // num_colonne
                column = i % num_colonne
                frame.grid(row=row, column=column, padx=5, pady=5)
                button = tk.Button(frame, image=images[i], bg="white", borderwidth=0, relief="flat", command=lambda index=i: toggle_selezione(index))
                button.grid(row=row, column=column, sticky="nsew")
                #frame.bind("<Button-1>", lambda event, index=i: frame_cliccato(event))
                #image_label = tk.Label(frame, image=images[i], bg="white",borderwidth=0)  # Imposta borderwidth a 0 per rimuovere il bordo dell'immagine
                #image_label.grid(row=row, column=column, sticky="nsew")  # Utilizza sticky per far espandere il label
                #image_label.bind("<Button-1>", lambda event, index=i: frame_cliccato(event))
                frameses.append(frame)

        carica_immagini()

        # Pulsante per confermare la selezione
        ok_button = tk.Button(root, text="OK", command=conferma_selezioni)
        ok_button.grid(row=num_righe, column=0, columnspan=num_colonne, pady=10)

        # Avvia l'interfaccia grafica
        root.mainloop()

        return np.where(selezioni)[0]



def aorta_segmentation(im_heart,mask_heart):



    im_heart_mod = MatToUint8(im_heart)

    plt.figure()
    plt.imshow(im_heart_mod,'gray')
    plt.draw()

    im_heart_mod = cv2.medianBlur(im_heart_mod, 5)

    plt.figure()
    plt.imshow(im_heart_mod, 'gray')
    plt.draw()

    im_heart_mod = cv2.GaussianBlur(im_heart_mod, (5, 5), sigmaX=0, sigmaY=0)

    plt.figure()
    plt.imshow(im_heart_mod, 'gray')
    plt.draw()

    Th2,_ = cv2.threshold(im_heart_mod, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Th1 = Th2 / 5

    edges = cv2.Canny(im_heart_mod, threshold1=Th1, threshold2=Th2)

    plt.figure()
    plt.imshow(edges, 'gray')
    plt.draw()

    kernel = np.ones((2,2)) # Puoi regolare le dimensioni del kernel
    #kernel = np.array(pd.read_excel('disk3.xlsx'))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)

    plt.figure()
    plt.imshow(closed_edges, 'gray')
    plt.draw()

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    aorta_mask = np.zeros_like(edges)
    lowest_y = -float('inf')
    for contour in contours:
        image = np.zeros_like(edges)
        cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)

        image = ndimage.binary_fill_holes(image)
        image = cv2.erode(MatToUint8(image),kernel,iterations=1)
        coordinate_bianche = np.argwhere(image != 0)
        if len(coordinate_bianche)!=0:
            centroide_y = np.mean(coordinate_bianche[:, 0])
            if centroide_y > lowest_y:
                lowest_y = centroide_y
                aorta_mask = image.copy()

    plt.figure()
    plt.imshow(aorta_mask, 'gray')
    plt.draw()

    kernel = skimage.morphology.disk(7)
    aorta_mask = cv2.dilate(MatToUint8(aorta_mask),kernel)

    plt.figure()
    plt.imshow(aorta_mask, 'gray')
    plt.draw()

    se = skimage.morphology.disk(3)
    result_mask =  cv2.subtract(MatToUint8(mask_heart),aorta_mask)
    result_mask = Mat2gray(result_mask)
    result_mask = Bwareafilt(result_mask)
    result_mask = np.array(skimage.morphology.binary_opening (result_mask,se),dtype=bool)

    plt.figure()
    plt.imshow(result_mask, 'gray')
    plt.draw()

    heart_without_aorta = cv2.bitwise_and(MatToUint8(im_heart), MatToUint8(im_heart), mask=MatToUint8(result_mask))

    plt.figure()
    plt.imshow(heart_without_aorta, 'gray')
    plt.show()

    #METTI HEART WITHOUT AORTA
    return heart_without_aorta, result_mask




def vena_cava_segmentation(im_heart,mask_heart):

    im_heart_mod = MatToUint8(im_heart)
    im_heart_mod = cv2.medianBlur(im_heart_mod, 5)
    im_heart_mod = cv2.GaussianBlur(im_heart_mod, (5, 5), sigmaX=1.5, sigmaY=1.5)

    Th2,_ = cv2.threshold(im_heart_mod, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Th1 = Th2 / 5

    edges = cv2.Canny(im_heart_mod, threshold1=Th1, threshold2=Th2)

    
    kernel = MatToUint8(np.ones((2,2)))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    vena_mask = np.zeros_like(edges)
    lowest_sum = float('inf')
    max_y = edges.shape[0] - 1
    for contour in contours:
        image = np.zeros_like(edges)
        cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
        image = ndimage.binary_fill_holes(image)
        image = cv2.erode(MatToUint8(image),kernel,iterations=1)
        coordinate_bianche = np.argwhere(image != 0)

        if len(coordinate_bianche)!=0:
            centroid_x = np.mean(coordinate_bianche[:, 1])
            centroid_y = np.mean(coordinate_bianche[:, 0])
            transformed_y = max_y - centroid_y
            current_sum = centroid_x + transformed_y

            if current_sum < lowest_sum:
                lowest_sum = current_sum
                vena_mask = image.copy()


    kernel = skimage.morphology.disk(5)
    vena_mask = cv2.dilate(MatToUint8(vena_mask),  kernel, iterations=1)


    se = skimage.morphology.disk(3)
    result_mask =  cv2.subtract(MatToUint8(mask_heart),vena_mask)
    result_mask = Mat2gray(result_mask)
    result_mask = Bwareafilt(result_mask)
    result_mask = np.array(skimage.morphology.binary_opening (result_mask,se),dtype=bool)


    heart_without_vena = cv2.bitwise_and(MatToUint8(im_heart), MatToUint8(im_heart), mask=MatToUint8(result_mask))

    return heart_without_vena, result_mask



def aorta_segmentation_snake(images):

    original_image = images[0].copy()

    # Scala per l'ingrandimento dell'immagine
    scale_percent = 500  # Modifica questa percentuale secondo necessità
    width = int(original_image.shape[1] * scale_percent / 100)
    height = int(original_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow('CT Image')
    cv2.imshow('CT Image', resized_image)

    # Funzione di callback per la selezione dei punti del contorno
    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(resized_image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow('CT Image', resized_image)

    def smooth_contour(contour, num_points=30):
        # Estrai le coordinate x e y dal contorno
        x, y = contour[:, 0], contour[:, 1]
        x = np.append(x, contour[0, 0])
        y = np.append(y, contour[0, 1])
        # Crea una funzione interpolante per x e y
        f_x = interp1d(np.arange(len(x)), x, kind='cubic', fill_value='extrapolate')
        f_y = interp1d(np.arange(len(y)), y, kind='cubic', fill_value='extrapolate')
        # Genera nuovi indici uniformemente spaziati
        new_indices = np.linspace(0, len(x) - 1, num_points)
        # Calcola le nuove coordinate x e y
        new_x, new_y = f_x(new_indices), f_y(new_indices)
        # Combina le nuove coordinate in un nuovo contorno
        new_contour = np.column_stack((new_x, new_y))
        return new_contour



    # Crea una lista per memorizzare i punti del contorno
    points = []

    # Attendi l'input interattivo per selezionare i punti del contorno
    cv2.setMouseCallback('CT Image', select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    scale_factor_x = original_image.shape[1] / width
    scale_factor_y = original_image.shape[0] / height
    original_points = [(int(x * scale_factor_x), int(y * scale_factor_y)) for x, y in points]

    initial_contour = np.array(original_points)
    # Estrai il contorno risultante dalla minimizzazione
    initial_contour = initial_contour.reshape((-1, 2))
    initial_contour = smooth_contour(initial_contour)



    # Imposta i parametri alpha e beta e sigma
    alpha_par = -0.015
    beta_par = 100
    gamma_par = 1E-2
    sigma_par = 1


    optimized_contour=[None]*len(images)
    for i in range(len(images)):
        if i==0:
            optimized_contour[i] = active_contour(images[i],initial_contour, alpha=alpha_par, beta=beta_par, gamma=gamma_par, boundary_condition='periodic',max_num_iter=2500)
        else:
            optimized_contour[i] = active_contour(images[i],optimized_contour[i-1], alpha=alpha_par, beta=beta_par, gamma=gamma_par,boundary_condition='periodic',max_num_iter=2500)
        plt.figure()
        plt.imshow(images[i], 'gray')
        for point in optimized_contour[i]:
            plt.scatter(point[0], point[1])
        plt.show()





def aorta_segmentation_watershed(images):

        def morphological_center_filter(image):
            # Definizione dell'elemento strutturante B1 come quadrato di dimensione 3
            kernel = np.ones((3, 3), np.uint8)

            # Esecuzione delle operazioni morfologiche: dilatazione, erosione, apertura e chiusura
            dilated = cv2.dilate(image, kernel, iterations=1)
            eroded = cv2.erode(image, kernel, iterations=1)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            # Calcolo di f1(x) = γB1(φB1(γB1(f))) e f2(x) = φB1(γB1(φB1(f)))
            f1 = cv2.erode(cv2.dilate(cv2.erode(image, kernel), kernel), kernel)
            f2 = cv2.dilate(cv2.erode(cv2.dilate(image, kernel), kernel), kernel)

            # Applicazione della formula β(f)(x) = min(max(f(x), f1(x)), f2(x))
            beta_filtered = cv2.min(cv2.max(image, f1), f2)

            plt.figure()
            plt.title('beta')
            plt.imshow(beta_filtered,'gray')
            plt.show()

            return beta_filtered

        def top_hat(image):
            # Carica l'immagine del filtro morfologico "morfological center" β(f)(x)
            beta_image = morphological_center_filter(image)  # L'immagine in scala di grigi

            # Calcola il cerchio di raggio variabile B2
            radius = aorta_diameter / 2

            # Crea il cerchio strutturante B2
            kernel_radius = np.sqrt(2 * radius)  # Il raggio del cerchio è il diametro della matrice del kernel
            kernel_size = int(2 * kernel_radius) + 1  # La dimensione del kernel deve essere dispari
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Applica l'apertura morfologica
            gamma_beta_image = cv2.morphologyEx(beta_image, cv2.MORPH_OPEN, kernel)

            # Calcola la trasformazione Top-Hat
            rho_plus_image = cv2.subtract(beta_image, gamma_beta_image)

            plt.figure()
            plt.title('rho plus')
            plt.imshow(rho_plus_image, 'gray')
            plt.show()

            return rho_plus_image




        def rho_tilde_plus_closing(image):
            # Load the image resulting from the previous Top-Hat transformation rho_plus_image
            rho_plus_image = top_hat(image)  # Assuming it's a grayscale image

            # Define the circular structuring element B3 with a radius of 7
            radius = 7
            kernel_size = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Apply the closing filter
            rho_tilde_plus_image = cv2.morphologyEx(rho_plus_image, cv2.MORPH_CLOSE, kernel)

            plt.figure()
            plt.title('rho tilde plus')
            plt.imshow(rho_tilde_plus_image, 'gray')
            plt.show()

            return rho_tilde_plus_image




        def rho_Image(image):
            # Load the image resulting from the previous closing operation rho_tilde_plus_image
            rho_tilde_plus_image = rho_tilde_plus_closing(image)  # Assuming it's a grayscale image

            # Define the circular structuring element B4 with a radius of 3
            radius = 3
            kernel_size = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Calculate the morphological gradient
            dilated = cv2.dilate(rho_tilde_plus_image, kernel, iterations=1)
            eroded = cv2.erode(rho_tilde_plus_image, kernel, iterations=1)
            rho_image = cv2.subtract(dilated, eroded)

            plt.figure()
            plt.title('rho image')
            plt.imshow(rho_image, 'gray')
            plt.show()

            return rho_image

        def get_internal_marker(segmented_aorta):
            y, x = np.nonzero(segmented_aorta)
            centroid_x = int(np.mean(x))
            centroid_y = int(np.mean(y))

            internal_marker = [centroid_x, centroid_y]
            return internal_marker

        def get_external_marker(segmented_aorta):
            # Dilatazione della regione segmentata dell'aorta

            kernel = np.array(pd.read_excel('disk3.xlsx'))
            dilated_aorta = skimage.morphology.dilation(segmented_aorta, kernel)  # Utilizzo di un elemento strutturante circolare con raggio 20

            plt.figure()
            plt.imshow(dilated_aorta,'gray')
            plt.show()

            # Calcolo del perimetro della regione dilatata
            perimeter,_ = cv2.findContours(MatToUint8(dilated_aorta), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = flat_contours(perimeter)

            return perimeter

        def select_slice(images):
            # input:
            # Imask: binary images of inner chest region

            # Output:
            # first_slice= number of slice from which the inner contour correction
            nums = str(len(images))
            numrow = int(nums[0])

            position = [np.round((np.size(images[0], 0)) / 2), 10]
            i = 0
            fnt = ImageFont.truetype("arial.ttf", 25)
            d = []
            while i < len(images):
                a = images[i].copy()
                image_data = MatToUint8(a)
                image = Image.fromarray(image_data, mode='L')
                draw = ImageDraw.Draw(image)
                draw.text((position[0], position[1]), str(i + 1), font=fnt, fill=255)
                d.append(image)
                i = i + 1

            new_images = []
            for image in d:
                new_images.append(np.array(image))

            montage = skimage.util.montage(new_images, grid_shape=(numrow, round(len(images) / numrow) + 1), fill=0,
                                           padding_width=10)

            def graph():
                fig = plt.figure()
                ax = fig.add_subplot(111)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                ax.imshow(montage, 'gray')
                ax.axis('off')
                fig.set_size_inches(18.5, 10.5, forward=True)
                plt.show()

            def visualize_select_3():
                try:
                    n = int(label1_input.get())
                    if n > len(images):
                        message.config(text=' Error in slice selection')
                    else:
                        return window.quit()
                except ValueError:
                    message.config(text=' Error in slice selection')

            window = tk.Tk()
            window.geometry("400x450")
            window.title("User Selection")
            window.configure(background="cyan")
            window.wm_resizable(False, False)

            label1 = tk.Label(window, text=" Select correct inner mask:", pady=10)
            label1.grid(row=0, column=1, sticky="N", padx=20, pady=10)
            label1_input = tk.Entry()
            label1_input.grid(row=0, column=2, sticky="WE", padx=10, pady=10)
            label1_input.focus()

            message = tk.Label(window, text="", pady=10)
            message.grid(row=2, column=1, sticky="N", padx=20, pady=10)

            Button1 = tk.Button(window, text="Show Images", command=graph)
            Button1.grid(row=1, column=1, sticky="WE", padx=50, pady=10)

            Button2 = tk.Button(window, text="Ok")
            Button2.grid(row=1, column=2, sticky="WE", padx=50, pady=10)
            Button2.config(command=visualize_select_3)

            window.mainloop()

            first_slice = int(label1_input.get()) - 1

            window.destroy()
            plt.close()
            return first_slice

        first_slice = select_slice(images)
        original_image = images[first_slice].copy()

        # Scala per l'ingrandimento dell'immagine
        scale_percent = 500  # Modifica questa percentuale secondo necessità
        width = int(original_image.shape[1] * scale_percent / 100)
        height = int(original_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)

        cv2.namedWindow('MRI Image')
        cv2.imshow('MRI Image', resized_image)

        # Funzione di callback per la selezione dei punti del contorno
        def select_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(resized_image, (x, y), 5, (255, 0, 0), -1)
                cv2.imshow('MRI Image', resized_image)

        def smooth_contour(contour, num_points=50):
            # Estrai le coordinate x e y dal contorno
            x = [point[0] for point in contour]
            y = [point[1] for point in contour]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            # Crea una funzione interpolante per x e y
            f_x = interp1d(np.arange(len(x)), x, kind='cubic', fill_value='extrapolate')
            f_y = interp1d(np.arange(len(y)), y, kind='cubic', fill_value='extrapolate')
            # Genera nuovi indici uniformemente spaziati
            new_indices = np.linspace(0, len(x) - 1, num_points)
            # Calcola le nuove coordinate x e y
            new_x, new_y = f_x(new_indices), f_y(new_indices)
            # Combina le nuove coordinate in un nuovo contorno
            new_contour = [ [int(x),int(y)] for x,y in zip(new_x,new_y) ]
            return new_contour

            # Crea una lista per memorizzare i punti del contorno

        points = []

        # Attendi l'input interattivo per selezionare i punti del contorno
        cv2.setMouseCallback('MRI Image', select_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        scale_factor_x = original_image.shape[1] / width
        scale_factor_y = original_image.shape[0] / height
        original_points = [[int(x * scale_factor_x), int(y * scale_factor_y)] for x, y in points]
        original_points = smooth_contour(original_points)

        x_points = [x[0] for x in original_points]
        aorta_diameter = np.max(x_points) - np.min(x_points)

        first_aorta_segmented = np.zeros_like(images[0])
        for point in original_points:
            x, y = point
            first_aorta_segmented[y,x] = 255

        kernel = np.ones((2, 2), np.uint8)
        first_aorta_segmented = cv2.dilate(first_aorta_segmented, kernel)

        first_aorta_segmented = ndimage.binary_fill_holes(first_aorta_segmented)

        aorta_masks = [None] * len(images)
        aorta_masks[first_slice] = first_aorta_segmented.copy()

        plt.figure()
        plt.imshow(aorta_masks[first_slice],'gray')
        plt.show()

        i = first_slice + 1
        while i < len(images):
            markers = np.zeros(images[i].shape[:2], dtype=np.int32)

            internal_marker = get_internal_marker(aorta_masks[i - 1])

            markers[internal_marker[1], internal_marker[0]] = 1

            plt.figure()
            plt.imshow(markers, 'gray')
            plt.show()

            external_marker = np.array(get_external_marker(aorta_masks[i - 1]))
            external_marker = external_marker.reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(markers, [external_marker], -1, -1, thickness=1)

            plt.figure()
            plt.title('markers prima')
            plt.imshow(markers, 'gray')
            plt.show()

            rho = MatToUint8(rho_Image(images[i]))
            rho = cv2.cvtColor(rho,cv2.COLOR_GRAY2BGR)

            plt.figure()
            plt.title('rho prima della watershed')
            plt.imshow(rho)
            plt.show()

            result = cv2.watershed(rho, markers)

            plt.figure()
            plt.title('markers dopo')
            plt.imshow(markers, 'gray')
            plt.show()

            plt.figure()
            plt.imshow(result, 'gray')
            plt.show()

            image_segmented = images[i].copy()
            image_segmented = cv2.cvtColor(MatToUint8(image_segmented),cv2.COLOR_GRAY2BGR)

            color = [255, 0, 0]
            image_segmented[result == -1] = color

            # Visualizza l'immagine segmentata
            cv2.imshow('Segmented Image', image_segmented)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Converti 'result' in un'immagine a scala di grigi (CV_8U)
            result = cv2.convertScaleAbs(result)

            # Creazione di una maschera binaria per i pixel segmentati
            mask = np.zeros_like(result, dtype=np.uint8)

            # Trova i contorni delle diverse regioni segmentate
            contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Disegna i contorni sulla maschera binaria
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

            aorta_masks[i] = mask.copy()

            plt.figure()
            plt.title('aorta mask')
            plt.imshow(aorta_masks[i], 'gray')
            plt.show()

            i = i + 1


        i = first_slice - 1
        while i >= 0:
            markers = np.zeros(images[i].shape[:2], dtype=np.int32)

            internal_marker = get_internal_marker(aorta_masks[i + 1])
            markers[internal_marker[1], internal_marker[0]] = 1

            external_marker = np.array(get_external_marker(aorta_masks[i + 1]))
            external_marker = external_marker.reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(markers, [external_marker], -1, 255, thickness=1)

            rho = MatToUint8(rho_Image(images[i]))
            rho = cv2.cvtColor(rho, cv2.COLOR_GRAY2BGR)
            result = cv2.watershed(rho, markers)

            plt.figure()
            plt.imshow(markers, 'gray')
            plt.show()

            plt.figure()
            plt.imshow(result,'gray')
            plt.show()

            image_segmented = images[i].copy()
            image_segmented = cv2.cvtColor(MatToUint8(image_segmented), cv2.COLOR_GRAY2BGR)

            color = [255, 0, 0]
            image_segmented[result == -1] = color

            # Visualizza l'immagine segmentata
            cv2.imshow('Segmented Image', image_segmented)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Converti 'result' in un'immagine a scala di grigi (CV_8U)
            result = cv2.convertScaleAbs(result)

            # Creazione di una maschera binaria per i pixel segmentati
            mask = np.zeros_like(result, dtype=np.uint8)

            # Trova i contorni delle diverse regioni segmentate
            contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Disegna i contorni sulla maschera binaria
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

            aorta_masks[i] = mask.copy()

            plt.figure()
            plt.title('aorta mask')
            plt.imshow(aorta_masks[i], 'gray')
            plt.show()

            i = i - 1

def mask_3d_visualization(masks,pixelX,pixelY,pixelZ):

    num_masks = len(masks)
    height, width = masks[0].shape
    mask_3d = np.zeros((height, width, num_masks), dtype=np.uint8)

    i = 0
    while i < num_masks:
        mask_3d[:, :, i] = masks[i]
        i = i + 1

    active_voxels = (mask_3d != 0).nonzero()
    voxel_dimensions = (pixelX, pixelY, pixelZ)

    x = active_voxels[0] * voxel_dimensions[0]
    y = active_voxels[1] * voxel_dimensions[1]
    z = active_voxels[2] * voxel_dimensions[2]

    # Creazione di una griglia tridimensionale
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
    z_grid = np.zeros_like(x_grid)  # Asse Z costante per la visualizzazione

    # Disegna una superficie interpolata
    ax.plot_trisurf(x, y, z, color='b', alpha=0.5)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2)

    # Imposta le etichette degli assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def liver_deletion(heart_image,heart_mask):
    #colori_viridis = plt.cm.viridis(np.linspace(0, 1, 3))
    #mappa_colori_personalizzata = ListedColormap(colori_viridis)

    #plt.figure()
    #plt.axis('off')
    #plt.imshow(heart_image, 'gray')
    #plt.draw()


    im_heart_mod = MatToUint8(heart_image.copy())
    im_heart_mod = cv2.medianBlur(im_heart_mod, 7)
    im_heart_mod = cv2.GaussianBlur(im_heart_mod, (7, 7), sigmaX=1.5, sigmaY=1.5)

    #plt.figure()
    #plt.imshow(im_heart_mod,'gray')
    #plt.axis('off')
    #plt.draw()


    Th2, _ = cv2.threshold(im_heart_mod, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Th1 = Th2 / 5

    edges = cv2.Canny(im_heart_mod, threshold1=Th1, threshold2=Th2)

    #plt.figure()
    #plt.imshow(edges, 'gray')
    #plt.axis('off')
    #plt.draw()


    kernel = skimage.morphology.disk(4)  # Puoi regolare le dimensioni del kernel
    # kernel = np.array(pd.read_excel('disk3.xlsx'))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    #plt.figure()
    #plt.imshow(closed_edges, 'gray')
    #plt.axis('off')
    #plt.draw()

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    sx_mask = np.zeros_like(edges)
    lowest_x = float('inf')
    for contour in contours:
        image = np.zeros_like(edges)
        cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
        image = ndimage.binary_fill_holes(image)
        image = cv2.erode(MatToUint8(image), kernel, iterations=1)
        coordinate_bianche = np.argwhere(image != 0)
        if len(coordinate_bianche) != 0:
            centroide_x = np.mean(coordinate_bianche[:, 1])
            if centroide_x < lowest_x:
                lowest_x = centroide_x
                sx_mask = image.copy()



    img_array = np.array(heart_image)
    # Reshape l'array in una lista di pixel


    pixels = img_array.reshape((-1, 1))
    # Applica l'algoritmo K-Means per dividere i pixel in cluster di intensità
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(pixels)
    # Predici il cluster di ciascun pixel
    labels = kmeans.predict(pixels)

    def compute_cluster_intensities(labels, data):
        unique_labels = np.unique(labels)
        cluster_intensities = []
        for label in unique_labels:
            cluster_data = data[labels == label]
            intensity_mean = np.mean(cluster_data)
            cluster_intensities.append((label, intensity_mean))
        return cluster_intensities

    cluster_intensities = compute_cluster_intensities(labels, pixels)

    # Ordina i label dei cluster in base all'intensità media dei pixel
    sorted_clusters = sorted(cluster_intensities, key=lambda x: x[1])

    # Crea un dizionario per mappare i vecchi label dei cluster con i nuovi label ordinati
    label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_clusters)}

    # Applica i nuovi label ordinati ai cluster
    sorted_labels = np.array([label_mapping[label] for label in labels])

    labels_matrix = sorted_labels.reshape(img_array.shape)

    #plt.figure()
    #plt.axis('off')
    #etichette = ['Gruppo 0', 'Gruppo 1', 'Gruppo 2']
    #handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colori_viridis]
    #plt.legend(handles, etichette, loc='upper right')
    #plt.imshow(labels_matrix,cmap='viridis')
    #plt.draw()

    sx_mask = Mat2gray(np.array(sx_mask))


    #plt.figure()
    #plt.axis('off')
    #plt.imshow(sx_mask,'gray')
    #plt.draw()


    labels_matrix_masked = labels_matrix * sx_mask


    #plt.figure()
    #plt.axis('off')
    #etichette = ['Gruppo 0', 'Gruppo 1', 'Gruppo 2']
    #handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colori_viridis]
    #plt.legend(handles, etichette, loc='upper right')
    #plt.imshow(labels_matrix_masked,cmap='viridis')
    #plt.draw()

    number_of_1 = np.count_nonzero(labels_matrix_masked==1) #Grey pixels
    number_of_2 = np.count_nonzero(labels_matrix_masked==2) #White pixels

    if number_of_1 > number_of_2:
        #print('si')
        #Allora la presenza del fegato risulta verificata
        #plt.figure()
        #plt.imshow(heart_mask,'gray')
        #plt.draw()

        kernel = skimage.morphology.disk(3)
        sx_mask = skimage.morphology.dilation(sx_mask,kernel)

        #plt.figure()
        #plt.imshow(sx_mask, 'gray')
        #plt.draw()

        mask_without_liver = cv2.subtract(MatToUint8(heart_mask),MatToUint8(sx_mask))

        #plt.figure()
        #plt.imshow(mask_without_liver, 'gray')
        #plt.draw()

        mask_without_liver = cv2.morphologyEx(mask_without_liver, cv2.MORPH_OPEN, MatToUint8(kernel))
        heart_without_liver = cv2.bitwise_and(MatToUint8(heart_image),MatToUint8(heart_image),mask=MatToUint8(mask_without_liver))
    else:
        #print('no')
        mask_without_liver = heart_mask.copy()
        heart_without_liver = heart_image.copy()


    mask_without_liver=MatToUint8(mask_without_liver)
    heart_without_liver=MatToUint8(heart_without_liver)

    #plt.figure()
    #plt.imshow(mask_without_liver,'gray')
    #plt.axis('off')
    #plt.draw()

    #plt.figure()
    #plt.imshow(heart_without_liver,'gray')
    #plt.axis('off')
    #plt.show()

    return heart_without_liver,mask_without_liver





def sternum_segmentation(image,inner_mask,min_ext,pmax1,pmax2):

    plt.figure()
    plt.imshow(image,'gray')
    plt.scatter(pmax1[0],pmax1[1])
    plt.scatter(pmax2[0],pmax2[1])


    internal_mask = MatToUint8(inner_mask)
    contours, _ = cv2.findContours(internal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    altezza, larghezza = image.shape[0], image.shape[1]
    medium_x = larghezza // 2


    x1 = int(np.mean([medium_x,pmax1[0]]))
    x2 = int(np.mean([medium_x,pmax2[0]]))
    y1 = int(min_ext[1])
    # Trova y2 con la stessa coordinata X di min_ext
    target_x = int(min_ext[0])  # Coordinata X di min_ext

    # Inizializza y2 come None per il caso in cui non venga trovata una coordinata con la stessa X
    y2 = float('inf')

    # Cerca y2 con la stessa coordinata X di min_ext e una coordinata Y diversa da y1
    for contour in contours:
        for point in contour:
            x, y = point[0]  # Estrai le coordinate x e y
            if x == target_x and y != y1 and y < y2:
                y2 = y

    plt.figure()
    plt.imshow(image,'gray')
    plt.scatter(x1,y1)
    plt.scatter(x1,y2)
    plt.scatter(x2,y1)
    plt.scatter(x2,y2)
    plt.show()

    external_mask = np.logical_not(inner_mask)

    roi_mask = np.zeros_like(external_mask, dtype=bool)
    roi_mask[y1:y2, x1:x2] = True


    mask = np.logical_and(external_mask,roi_mask)

    plt.figure()
    plt.imshow(mask, 'gray')
    plt.draw()

    external_image = cv2.bitwise_and(MatToUint8(image),MatToUint8(image),mask=MatToUint8(mask))

    # Applica il filtro di Sobel lungo l'asse x (orizzontale)
    sobel_x = cv2.Sobel(external_image, cv2.CV_64F, 1, 0, ksize=3)

    # Applica il filtro di Sobel lungo l'asse y (verticale)
    sobel_y = cv2.Sobel(external_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calcola il modulo del gradiente per evidenziare i bordi
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Normalizza l'immagine risultante tra 0 e 255
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    plt.figure()
    plt.imshow(sobel_combined,'gray')
    plt.draw()


    kernel = MatToUint8(np.ones((2,2)))
    sobel_combined = cv2.erode(sobel_combined,kernel)

    plt.figure()
    plt.imshow(sobel_combined, 'gray')
    plt.draw()

    sobel_thresholded = np.array(sobel_combined > 30)
    sobel_thresholded = skimage.morphology.remove_small_holes(sobel_thresholded, area_threshold=20)
    sobel_thresholded = skimage.morphology.remove_small_objects(sobel_thresholded, min_size=10, connectivity=2)

    plt.figure()
    plt.imshow(sobel_thresholded, 'gray')
    plt.draw()

    contours, _ = cv2.findContours(MatToUint8(sobel_thresholded), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    contours = [cnt for cnt in contours if not np.array_equal(cnt, largest_contour)]

    best_rect = None
    best_diff = float('inf')
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area_rect = cv2.contourArea(box)
        area_contour = cv2.contourArea(contour)
        diff = abs(area_rect - area_contour)
        if diff < best_diff:
            best_diff = diff
            best_rect = rect


    # Disegna il rettangolo migliore sull'immagine
    box = cv2.boxPoints(best_rect)
    box = np.int0(box)
    result_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    result_image = cv2.drawContours(result_image, [box], 0, (0, 255, 0), 2)

    # Visualizza l'immagine con il rettangolo evidenziato
    plt.figure()
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.show()

    return box



def heart_concavity(image,mask, pixel_spacing, pcontour, pmax1, pmax2):
    # minimum x coordinate of inner contour


    x_max = np.argmax(np.array([x[0] for x in pcontour]))

    pcontour_x = [x[0] for x in pcontour]
    pcontour_x = pcontour_x[0:x_max+1]

    pcontour_y = [x[1] for x in pcontour]
    pcontour_y = pcontour_y[0:x_max+1]


    ## Sternum position
    # point corresponding to max1 point on the inner contour (upper half of contour)
    imax1 = np.where(pcontour_x == np.array(pmax1[0]).astype(np.float64))[0]
    if len(imax1)!=0:
        imax1 = imax1.min()
    else:
        imax1 = np.abs(pcontour_x - np.array(pmax1[0]).astype(np.float64)).argmin()
    # point corresponding to max2 point on the inner contour (upper half of contour)
    imax2 = np.where(pcontour_x == np.array(pmax2[0]).astype(np.float64))[0]
    if len(imax2)!=0:
        imax2 = imax2.min()
    else:
        imax2 = np.abs(pcontour_x - np.array(pmax2[0]).astype(np.float64)).argmin()


    # minimum point of inner contour: sternum position (point with the maximum y
    # coordinate among the points found above)

    minsternumv = np.max(pcontour_y[imax1:imax2 + 1])
    iminsternuma = np.where(pcontour_y[imax1:imax2 + 1] == minsternumv)[0]
    # Among the points with the same y coordinate it is selected the middle one
    iminsternum = np.abs((iminsternuma[-1] + iminsternuma[0]) / 2) + imax1 - 1
    min_coord = pcontour[int(iminsternum)]
    min_coord = [min_coord[1],min_coord[0]]


    #plt.figure()
    #plt.imshow(image, 'gray')
    #plt.scatter(min_coord[0],min_coord[1])
    #plt.draw()

    #plt.figure()
    #plt.imshow(mask, 'gray')
    #plt.draw()

    convex_hull = skimage.morphology.convex_hull_image(mask)


    #plt.figure()
    #plt.imshow(convex_hull,'gray')
    #plt.draw()

    area = cv2.subtract(MatToUint8(convex_hull),MatToUint8(mask))
    area = np.array(area,dtype=bool)

    #plt.figure()
    #plt.imshow(area, 'gray')
    #plt.draw()

    area = skimage.morphology.remove_small_objects(area, min_size=5)

    #plt.figure()
    #plt.imshow(area, 'gray')
    #plt.scatter(min_coord[0],min_coord[1])
    #plt.draw()

    #labeled_mask, num_features = skimage.measure.label(area, connectivity=2, return_num=True)
    #regions = skimage.measure.regionprops(labeled_mask)
    #centroids = [region.centroid for region in regions]
    #distances = [np.linalg.norm(centroid - np.array(min_coord)) for centroid in centroids]
    #closest_region_idx = np.argmin(distances)
    #closest_label = regions[closest_region_idx].label
    #filtered_mask = np.where(labeled_mask == closest_label, 1, 0)

    labeled_area = measure.label(area)
    min_label = labeled_area[int(min_coord[1]), int(min_coord[0])]

    background_label = 0
    if min_label == background_label:
        min_label = None
        # Trova l'etichetta successiva diversa dal background
        unique_labels = np.unique(labeled_area)
        unique_labels = unique_labels[unique_labels != background_label]

        for label in unique_labels:
            if label in labeled_area:

                points_in_label = np.argwhere(labeled_area == label)
                distances = [distance.euclidean(point, min_coord) for point in points_in_label]

                if any(dist < 5 for dist in distances):
                    min_label = label
                    break

    if min_label is not None:
        filtered_mask = np.where(labeled_area == min_label, 1, 0)

        #plt.figure()
        #plt.imshow(filtered_mask, 'gray')
        #plt.scatter(min_coord[0], min_coord[1])
        #plt.show()

        mask_convered = cv2.add(MatToUint8(mask), MatToUint8(filtered_mask))

        # plt.figure()
        # plt.imshow(mask_convered, 'gray')
        # plt.draw()

        original_area = np.count_nonzero(mask)
        convex_area = np.count_nonzero(mask_convered)

        concavity = (1 - (original_area / convex_area)) * 100

        # Sovrapponi l'immagine originale e le parti corrispondenti alla concavità colorate
        #plt.figure(figsize=(8, 8))

        # Mostra l'immagine originale
        #plt.imshow(image, cmap='gray')

        # Sovrapponi le parti corrispondenti alla concavità colorate sull'immagine originale
        points = np.argwhere(filtered_mask)
        number_of_points = len(points)
        #for point in points:
            #plt.scatter(point[1], point[0], s=2, alpha=0.5, c='red')

        #plt.title(f'Concavità: {concavity:.2f}%')
        #plt.axis('off')
        #plt.show()
        area_loss = round(number_of_points * pixel_spacing[0] * pixel_spacing[1] / 100, 2)
    else:
        points = None
        concavity = 0
        area_loss = 0

    return points, concavity, area_loss



def heart_new_index(image,mask):


    center_of_mass = ndimage.measurements.center_of_mass(mask)
    center_of_mass = np.array([center_of_mass[1],center_of_mass[0]])


    num_raggi = 100
    lunghezza_max = 500
    angoli = np.linspace(0, 360, num_raggi, endpoint=False)


    contour, _ = cv2.findContours(MatToUint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordinate_raggi = []
    for angolo in angoli:
        for distanza in range(1, lunghezza_max + 1):
            x_new = int(center_of_mass[1] + np.cos(angolo) * distanza)
            y_new = int(center_of_mass[0] + np.sin(angolo) * distanza)
            if mask[x_new, y_new] == 0:
                coordinate_raggi.append([x_new, y_new])
                break

    distanze = []
    for point in coordinate_raggi:
        point = [point[1],point[0]]
        distanze.append (np.linalg.norm( np.array(point) - center_of_mass ))

    mean = np.mean(distanze)
    std = np.std(distanze)
    coeff_variability = np.round((std/mean) * 100,1)

    plt.figure()
    plt.imshow(image,'gray')
    plt.scatter(center_of_mass[0],center_of_mass[1],s=2,zorder=3,c='red')
    for point in coordinate_raggi:
        plt.scatter(point[1], point[0],s=2,zorder=3,c='red')
        plt.plot([center_of_mass[0],point[1]],[center_of_mass[1],point[0]],zorder=2,c='blue')
    plt.axis('off')
    plt.show()

    return coeff_variability






def concavity_slider( images, points, area_concavity, slice_spacing, slice_thickness):


    volume_of_concavity = round(np.sum(area_concavity) * ( (slice_thickness + slice_spacing) / 10 ),2)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    max_index = np.argmax(area_concavity)

    ax.imshow(images[max_index],'gray')
    #for point in points[max_index]:
        #ax.scatter(point[1], point[0], s=2, alpha=0.2, c='red')
    if points[max_index] is not None:
        ax.scatter(points[max_index][:, 1], points[max_index][:, 0], s=2, alpha=0.5, c='red')
    ax.text(0.5, -0.1, f"Area Concavity: {area_concavity[max_index]} cm\u00B2", horizontalalignment='center',transform=ax.transAxes)
    ax.text(0.5, -0.15, f"Volume Concavity: {volume_of_concavity} cm\u00B3" , horizontalalignment='center', transform=ax.transAxes)

    def update(val):
        index = int(val)
        ax.clear()
        ax.imshow(images[index],'gray')
        #for point in points[index]:
            #ax.scatter(point[1], point[0], s=2, alpha=0.5, c='red')
        if points[index] is not None:
            ax.scatter(points[index][:, 1], points[index][:, 0], s=2, alpha=0.5, c='red')
        ax.text(0.5, -0.1, f"Area Concavity: {area_concavity[index]} cm\u00B2", horizontalalignment='center',transform=ax.transAxes)
        ax.text(0.5, -0.15, f"Volume Concavity: {volume_of_concavity} cm\u00B3", horizontalalignment='center',transform=ax.transAxes)
        plt.draw()


    # Creazione dello slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Image', 0, len(images) - 1, valinit=max_index, valfmt='%d')
    slider.on_changed(update)

    plt.show()


def aorta_segmentation_new ( images , im_hearts  , mask_hearts , inner_masks , indexes ):


    images_selected = [x for i, x in enumerate(images) if i in indexes]
    im_hearts_selected = [x for i, x in enumerate(im_hearts) if i in indexes]
    mask_hearts_selected = [x for i, x in enumerate(mask_hearts) if i in indexes]
    inner_masks_selected = [x for i, x in enumerate(inner_masks) if i in indexes]

    im_hearts_not_selected = [x for i, x in enumerate(im_hearts) if i not in indexes]
    mask_hearts_not_selected = [x for i, x in enumerate(mask_hearts) if i not in indexes]


    means = np.array([np.mean(x) for x in images_selected])

    above_threshold_images = [im[im > mean] for im, mean in zip(images_selected, means)]

    means_sup = np.array([np.mean(x) for x in above_threshold_images])
    sigma = np.array([np.std(x) for x in above_threshold_images])

    means_sup_plus_sigma = means_sup + sigma

    thresholded = [im > th for im, th in zip(images_selected, means_sup_plus_sigma)]
    thresholded_masked = [th * inner for th, inner in zip(thresholded, inner_masks_selected)]

    Montage_Matlab_draw(thresholded_masked, 'ga')


    thresholded_filtered = [Mat2gray(cv2.medianBlur(MatToUint8(im),1)) > th for im, th in zip(images_selected, means_sup_plus_sigma)]
    thresholded_filtered_masked = [th * inner for th, inner in zip(thresholded_filtered, inner_masks_selected)]


    se = skimage.morphology.rectangle(3,3)
    thresholded_masked_dil = [ skimage.morphology.dilation(skimage.morphology.remove_small_objects(im,10),se) for im in thresholded_masked]

    Montage_Matlab_draw(thresholded_masked_dil,'ga')


    complexive_AND = np.ones_like(thresholded_masked_dil[0])
    i = 0
    while i < len(thresholded_masked_dil):
        complexive_AND = np.logical_and(complexive_AND, thresholded_masked_dil[i])
        i = i + 1


    complexive_AND = skimage.morphology.remove_small_objects(complexive_AND,5)

    plt.figure()
    plt.imshow(complexive_AND,'gray')
    plt.draw()

    contours, _ = cv2.findContours(MatToUint8(complexive_AND), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    aorta_location = np.zeros_like(complexive_AND)
    lowest_y = -float('inf')
    for contour in contours:
        image = MatToUint8(np.zeros_like(complexive_AND))
        cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
        image = ndimage.binary_fill_holes(image)
        #image = cv2.erode(MatToUint8(image), kernel, iterations=1)
        coordinate_bianche = np.argwhere(image != 0)
        if len(coordinate_bianche) != 0:
            centroide_y = np.mean(coordinate_bianche[:, 0])
            if centroide_y > lowest_y:
                lowest_y = centroide_y
                aorta_location = image.copy()

    plt.figure()
    plt.imshow(aorta_location,'gray')
    plt.draw()


    aorta_masks = []
    points_of_location = np.argwhere(aorta_location!=0)

    def elemento_comune(lista1, lista2):
        for coord1 in lista1:
            for coord2 in lista2:
                if np.array_equal(np.array(coord1), np.array(coord2)):
                    return True
        return False


    se=skimage.morphology.disk(1)
    for th in thresholded_filtered_masked:
        mask = MatToUint8(np.zeros_like(th))
        contours, _ = cv2.findContours(MatToUint8(skimage.morphology.erosion(th,se)), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        common_elements = 0
        for contour in contours:
            contour_image = MatToUint8(np.zeros_like(th))
            cv2.drawContours(contour_image, contour, -1, (255, 255, 255), thickness=1)
            contour_image = ndimage.binary_fill_holes(contour_image)
            white_points = np.argwhere(contour_image!=0)
            if elemento_comune(points_of_location,white_points)==True:
                mask = skimage.morphology.remove_small_objects(cv2.add(MatToUint8(mask),MatToUint8(contour_image)),10)
        aorta_masks.append(mask)

    Montage_Matlab_draw(aorta_masks,'lol')

    aorta_masks_one_area = []
    for mask in aorta_masks:
        num_labels, labeled_image = cv2.connectedComponents(mask)
        num_areas = num_labels - 1
        if num_areas == 1:
            aorta_masks_one_area.append(mask)
        else:
            contours, _ = cv2.findContours(MatToUint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area = np.zeros_like(mask)
            lowest_y = -float('inf')
            for contour in contours:
                image = MatToUint8(np.zeros_like(mask))
                cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
                image = ndimage.binary_fill_holes(image)
                # image = cv2.erode(MatToUint8(image), kernel, iterations=1)
                coordinate_bianche = np.argwhere(image != 0)
                if len(coordinate_bianche) != 0:
                    centroide_y = np.mean(coordinate_bianche[:, 0])
                    if centroide_y > lowest_y:
                        lowest_y = centroide_y
                        area = image.copy()
            aorta_masks_one_area.append(area)

    Montage_Matlab_draw(aorta_masks_one_area,'gray')



    non_zero_elements = [np.count_nonzero(x) for x in aorta_masks_one_area]
    mean_value = np.mean(non_zero_elements)
    std_deviation = np.std(non_zero_elements)
    outliers = [abs(value - mean_value) > (2 * std_deviation) for value in non_zero_elements]
    print(outliers)

    aorta_masks_final = [None] *  len(outliers)
    kernel = skimage.morphology.disk(7)
    i = 0
    while i < len(outliers):
        if outliers[i] == False:
            aorta_masks_final[i] = aorta_masks_one_area[i].copy()
        else:
            im_eroded = cv2.erode(MatToUint8(aorta_masks_one_area[i]), kernel, iterations=1)
            contours, _ = cv2.findContours(MatToUint8(im_eroded), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area = np.zeros_like(im_eroded)
            lowest_y = -float('inf')
            for contour in contours:
                image = MatToUint8(np.zeros_like(im_eroded))
                cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
                image = ndimage.binary_fill_holes(image)
                coordinate_bianche = np.argwhere(image != 0)
                if len(coordinate_bianche) != 0:
                    centroide_y = np.mean(coordinate_bianche[:, 0])
                    if centroide_y > lowest_y:
                        lowest_y = centroide_y
                        area = image.copy()
            aorta_masks_final[i] = area.copy()
        i=i+1

    Montage_Matlab(aorta_masks_final,'lol')

    se = skimage.morphology.disk(5)
    aorta_masks_final_dilated = [cv2.dilate(MatToUint8(x),se) for x in aorta_masks_final]

    se2 = skimage.morphology.disk(3)
    mask_heart_without_aorta = [ Bwareafilt(cv2.morphologyEx( cv2.subtract(MatToUint8(mask),MatToUint8(aorta)),cv2.MORPH_OPEN,se2)).astype(np.uint8) for mask,aorta in zip(mask_hearts_selected,aorta_masks_final_dilated)]

    heart_without_aorta = [cv2.bitwise_and(heart, heart, mask=Mask) for heart,Mask in zip(im_hearts_selected,mask_heart_without_aorta)]


    heart_mask_final = []
    heart_image_final = []
    aorta_mask_final = []

    for i in range(len(mask_hearts)):
        if i in indexes:
            heart_image_final.append(heart_without_aorta.pop(0))
            heart_mask_final.append(mask_heart_without_aorta.pop(0))
            aorta_mask_final.append(aorta_masks_final_dilated.pop(0))
        else:
            heart_image_final.append(im_hearts_not_selected.pop(0))
            heart_mask_final.append(mask_hearts_not_selected.pop(0))
            aorta_mask_final.append(MatToUint8(np.zeros_like(mask_hearts[0])))


    return heart_image_final, heart_mask_final, aorta_mask_final


def vena_segmentation_new(images, im_hearts, mask_hearts, inner_masks, indexes):



    images_selected = [x for i, x in enumerate(images) if i in indexes]
    im_hearts_selected = [x for i, x in enumerate(im_hearts) if i in indexes]
    mask_hearts_selected = [x for i, x in enumerate(mask_hearts) if i in indexes]
    inner_masks_selected = [x for i, x in enumerate(inner_masks) if i in indexes]

    Montage_Matlab_draw(im_hearts_selected,'lol')

    im_hearts_not_selected = [x for i, x in enumerate(im_hearts) if i not in indexes]
    mask_hearts_not_selected = [x for i, x in enumerate(mask_hearts) if i not in indexes]

    means = np.array([np.mean(x) for x in images_selected])

    above_threshold_images = [im[im > mean] for im, mean in zip(images_selected, means)]

    means_sup = np.array([np.mean(x) for x in above_threshold_images])
    sigma = np.array([np.std(x) for x in above_threshold_images])

    means_sup_plus_sigma = means_sup + sigma

    thresholded = [im > th for im, th in zip(images_selected, means_sup_plus_sigma)]
    thresholded_masked = [th * inner for th, inner in zip(thresholded, inner_masks_selected)]

    thresholded_filtered = [im > th for im, th in zip(images_selected, means_sup_plus_sigma)]

    thresholded_filtered_masked = [th * inner for th, inner in zip(thresholded_filtered, inner_masks_selected)]

    Montage_Matlab_draw(thresholded_filtered_masked, 'lol')

    se = skimage.morphology.disk(2)
    #skimage.morphology.dilation(skimage.morphology.erosion(skimage.morphology.remove_small_objects(im,10),se),se
    thresholded_masked_dil = [ skimage.morphology.erosion(skimage.morphology.remove_small_objects(im,15),se) for im in thresholded_masked]
    thresholded_masked_dil = [ ndimage.binary_fill_holes(x) for x in thresholded_masked_dil]


    Montage_Matlab_draw(thresholded_masked_dil, 'lol')

    complexive_AND = np.ones_like(thresholded_masked_dil[0])
    i = 0
    while i < len(thresholded_masked_dil):
        complexive_AND = np.logical_and(complexive_AND, thresholded_masked_dil[i])
        i = i + 1

    complexive_AND = skimage.morphology.remove_small_objects(complexive_AND, 5)

    plt.figure()
    plt.imshow(complexive_AND,'gray')
    plt.draw()




    contours, _ = cv2.findContours(MatToUint8(complexive_AND), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    vena_location = np.zeros_like(complexive_AND)
    lowest_x = float('inf')
    for contour in contours:
        image = MatToUint8(np.zeros_like(complexive_AND))
        cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
        image = ndimage.binary_fill_holes(image)
        # image = cv2.erode(MatToUint8(image), kernel, iterations=1)
        coordinate_bianche = np.argwhere(image != 0)
        if len(coordinate_bianche) != 0:
            centroide_x = np.mean(coordinate_bianche[:, 1])
            if centroide_x < lowest_x:
                lowest_x = centroide_x
                vena_location = image.copy()




    vena_masks = []
    points_of_location = np.argwhere(vena_location != 0)

    def elemento_comune(lista1, lista2):
        for coord1 in lista1:
            for coord2 in lista2:
                if np.array_equal(np.array(coord1), np.array(coord2)):
                    return True
        return False



    for th in thresholded_masked_dil:
        mask = MatToUint8(np.zeros_like(th))
        contours, _ = cv2.findContours(MatToUint8(th), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        common_elements = 0
        for contour in contours:
            contour_image = MatToUint8(np.zeros_like(th))
            cv2.drawContours(contour_image, contour, -1, (255, 255, 255), thickness=1)
            contour_image = ndimage.binary_fill_holes(contour_image)
            white_points = np.argwhere(contour_image != 0)
            if (elemento_comune(points_of_location, white_points) == True) & (len(white_points)>5):
                mask = skimage.morphology.remove_small_objects(cv2.add(MatToUint8(mask),MatToUint8(contour_image)),10)
        vena_masks.append(mask)

    Montage_Matlab_draw(vena_masks,'gray')


    vena_masks_one_area = []
    for mask in vena_masks:
        num_labels, labeled_image = cv2.connectedComponents(mask)
        num_areas = num_labels - 1
        if num_areas == 1:
            vena_masks_one_area.append(mask)
        else:
            contours, _ = cv2.findContours(MatToUint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area = np.zeros_like(mask)
            lowest_x = float('inf')
            for contour in contours:
                image = MatToUint8(np.zeros_like(mask))
                cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
                image = ndimage.binary_fill_holes(image)
                # image = cv2.erode(MatToUint8(image), kernel, iterations=1)
                coordinate_bianche = np.argwhere(image != 0)
                if len(coordinate_bianche) != 0:
                    centroide_x = np.mean(coordinate_bianche[:, 1])
                    if centroide_x < lowest_x:
                        lowest_x = centroide_x
                        area = image.copy()
            vena_masks_one_area.append(area)


    Montage_Matlab(vena_masks_one_area,'gray')


    non_zero_elements = [np.count_nonzero(x) for x in vena_masks_one_area]
    print(non_zero_elements)
    mean_value = np.mean(non_zero_elements)
    std_deviation = np.std(non_zero_elements)
    outliers = [abs(value - mean_value) > (1 * std_deviation) for value in non_zero_elements]
    print(outliers)

    vena_masks_final = [None] * len(outliers)
    kernel = skimage.morphology.disk(7)
    i = 0
    while i < len(outliers):
        if outliers[i] == False:
            vena_masks_final[i] = vena_masks_one_area[i].copy()
        else:
            im_eroded = cv2.erode(MatToUint8(vena_masks_one_area[i]), kernel, iterations=1)
            contours, _ = cv2.findContours(MatToUint8(im_eroded), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area = np.zeros_like(im_eroded)
            lowest_x = float('inf')
            for contour in contours:
                image = MatToUint8(np.zeros_like(im_eroded))
                cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
                image = ndimage.binary_fill_holes(image)
                coordinate_bianche = np.argwhere(image != 0)
                if len(coordinate_bianche) != 0:
                    centroide_x = np.mean(coordinate_bianche[:, 1])
                    if centroide_x < lowest_x:
                        lowest_x = centroide_x
                        area = image.copy()
            vena_masks_final[i] = area.copy()
        i = i + 1

    se = skimage.morphology.disk(3)
    vena_masks_final_dilated = [cv2.dilate(MatToUint8(x), se) for x in vena_masks_final]

    se2 = skimage.morphology.disk(3)
    mask_heart_without_vena = [Bwareafilt(cv2.morphologyEx(cv2.subtract(MatToUint8(mask), MatToUint8(vena)), cv2.MORPH_OPEN, se2)).astype(
            np.uint8) for mask, vena in zip(mask_hearts_selected, vena_masks_final_dilated)]

    heart_without_vena = [cv2.bitwise_and(heart, heart, mask=Mask) for heart, Mask in
                           zip(im_hearts_selected, mask_heart_without_vena)]

    heart_mask_final = []
    heart_image_final = []
    vena_mask_final = []

    for i in range(len(mask_hearts)):
        if i in indexes:
            heart_image_final.append(heart_without_vena.pop(0))
            heart_mask_final.append(mask_heart_without_vena.pop(0))
            vena_mask_final.append(vena_masks_final_dilated.pop(0))
        else:
            heart_image_final.append(im_hearts_not_selected.pop(0))
            heart_mask_final.append(mask_hearts_not_selected.pop(0))
            vena_mask_final.append(MatToUint8(np.zeros_like(mask_hearts[0])))

    return heart_image_final, heart_mask_final, vena_mask_final




def left_cardiac_lateral_shift (images, heart_masks, index_max_compression, thorax_mask, inner_mask, inner_contour, pmax1_H, pmax2_H):

    # # SPINE LOCATION
    means = np.array([np.mean(x) for x in images])
    #
    above_threshold_images = [im[im > mean] for im, mean in zip(images, means)]
    #
    means_sup = np.array([np.mean(x) for x in above_threshold_images])
    sigma = np.array([np.mean(x) for x in above_threshold_images])
    #
    means_sup_plus_sigma = means_sup + sigma
    #
    images_thresholded = [im > m for im, m in zip(images, means_sup)]
    #
    # Montage_Matlab(images_thresholded,'lol')
    #
    complexive_AND = np.ones_like(images_thresholded[0])
    se = skimage.morphology.rectangle(1, 5)
    se2 = skimage.morphology.disk(10)
    i = 0
    while i < len(images_thresholded):
         im_dilated = skimage.morphology.dilation(images_thresholded[i], se)
         mask2 = skimage.morphology.erosion(thorax_mask[i],se2)
         im_dilated = np.logical_and(im_dilated,mask2)
         complexive_AND = np.logical_and(complexive_AND, im_dilated)
         i = i + 1
    #
    complexive_AND = skimage.morphology.remove_small_objects(complexive_AND, 15)

    #
    plt.figure()
    plt.imshow(complexive_AND,'gray')
    plt.show()
    #
    width = complexive_AND.shape[0]
    third_width = width // 3
    #
    complexive_AND_no_artifacts = np.copy(complexive_AND)
    complexive_AND_no_artifacts[:, :third_width+20] = 0
    complexive_AND_no_artifacts[:, (2 * third_width)-20:] = 0
    #
    se = skimage.morphology.disk(5)
    contours, _ = cv2.findContours(MatToUint8(complexive_AND_no_artifacts), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    spine_location = np.zeros_like(complexive_AND_no_artifacts)
    lowest_y = -float('inf')
    for contour in contours:
         image = MatToUint8(np.zeros_like(complexive_AND_no_artifacts))
         cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
         image = ndimage.binary_fill_holes(image)
         coordinate_bianche = np.argwhere(image != 0)
         if len(coordinate_bianche) != 0:
             centroide_y = np.mean(coordinate_bianche[:, 0])
             centroide_x = np.mean(coordinate_bianche[:, 1])
             if centroide_y > lowest_y  :
                 lowest_y = centroide_y
                 spine_location = image.copy()
    #
    points_of_location = np.argwhere(spine_location!=0)
    #
    def elemento_comune(lista1, lista2):
         for coord1 in lista1:
             for coord2 in lista2:
                 if np.array_equal(np.array(coord1), np.array(coord2)):
                     return True
         return False
    #
    #
    image_of_max_compression = images[index_max_compression] > means_sup[index_max_compression]
    image_of_max_compression = skimage.morphology.remove_small_objects(image_of_max_compression, 5)
    image_of_max_compression_masked = np.logical_and(image_of_max_compression,np.logical_not(inner_mask[index_max_compression]))
    #
    #
    spine_mask = MatToUint8(np.zeros_like(image_of_max_compression_masked))
    contours, _ = cv2.findContours(MatToUint8(image_of_max_compression_masked), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
         contour_image = MatToUint8(np.zeros_like(image_of_max_compression_masked))
         cv2.drawContours(contour_image, contour, -1, (255, 255, 255), thickness=1)
         contour_image = ndimage.binary_fill_holes(contour_image)
         white_points = np.argwhere(contour_image != 0)
         if elemento_comune(points_of_location, white_points) == True:
              spine_mask = cv2.add(MatToUint8(spine_mask), MatToUint8(contour_image))
    #
    plt.figure()
    plt.imshow(spine_mask,'gray')
    plt.show()
    #
    spine_mask = keep_n_largest_objects(spine_mask,1)
    white_pixels_spine = np.argwhere(spine_mask != 0)
    #
    centroid_x = np.mean([x[0] for x in white_pixels_spine])
    centroid_y = np.mean([x[1] for x in white_pixels_spine])
    #
    spine_point = [centroid_y,centroid_x]
    #
    #
    # #STERNUM LOCATION
    #
    # pcontour = inner_contour[index_max_compression].copy()
    #
    # x_max = np.argmax(np.array([x[0] for x in pcontour]))
    #
    # pcontour_x = [x[0] for x in pcontour]
    # pcontour_x = pcontour_x[0:x_max + 1]
    #
    # pcontour_y = [x[1] for x in pcontour]
    # pcontour_y = pcontour_y[0:x_max + 1]
    #
    # ## Sternum position
    # # point corresponding to max1 point on the inner contour (upper half of contour)
    # pmax1 = pmax1_H[index_max_compression].copy()
    # pmax2 = pmax2_H[index_max_compression].copy()
    #
    # imax1 = np.where(pcontour_x == np.array(pmax1[0]).astype(np.float64))[0]
    # if len(imax1) != 0:
    #     imax1 = imax1.min()
    # else:
    #     imax1 = np.abs(pcontour_x - np.array(pmax1[0]).astype(np.float64)).argmin()
    # # point corresponding to max2 point on the inner contour (upper half of contour)
    # imax2 = np.where(pcontour_x == np.array(pmax2[0]).astype(np.float64))[0]
    # if len(imax2) != 0:
    #     imax2 = imax2.min()
    # else:
    #     imax2 = np.abs(pcontour_x - np.array(pmax2[0]).astype(np.float64)).argmin()
    #
    # # coordinate among the points found above)
    #
    # minsternumv = np.max(pcontour_y[imax1:imax2 + 1])
    # iminsternuma = np.where(pcontour_y[imax1:imax2 + 1] == minsternumv)[0]
    # # Among the points with the same y coordinate it is selected the middle one
    # iminsternum = np.abs((iminsternuma[-1] + iminsternuma[0]) / 2) + imax1 - 1
    # min_coord = pcontour[int(iminsternum)]
    # min_coord = [min_coord[0], min_coord[1]]

    #sternum_spinal_line = np.mean(np.array([min_coord[1] , spine_point[1]]))


    sternum_spinal_line = np.mean(np.array([spine_point[0], spine_point[0]]))


    heart_contours, _ = cv2.findContours(MatToUint8(heart_masks[index_max_compression]), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    heart_contours = flat_contours(heart_contours)


    heart_contours_right  = [x for x in heart_contours if x[0] < sternum_spinal_line]
    heart_contours_left = [x for x in heart_contours if x[0] > sternum_spinal_line]

    if heart_contours_right:  # Verifica se heart_contours_right non è vuoto
        right_max_distance_point = min(heart_contours_right, key=lambda punto: punto[0])
        # Fai qualcosa con right_max_distance_point
    else:
        left_shift = 100
        return left_shift

    left_max_distance_point = max(heart_contours_left, key=lambda punto: punto[0])

    right_distance = np.abs(sternum_spinal_line - right_max_distance_point[0])
    left_distance = np.abs(sternum_spinal_line - left_max_distance_point[0])

    left_shift = (left_distance / (left_distance + right_distance)) * 100


    plt.figure()
    plt.imshow(images[index_max_compression],'gray')
    plt.scatter(right_max_distance_point[0], right_max_distance_point[1], color='red', s=2)
    plt.scatter(left_max_distance_point[0], left_max_distance_point[1], color='red', s=2)
    plt.plot([sternum_spinal_line, sternum_spinal_line],[0  , images[index_max_compression].shape[0]-1], color='red')
    plt.scatter(sternum_spinal_line,right_max_distance_point[1],color='red',s=2)
    plt.scatter(sternum_spinal_line,left_max_distance_point[1], color='red', s=2)
    plt.plot([right_max_distance_point[0], sternum_spinal_line],[right_max_distance_point[1], right_max_distance_point[1]], color='blue')
    plt.plot([left_max_distance_point[0], sternum_spinal_line],[left_max_distance_point[1], left_max_distance_point[1]], color='blue')
    plt.text(0.1, 0.1, f'Left Cardiac Shift: {left_shift}%', color='black', fontsize=8, transform=plt.gca().transAxes)
    plt.show()

    return left_shift


def find_closest_contour(mask, reference_point):
    # Trova i contorni della maschera
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_distance = float('inf')
    closest_contour = None

    # Calcola la distanza di ciascun contorno dal punto di riferimento
    for contour in contours:
        for point in contour.squeeze():
            distance = np.linalg.norm(point - reference_point)
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

    closest_mask = np.zeros_like(mask)
    cv2.drawContours(closest_mask, [closest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    closest_mask = ndimage.binary_fill_holes(closest_mask.astype(bool))

    return closest_mask


def elemento_comune(lista1, lista2):
    for coord1 in lista1:
        for coord2 in lista2:
            if np.array_equal(np.array(coord1), np.array(coord2)):
                 return True
    return False




def heart_ellipse (image, mask , contour, max1, max2, innermask , pixel_distance, slice_distance):

    pcontour = contour[0].copy()
    pmax1=max1[0].copy()
    pmax2=max2[0].copy()
    #Sternum Point
    x_max = np.argmax(np.array([x[0] for x in pcontour]))

    pcontour_x = [x[0] for x in pcontour]
    pcontour_x = pcontour_x[0:x_max + 1]

    pcontour_y = [x[1] for x in pcontour]
    pcontour_y = pcontour_y[0:x_max + 1]

    ## Sternum position
    # point corresponding to max1 point on the inner contour (upper half of contour)
    imax1 = np.where(pcontour_x == np.array(pmax1[0]).astype(np.float64))[0]
    if len(imax1) != 0:
        imax1 = imax1.min()
    else:
        imax1 = np.abs(pcontour_x - np.array(pmax1[0]).astype(np.float64)).argmin()
    # point corresponding to max2 point on the inner contour (upper half of contour)
    imax2 = np.where(pcontour_x == np.array(pmax2[0]).astype(np.float64))[0]
    if len(imax2) != 0:
        imax2 = imax2.min()
    else:
        imax2 = np.abs(pcontour_x - np.array(pmax2[0]).astype(np.float64)).argmin()

    # minimum point of inner contour: sternum position (point with the maximum y
    # coordinate among the points found above)

    minsternumv = np.max(pcontour_y[imax1:imax2 + 1])
    iminsternuma = np.where(pcontour_y[imax1:imax2 + 1] == minsternumv)[0]
    # Among the points with the same y coordinate it is selected the middle one
    iminsternum = np.abs((iminsternuma[-1] + iminsternuma[0]) / 2) + imax1 - 1
    min_coord = pcontour[int(iminsternum)]
    min_coord = [min_coord[0], min_coord[1]]
    ###########################

    i = 0
    concavity_area_one=[]
    while i < len(image):

        contours, _ = cv2.findContours(MatToUint8(mask[i]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(flat_contours(contours)).astype(np.float32)
        ellipse = cv2.fitEllipse(contours)

        ellipse_mask = np.zeros_like(mask[i])
        cv2.ellipse(ellipse_mask, ellipse, (255, 255, 255), -1)
        concavity_area = MatToUint8(cv2.subtract(MatToUint8(ellipse_mask), MatToUint8(mask[i])))


        if i==0:
            concavity_area_one.append(MatToUint8(find_closest_contour(concavity_area, min_coord)))

        else:
            contours, _ = cv2.findContours(MatToUint8(concavity_area), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_area = MatToUint8(np.zeros_like(concavity_area))

            for contour in contours:
                contour_image = np.zeros_like(concavity_area)

                cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                contour_image = MatToUint8(ndimage.binary_fill_holes(contour_image))

                se = skimage.morphology.disk(1)
                concavity_area_precedent = skimage.morphology.dilation(concavity_area_one[i-1],se)

                white_points_precedent = np.argwhere(concavity_area_precedent !=0)
                white_points_actual = np.argwhere(contour_image != 0)
                if elemento_comune(white_points_precedent, white_points_actual) == True:
                    final_area = cv2.add(final_area, contour_image)
            final_area = keep_n_largest_objects(final_area,1)

            concavity_area_one.append(final_area)

        i=i+1



    Heart_with_ellipse = [cv2.add(MatToUint8(heart), MatToUint8(white)) for heart, white in zip(image, concavity_area_one)]
    Montage_Matlab(Heart_with_ellipse,'Cardiac Depression Factor')

    # number of nonzero elements in matrix representing depression area
    depression_area_p = [np.count_nonzero(concavity) for concavity in concavity_area_one]

    # single pixel area in mm
    pixel_area = pixel_distance[0] * pixel_distance[1]

    # depression area in mm
    depression_area = [depr * pixel_area for depr in depression_area_p]

    ##correct chest area computation
    # number of nonzero elements in matrix representing correct heart

    correct_heart = [cv2.add(MatToUint8(concavity),MatToUint8(h_mask)) for concavity,h_mask in zip(concavity_area_one,mask)]

    correct_area_p = [np.count_nonzero(corr) for corr in correct_heart]
    # correct chest area in mm
    correct_area = [corr_area * pixel_area for corr_area in correct_area_p]

    first_non_zero = np.where(depression_area_p!=0)[0]
    first_non_zero = first_non_zero[-1]+1
    depression_volume = np.sum(np.multiply(depression_area[:first_non_zero+1],slice_distance)) / 1000
    corrheart_volume =  np.sum(np.multiply(correct_area[:first_non_zero+1],slice_distance))  / 1000
    depress_fraction =  np.divide(depression_volume,corrheart_volume) * 100

    return depress_fraction









def cardiac_rotation (images, heart_masks, index_max_compression, thorax_mask, inner_mask, inner_contour, pmax1_H, pmax2_H):
    # SPINE LOCATION
    means = np.array([np.mean(x) for x in images])

    above_threshold_images = [im[im > mean] for im, mean in zip(images, means)]

    means_sup = np.array([np.mean(x) for x in above_threshold_images])
    sigma = np.array([np.mean(x) for x in above_threshold_images])

    means_sup_plus_sigma = means_sup + sigma

    images_thresholded = [im > m for im, m in zip(images, means_sup)]

    complexive_AND = np.ones_like(images_thresholded[0])
    se = skimage.morphology.rectangle(1, 10)
    se2 = skimage.morphology.disk(10)
    i = 0
    while i < len(images_thresholded):
        im_dilated = skimage.morphology.dilation(images_thresholded[i], se)
        mask2 = skimage.morphology.erosion(thorax_mask[i], se2)
        im_dilated = np.logical_and(im_dilated, mask2)
        complexive_AND = np.logical_and(complexive_AND, im_dilated)
        i = i + 1

    complexive_AND = skimage.morphology.remove_small_objects(complexive_AND, 5)


    width = complexive_AND.shape[0]
    third_width = width // 3

    complexive_AND_no_artifacts = np.copy(complexive_AND)
    complexive_AND_no_artifacts[:, :third_width] = 0
    complexive_AND_no_artifacts[:, 2 * third_width:] = 0

    contours, _ = cv2.findContours(MatToUint8(complexive_AND_no_artifacts), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    spine_location = np.zeros_like(complexive_AND_no_artifacts)
    lowest_y = -float('inf')
    for contour in contours:
        image = MatToUint8(np.zeros_like(complexive_AND_no_artifacts))
        cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
        image = ndimage.binary_fill_holes(image)
        coordinate_bianche = np.argwhere(image != 0)
        if len(coordinate_bianche) != 0:
            centroide_y = np.mean(coordinate_bianche[:, 0])
            if centroide_y > lowest_y:
                lowest_y = centroide_y
                spine_location = image.copy()

    points_of_location = np.argwhere(spine_location != 0)

    def elemento_comune(lista1, lista2):
        for coord1 in lista1:
            for coord2 in lista2:
                if np.array_equal(np.array(coord1), np.array(coord2)):
                    return True
        return False

    image_of_max_compression = images[index_max_compression] > means_sup[index_max_compression]
    image_of_max_compression = skimage.morphology.remove_small_objects(image_of_max_compression, 5)
    image_of_max_compression_masked = np.logical_and(image_of_max_compression,
                                                     np.logical_not(inner_mask[index_max_compression]))

    spine_mask = MatToUint8(np.zeros_like(image_of_max_compression_masked))
    contours, _ = cv2.findContours(MatToUint8(image_of_max_compression_masked), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        contour_image = MatToUint8(np.zeros_like(image_of_max_compression_masked))
        cv2.drawContours(contour_image, contour, -1, (255, 255, 255), thickness=1)
        contour_image = ndimage.binary_fill_holes(contour_image)
        white_points = np.argwhere(contour_image != 0)
        if elemento_comune(points_of_location, white_points) == True:
            spine_mask = cv2.add(MatToUint8(spine_mask), MatToUint8(contour_image))


    spine_mask = keep_n_largest_objects(spine_mask, 1)
    white_pixels_spine = np.argwhere(spine_mask != 0)

    centroid_x = np.mean([x[0] for x in white_pixels_spine])
    centroid_y = np.mean([x[1] for x in white_pixels_spine])

    spine_point = [centroid_y, centroid_x]

    # sternum_spinal_line = np.mean(np.array([min_coord[1] , spine_point[1]]))
    sternum_spinal_line = int(np.mean(np.array([spine_point[0], spine_point[0]])))

    heart_contours, _ = cv2.findContours(MatToUint8(heart_masks[index_max_compression]), cv2.RETR_LIST,
                                         cv2.CHAIN_APPROX_NONE)
    heart_contours = flat_contours(heart_contours)

    white_points_inner_mask = np.argwhere(inner_mask[index_max_compression]!=0)

    Min_heart_point = [point.tolist() for point in white_points_inner_mask if point[1]==sternum_spinal_line]

    Min_heart_point = max(Min_heart_point, key=lambda punto: punto[0])
    Min_heart_point = [Min_heart_point[1], Min_heart_point[0]]

    distanze = [np.linalg.norm(np.array(punto) - np.array(Min_heart_point)) for punto in heart_contours]

    indice_punto_max_dist = np.argmax(distanze)

    punto_max_dist = heart_contours[indice_punto_max_dist]

    punto1 = [Min_heart_point[0], images[0].shape[1] - Min_heart_point[1]]
    punto2 = [punto_max_dist[0], images[0].shape[1] - punto_max_dist[1]]
    angolo_gradi = 90 - np.degrees(np.arctan2(punto2[1]-punto1[1],punto2[0]-punto1[0]))


    plt.figure()
    plt.imshow(images[index_max_compression], 'gray')
    plt.plot([sternum_spinal_line, sternum_spinal_line], [0, images[index_max_compression].shape[0] - 1], color='red')
    plt.plot([Min_heart_point[0], punto_max_dist[0]], [Min_heart_point[1], punto_max_dist[1]])
    plt.text(10, 10, f'Angolo: {angolo_gradi}', color='white')
    plt.show()




def left_cardiac_lateral_shift_correction (images, heart_masks, index_max_compression, thorax_mask, inner_mask, inner_contour, pmax1_H, pmax2_H, mask_to_correct):

    # SPINE LOCATION
    means = np.array([np.mean(x) for x in images])

    above_threshold_images = [im[im > mean] for im, mean in zip(images, means)]

    means_sup = np.array([np.mean(x) for x in above_threshold_images])
    sigma = np.array([np.mean(x) for x in above_threshold_images])

    means_sup_plus_sigma = means_sup + sigma

    images_thresholded = [im > m for im, m in zip(images, means_sup)]

    complexive_AND = np.ones_like(images_thresholded[0])
    se = skimage.morphology.rectangle(1, 10)
    se2 = skimage.morphology.disk(10)
    i = 0
    while i < len(images_thresholded):
        im_dilated = skimage.morphology.dilation(images_thresholded[i], se)
        mask2 = skimage.morphology.erosion(thorax_mask[i],se2)
        im_dilated = np.logical_and(im_dilated,mask2)
        complexive_AND = np.logical_and(complexive_AND, im_dilated)
        i = i + 1

    complexive_AND = skimage.morphology.remove_small_objects(complexive_AND, 5)



    width = complexive_AND.shape[0]
    third_width = width // 3

    complexive_AND_no_artifacts = np.copy(complexive_AND)
    complexive_AND_no_artifacts[:, :third_width] = 0
    complexive_AND_no_artifacts[:, 2 * third_width:] = 0

    contours, _ = cv2.findContours(MatToUint8(complexive_AND_no_artifacts), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    spine_location = np.zeros_like(complexive_AND_no_artifacts)
    lowest_y = -float('inf')
    for contour in contours:
        image = MatToUint8(np.zeros_like(complexive_AND_no_artifacts))
        cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=1)
        image = ndimage.binary_fill_holes(image)
        coordinate_bianche = np.argwhere(image != 0)
        if len(coordinate_bianche) != 0:
            centroide_y = np.mean(coordinate_bianche[:, 0])
            if centroide_y > lowest_y:
                lowest_y = centroide_y
                spine_location = image.copy()

    points_of_location = np.argwhere(spine_location!=0)

    def elemento_comune(lista1, lista2):
        for coord1 in lista1:
            for coord2 in lista2:
                if np.array_equal(np.array(coord1), np.array(coord2)):
                    return True
        return False


    image_of_max_compression = images[index_max_compression] > means_sup[index_max_compression]
    image_of_max_compression = skimage.morphology.remove_small_objects(image_of_max_compression, 5)
    image_of_max_compression_masked = np.logical_and(image_of_max_compression,np.logical_not(inner_mask[index_max_compression]))


    spine_mask = MatToUint8(np.zeros_like(image_of_max_compression_masked))
    contours, _ = cv2.findContours(MatToUint8(image_of_max_compression_masked), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        contour_image = MatToUint8(np.zeros_like(image_of_max_compression_masked))
        cv2.drawContours(contour_image, contour, -1, (255, 255, 255), thickness=1)
        contour_image = ndimage.binary_fill_holes(contour_image)
        white_points = np.argwhere(contour_image != 0)
        if elemento_comune(points_of_location, white_points) == True:
            spine_mask = cv2.add(MatToUint8(spine_mask), MatToUint8(contour_image))



    spine_mask = keep_n_largest_objects(spine_mask,1)
    white_pixels_spine = np.argwhere(spine_mask != 0)

    centroid_x = np.mean([x[0] for x in white_pixels_spine])
    centroid_y = np.mean([x[1] for x in white_pixels_spine])

    spine_point = [centroid_y,centroid_x]


    #STERNUM LOCATION

    pcontour = inner_contour[index_max_compression].copy()

    x_max = np.argmax(np.array([x[0] for x in pcontour]))

    pcontour_x = [x[0] for x in pcontour]
    pcontour_x = pcontour_x[0:x_max + 1]

    pcontour_y = [x[1] for x in pcontour]
    pcontour_y = pcontour_y[0:x_max + 1]

    ## Sternum position
    # point corresponding to max1 point on the inner contour (upper half of contour)
    pmax1 = pmax1_H[index_max_compression].copy()
    pmax2 = pmax2_H[index_max_compression].copy()

    imax1 = np.where(pcontour_x == np.array(pmax1[0]).astype(np.float64))[0]
    if len(imax1) != 0:
        imax1 = imax1.min()
    else:
        imax1 = np.abs(pcontour_x - np.array(pmax1[0]).astype(np.float64)).argmin()
    # point corresponding to max2 point on the inner contour (upper half of contour)
    imax2 = np.where(pcontour_x == np.array(pmax2[0]).astype(np.float64))[0]
    if len(imax2) != 0:
        imax2 = imax2.min()
    else:
        imax2 = np.abs(pcontour_x - np.array(pmax2[0]).astype(np.float64)).argmin()

    # coordinate among the points found above)

    minsternumv = np.max(pcontour_y[imax1:imax2 + 1])
    iminsternuma = np.where(pcontour_y[imax1:imax2 + 1] == minsternumv)[0]
    # Among the points with the same y coordinate it is selected the middle one
    iminsternum = np.abs((iminsternuma[-1] + iminsternuma[0]) / 2) + imax1 - 1
    min_coord = pcontour[int(iminsternum)]
    min_coord = [min_coord[0], min_coord[1]]

    #sternum_spinal_line = np.mean(np.array([min_coord[1] , spine_point[1]]))
    sternum_spinal_line = np.mean(np.array([spine_point[0], spine_point[0]]))



    heart_contours, _ = cv2.findContours(MatToUint8(mask_to_correct), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    heart_contours = flat_contours(heart_contours)


    heart_contours_right  = [x for x in heart_contours if x[0] < sternum_spinal_line]
    heart_contours_left = [x for x in heart_contours if x[0] > sternum_spinal_line]

    right_max_distance_point = min(heart_contours_right, key=lambda punto: punto[0])
    left_max_distance_point = max(heart_contours_left, key=lambda punto: punto[0])

    right_distance = np.abs(sternum_spinal_line - right_max_distance_point[0])
    left_distance = np.abs(sternum_spinal_line - left_max_distance_point[0])

    left_shift = (left_distance / (left_distance + right_distance)) * 100


    return left_shift


def intersect(a, b):
    common_elements = []
    indices_a = []
    indices_b = []

    for idx, val in enumerate(a):
        if val in b:
            common_elements.append(val)
            indices_a.append(idx)
            indices_b.append(b.index(val))

    return common_elements, indices_a, indices_b





def innercontour_seg_2(I_imadjust, BWlung, contour, xhalf, lung1, lung2):

    nsteps = 12
    nrow = int(np.round((len(contour) - nsteps) / nsteps))

    midX = np.zeros((nrow+1, 1))
    midY = np.zeros((nrow+1, 1))

    inters = [None] * (nrow+1)

    distancemidint = np.zeros((nrow+1, 1))
    distancem = np.zeros((nrow+1, 1))
    stdm = np.zeros((nrow+1, 1))

    a = -1

    contour_x = np.array([x[0] for x in contour])
    contour_y = np.array([x[1] for x in contour])

    icx = np.where(contour_x == xhalf)[0]
    ic2 = icx[0]
    ic4 = icx[1]

    ic1 = 0
    ic3x = np.where(contour_y == contour_y[0])[0]
    ic3delete = np.where(np.diff(ic3x) == 1)[0]
    if len(ic3delete) != 0:
        ic3delete = ic3delete + 1
        ic3x = [x for index, x in enumerate(ic3x) if index not in ic3delete]
    ic3 = ic3x[1]

    max_i_value = len(contour) - nsteps
    for i in range(0, max_i_value + 1, nsteps):
        a = a + 1

        p1 = contour[i]
        p2 = contour[i + nsteps]


        midX[a] = np.round(np.mean([p1[0], p2[0]]))
        midY[a] = np.round(np.mean([p1[1], p2[1]]))

        if p2[1] == p1[1]:
            y = np.linspace(0, I_imadjust.shape[0], I_imadjust.shape[1])
            x = midX[a] * np.ones((I_imadjust.shape[1]))
            slope = 10000000000

        elif p2[0] == p1[0]:
            x = np.linspace(0, I_imadjust.shape[0], I_imadjust.shape[1])
            y = midY[a] * np.ones((I_imadjust.shape[1]))
            slope = 0

        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            slope = -1 / slope

            x = np.linspace(0, I_imadjust.shape[0], I_imadjust.shape[1])
            x = [np.round(p) for p in x]

            y = slope * (x - midX[a]) + midY[a]
            y = [np.round(p) for p in y]

            idelete = np.where( (np.array(y) > I_imadjust.shape[0]) | (np.array(y) < 0) )[0]
            x = [p for i, p in enumerate(x) if i not in idelete]
            y = [p for i, p in enumerate(y) if i not in idelete]

        mat = [[px, py] for px, py in zip(x, y)]
        mat_x = [x[0] for x in mat]
        mat_y = [x[1] for x in mat]

        if i >= ic1 & i <= ic2:
            v,_, ib = intersect(lung1, mat)
            if len(ib) != 0:
                if v[0][1] > v[-1][1]:
                    ib = ib[::-1]
                inters[a] = mat[ib[0]]

        elif i >= ic2 & i <= ic3:
            v,_, ib = intersect(lung2, mat)
            if len(ib) != 0:
                if v[0][1] >= V[-1][1]:
                    ib = ib[::-1]
                inters[a] = mat[ib[0]]

        elif i >= ic3 & i <= ic4:
            v,_, ib = intersect(lung2, mat)
            if (len(ib)) != 0:
                if v[0][1] > v[-1][1]:
                    ib = ib[::-1]
                inters[a] = mat[ib[-1]]

        else:
            v,_, ib = intersect(lung1, mat)
            if len(ib) != 0:
                if v[0][1] >= v[-1][1]:
                    ib = ib[::-1]
                inters[a] = mat[ib[-1]]

        mid = [[x, y] for x, y in zip(midX, midY)]

        if inters[a] != None:

            distancemidint[a] = np.sqrt((midX[a] - inters[a][0]) ** 2 + (midY[a] - inters[a][1]) ** 2)
            distancem[a] = np.mean(distancemidint[0:a + 1])
            stdm[a] = np.std(distancemidint[0:a + 1])

            if a > 0:
                if distancemidint[a] >= ((2 * np.floor(distancem[a]) - np.floor(stdm[a]))):
                    distancemidint[a] = distancemidint[a - 1]
                    distancem[a] = np.mean(distancemidint[0:a + 1])
                    distmatmid = np.round(np.sqrt((midX[a] - mat_x) ** 2 + (midY[a] - mat_y) ** 2))
                    imid = np.where(distmatmid == 0)[0]
                    if len(imid)!=0:
                        imid = int(imid[0])
                    else:
                        imid = 0
                    slope = np.round(slope)

                    if (i >= ic1) & (i <= ic2):
                        if slope < 0:
                            index_delete = range(imid + 1, len(distmatmid))
                        else:
                            index_delete = range(0, imid)
                        distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                        mat = [x for i, x in enumerate(mat) if i not in index_delete]
                        mat_x = [x[0] for x in mat]
                        mat_y = [x[1] for x in mat]

                    elif (i > ic2) & (i <= ic3):
                        if slope <= 0:
                            index_delete = range(imid + 1, len(distmatmid))
                        else:
                            index_delete = range(0, imid)
                        distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                        mat = [x for i, x in enumerate(mat) if i not in index_delete]
                        mat_x = [x[0] for x in mat]
                        mat_y = [x[1] for x in mat]

                    elif (i > ic3) & (i <= ic4):
                        if slope < 0:
                            index_delete = range(0, imid)
                        else:
                            index_delete = range(imid + 1, len(distmatmid))
                        distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                        mat = [x for i, x in enumerate(mat) if i not in index_delete]
                        mat_x = [x[0] for x in mat]
                        mat_y = [x[1] for x in mat]

                    else:
                        if slope <= 0:
                            index_delete = range(0, imid)
                        else:
                            index_delete = range(imid + 1, len(distmatmid))
                        distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                        mat = [x for i, x in enumerate(mat) if i not in index_delete]
                        mat_x = [x[0] for x in mat]
                        mat_y = [x[1] for x in mat]

                    iinters = np.argmin(np.abs(distmatmid - distancemidint[a]))
                    if iinters is not None:
                        inters[a] = mat[iinters]


        else:
            distancemidint[a] = distancemidint[a - 1]
            distancem[a] = distancem[a - 1]
            stdm[a] = stdm[a-1]
            distmatmid = np.round(np.sqrt((midX[a] - mat_x) ** 2 + (midY[a] - mat_y) ** 2))
            imid = np.where(distmatmid == 0)[0]
            imid = int(imid[0])
            slope = np.round(slope)

            if (i >= ic1) & (i <= ic2):
                if slope < 0:
                    index_delete = range(imid + 1, len(distmatmid))
                else:
                    index_delete = range(0, imid)
                distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                mat = [x for i, x in enumerate(mat) if i not in index_delete]
                mat_x = [x[0] for x in mat]
                mat_y = [x[1] for x in mat]

            elif (i > ic2) & (i <= ic3):
                if slope <= 0:
                    index_delete = range(imid + 1, len(distmatmid))
                else:
                    index_delete = range(0, imid)
                distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                mat = [x for i, x in enumerate(mat) if i not in index_delete]
                mat_x = [x[0] for x in mat]
                mat_y = [x[1] for x in mat]

            elif (i > ic3) & (i <= ic4):
                if slope < 0:
                    index_delete = range(0, imid)
                else:
                    index_delete = range(imid + 1, len(distmatmid))
                distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                mat = [x for i, x in enumerate(mat) if i not in index_delete]
                mat_x = [x[0] for x in mat]
                mat_y = [x[1] for x in mat]

            else:
                if slope <= 0:
                    index_delete = range(0, imid)
                else:
                    index_delete = range(imid + 1, len(distmatmid))
                distmatmid = [x for i, x in enumerate(distmatmid) if i not in index_delete]
                mat = [x for i, x in enumerate(mat) if i not in index_delete]
                mat_x = [x[0] for x in mat]
                mat_y = [x[1] for x in mat]

            iinters = np.argmin(np.abs(distmatmid - distancemidint[a]))
            if iinters is not None:
                inters[a] = mat[iinters]




    plt.figure()
    plt.imshow(I_imadjust,'gray')
    for point in inters:
        if point!=None:
            plt.scatter(point[0],point[1])
    plt.show()

    return inters


def delete_area(ccorr,cwrong, im):
    acorr = np.zeros_like(im)
    awrong = np.zeros_like(im)

    a = cv2.subtract(awrong,acorr)
    a = skimage.morphology.remove_small_objects(minsize=50)

    if np.nonzero(a)!=0:
        se = skimage.morphology.disk(5)
        final = cv2.subtract(awrong, a)
        final = skimage.morphology.opening(final,se)
    else:
        final = awrong.copy()

    return final


def add_area(ccorr, cwrong, im):
    a = cv2.subtract(acorr,awrong)
    a = skimage.morphology.remove_small_objects(minsize=50)
    if np.nonzero(a) != 0:
        se = skimage.morphology.disk(5)
        final = cv2.add(awrong, a)
        final = skimage.morphology.opening(final, se)
    else:
        final = awrong.copy()
    return final


def eliminazione_diaframma(im,mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Prendi il contorno principale (assumendo che ci sia un solo contorno significativo)
    contours = flat_contours(contours)
    contours.reverse()

    # contours_pre = measure.find_contours(BWt)

    x_contours = np.array([x[0] for x in contours])
    y_contours = np.array([y[1] for y in contours])

    ymin = np.min(y_contours)
    iymin = np.where(y_contours == ymin)[0].min()
    contours = np.roll(contours, -iymin, axis=0)

    slopes = []

    # Itera attraverso i punti del contorno
    for i in range(len(contours)):
        # Indici dei punti consecutivi
        x1 = x_contours[i]  # Punto corrente
        x2 = x_contours[i+1]  # Punto successivo
        y1 = y_contours[i]  # Punto corrente
        y2 = y_contours[i+1]  # Punto successivo
        # Calcola la pendenza, gestendo il caso di divisione per zero
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = float('inf')  # Pendenza infinita per linee verticali
        slopes.append(slope)

    slope_changes = []
    # Itera per calcolare la differenza di pendenza
    for i in range(len(slopes)):
        # Calcola la variazione di pendenza
        delta_slope = abs(slopes[i] - slopes[(i + 1)])
        slope_changes.append(delta_slope)

    top_indices = np.argsort(slope_changes)[-2:]  # Prendi i 2 indici più grandi
    top_indices = sorted(top_indices)  # Ordina gli indici

    # Ottieni le coordinate dei due punti angolosi
    point1 = contour[top_indices[0]][0]
    point2 = contour[top_indices[1]][0]

    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Converti in immagine a colori
    cv2.circle(output, tuple(point1), 5, (0, 0, 255), -1)  # Punto 1 (rosso)
    cv2.circle(output, tuple(point2), 5, (255, 0, 0), -1)  # Punto 2 (blu)

    # Mostra o salva il risultato
    cv2.imwrite("output.png", output)
    cv2.imshow("Punti Angolosi", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






