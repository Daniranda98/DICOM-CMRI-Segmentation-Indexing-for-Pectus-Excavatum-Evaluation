from tkinter.filedialog import askdirectory
from pathlib import Path
import numpy
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import util
from scipy import ndimage, signal
from os.path import dirname, join as pjoin
import scipy.io as sio
import cv2
import pandas as pd
import os
import time
from Functions import Imadjust, Border_control,maskborder,Mat2gray, lung_segmentation, Montage_Matlab, Montage_Matlab_draw, sternum_segmentation, MatToUint8, innercontour_seg_2
from Functions import Hist_threshold, outer_contour, visualize_select, visualize_test, inner_index, aorta_segmentation_snake, liver_deletion, flat_contours
from Functions import depression_eval, inner_contours_seg, contourinterpolation, inner_analysis, Bwareafilt, aorta_segmentation_watershed, left_cardiac_lateral_shift_correction
from Functions import innermask_seg, innermask_select, inner_contour_correction, contcorrinterpolation, MatToUint8, mask_3d_visualization, cardiac_rotation
from Functions import Heart_segmentation, heart_index, delete_select, aorta_segmentation, visualize_select_heart, vena_cava_segmentation, heart_ellipse
from Functions import heart_concavity, heart_new_index, concavity_slider, aorta_segmentation_new, vena_segmentation_new, left_cardiac_lateral_shift, eliminazione_diaframma



#PRE-PROCESSING PART
#FOLDER SELECTION
path_selected =  '{}'.format(askdirectory(title='Select folder with .dcm files', mustexist=True))
path= Path(path_selected)
all_files=path.glob("*.dcm")

#DICOM IMPORTATION
mri_data=[]
for file in all_files:
    data = pydicom.read_file(file)
    mri_data.append(data)


#IMAGE SELECTION FROM DICOM FILES
image_data=[]
for mri_slice in mri_data:
    image_data.append(mri_slice.pixel_array)


#IMAGE NUMBERS FROM 0 T0 1
image_data_2=[]
for image in image_data:
    image_data_2.append(Mat2gray(image))

#Montage_Matlab(image_data_2,'immagini originali')

#CONTRAST ADJUSTMENT ( Function:imadjust in matlab )
image_data_corrected=[]
for image_slice in image_data_2:
    image_data_corrected.append(Imadjust(image_slice))

Montage_Matlab(image_data_corrected,'immagini con contrasto aumentato')



#DICOM FILE INFO


pixel_spacing=mri_data[0].PixelSpacing
slice_spacing=mri_data[0].SpacingBetweenSlices
slice_thickness=mri_data[0].SliceThickness
Voxel_volume=pixel_spacing[0]*pixel_spacing[1]*slice_spacing
number_of_slices=len(mri_data)
patient_sex = mri_data[0].PatientSex




# -------------------------------------------- CHEST PART --------------------------------------------------------#

# #VISUALIZE ALL SLICES, AND SELECTION
#
n1,ndend,ns,s1,send,numimages,numdep,srange,gender=visualize_select(image_data_corrected,number_of_slices,patient_sex)
#
# #BORDER CONTROL FUNCTION
Threshold=0.1
Bw_image=[]
i1_mask=[]
i2_mask=[]
i = 0
#
while i<numimages:
     Bw,i1,i2 = Border_control(image_data_corrected[i+n1],Threshold)
     Bw_image.append(Bw)
     i1_mask.append(i1)
     i2_mask.append(i2)
     i=i+1
#
#Montage_Matlab(Bw_image,'Immagini binarie')
#
#MASK BORDER FUNCTION
i=0
Im_filt=[]
Bw_filt=[]
x_half=[]
y_half=[]
i1 = []
i2 = []
while i<numimages:
     i1.append(i1_mask[i])
     i2.append(i2_mask[i])
     im,bw,x,y=maskborder(image_data_corrected[i+n1],Bw_image[i],i1,i2)
     Im_filt.append(im)
     Bw_filt.append(bw)
     x_half.append(x)
     y_half.append(y)
     i=i+1
#
#
Montage_Matlab(Im_filt,'immagini con petto isolato da braccia')
#
#
# # DEPRESSION QUANTIFICATION
lowering_factor =  int(np.round((15*(Im_filt[0].shape[1]))/320))
i=0
pmax1=[]
pmax2=[]
pmin=[]
Bcontest=[]
contornocorr=[]
BWcorrect=[]
Iellipse=[]
BWdepression=[]
depression_area=[]
corrchest_area=[]
#
#
while i<numimages:
     a,b,c,d,e=outer_contour(Bw_filt[i],x_half[i],y_half[i])

     #plt.figure()
     #plt.imshow(Bw_filt[i],'gray')
     #plt.scatter(a[0],a[1],color='red',s=30)
     #plt.scatter(b[0],b[1],color='red',s=30)
#
#
     a[1]=a[1]+lowering_factor
     b[1]=b[1]+lowering_factor
     pmax1.append(a)
     pmax2.append(b)
     pmin.append(c)
     Bcontest.append(d)
     contornocorr.append(e)
#     # further analysis is done only if patient is male and by considering slices selected for depression quantification
     if ("M" in str(gender)) and (i<numdep):
         BWc,Ell,BWd,dep,corr=depression_eval(pmax1[i], pmax2[i], Bw_filt[i] , Im_filt[i], y_half[i], pixel_spacing)
         BWcorrect.append(BWc)
         Iellipse.append(Ell)
         BWdepression.append(BWd)
         depression_area.append(dep)
         corrchest_area.append(corr)
     i=i+1
#
#
#
#
if ("M" in str(gender)):
     Montage_Matlab(Iellipse, 'Area di petto mancante')
     print('.')
#
#
#
# # CALCOLI SE PAZIENTE UOMO
if 'M' in str(gender):
     depression_volume = np.sum(np.multiply(depression_area,slice_spacing)) / 1000
     corrchest_volume =  np.sum(np.multiply(corrchest_area,slice_spacing))  / 1000
     depress_fraction =  np.divide(depression_volume,corrchest_volume) * 100
     print(depress_fraction)

#
#
# # INNER CONTOUR ANALYSIS
# # inner contour analysis: elimination of slices that preceed the one selected for indices computation
#
Im_filt_s=Im_filt[s1:s1+srange]
Bw_filt_s=Bw_filt[s1:s1+srange]
Bcontest_s=Bcontest[s1:s1+srange]
contornocorr_s=contornocorr[s1:s1+srange]
x_half_s=x_half[s1:s1+srange]
y_half_s=y_half[s1:s1+srange]
pmax1_s=pmax1[s1:s1+srange]
pmax2_s=pmax2[s1:s1+srange]
pmin_s=pmin[s1:s1+srange]
#
#
#
# #inner contour analysis: lung and heart segmentation by using histogram partitioning method
#
counts=[]
bins=[]
tb=[]
itb=[]
#
for image in Im_filt_s:
     count, bin = np.histogram(np.ravel(image), 256, [0,1])
     counts.append(count)
     bins.append(bin)
     t, it = Hist_threshold(count, bin, 0, 0.3)
     tb.append(t)
     itb.append(it)
#
#
countsbd=[]
binsbd=[]
i=0
while i<srange:
     c = list(counts[i])
     b = list(bins[i])
     c[0:itb[i]+1]=[]
     b[0:itb[i]+1]=[]
     countsbd.append(c)
     binsbd.append(b)
     i=i+1
#
#
#
t=[]
th=[]
i=0
while i<srange:
     t1,_ = Hist_threshold(countsbd[i], binsbd[i], tb[i], 0.4)
     t2,_ = Hist_threshold(countsbd[i], binsbd[i], 0.3, 0.8)
     t.append(t1)
     th.append(t2)
     i=i+1
#
#
#
# # inner contour analysis: correction of thresholds
t_mean = np.mean(t)
t_delete=np.where((np.round(np.abs(t-t_mean),2))>0.05)[0]
t_corr = np.array(t)
t_corr[t_delete]=t_mean
#
#
t_mean_heart = np.mean(th)
t_delete_heart=np.where((np.round(np.abs(th-t_mean_heart),2))>0.05)[0]
t_corr_heart = np.array(th)
t_corr_heart[t_delete_heart]=t_mean_heart
#
#
# # inner contour analysis: application of thresholds for lung segmentation
#
BWlung = []
polm1 = []
polm2 = []
lung_fraction = []
i = 0
#
#
while i < srange:
     BW,pol1,pol2,fraction = lung_segmentation(Im_filt_s[i],t_corr[i],Bcontest_s[i],x_half_s[i],Bw_filt_s[i])
     BWlung.append(BW)
     polm1.append(pol1)
     polm2.append(pol2)
     lung_fraction.append(fraction)
     i = i + 1
#
#
#
#
#
Montage_Matlab(BWlung,'Maschera Polmoni')
#
# #inner contour analysis: elimination of slices where lungs are not visible
threshold_2=0.2
incorr_slice=np.array(lung_fraction)<threshold_2
srange =srange - np.count_nonzero(incorr_slice)
if srange>15:
     srange = 15
index_incorr = np.where(incorr_slice==True)[0]
#
#
#
if len(index_incorr)!=0:
     Im_filt_s=[x for index,x in enumerate(Im_filt_s) if index not in index_incorr]
     Bcontest_s = [x for  index,x in enumerate(Bcontest_s) if index not in index_incorr]
     contornocorr_s = [x for  index,x in enumerate(contornocorr_s) if index not in index_incorr]
     x_half_s = [x for  index,x in enumerate(x_half_s) if index not in index_incorr]
     y_half_s = [x for  index,x in enumerate(y_half_s) if index not in index_incorr]
     pmax1_s = [x for  index,x in enumerate(pmax1_s) if index not in index_incorr]
     pmax2_s = [x for  index,x in enumerate(pmax2_s) if index not in index_incorr]
     pmin_s = [x for  index,x in enumerate(pmin_s) if index not in index_incorr]
     polm1 = [x for  index,x in enumerate(polm1) if index not in index_incorr]
     polm2 = [x for  index,x in enumerate(polm2) if index not in index_incorr]
     t_corr = [x for  index,x in enumerate(t_corr) if index not in index_incorr]
     t_corr_heart = [x for  index,x in enumerate(t_corr_heart) if index not in index_incorr]
     Bw_filt_s = [x for  index,x in enumerate(Bw_filt_s) if index not in index_incorr]
#
#
# #inner contour analysis: inner contour segmentation
intpoint=[]
pcontorno=[]
contourmask=[]
#
#
i=0
while i < srange:
     #inters = inner_contours_seg(Im_filt_s[i],BWlung[i],contornocorr_s[i],x_half_s[i],polm1[i],polm2[i])
     inters = inner_contours_seg(Im_filt_s[i], BWlung[i], contornocorr_s[i], x_half_s[i], polm1[i], polm2[i])
     pcont,contmask=contourinterpolation(pmin_s[i],pmax1_s[i],pmax2_s[i],inters,Im_filt_s[i],contornocorr_s[i])
     pcontorno.append(pcont)
     contourmask.append(contmask)
     i=i+1
#
#
#
#
Montage_Matlab(contourmask,'Maschera parte toracica')
# #inner contour analysis: vertebral body exclusion
Icont=[]
contpoint = []
contpointd = []
#
i=0
while i < srange:
     Ic , cp = innermask_seg(Im_filt_s[i], contourmask[i] , Bcontest_s[i] , t_corr[i], t_corr_heart[i])
     Icont.append(Ic)
     contpoint.append(cp)
     x_cp = [x[0] for x in cp]
     if len(x_cp)>=27:
         x_cp = signal.decimate(x_cp, 3, zero_phase=True)
     y_cp = [x[1] for x in cp]
     if len(y_cp)>=27:
         y_cp = signal.decimate(y_cp, 3, zero_phase=True)
     contpointd.append( [[np.round(x),np.round(y)] for x,y in zip(x_cp,y_cp)])
     i=i+1
#
#
Montage_Matlab(Icont,'Contorni non corretti')
#
ncorr = innermask_select(Icont)
#
#
plt.figure()
plt.imshow(Im_filt_s[ncorr],'gray')
plt.scatter([x[0] for x in contpointd[ncorr]],[x[1] for x in contpointd[ncorr]])
plt.show()
#
#
# #inner contour analysis: correction of slices preceeding the one selected by user
c2corr=[None]*srange
innermask=[None]*srange
innermaskn=[None]*srange
Iinner=[None]*srange
Ipolmoni=[None]*srange
#
#
#
if ncorr > 0:
     for u in range(ncorr, -1, -1):
         if u == ncorr:
             c2corr[u] = contpointd[ncorr]
         else:
             c2corr[u] = inner_contour_correction(c2corr[u+1], contpointd[u], y_half_s[u],Im_filt_s[u])
         innermask[u], Iinner[u], Ipolmoni[u] = inner_analysis(c2corr[u], Im_filt_s[u])
#
#     # inner contour analysis: correction of slices following the one selected by user
#
#
     for u in range(ncorr, srange):
         if u == ncorr:
             c2corr[u] = contpointd[ncorr]
         else:
             c2corr[u] = inner_contour_correction(c2corr[u-1], contpointd[u], y_half_s[u],Im_filt_s[u])
         innermask[u], Iinner[u], Ipolmoni[u] = inner_analysis(c2corr[u], Im_filt_s[u])
#
#
#
Montage_Matlab(innermask,'maschera interna corretta')
Montage_Matlab(Iinner,'Parte interna corretta')
Montage_Matlab(Ipolmoni,'Parte polmoni corretta')
#
#
#
# ## index computation: visualization of the slice selected by user and the one picked by algorithm
indslice=0
selslice = ns
#
#
# ## index computation: inner contour preparation
pcontornoint,p1in,p2in,ipr = contcorrinterpolation(c2corr[indslice],Im_filt_s[indslice],pmin_s[indslice],pmax1_s[indslice],y_half_s[indslice])
#
while ipr>1:
     indslice=indslice+1
     [pcontornoint,p1in,p2in,ipr] = contcorrinterpolation(c2corr[indslice],Im_filt_s[indslice],pmin_s[indslice],pmax1_s[indslice],y_half_s[indslice])
#
#
#
if np.abs(p1in[0]-pmax1_s[indslice][0])>20:
     pmax1_s[indslice][0] = p1in[0]
#
#
# ##index computation on the slice picked by algorithm (METTERE 1)
diamtrasv,d_emitsx,d_emitdx,iAsymetry,iFlatness,minsternum,maxAPd,minAPd,iHaller,iCorrection,iDepression,pcvert= inner_index(pcontornoint,pmax1_s[indslice],pmax2_s[indslice],Im_filt_s[indslice],pixel_spacing,1)
#
# #creation of a table containing results of inner thoracic distances,
# #thoracic indices and depression factor resulting from depression quantification
if 'M' in str(gender):
     results = ['transversed diameter (cm)', 'min APd (cm)', 'max APd (cm)', 'APd right emitorax (cm)',
                'APd left emitorax (cm)', 'iHaller', 'iCorrection (%)', 'iDepression', 'iAsymetry', 'iFlatness', 'depression_factor(%)']
     values = [diamtrasv, minAPd, maxAPd, d_emitdx, d_emitsx, iHaller, iCorrection, iDepression, iAsymetry, iFlatness,
               depress_fraction]
     Tdresult = pd.DataFrame({'Results': results, 'Values': values})
     print(Tdresult)
else:
     results = ['transversed diameter (cm)', 'min APd (cm)', 'max APd (cm)', 'APd right emitorax (cm)',
                'APd left emitorax (cm)', 'iHaller', 'iCorrection (%)', 'iDepression', 'iAsymetry', 'iFlatness']
     values = [diamtrasv, minAPd, maxAPd, d_emitdx, d_emitsx, iHaller, iCorrection, iDepression, iAsymetry, iFlatness]
     Tdresult = pd.DataFrame({'Results': results, 'Values': values})
     print(Tdresult)
#
#
# #results saved as an Excel table in the same folder where images are located
table_path_format = os.path.join(path_selected, 'results_1_chest.xlsx')
print(table_path_format)
Tdresult.to_excel(table_path_format, index=True)
#
#
#
# # ## index computation on the slice selected by user (if it is different from the one picked by algorithm)
if (incorr_slice[0]==1) | (indslice>1):


      diamtrasv2, d_emitsx2, d_emitdx2, iAsymetry2, iFlatness2, minsternum2, maxAPd2, minAPd2, iHaller2, iCorrection2, iDepression2, pcvert2 = inner_index(pcontornoint, pmax1_s[indslice], pmax2_s[indslice], image_data_corrected[selslice], pixel_spacing, 2)
      if 'M' in str(gender):
          results = ['transversed diameter (cm)', 'min APd (cm)', 'max APd (cm)', 'APd right emitorax (cm)',
                     'APd left emitorax (cm)', 'iHaller', 'iCorrection (%)', 'iDepression', 'iAsymetry', 'iFlatness',
                     'depression_factor(%)']
          values2 = [diamtrasv2, minAPd2, maxAPd2, d_emitdx2, d_emitsx2, iHaller2, iCorrection2, iDepression2, iAsymetry2, iFlatness2,depress_fraction]
      else:
          results = ['transversed diameter (cm)', 'min APd (cm)', 'max APd (cm)', 'APd right emitorax (cm)',
                     'APd left emitorax (cm)', 'iHaller', 'iCorrection (%)', 'iDepression', 'iAsymetry', 'iFlatness']
          values2 = [diamtrasv2, minAPd2, maxAPd2, d_emitdx2, d_emitsx2, iHaller2, iCorrection2, iDepression2, iAsymetry2, iFlatness2]
      Tdresult = pd.DataFrame({'Results': results, 'Values': values2})
      print(Tdresult)
#     #Table containing results
      table_path_format = os.path.join(path_selected, 'results_2_chest.xlsx')
      print(table_path_format)
      Tdresult.to_excel(table_path_format, index=True)


#  ------------------------------------------------- HEART PART ---------------------------------------------


#VISUALIZE ALL SLICES, AND SELECTION

n1,n2,ns = visualize_select_heart(image_data_corrected)
heart_range = n2 - n1 + 1
ns = ns - n1
#s2 = s1 + srange
#BORDER CONTROL FUNCTION

Threshold=0.1
Bw_image_H=[]
i1_mask_H=[]
i2_mask_H=[]

i = 0
while i<heart_range:
    Bw,i1,i2 = Border_control(image_data_corrected[i+n1],Threshold)
    Bw_image_H.append(Bw)
    i1_mask_H.append(i1)
    i2_mask_H.append(i2)
    i=i+1

Montage_Matlab(Bw_image_H,'Immagini binarie cuore')

#MASK BORDER FUNCTION
i=0
Im_filt_H=[]
Bw_filt_H=[]
x_half_H=[]
y_half_H=[]
i1_H = []
i2_H = []
while i<heart_range:
    i1_H.append(i1_mask_H[i])
    i2_H.append(i2_mask_H[i])
    im,bw,x,y=maskborder(image_data_corrected[i+n1],Bw_image_H[i],i1_H,i2_H)
    Im_filt_H.append(im)
    Bw_filt_H.append(bw)
    x_half_H.append(x)
    y_half_H.append(y)
    i=i+1


Montage_Matlab(Im_filt_H,'immagini con petto isolato da braccia (cuore)')


# DEPRESSION QUANTIFICATION
lowering_factor =  int(np.round((15*(Im_filt_H[0].shape[1]))/320))
i=0
pmax1_H=[]
pmax2_H=[]
pmin_H=[]
Bcontest_H=[]
contornocorr_H=[]


while i<heart_range:
    a,b,c,d,e=outer_contour(Bw_filt_H[i],x_half_H[i],y_half_H[i])
    a[1]=a[1]+lowering_factor
    b[1]=b[1]+lowering_factor
    pmax1_H.append(a)
    pmax2_H.append(b)
    pmin_H.append(c)
    Bcontest_H.append(d)
    contornocorr_H.append(e)
    i=i+1



#inner contour analysis: lung and heart segmentation by using histogram partitioning method
counts_H=[]
bins_H=[]
tb_H=[]
itb_H=[]
for image in Im_filt_H:
    count, bin = np.histogram(np.ravel(image), 256, [0,1])
    counts_H.append(count)
    bins_H.append(bin)
    t, it = Hist_threshold(count, bin, 0, 0.3)
    tb_H.append(t)
    itb_H.append(it)




countsbd_H=[]
binsbd_H=[]
i=0
while i<heart_range:
    c = list(counts_H[i])
    b = list(bins_H[i])
    c[0:itb_H[i]+1]=[]
    b[0:itb_H[i]+1]=[]
    countsbd_H.append(c)
    binsbd_H.append(b)
    i=i+1


t_H=[]
th_H = []
i=0
while i<heart_range:
    t1,_ = Hist_threshold(countsbd_H[i], binsbd_H[i], tb_H[i], 0.4)
    t2, _ = Hist_threshold(countsbd_H[i], binsbd_H[i], 0.3, 0.8)
    t_H.append(t1)
    th_H.append(t2)
    i=i+1



# inner contour analysis: correction of thresholds
t_mean_H = np.mean(t_H)
t_delete_H=np.where((np.round(np.abs(t_H-t_mean_H),2))>0.05)[0]
t_corr_H = np.array(t_H)
t_corr_H[t_delete_H]=t_mean_H

t_mean_heart_H = np.mean(th_H)
t_delete_heart_H=np.where((np.round(np.abs(th_H-t_mean_heart_H),2))>0.05)[0]
t_corr_heart_H = np.array(th_H)
t_corr_heart_H[t_delete_heart_H]=t_mean_heart_H

# inner contour analysis: application of thresholds for lung segmentation

BWlung_H = []
polm1_H = []
polm2_H = []
lung_fraction_H = []
i = 0
while i < heart_range:
    BW,pol1,pol2,fraction = lung_segmentation(Im_filt_H[i],t_corr_H[i],Bcontest_H[i],x_half_H[i],Bw_filt_H[i])
    BWlung_H.append(BW)
    polm1_H.append(pol1)
    polm2_H.append(pol2)
    lung_fraction_H.append(fraction)
    i = i + 1


Montage_Matlab(BWlung_H,'Maschera Polmoni')





#inner contour analysis: inner contour segmentation
intpoint_H=[]
pcontorno_H=[]
contourmask_H=[]


i=0
while i < heart_range:
    inters = inner_contours_seg(Im_filt_H[i],BWlung_H[i],contornocorr_H[i],x_half_H[i],polm1_H[i],polm2_H[i])
    pcont,contmask=contourinterpolation(pmin_H[i],pmax1_H[i],pmax2_H[i],inters,Im_filt_H[i],contornocorr_H[i])
    pcontorno_H.append(pcont)
    contourmask_H.append(contmask)
    i=i+1


Montage_Matlab(contourmask_H,'Maschera parte toracica (cuore)')
#inner contour analysis: vertebral body exclusion
Icont_H=[]
contpoint_H = []
contpointd_H = []

i=0
while i < heart_range:
    Ic , cp = innermask_seg(Im_filt_H[i], contourmask_H[i] , Bcontest_H[i] , t_corr_H[i], t_corr_heart_H[i])
    Icont_H.append(Ic)
    contpoint_H.append(cp)
    x_cp = [x[0] for x in cp]
    x_cp = signal.decimate(x_cp,3,zero_phase=True)
    y_cp = [x[1] for x in cp]
    y_cp = signal.decimate(y_cp,3,zero_phase=True)
    contpointd_H.append( [[np.round(x),np.round(y)] for x,y in zip(x_cp,y_cp)])
    i=i+1


Montage_Matlab(Icont_H,'Contorni non corretti')

ncorr_H = innermask_select(Icont_H)

plt.figure()
plt.imshow(Im_filt_H[ncorr_H],'gray')
plt.scatter([x[0] for x in contpointd_H[ncorr_H]],[x[1] for x in contpointd_H[ncorr_H]])
plt.show()


#inner contour analysis: correction of slices preceeding the one selected by user
c2corr_H=[None]*heart_range
innermask_H=[None]*heart_range
innermaskn_H=[None]*heart_range
Iinner_H=[None]*heart_range
Ipolmoni_H=[None]*heart_range


if ncorr_H > 0:
    for u in range(ncorr_H, -1, -1):
        if u == ncorr_H:
            c2corr_H[u] = contpointd_H[ncorr_H]
        else:
            c2corr_H[u] = inner_contour_correction(c2corr_H[u+1], contpointd_H[u], y_half_H[u],Im_filt_H[u])
        innermask_H[u], Iinner_H[u], Ipolmoni_H[u] = inner_analysis(c2corr_H[u], Im_filt_H[u])

    # inner contour analysis: correction of slices following the one selected by user

    for u in range(ncorr_H, heart_range):
        if u == ncorr_H:
            c2corr_H[u] = contpointd_H[ncorr_H]
        else:
            c2corr_H[u] = inner_contour_correction(c2corr_H[u-1], contpointd_H[u], y_half_H[u],Im_filt_H[u])
        innermask_H[u], Iinner_H[u], Ipolmoni_H[u] = inner_analysis(c2corr_H[u], Im_filt_H[u])



Montage_Matlab(innermask_H,'maschera interna corretta')
Montage_Matlab(Iinner_H,'Parte interna corretta')
Montage_Matlab(Ipolmoni_H,'Parte polmoni corretta')



#----------------------------------- END CODE REPETITION


Heart_mask = [None] * len(Im_filt_H)
Heart_image = [None] * len(Im_filt_H)
i = 0
while i < len(Ipolmoni_H):
    mask,heart = Heart_segmentation(Im_filt_H[i],innermask_H[i],Ipolmoni_H[i])
    Heart_mask[i]=mask.copy()
    Heart_image[i]=heart.copy()
    i=i+1


Montage_Matlab(Heart_mask,'Mantenendo area maggiore')
Montage_Matlab(Heart_image,'Cuore')


#----------------------------------- ELIMINAZIONE AORTA


for i in range(len(Im_filt_H)):
    im, mask = liver_deletion(Heart_image[i], Heart_mask[i])
    Heart_image[i] = im.copy()
    Heart_mask[i] = mask.copy()


selezioni_aorta = delete_select(Heart_image)

im_heart_No_aorta = [None] * len(Im_filt_H)
mask_heart_No_aorta = [None] * len(Im_filt_H)
aorta_mask = [None] * len(Im_filt_H)

if len(selezioni_aorta)!=0:
    im_heart_No_aorta, mask_heart_No_aorta, aorta_mask = aorta_segmentation_new( Im_filt_H , Heart_image , Heart_mask , innermask_H , selezioni_aorta)
else:
    im_heart_No_aorta = [x for x in Heart_image]
    mask_heart_No_aorta = [x for x in Heart_mask]
    aorta_mask = [ np.zeros_like(x[0]) for x in Heart_mask]


#for i in range(len(Im_filt_H)):
    #if i in selezioni_aorta:
        #im , mask = aorta_segmentation(Heart_image[i],Heart_mask[i])
        #im_heart_No_aorta[i] = im.copy()
        #mask_heart_No_aorta[i] = np.array(mask.copy(),dtype=bool)
    #else:
        #im_heart_No_aorta[i] = Heart_image[i].copy()
        #mask_heart_No_aorta[i] = Heart_mask[i].copy()


Montage_Matlab_draw(Heart_image,'cuore')
Montage_Matlab(im_heart_No_aorta,'cuore senza aorta')

#----------------------------------- ELIMINAZIONE VENA

selezioni_vena = delete_select(im_heart_No_aorta)

im_heart_No_vessels= [None] * len(Im_filt_H)
mask_heart_No_vessels= [None] * len(Im_filt_H)
vena_mask = [None] * len(Im_filt_H)

if len(selezioni_vena)!=0:
    im_heart_No_vessels, mask_heart_No_vessels, vena_mask = vena_segmentation_new(Im_filt_H, im_heart_No_aorta,mask_heart_No_aorta, innermask_H,selezioni_vena)
else:
    im_heart_No_vessels = [x for x in im_heart_No_aorta]
    mask_heart_No_vessels = [x for x in mask_heart_No_aorta]
    vena_mask = [np.zeros_like(x[0]) for x in mask_heart_No_vessels]

Montage_Matlab_draw(Heart_image,'cuore')
Montage_Matlab(im_heart_No_vessels,'cuore senza vena')


selezioni_diaframma = delete_select(im_heart_No_vessels)

im_heart_No_diaphragm= [None] * len(Im_filt_H)
mask_heart_No_diaphragm= [None] * len(Im_filt_H)
diaphragm_mask = [None] * len(Im_filt_H)

for i in range(len(Im_filt_H)):
    if i in selezioni_diaframma:
        im_heart_No_diaphragm[i], mask_heart_No_diaphragm[i] = eliminazione_diaframma(im_heart_No_vessels[i],mask_heart_No_vessels[i])
    else:
        im_heart_No_diaphragm[i] = im_heart_No_vessels[i].copy()
        mask_heart_No_diaphragm[i] = mask_heart_No_vessels[i].copy()


exit()
#for i in range(len(Im_filt_H)):
    #if i in selezioni_vena:
        #im_heart_No_vessels[i], mask_heart_No_vessels[i] = vena_cava_segmentation(im_heart_No_aorta[i],mask_heart_No_aorta[i])
    #else:
        #im_heart_No_vessels[i] = im_heart_No_aorta[i].copy()
        #mask_heart_No_vessels[i] = mask_heart_No_aorta[i].copy()

#Montage_Matlab_draw(Heart_image,'cuore')
#Montage_Matlab(im_heart_No_vessels,'cuore senza vena')



# VISUALIZZAZIONE

#mask_3d_visualization(mask_heart_No_vessels,pixel_spacing[0],pixel_spacing[1],slice_spacing)

#-----------------

#rld,lld,wpapd,ndh,td,clls,ai,cc=heart_index(Im_filt_s[ns],Heart_mask[ns],pixel_spacing)

#results = ['Right lateral distance (cm)', 'Left lateral distance(cm)', 'widest paramedian antero posterior diameter (cm)', 'narrowest diameter at xiphoid process (cm)',
           #'transverse diameter (cm)', ' Cardiac left lateral shift (%)', 'asymmetry index', 'cardiac compression index']
#values = [rld,lld,wpapd,ndh,td,clls,ai,cc]
#control = ["","","","","", "65.2 ± 5.8 %" , "< 1.15" , "< 1.82"]
#Tdresult = pd.DataFrame({'Results': results, 'Values': values , 'Control':control })
#print(Tdresult)



#i = 0
#while i < len(Im_filt_H):
    #plt.figure()
    #plt.imshow(Im_filt_H[i],'gray')
    #points = points_of_concavity[i]
    #if points is not None:
        #for point in points:
            #plt.scatter(point[1], point[0], s=2, alpha=0.5, c='red')
    #plt.title(f'Concavità: {percentage_of_concavity[i]:.2f}%')
    #plt.axis('off')
    #plt.draw()
    #i=i+1



left_shift = left_cardiac_lateral_shift( Im_filt_H, mask_heart_No_vessels, ns, Bw_filt_H, innermask_H, c2corr_H, pmax1_H, pmax2_H)
#heart_rotation = cardiac_rotation( Im_filt_H, mask_heart_No_vessels, ns, Bw_filt_H, innermask_H, pcontorno_H, pmax1_H, pmax2_H)

#coeff_var=[]
#i=0
#while i<len(Im_filt_H):
    #coeff_var.append(heart_new_index(Im_filt_H[i],mask_heart_No_vessels[i]))
    #i=i+1

coeff_var = heart_new_index(Im_filt_H[ns],mask_heart_No_vessels[ns])

cardiac_depression_fraction = heart_ellipse(Im_filt_H,mask_heart_No_vessels, c2corr_H, pmax1_H, pmax2_H, innermask_H, pixel_spacing, slice_spacing)

if left_shift != 100:
    wpapd, ndh, td, ai, cc = heart_index(Im_filt_H[ns], mask_heart_No_vessels[ns], pixel_spacing, c2corr_H[ns],pmax1_H[ns], pmax2_H[ns])
else:
    wpapd = None
    ndh = None
    td = None
    ai = None
    cc = None


results = [ 'widest paramedian antero posterior diameter (cm)', 'narrowest diameter at xiphoid process (cm)',
           'transverse diameter (cm)', 'asymmetry index', 'cardiac compression index', 'left cardiac lateral shift (%)','Heart new index (%)'
            , 'cardiac depression fraction (%)']
values = [wpapd,ndh,td,ai,cc,left_shift,coeff_var,cardiac_depression_fraction]
control = ["","","", "< 1.15" , "< 1.82", "","",""]
Tdresult = pd.DataFrame({'Results': results, 'Values': values , 'Control':control })
print(Tdresult)

exit()

#results saved as an Excel table in the same folder where images are located
table_path_format = os.path.join(path_selected, 'results_1_heart.xlsx')
print(table_path_format)
Tdresult.to_excel(table_path_format, index=True)


# Heart Confront
original_heart_chosen = mask_heart_No_vessels[ns].copy()
contours, _ = cv2.findContours(MatToUint8(original_heart_chosen), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = np.array(flat_contours(contours)).astype(np.float32)
ellipse = cv2.fitEllipse(contours)
ellipse_mask = np.zeros_like(original_heart_chosen)
cv2.ellipse(ellipse_mask, ellipse, (255, 255, 255), -1)
ellipse_mask = cv2.subtract(ellipse_mask,vena_mask[ns])
ellipse_mask = cv2.subtract(ellipse_mask,aorta_mask[ns])


corrected_mask = ellipse_mask.copy()
translation_matrix = np.float32([[1, 0, -1], [0, 1, 0]])
left_shift_corrected = left_cardiac_lateral_shift_correction(Im_filt_H, mask_heart_No_vessels, ns, Bw_filt_H, innermask_H, pcontorno_H, pmax1_H,pmax2_H,corrected_mask)
while left_shift_corrected > 66:
    corrected_mask = cv2.warpAffine(corrected_mask, translation_matrix, (corrected_mask.shape[1], corrected_mask.shape[0]))
    left_shift_corrected = left_cardiac_lateral_shift_correction(Im_filt_H, mask_heart_No_vessels, ns, Bw_filt_H,innermask_H, pcontorno_H, pmax1_H, pmax2_H,corrected_mask)



if ("M" in str(gender)):
    _,Imcorrected,_,_,_ = depression_eval(pmax1_H[ns], pmax2_H[ns], Bw_filt_H[ns], Im_filt_H[ns], y_half_H[ns], pixel_spacing)
else:
    Imcorrected = Im_filt_H[ns].copy()

corr_contours, _ = cv2.findContours(corrected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
wrong_contours, _ = cv2.findContours(mask_heart_No_vessels[ns], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Imcorrected = cv2.cvtColor(MatToUint8(Imcorrected), cv2.COLOR_GRAY2BGR)
cv2.drawContours(Imcorrected, wrong_contours, -1, (0, 255, 0), 2)
cv2.drawContours(Imcorrected, corr_contours, -1, (255, 0, 0), 2)

plt.figure()
plt.imshow(Imcorrected)
plt.show()

#FINE <3