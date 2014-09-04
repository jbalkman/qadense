# Math (can still use '//' if want integer output; this library gives floats)
from __future__ import division

# General
import os
import re
import StringIO
import cStringIO
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file

# Image Processingx
import math
import numpy as np
import cv2, cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mahotas
import dicom
from matplotlib.figure import Figure

# Image Drawing
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image

# Storage
from boto.s3.connection import S3Connection
from boto.s3.key import Key

# Scientific Image Manipulation
from scipy import ndimage
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filter import rank, threshold_otsu
from skimage.util import img_as_ubyte

# Globals
DEBUG = False
MAX_DIST = 512 # arbritrary
BLKSZ = 101

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # change to relative /var/www/ file

ALLOWED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'dcm', 'dicom']
TIFF_EXTENSIONS = ['tif', 'tiff']
DICOM_EXTENSIONS = ['dcm', 'dicom']
PNG_EXTENSIONS = ['png']
JPG_EXTENSIONS = ['jpg', 'jpeg']
DEBUG = False
FONT_PATH = 'static/fonts/'
ACCESS_KEY = ''
SECRET_KEY = ''

@app.route('/')
def hello_world():
   print 'Hello World!'
   return render_template('index.html')

@app.route('/process_serve_mri', methods=['GET'])
def process_serve_img_mri():

   # Init
   data_encode = None
   ifile = None
   imgarr = []

   # Initialize submitted variables
   total_seg = 0
   total_fat = 0
   total_seg_l = 0
   total_fat_l = 0
   total_seg_r = 0
   total_fat_r = 0

   imgfile = request.args.get('imgfile')
   print "Process/Serving Image: "+imgfile
   fnamesplit = imgfile.rsplit('.', 1)
   ext = fnamesplit[1]
   imgprefix = fnamesplit[0]

   prfxsplt = imgprefix.rsplit('-', 2) 
   prefix = prfxsplt[0]
   nfiles = int(prfxsplt[1])
   idx = int(prfxsplt[2])

   conn = S3Connection(ACCESS_KEY, SECRET_KEY)
   bkt = conn.get_bucket('qad_imgs')
   
   k = Key(bkt)
   try:
      for x in range(0,nfiles):
         mykey = prefix+'-'+str(nfiles)+'-'+str(x)
         k.key = mykey

         # NEW: Process file here...
         fout = cStringIO.StringIO()
         k.get_contents_to_file(fout)
         #d, v = processSliceMRI(fout,k)
         print "before process MRI file"
         s, f, sl, fl, sr, fr = processMRIFile(fout,k)
         print "after process MRI file"

         # Calculate Totals
         total_seg = total_seg + s
         total_fat = total_fat + f
         
         total_seg_l = total_seg_l + sl
         total_fat_l = total_fat_l + fl
         
         total_seg_r = total_seg_r + sr
         total_fat_r = total_fat_r + fr

         data = k.get_contents_as_string()
         #k.delete() # putting the delete here causes premature loss of the image; need to find somewhere else to do it probably performed via outside function when called from javascript
         data_encode = data.encode("base64")
         imgarr.append(k.generate_url(3600))
      result = 1
   except:
      result = 0

   bilateral_density = (total_seg - total_fat)*100//total_seg
   left_density = (total_seg_l - total_fat_l)*100//total_seg_l
   right_density = (total_seg_r - total_fat_r)*100//total_seg_r
   volstr = 'L = '+str(left_density)+', R ='+str(right_density)

   return jsonify({"success":result,"imagefile":data_encode,"imgarr":imgarr, "area_d":'N/A', "volumetric_d":volstr, "dcat_a":'N/A', "dcat_v":'N/A', "side":'Bilateral', "view":'Axial'})

@app.route('/process_serve_mammo', methods=['GET']) # remove golden to retain functionality
def process_serve_img_mammo():
   # Init
   data_encode = None
   ifile = None
   imgarr = []

   imgfile = request.args.get('imgfile')
   print "Process/Serving Image: "+imgfile
   fnamesplit = imgfile.rsplit('.', 1)
   ext = fnamesplit[1]
   imgprefix = fnamesplit[0]

   prfxsplt = imgprefix.rsplit('-', 2) 
   prefix = prfxsplt[0]
   nfiles = int(prfxsplt[1])
   idx = int(prfxsplt[2])

   # S3 Get File
   conn = S3Connection(ACCESS_KEY, SECRET_KEY)
   bkt = conn.get_bucket('qad_imgs')
   k = Key(bkt)
   mykey = prefix+'-'+str(nfiles)+'-'+'0' # replace the '0' with str(index) if we deal with more than one file (see MRI); nfiles shoulder be '1'
   k.key = mykey

   # Initialize submitted variables
   a = None
   d = None
   ca  = None
   cv  = None
   s = None
   v = None
   data_encode = None

   try:
      fout = cStringIO.StringIO()
      k.get_contents_to_file(fout)
      a, d, ca, cv, s, v = processMammoFile(fout,k) # returns density, density category, side, and view 
      data = k.get_contents_as_string()
 
      print a, d, ca, cv, s, v
   
      data_encode = data.encode("base64")
      imgarr.append(k.generate_url(3600))
   
      result = 1
   except:
      result = 0

   return jsonify({"success":result, "imagefile":data_encode, "imgarr":imgarr, "area_d":a, "volumetric_d":d, "dcat_a":ca, "dcat_v":cv, "side":s, "view":v})

@app.route('/upload', methods=['POST'])
def upload():

   if request.method == 'POST':
      file = request.files['file']
      if file and allowed_file(file.filename):
         now = datetime.now()

         # Naming and storage to S3 database
         prefix = file.filename.rsplit('.', 1)[0]

         conn = S3Connection(ACCESS_KEY, SECRET_KEY)
         bkt = conn.get_bucket('qad_imgs')
         k = Key(bkt)
         k.key = prefix
         if istiff(file.filename):
            k.set_contents_from_file(file, headers={"Content-Type":"image/tiff"})
         elif isjpg(file.filename):
            k.set_contents_from_file(file, headers={"Content-Type":"image/jpeg"})
         elif ispng(file.filename):
            k.set_contents_from_file(file, headers={"Content-Type":"image/png"})
         elif isdicom(file.filename):
            ds = dicom.read_file(file)
            pil_dcm = get_dicom_PIL(ds)
            pil_dcm_str = cStringIO.StringIO()
            pil_dcm.save(pil_dcm_str, format='tiff')
            pil_dcm_str.seek(0)
            k.set_contents_from_file(pil_dcm_str, headers={"Content-Type":"image/tiff"})
         else:
            k.set_contents_from_file(file) # don't suspect that this will work

         return jsonify({"success":True, "file": file.filename}) # passes to upload.js, function uploadFinished

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def isjpg(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in JPG_EXTENSIONS

def ispng(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in PNG_EXTENSIONS

def istiff(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in TIFF_EXTENSIONS

def isdicom(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in DICOM_EXTENSIONS

def processMRIFile(f, k):
   density = 1
   volume = 1
   
   # Read the file into an array
   array = np.frombuffer(f.getvalue(), dtype='uint16')
   origimg = cv2.imdecode(array, cv2.CV_LOAD_IMAGE_GRAYSCALE)
   
   # Chop off the bottom of the image b/c there is often noncontributory artifact; then make numpy arrays
   img = origimg[:,30:482]
   imarray = np.array(img)

   hist_orig, bins_orig = np.histogram(imarray, bins=255, normed=False, range=(10, 255))
   
   imarraymarkup = imarray
   maskarray = np.zeros_like(imarray)
   contoursarray = np.zeros_like(imarray)
   onesarray = np.ones_like(imarray)
   
   # Option for printing entire arrays
   #np.set_printoptions(threshold='nan')
   
   # THRESHOLDING FOR BREAST SEGMENTATION
   
   #---------------------#
   #      THRESHOLD      #
   #---------------------#
   ret,thresh = cv2.threshold(imarray,10,255,cv2.THRESH_BINARY) # threshold out the background by arbitrary number
   
   #---------------------#
   #    GET CONTOURS     #
   #---------------------#
   contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
   biggest_contour = []
   for n, contour in enumerate(contours):
      if len(contour) > len(biggest_contour):
         biggest_contour = contour
         
   # Fill in the biggest contour
   cv2.fillPoly(maskarray, pts = [biggest_contour], color=cv.RGB(255,255,255))
         
   # Use the mask array to crop the original
   imcrop_orig = cv2.bitwise_and(imarray, maskarray)

   # Adaptive Threshold
   fgt_thresh = cv2.adaptiveThreshold(imcrop_orig,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,BLKSZ, 0)
   #fgt_thresh = cv2.adaptiveThreshold(imcrop_orig,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101, 0)

   # Absolute Val Threshold
   #ret,thresh = cv2.threshold(imarray,100,255,cv2.THRESH_BINARY) # threshold out the background by arbitrary number

   # Calculate R/L sidedness using centroid
   #M = cv2.moments(biggest_contour)
   M = cv2.moments(maskarray)
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   
   # Remove the aberrant portion of the contour below the center of mass
   biggest_contour = biggest_contour[biggest_contour[:,0,1] < cy] # wow indexing is nice for ndarrays

   # Draw the biggest contour on an array map
   cv2.drawContours(contoursarray,biggest_contour,-1,cv.RGB(255,255,255),5)            

   # Plot the center of mass
   cv2.circle(contoursarray,(cx,cy),15,[255,0,255],-1)            
   
   #---------------------#
   #    SEGMENTATION     #
   #---------------------#
   
   # Grab a short segment of the big contour near the centroid
   short_segment = biggest_contour[biggest_contour[:,0,0] < (cx + 20)]
   short_segment = short_segment[short_segment[:,0,0] > (cx - 20)]
   
   dft_cntr = (cx, cy)
   
   pdist = MAX_DIST
   cdist = 0
   
   for i in range(short_segment.shape[0]):
      cdist = abs(cx - short_segment[i,0,0])
      if cdist < pdist: # test for minimum distance from the centroid
         dft_cntr = (short_segment[i,0,0],short_segment[i,0,1])
         pdist = cdist
         
   if dft_cntr:
      cv2.circle(contoursarray,dft_cntr,10,cv.RGB(255,0,255),-1)
            
   # Draw thick black contour to eliminate the skin and nipple from the image
   cv2.drawContours(fgt_thresh,biggest_contour,-1,(0,0,0),5) # 
   cv2.drawContours(maskarray,biggest_contour,-1,(0,0,0),5) #

   #---------------------#
   #      CROPPING       #
   #---------------------#

   # Crop the arrays based on segmentation; this may have to change with orientation; just switch the colons to the opposite sides
   fgt_thresh_crop = fgt_thresh[:dft_cntr[1],:]
   mask_crop = maskarray[:dft_cntr[1],:]
   
   # Isolate the breasts; this will have to be changed with orientation
   fgt_thresh_crop_r = fgt_thresh[:dft_cntr[1],:dft_cntr[0]]
   fgt_thresh_crop_l = fgt_thresh[:dft_cntr[1],dft_cntr[0]:]
   
   mask_crop_r = maskarray[:dft_cntr[1],:dft_cntr[0]]
   mask_crop_l = maskarray[:dft_cntr[1],dft_cntr[0]:]
   
   # Find the proportions of fibroglandular tissue
   segmented = mask_crop > 0
   segmented = segmented.astype(int)
   segmented_sum = segmented.sum()
   
   fat = fgt_thresh_crop > 0
   fat = fat.astype(int)
   fat_sum = fat.sum()
   
   # Perform calculations for each breast
   segmented_l = mask_crop_l > 0
   segmented_l = segmented_l.astype(int)
   segmented_l_sum = segmented_l.sum()
   
   fat_l = fgt_thresh_crop_l > 0
   fat_l = fat_l.astype(int)
   fat_l_sum = fat_l.sum()
   
   segmented_r = mask_crop_r > 0
   segmented_r = segmented_r.astype(int)
   segmented_r_sum = segmented_r.sum()
   
   fat_r = fgt_thresh_crop_r > 0
   fat_r = fat_r.astype(int)
   fat_r_sum = fat_r.sum()
   
   # Total tissue measured minus fat tissue over the total tissue = fibroglandular percent
   tfgt = (segmented_sum - fat_sum)*100//segmented_sum
   lfgt = (segmented_l_sum - fat_l_sum)*100//segmented_l_sum
   rfgt = (segmented_r_sum - fat_r_sum)*100//segmented_r_sum
   print "Percentage Fibrogalandular Tissue: ", tfgt
   print "Percentage Left Fibrogalandular Tissue: ", lfgt
   print "Percentage Right Fibrogalandular Tissue: ", rfgt
   
   # Plot a 4x4
   pil_imarray = Image.fromarray(imarray)
   pil_thresh = Image.fromarray(thresh)
   pil_fgt_thresh_l = Image.fromarray(fgt_thresh_crop_l)
   pil_fgt_thresh_r = Image.fromarray(fgt_thresh_crop_r)
   pil_otsu = Image.fromarray(fgt_thresh)
   pil_segmented_l = Image.fromarray(mask_crop_l)
   pil_segmented_r = Image.fromarray(mask_crop_r)
   pil_markup = Image.fromarray(contoursarray)
   
   # Paste Results
   pil_backdrop = Image.new('RGB', (587,587), "white")
   pil_backdrop.paste(pil_imarray.resize((256,256),Image.ANTIALIAS),(25,25))
   pil_backdrop.paste(pil_markup.resize((256,256),Image.ANTIALIAS),(306,25))
   
   w, h = pil_fgt_thresh_r.size
   W = 256*w//h
   pil_backdrop.paste(pil_fgt_thresh_r.resize((W,256),Image.ANTIALIAS),(25,306))
   
   w, h = pil_fgt_thresh_l.size
   W = 256*w//h
   pil_backdrop.paste(pil_fgt_thresh_l.resize((W,256),Image.ANTIALIAS),(306,306))
   
   # Add Text
   font = ImageFont.truetype(FONT_PATH+"Arial Black.ttf",14)
   draw = ImageDraw.Draw(pil_backdrop)
   draw.text((25,5),"Original Image",0,font=font)
   draw.text((306,5),"Breast Contour",0,font=font)
   draw.text((25,286),"Right Breast FGT: "+str(rfgt)+'%',0,font=font)
   draw.text((306,286),"Left Breast FGT: "+str(lfgt)+'%',0,font=font)
   
   # Save File
   #sfname = fname.rsplit('/',1)[1].split('.',1)[0]
   #pil_backdrop.save('./OUT/p'+sfname+'.jpg')
   
   imgout = cStringIO.StringIO()
   pil_backdrop.save(imgout, format='png')
   k.set_contents_from_string(imgout.getvalue())
   imgout.close()
   
   return segmented_sum, fat_sum, segmented_l_sum, fat_l_sum, segmented_r_sum, fat_r_sum

def processSliceMRI(f,k): # pass the key so we can replace the original image with the processed image for display
   density = 1
   volume = 1

   # Read the file into an array
   array = np.frombuffer(f.getvalue(), dtype='uint16')
   origimg = cv2.imdecode(array, cv2.CV_LOAD_IMAGE_GRAYSCALE)

   # Chop off the bottom of the image b/c there is often noncontributory artifact; then make numpy arrays
   img = origimg[:,30:482]
   imarray = np.array(img)
   print imarray[250:300,350:400]

   hist_orig, bins_orig = np.histogram(imarray, bins=255, normed=False, range=(10, 255))

   imarraymarkup = imarray
   maskarray = np.zeros_like(imarray)
   contoursarray = np.zeros_like(imarray)
   onesarray = np.ones_like(imarray)

   # Minimum Val Threshold
   #ret,thresh = cv2.threshold(imarray,np.amin(imarray),255,cv2.THRESH_BINARY)

   # Otsu Threshold
   '''t = mahotas.otsu(imarray, ignore_zeros = True)
   otsu_bool_out_1 = imarray > 25
   otsu_binary_out_1 = otsu_bool_out_1.astype('uint8')
   thresh = otsu_binary_out_1 * 255'''

   # Absolute Val Threshold
   ret,thresh = cv2.threshold(imarray,10,255,cv2.THRESH_BINARY) # threshold out the background by arbitrary number

   # Adaptive Threshold
   #thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

   # Get the biggest contour found on the image
   contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   biggest_contour = []
   for n, contour in enumerate(contours):
      if len(contour) > len(biggest_contour):
         biggest_contour = contour
         
   # Fill in the biggest contour
   cv2.fillPoly(maskarray, pts = [biggest_contour], color=cv.RGB(255,255,255))

   # Use the mask array to crop the original
   imcrop_orig = cv2.bitwise_and(imarray, maskarray)

   # THRESHOLDING FOR FIBROGLANDULAR SEGMENTATION

   # Adaptive Threshold
   fgt_thresh = cv2.adaptiveThreshold(imcrop_orig,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,BLKSZ, 0)

   # Draw the biggest contour on an array map
   cv2.drawContours(contoursarray,biggest_contour,-1,cv.RGB(255,255,255),5)            

   # Calculate R/L sidedness using centroid
   M = cv2.moments(biggest_contour)
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   
   # Plot the center of mass
   cv2.circle(contoursarray,(cx,cy),10,[255,0,255],-1)            

   # Approximate the breast
   #epsilon = 0.001*cv2.arcLength(biggest_contour,True)
   epsilon = 0.02*cv2.arcLength(biggest_contour,True)
   approx = cv2.approxPolyDP(biggest_contour,epsilon,True)

   # Calculate the hull and convexity defects
   drawhull = cv2.convexHull(approx)
   #cv2.drawContours(contoursarray,drawhull,-1,(0,255,0),60)
   hull = cv2.convexHull(approx, returnPoints = False)
   defects = cv2.convexityDefects(approx,hull)

   # Find the defect closest to the centroid
   dft_sdist_c = MAX_DIST # defect short distance from centroid
   dft_cntr = [] # tuple for center defect
   
   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         dist = math.hypot(cx - far[0], cy - far[1])
         if dist < dft_sdist_c: # test for minimum distance from the centroid
            dft_sdist_c = dist
            dft_cntr = far
            cv2.circle(contoursarray,far,10,cv.RGB(255,0,255),-1)

   if dft_cntr:
      cv2.circle(contoursarray,dft_cntr,20,cv.RGB(255,0,255),-1)

   # Draw thick black contour to eliminate the skin and nipple from the image
   cv2.drawContours(fgt_thresh,biggest_contour,-1,(0,0,0),5) # 
   cv2.drawContours(maskarray,biggest_contour,-1,(0,0,0),5) # 

   # Crop the arrays based on segmentation; this may have to change with orientation; just switch the colons to the opposite sides
   fgt_thresh_crop = fgt_thresh[:dft_cntr[1],:]
   mask_crop = maskarray[:dft_cntr[1],:]
    
   # Isolate the breasts; this will have to be changed with orientation
   fgt_thresh_crop_r = fgt_thresh[:dft_cntr[1],:dft_cntr[0]]
   fgt_thresh_crop_l = fgt_thresh[:dft_cntr[1],dft_cntr[0]:]
    
   mask_crop_r = maskarray[:dft_cntr[1],:dft_cntr[0]]
   mask_crop_l = maskarray[:dft_cntr[1],dft_cntr[0]:]

   # Find the proportions of fibroglandular tissue
   segmented = mask_crop > 0
   segmented = segmented.astype(int)
   segmented_sum = segmented.sum()
   
   fat = fgt_thresh_crop > 0
   fat = fat.astype(int)
   fat_sum = fat.sum()

   # Perform calculations for each breast
   segmented_l = mask_crop_l > 0
   segmented_l = segmented_l.astype(int)
   segmented_l_sum = segmented_l.sum()
   
   fat_l = fgt_thresh_crop_l > 0
   fat_l = fat_l.astype(int)
   fat_l_sum = fat_l.sum()
   
   segmented_r = mask_crop_r > 0
   segmented_r = segmented_r.astype(int)
   segmented_r_sum = segmented_r.sum()
   
   fat_r = fgt_thresh_crop_r > 0
   fat_r = fat_r.astype(int)
   fat_r_sum = fat_r.sum()
   
   # Total tissue measured minus fat tissue over the total tissue = fibroglandular percent
   print "Percentage Fibrogalandular Tissue: ", (segmented_sum - fat_sum)/segmented_sum
   print "Percentage Left Fibrogalandular Tissue: ", (segmented_l_sum - fat_l_sum)/segmented_l_sum
   print "Percentage Right Fibrogalandular Tissue: ", (segmented_r_sum - fat_r_sum)/segmented_r_sum
    
   # Plot a 4x4
   pil_imarray = Image.fromarray(imarray)
   pil_thresh = Image.fromarray(thresh)
   copyarr = fgt_thresh_crop_l.copy() # need to copy array b/c of some obscure ubuntu/PIL issue: http://stackoverflow.com/questions/10854903/what-is-causing-dimension-dependent-attributeerror-in-pil-fromarray-function
   pil_fgt_thresh_l = Image.fromarray(copyarr)
   copyarr = fgt_thresh_crop_r.copy() # see above
   pil_fgt_thresh_r = Image.fromarray(copyarr)
   pil_otsu = Image.fromarray(fgt_thresh)
   copyarr = mask_crop_l.copy() # see above
   pil_segmented_l = Image.fromarray(copyarr)
   copyarr = mask_crop_r.copy() # see above
   pil_segmented_r = Image.fromarray(copyarr)
   pil_markup = Image.fromarray(contoursarray)

   plt.figure(figsize=(18,18))
   plt.subplot(2,2,1),plt.imshow(pil_imarray, 'gray')
   plt.title('Selected Slice from Original\nMRI T1 Axial Breast MRI',fontsize=20)
   plt.subplot(2,2,2),plt.imshow(pil_markup, 'gray')
   plt.title('Breast Contouring',fontsize=20)
   #plt.subplot(2,2,2),plt.imshow(pil_segmented_r, 'gray')
   #plt.title('Right Breast Segmented',fontsize=28)
   #plt.subplot(2,2,2),plt.imshow(pil_segmented_l, 'gray')
   #plt.title('Left Breast Segmented',fontsize=35)
   plt.subplot(2,2,3),plt.imshow(pil_fgt_thresh_r, 'gray')
   plt.title('Right Fibroglandular Segmentation\n13% Calculated Volumetric Density (all slices)',fontsize=20)
   plt.subplot(2,2,4),plt.imshow(pil_fgt_thresh_l, 'gray')
   plt.title('Left Fibroglandular Segmentation\n14% Calculated Volumetric Density (all slices)',fontsize=20)
   plt.tight_layout() # didn't work...

   imgout = cStringIO.StringIO()
   plt.savefig(imgout, format='png') # can't save as a JPG; use PNG and PIL to convert to JPG if necessary
   k.set_contents_from_string(imgout.getvalue())
   imgout.close()
   plt.close('all')   

   return density, volume

def processMammoFile(f, k):

   # New method to read ead the file into an array
   array = np.frombuffer(f.getvalue(), dtype='uint8') # uint16 works for some reason; 
   origimg = cv2.imdecode(array, cv2.CV_LOAD_IMAGE_GRAYSCALE)
      
   # Chop off the top of the image b/c there is often noncontributory artifact & make numpy arrays
   img = origimg[25:,:]
   imarray = np.array(img)
   
   imarraymarkup = imarray
   maskarray = np.zeros_like(imarray)
   contoursarray = np.zeros_like(imarray)
   onesarray = np.ones_like(imarray)
   
    # Store dimensions for subsequent calculcations
   max_imheight = maskarray.shape[0]
   max_imwidth = maskarray.shape[1]
   
   if DEBUG: print max_imwidth, max_imheight
    
   # Choose the minimum in the entire array as the threshold value b/c some mammograms have > 0 background which screws up the contour finding if based on zero or some arbitrary number
   ret,thresh = cv2.threshold(imarray,np.amin(imarray),255,cv2.THRESH_BINARY)
   contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   biggest_contour = []
   for n, contour in enumerate(contours):
      if len(contour) > len(biggest_contour):
         biggest_contour = contour

    # Get the lower most extent of the contour (biggest y-value)
   max_vals = np.argmax(biggest_contour, axis = 0)
   min_vals = np.argmin(biggest_contour, axis = 0)

   bc_max_y = biggest_contour[max_vals[0,1],0,1] # get the biggest contour max y
   bc_min_y = biggest_contour[min_vals[0,1],0,1] # get the biggest contour min y
   
   cv2.drawContours(contoursarray,biggest_contour,-1,(255,255,0),15)            

   # Calculate R/L sidedness using centroid
   M = cv2.moments(biggest_contour)
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   right_side = cx > max_imwidth/2
    
    # Plot the center of mass
   cv2.circle(contoursarray,(cx,cy),100,[255,0,255],-1)            

    # Approximate the breast
   epsilon = 0.001*cv2.arcLength(biggest_contour,True)
   approx = cv2.approxPolyDP(biggest_contour,epsilon,True)
            
    # Calculate the hull and convexity defects
   drawhull = cv2.convexHull(approx)
   #cv2.drawContours(contoursarray,drawhull,-1,(0,255,0),60)
   hull = cv2.convexHull(approx, returnPoints = False)
   defects = cv2.convexityDefects(approx,hull)
   
    # Plot the defects and find the most superior. Note: I think the superior and inferior ones have to be kept separate
    # Also make sure that these are one beyond a reasonable distance from the centroid (arbitrarily cdist_factor = 80%) to make sure that nipple-related defects don't interfere
   supdef_y = maskarray.shape[0]
   supdef_tuple = []
   
   cdist_factor = 0.80

   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if far[1] < (cy*cdist_factor) and far[1] < supdef_y:
            supdef_y = far[1]
            supdef_tuple = far
            cv2.circle(contoursarray,far,50,[255,0,255],-1)

   # Find lower defect if there is one
   # Considering adding if a lower one is at least greater than 1/2 the distance between the centroid and the lower most border of the contour (see IMGS_MLO/IM4010.tif)
   infdef_y = 0
   infdef_tuple = []
   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if far[1] > infdef_y and supdef_tuple: # cy + 3/4*(bc_max_y - cy) = (bc_max_y + cy)/2
            if (right_side and far[0] > supdef_tuple[0]) or (not right_side and far[0] < supdef_tuple[0]):
               infdef_y = far[1]
               infdef_tuple = far
               cv2.circle(contoursarray,far,50,[255,0,255],-1)

    # Try cropping contour beyond certain index; get indices of supdef/infdef tuples, and truncate vector beyond those indices
   cropped_contour = biggest_contour[:,:,:]
               
   if supdef_tuple:
      sup_idx = [i for i, v in enumerate(biggest_contour[:,0,:]) if v[0] == supdef_tuple[0] and v[1] == supdef_tuple[1]]
      if sup_idx:
         if right_side:
            cropped_contour = cropped_contour[sup_idx[0]:,:,:]
         else:
            cropped_contour = cropped_contour[:sup_idx[0],:,:]
            
   if infdef_tuple:
      inf_idx = [i for i, v in enumerate(cropped_contour[:,0,:]) if v[0] == infdef_tuple[0] and v[1] == infdef_tuple[1]]
      if inf_idx:
         if right_side:
            cropped_contour = cropped_contour[:inf_idx[0],:,:]
         else:
            cropped_contour = cropped_contour[inf_idx[0]:,:,:]
         
   if right_side:
      cropped_contour = cropped_contour[cropped_contour[:,0,1] != 1]
   else:
      cropped_contour = cropped_contour[cropped_contour[:,0,0] != 1]

    # Fill in the cropped polygon to mask
   cv2.fillPoly(maskarray, pts = [cropped_contour], color=(255,255,255))

   # Multiply original image to the mask to get the cropped image
   imarray2 = imarray + onesarray
   imcrop_orig = cv2.bitwise_and(imarray2, maskarray)
   
   # Draw thick black contour to eliminate the skin and nipple from the image
   cv2.drawContours(imcrop_orig,cropped_contour,-1,(0,0,0),175) # 
   cv2.drawContours(maskarray,cropped_contour,-1,(0,0,0),175) # 

    # Apply Otsu thresholding to generate a new matrix and convert to int type
   thresh = mahotas.otsu(imcrop_orig, ignore_zeros = True)
   otsu_bool_out = imcrop_orig > thresh
   otsu_binary_out = otsu_bool_out.astype('uint8')
   otsu_binary_sum = otsu_binary_out.sum()
   otsu_int_out = otsu_binary_out * 255

   # Crop out the fibroglandular tissue
   imcrop_fgt = cv2.bitwise_and(imarray2, otsu_int_out) # both arrays are uint8 type
   pix_min = np.min(imcrop_fgt[np.nonzero(imcrop_fgt)])
   pix_inc = (255 - pix_min)/4
   hist, bins = np.histogram(imcrop_fgt, bins=8, normed=False, range=(pix_min,255))

   # Volumetric sum attained by scaling the histogram curve
   fgt_scaled_sum = hist[0]*0.50 + hist[1]*0.50 + hist[2]*0.75 + hist[3]*0.75 + hist[4] + hist[5] + hist[6] + hist[7]

   # Find area of the segmented breast
   segmented = maskarray > 0
   segmented = segmented.astype(int)
   segmented_sum = segmented.sum()

   # Calculate area & volumetric based densities
   area_d = (otsu_binary_sum*100/segmented_sum).astype(int)  
   volumetric_d = (fgt_scaled_sum*100/segmented_sum).astype(int)

   # Determine Area-based BI-RADS Category
   if area_d < 25:
      dcat_a = 'Fatty'
   elif area_d < 50:
      dcat_a = 'Scattered'
   elif area_d < 75:
      dcat_a = 'Heterogenous'
   else:
      dcat_a = 'Extremely Dense'

   # Determine Volumetric BI-RADS Category
   if volumetric_d < 25:
      dcat_v = 'Fatty'
   elif volumetric_d < 50:
      dcat_v = 'Scattered'
   elif volumetric_d < 75:
      dcat_v = 'Heterogenous'
   else:
      dcat_v = 'Extremely Dense'

   # Determine Sidedness
   if right_side:
      side = 'Right'
   else:
      side = 'Left'

   # Determine View
   if bc_min_y > 1:
      view = 'CC'
   else:
      view = 'MLO'
      
   # Print results
   print side, view, otsu_binary_sum, fgt_scaled_sum, segmented_sum, area_d, volumetric_d, dcat_a, dcat_v
    
   # Create Histogram Plot
   width = 0.8 * (bins[1] - bins[0])
   center = (bins[:-1] + bins[1:]) / 2

   plt.subplot(1,2,1),plt.bar(center, hist, align='center', width=width)
   plt.title('Area-based Histogram')
   plt.xlabel("Pixel Value")
   plt.ylabel("Number of Pixels")

   # Rescale to plot volumetrics
   hist[0] = hist[0]*0.50
   hist[1] = hist[1]*0.50
   hist[2] = hist[2]*0.75
   hist[3] = hist[3]*0.75
   plt.subplot(1,2,2),plt.bar(center, hist, align='center', width=width)
   plt.title('Volumetric-based Histogram')
   plt.xlabel("Pixel Value")

   # Save plot as Image and close
   plt.tight_layout()
   #plt.savefig(fname_hist) # old method of saving to disk
   histimg = cStringIO.StringIO()
   plt.savefig(histimg, format='png') # can't save as a JPG; use PNG and PIL to convert to JPG if necessary
   histimg.seek(0) # apparently this is necessary :)
   plt.close('all')

   # Create PIL images
   pil1 = Image.fromarray(imarray2)
   pil2 = Image.fromarray(contoursarray)
   pil3 = Image.fromarray(maskarray)
   pil4 = Image.fromarray(imcrop_fgt)
   #pil5 = Image.open(fname_hist) # old method
   pil5 = Image.open(histimg)

   # Pasting images above to a pil background along with text. There's a lot of particular measurements sizing the fonts & pictures so that everything fits.  It's somewhat arbitrary with lots of trial and error, but basically everything is based off the resized width of the first image.  Images needed to be resized down b/c they were too high resolution for canvas.
   rf = 3 # rf = resize factor

   w1,h1 = pil1.size
   pil1_sm = pil1.resize((w1//rf,h1//rf))
   w1_sm,h1_sm = pil1_sm.size
   print "Resize", int(h1_sm*0.9)

   pil2_sm = pil2.resize((w1_sm,h1_sm))
   pil3_sm = pil3.resize((w1_sm,h1_sm))
   pil4_sm = pil4.resize((w1_sm,h1_sm))
   pil5_sm = pil5.resize((w1_sm*2,int(h1_sm*0.8)))

   pil_backdrop = Image.new('RGB', (100+2*w1_sm,3*h1_sm+3*h1_sm//8), "white")

   pil_backdrop.paste(pil1_sm, (0,h1_sm//8))
   pil_backdrop.paste(pil2_sm, (100+w1_sm,h1_sm//8))
   pil_backdrop.paste(pil3_sm, (0,h1_sm//4+h1_sm))
   pil_backdrop.paste(pil4_sm, (100+w1_sm,h1_sm//4+h1_sm))
   pil_backdrop.paste(pil5_sm, (0,3*h1_sm//8+2*h1_sm))

   font = ImageFont.truetype(FONT_PATH+"Arial Black.ttf",w1_sm//20)

   draw = ImageDraw.Draw(pil_backdrop)
   draw.text((0,0),"Original Image",0,font=font)
   draw.text((100+w1_sm,0),"Fibroglandular Tissue",0,font=font)
   draw.text((0,h1_sm+h1_sm//6),"Breast Contouring",0,font=font)
   draw.text((100+w1_sm,h1_sm+h1_sm//6),"Breast Segmentation",0,font=font)

   imgout = cStringIO.StringIO()
   pil_backdrop.save(imgout, format='png')
   k.set_contents_from_string(imgout.getvalue())
   imgout.close()
   histimg.close()

   print "returning from process file..."
   print area_d, volumetric_d, dcat_a, dcat_v, side, view
   return area_d, volumetric_d, dcat_a, dcat_v, side, view

def get_LUT_value(data, window, level):
    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])

# Display an image using the Python Imaging Library (PIL)
def get_dicom_PIL(dataset):
    if ('PixelData' not in dataset):
        raise TypeError("Cannot show image -- DICOM dataset does not have pixel data")
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):  # can only apply LUT if these values exist
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            mode = "I;16"  # not sure about this -- PIL source says is 'experimental' and no documentation. Also, should bytes swap depending on endian of file and system??
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated and %d SamplesPerPixel" % (bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)

        im = Image.frombuffer(mode, size, dataset.PixelData, "raw", mode, 0, 1)  # Recommended from the original code to specify all details by http://www.pythonware.com/library/pil/handbook/image.html; this is experimental...
    else:
        image = get_LUT_value(dataset.pixel_array, dataset.WindowWidth, dataset.WindowCenter)
        im = Image.fromarray(np.uint8(image)).convert('L')  # Convert mode to L since LUT has only 256 values: http://www.pythonware.com/library/pil/handbook/image.htm

    return im

if __name__ == '__main__':
    app.run(debug=True)
