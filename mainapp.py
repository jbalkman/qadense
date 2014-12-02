# Math (can still use '//' if want integer output; this library gives floats)
from __future__ import division

# General
import os
import re
import StringIO
import cStringIO
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file

# Image Processing
import math
import numpy as np
import numpy.ma as ma
from numpy.linalg import inv
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
from scipy import signal
from scipy.signal import argrelextrema
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filter import rank, threshold_otsu, canny
from skimage.util import img_as_ubyte
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

# Globals
DEBUG = False
MAX_DIST = 512 # arbritrary
SKIN = 45

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
   result = 0
   result_str = 'No result'
   data_encode = None
   imgarr = []
   bpe = False

   imgfile = request.args.get('imgfile')
   if request.args.get('bpe') == '1':
      bpe = True
   print "Process/Serving Image: "+imgfile+" BPE: "+str(bpe)
   fnamesplit = imgfile.rsplit('.', 1)
   imgprefix = fnamesplit[0]

   prfxsplt = imgprefix.rsplit('-', 2) 
   prefix = prfxsplt[0]
   nfiles = int(prfxsplt[1])

   conn = S3Connection(ACCESS_KEY, SECRET_KEY)
   bkt = conn.get_bucket('qad_imgs')   
   key = Key(bkt)

   result, result_str, data_encode, imgarr = processMRI(key, prefix, nfiles, bpe)

   return jsonify({"success":result,"imagefile":data_encode,"imgarr":imgarr, "area_d":'N/A', "volumetric_d":result_str, "dcat_a":'N/A', "dcat_v":'N/A', "side":'Bilateral', "view":'Axial'})

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
      print 'contouring mammo file...'
      img, msk, cnt, right, view, ocnt = contourMammoFile(fout) # ocnt = original contour not cropped
      print 'processing mammo file...'
      a, d, ca, cv, s, v = processMammoFile(k, img, msk, cnt, right, view, ocnt) # returns density, density category, side, and view 
      print 'done processing...'
      data = k.get_contents_as_string()
 
      print a, d, ca, cv, s, v
   
      data_encode = data.encode("base64")
      imgarr.append(k.generate_url(3600))
   
      result = 1
   except:
      result = 0

   return jsonify({"success":result, "imagefile":data_encode, "imgarr":imgarr, "area_d":a, "volumetric_d":d, "dcat_a":ca, "dcat_v":cv, "side":s, "view":v})

'''def process_serve_img_mammo_bkup():
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

   return jsonify({"success":result, "imagefile":data_encode, "imgarr":imgarr, "area_d":a, "volumetric_d":d, "dcat_a":ca, "dcat_v":cv, "side":s, "view":v})'''


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

#------------------------------#
#     PROCESSING FUNCTIONS     #
#------------------------------#

def contourMammoFile(f):

   print 'starting contour function...'
   # Use this method for actual implementation
   origimg = np.frombuffer(f.getvalue(), dtype='uint8') # or use uint16?
   img = cv2.imdecode(origimg, cv2.CV_LOAD_IMAGE_GRAYSCALE)      
   imarray = np.array(img)

   # Chop off the top of the image b/c there is often noncontributory artifact & make numpy arrays
   print 'chop original matrix...'
   imarray = imarray[25:,:]
   
   # Prepare processing matrices
   print 'make necessary matrices...'
   imarraymarkup = imarray
   maskarray = np.zeros_like(imarray)
   skinarray = np.zeros_like(imarray)
   contoursarray2D = np.zeros_like(imarray)
   contoursarray3D = np.zeros((imarray.shape[0],imarray.shape[1],3), 'uint8')
   onesarray = np.ones_like(imarray)
   
    # Store dimensions for subsequent calculcations
   max_imheight = maskarray.shape[0]
   max_imwidth = maskarray.shape[1]
   
   # Choose the minimum in the entire array as the threshold value b/c some mammograms have > 0 background which screws up the contour finding if based on zero or some arbitrary number
   ret,thresh = cv2.threshold(imarray,np.amin(imarray),255,cv2.THRESH_BINARY)
   
   #edges = cv2.Canny(imarray,0,50)
   print 'applying blur...'
   blur = cv2.GaussianBlur(thresh,(25,25),0)
   blur = blur.astype('int')
   np.clip(blur*100, 0, 255, out=blur)
   blur = blur.astype('uint8')
   #contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   contours, hierarchy = cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   biggest_contour = []
   for n, contour in enumerate(contours):
      if len(contour) > len(biggest_contour):
         biggest_contour = contour

    # Get the lower most extent of the contour (biggest y-value)
   max_vals = np.argmax(biggest_contour, axis = 0)
   min_vals = np.argmin(biggest_contour, axis = 0)

   bc_max_y = biggest_contour[max_vals[0,1],0,1] # get the biggest contour max y
   bc_min_y = biggest_contour[min_vals[0,1],0,1] # get the biggest contour min y
   print 'Cnt Max/Min: ', bc_max_y, bc_min_y
   
   cv2.drawContours(contoursarray2D,biggest_contour,-1,(255,255,0),15)            
   cv2.drawContours(contoursarray3D,biggest_contour,-1,(255,255,0),90)            
   #cv2.drawContours(blur,biggest_contour,-1,(255,255,0),15)               

   # Calculate R/L sidedness using centroid
   M = cv2.moments(biggest_contour)
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   right_side = cx > max_imwidth/2
    
    # Plot the center of mass
   cv2.circle(contoursarray2D,(cx,cy),100,[255,0,255],-1)            
   cv2.circle(contoursarray3D,(cx,cy),100,[0,255,255],-1)            

    # Approximate the breast
   epsilon = 0.001*cv2.arcLength(biggest_contour,True)
   #epsilon = 0.007*cv2.arcLength(biggest_contour,True)
   approx = cv2.approxPolyDP(biggest_contour,epsilon,True)
            
    # Calculate the hull and convexity defects
   drawhull = cv2.convexHull(approx)
   #cv2.drawContours(contoursarray2D,drawhull,-1,(0,255,0),60)
   hull = cv2.convexHull(approx, returnPoints = False)
   defects = cv2.convexityDefects(approx,hull)
   
    # Plot the defects and find the most superior. Note: I think the superior and inferior ones have to be kept separate
    # Also make sure that these are one beyond a reasonable distance from the centroid (arbitrarily cdist_factor = 80%) to make sure that nipple-related defects don't interfere
   supdef_y = maskarray.shape[0]
   supdef_tuple = []
   
   cdist_factor = 0.65

   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if far[1] < (cy*cdist_factor) and far[1] < supdef_y:
            supdef_y = far[1]
            supdef_tuple = far
            cv2.circle(contoursarray2D,far,50,[255,0,255],-1)
            cv2.circle(contoursarray3D,far,75,[255,0,255],-1)

   # Find lower defect if there is one
   # Considering adding if a lower one is at least greater than 1/2 the distance between the centroid and the lower most border of the contour (see IMGS_MLO/IM4010.tif)
   infdef_y = 0
   infdef_tuple = []
   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if (far[1] > infdef_y) and (far[1] > 2*(bc_max_y - cy)/3+cy) and supdef_tuple: # cy + 3/4*(bc_max_y - cy) = (bc_max_y + cy)/2
            if (right_side and far[0] > supdef_tuple[0]) or (not right_side and far[0] < supdef_tuple[0]):
               infdef_y = far[1]
               infdef_tuple = far
               cv2.circle(contoursarray2D,far,50,[255,0,255],-1)
               cv2.circle(contoursarray3D,far,75,[255,0,255],-1)

    # Try cropping contour beyond certain index; get indices of supdef/infdef tuples, and truncate vector beyond those indices
   cropped_contour = biggest_contour[:,:,:]
               
   # Find the index of the supdef_tuple within the contour so that it can be cropped appropriately
   if supdef_tuple:
      sup_idx = [i for i, v in enumerate(biggest_contour[:,0,:]) if v[0] == supdef_tuple[0] and v[1] == supdef_tuple[1]]
      if sup_idx:
         if right_side:
            cropped_contour = cropped_contour[sup_idx[0]:,:,:]
         else:
            cropped_contour = cropped_contour[:sup_idx[0],:,:]

   # Find the index of the infdef_tuple within the contour so that it can be cropped appropriately
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
   cv2.drawContours(imcrop_orig,cropped_contour,-1,(0,0,0),255) 
   cv2.drawContours(maskarray,cropped_contour,-1,(0,0,0),255) 

   #piltest = Image.fromarray(contoursarray2D)
   #piltest.save(myfile.rsplit('/',1)[0]+'/test.jpg')
   
   return imcrop_orig, maskarray, cropped_contour, right_side, bc_min_y, contoursarray3D

def processMammoFile(k,i,m,c,s,v,o): # k = key; i = img; m = mask; c = contour; s = side (right); v = view; ocnt = original contours (not cropped)

   #-----------------------------#
   #        MAKE CONTOUR         #
   #-----------------------------#
   contoursarray2D = np.zeros_like(i)
   cv2.drawContours(contoursarray2D,c,-1,(255,255,255),25)            

   #-----------------------------#
   #           ROTATE            #
   #-----------------------------#
   print "Running rotation..."
   x1, y1 = c[0][0]
   x2, y2 = c[-1][0]
   xdel = x1 - x2
   ydel = y1 - y2
   
   # Prevent a corner case of unknown cause from happening where the first and last contour coordinate have the same y-val 
   # See: /IMGS_QA/IM-0002-1084.tif
   if abs(ydel) < 100:
       x2, y2 = c[-2][0]
       xdel = x1 - x2
       ydel = y1 - y2

   print "X-delta: ", xdel
   print "Y-delta: ", ydel
   rotcoef = 1
   if xdel < 0:
       rotcoef = -1    
   elif xdel > 0:
       rotcoef = 1
   elif xdel == 0:
       rotcoef = 0

   rotdeg = math.degrees(math.atan(xdel/ydel))
   imcrop_orig = ndimage.rotate(i,rotdeg*rotcoef,mode='constant',cval=0)
   maskarray = ndimage.rotate(m,rotdeg*rotcoef,mode='constant',cval=0)

   #-----------------------------#
   #            CROP             #
   #-----------------------------#
   print "Running crop..."
   nza = np.nonzero(maskarray)
   nza_sx, nza_sy = np.sort(nza, 1)
   print nza_sx, nza_sy
   xmin = nza_sx[0]
   xmax = nza_sx[-1]
   ymin = nza_sy[0]
   ymax = nza_sy[-1]
   print xmin, xmax, ymin, ymax

   imcrop_orig = imcrop_orig[xmin:xmax, ymin:ymax]
   maskarray = maskarray[xmin:xmax, ymin:ymax]
   #print maskarray

   #------------------------#
   #        DENOISING       #
   #------------------------#
   #pretv = denoise_tv_chambolle(imcrop_orig, weight=0.1, multichannel=False)
   #pretv = (pretv*255).astype('uint8')

   #-----------------------------#
   #            LPF              #
   #-----------------------------#
   print "Starting LPF..."
   sub1, lpf1 = lpf(imcrop_orig, 10, 0, 1)
   sub2, lpf2 = lpf(imcrop_orig, 500, 0, 0.6)
   #sub2, lpf2 = lpf(imcrop_orig, 550, 0, 0.6)   
   print 'finished LPF filters...'

   sub = np.zeros_like(imcrop_orig)
   #np.clip(lpf1-lpf2*0.9, 0, 255, out=sub)
   np.clip(lpf1-lpf2, 0, 255, out=sub)

   #lpfsub_int, lpfsub_sum = bintoint(sub, 0)
   onesarray = np.ones_like(sub)
   subones = sub + onesarray
   subcrop = cv2.bitwise_and(subones.astype('uint8'), maskarray) # Crop out the fibroglandular tissue; both arrays are uint8 type
   sub_mask, sub_sum, sub_mask_inv, sub_sum_inv = bintoint(sub, 0)

   #a = b.astype('int')
   #np.clip(a+b, 0, 255, out=subtracted)
   
   #-----------------------------#
   #             OTSU            #
   #-----------------------------#
   print "Starting Otsu Thresholding..."
   #otsu_binary_sum, otsu_int_out = motsu(imcrop_orig)
   #otsu_fgt = cv2.bitwise_and(imcrop_orig, otsu_int_out) # Crop out the fibroglandular tissue; both arrays are uint8 type
   otsu_binary_sum, otsu_mask, otsu_binary_sum_inv, otsu_mask_inv = motsu(subcrop, True)
   imcrop_fgt = cv2.bitwise_and(imcrop_orig, sub_mask) # Crop out the fibroglandular tissue; both arrays are uint8 type
   #print "Binary Sum: ", otsu_binary_sum

   #-----------------------------#
   #         STATISTICS          #
   #-----------------------------#
   #print "Starting statistics..."   

   # Create histogram
   pix_min = np.min(imcrop_fgt[np.nonzero(imcrop_fgt)])
   pix_inc = (255 - pix_min)/4
   hist, bins = np.histogram(imcrop_fgt, bins=8, normed=False, range=(pix_min,255))

   # Volumetric sum attained by scaling the histogram curve
   #fgt_scaled_sum = hist[0] + hist[1] + hist[2] + hist[3]
   #fgt_scaled_sum2 = hist[0] + hist[1] + hist[2] + hist[3]
   fgt_scaled_sum = hist[1]*0.5 + hist[2]*0.5 + hist[3] + hist[4] + hist[5] + hist[6] + hist[7]
   fgt_scaled_sum_orig = hist[0] + hist[1] + hist[2] + hist[3] + hist[4] + hist[5] + hist[6] + hist[7]
   print fgt_scaled_sum_orig,fgt_scaled_sum,hist[0],hist[1],hist[2],hist[3],hist[4],hist[5],hist[6],hist[7]

   # Find area of the segmented breast
   segmented = maskarray > 0
   segmented = segmented.astype(int)
   segmented_sum = segmented.sum()
   print "Binary Sum: ", otsu_binary_sum
   print "Segmented Sum: ", segmented_sum

   #fgt_sum = (sub > 0).astype('uint8').sum() # change this to reflect the post-otus sum of the imcrop_orig image
   fgt_sum = sub_sum
   area_sub_d = (fgt_sum*100//segmented_sum)
   area_otsu_d = (otsu_binary_sum*100//segmented_sum)
   vol_sub_d = (fgt_scaled_sum*100//segmented_sum)

   # Determine Area-based BI-RADS Category
   if area_sub_d < 25:
      dcat_sub_a = 'Fatty'
   elif area_sub_d < 50:
      dcat_sub_a = 'Scattered'
   elif area_sub_d < 75:
      dcat_sub_a = 'Heterogenous'
   else:
      dcat_sub_a = 'Extremely Dense'

   # Determine Volumetric BI-RADS Category
   if vol_sub_d < 25:
      dcat_v = 'Fatty'
   elif vol_sub_d < 50:
      dcat_v = 'Scattered'
   elif vol_sub_d < 75:
      dcat_v = 'Heterogenous'
   else:
      dcat_v = 'Extremely Dense'
   
   # Determine Sidedness
   if s:
      side = 'Right'
   else:
      side = 'Left'

   # Determine View
   if v > 1:
      view = 'CC'
   else:
      view = 'MLO'

   print 'SIDE,'+side+',VIEW,'+view+',OTSU,'+str(area_otsu_d)+',SUB,'+str(area_sub_d)+',SUB-VOL,'+str(vol_sub_d)

   #-------------------#
   #    PIL RESULTS    #
   #-------------------#
   
   # Create Histogram Plot
   '''width = 0.8 * (bins[1] - bins[0])
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
   histimg = cStringIO.StringIO()
   plt.savefig(histimg, format='png') # can't save as a JPG; use PNG and PIL to convert to JPG if necessary
   histimg.seek(0) # apparently this is necessary :)
   plt.close('all')'''

   # Create PIL images
   print 'make pil1...'
   pil1 = makepil(i) 
   print 'make pil2...'
   pil2 = makepil(o) 
   print 'make pil3...'
   pil3 = makepil(sub_mask) 
   print 'make pil4...'
   pil4 = makepil(imcrop_fgt) 

   
   # Pasting images above to a pil background along with text. There's a lot of particular measurements sizing the fonts & pictures so that everything fits.  It's somewhat arbitrary with lots of trial and error, but basically everything is based off the resized width of the first image.  Images needed to be resized down b/c they were too high resolution for canvas.
   rf = 3 # rf = resize factor

   w1,h1 = pil1.size
   w2,h2 = pil2.size
   w3,h3 = pil3.size
   w4,h4 = pil4.size
   print w1, h1, w2, h2, w3, h3, w4, h4

   pil1_sm = pil1.resize((w1//rf,h1//rf))
   pil2_sm = pil2.resize((w2//rf,h2//rf))
   pil3_sm = pil3.resize((w3//rf,h3//rf))
   pil4_sm = pil4.resize((w4//rf,h4//rf))

   w1_sm,h1_sm = pil1_sm.size
   w2_sm,h2_sm = pil2_sm.size
   w3_sm,h3_sm = pil3_sm.size
   w4_sm,h4_sm = pil4_sm.size
   print w1_sm, h1_sm, w2_sm, h2_sm, w3_sm, h3_sm, w4_sm, h4_sm

   BUFF = 80
   pil_backdrop = Image.new('RGB', (w1_sm+w2_sm+BUFF*3,h1_sm+h3_sm+BUFF*3), "white")
   pil_backdrop.paste(pil1_sm, (BUFF,BUFF))
   pil_backdrop.paste(pil2_sm, (w1_sm+BUFF*2,BUFF))
   pil_backdrop.paste(pil3_sm, (BUFF,h1_sm+BUFF*2))
   pil_backdrop.paste(pil4_sm, (w1_sm+BUFF*2,h1_sm+BUFF*2))
   #pil_backdrop.paste(pil5_sm, (0,3*h1_sm//8+2*h1_sm))

   font = ImageFont.truetype(FONT_PATH+"Arial Black.ttf",w1_sm//20)
   draw = ImageDraw.Draw(pil_backdrop)
   draw.text((BUFF,BUFF//8),"Breast Segmentation",0,font=font)
   draw.text((w1_sm+BUFF*2,BUFF//8),"Original Contour",0,font=font)
   draw.text((BUFF,h1_sm+BUFF+BUFF//8),"FGT Segmentation Area = "+str(area_sub_d)+"%",0,font=font)
   draw.text((w1_sm+BUFF*2,h1_sm+BUFF+BUFF//8),"FGT Volumetric Est. = "+str(vol_sub_d)+"%",0,font=font)
   draw.text((BUFF*2,h1_sm+h3_sm+BUFF*2+BUFF//8),"NOTE: Bottom post-processed (FGT) images are cropped and rotated.",0,font=font)

   # Save File
   print 'saving file...'
   imgout = cStringIO.StringIO()
   pil_backdrop.save(imgout, format='png')
   k.set_contents_from_string(imgout.getvalue())
   imgout.close()
   #histimg.close()

   #print sub_area_d, sub_volumetric_d, sub_dcat_a, sub_dcat_v, side, view

   return area_sub_d, vol_sub_d, dcat_sub_a, dcat_v, side, view

def processMRI(k,p,n,b): # k = key; p = prefix; n = number of files; b = report BPE (True) or breast density (False)

   # Initialize
   tvolr = 0
   tvoll = 0   
   tfgtr = 0
   tfgtl = 0
   tbper = 0
   tbpel = 0

   r = 0
   rstr = 'No result available'
   de = None
   precon = None
   postcon = None
   arr = []
   itr = n

   if b:
      if n%2 > 0:
         print 'The BPE stack must contain an even number of files containing an equal number of pre and post-contrast images.'
      itr = n//2

   try:
      for x in range(0,itr):
         mykey = p+'-'+str(n)+'-'+str(x)
         k.key = mykey

         precon = cStringIO.StringIO() # rf = pre-contrast file
         k.get_contents_to_file(precon)
         
         # If BPE, get the post-contrast file in the stack
         if b:
            mykey = p+'-'+str(n)+'-'+str(x+itr)
            k.key = mykey

            postcon = cStringIO.StringIO() # rf = pre-contrast file
            k.get_contents_to_file(postcon)            

         print "before process MRI file"
         svolr, svoll, sfgtr, sfgtl, sbper, sbpel = processSlice(k, rf=precon, of=postcon)
         print "after process MRI file"

         # Calculate Totals
         tvolr += svolr 
         tvoll += svoll
         
         tfgtr += sfgtr
         tfgtl += sfgtl
         
         if b:
            tbper += sbper
            tbpel += sbpel
         
         data = k.get_contents_as_string()
         de = data.encode("base64")
         arr.append(k.generate_url(3600))
      r = 1
   except:
      r = 0

   if b: # prepare BPE string
      bilateral_bpe = str((tbper+tbpel)*100//(tfgtr+tfgtl))
      left_bpe = str(tbpel*100//tfgtl)
      right_bpe = str(tbper*100//tfgtr)
      rstr = 'BPE R = '+right_bpe+', BPE L = '+left_bpe
   else: # prepare BD string
      bilateral_density = str((tfgtr+tfgtl)*100//(tvolr+tvoll))
      left_density = str(tfgtl*100//tvoll)
      right_density = str(tfgtr*100//tvolr)
      rstr = 'R = '+right_density+', L = '+left_density

   return r, rstr, de, arr

def processSlice(k, rf=None, of=None): # k = key; rf = pre-contrast image; of = post-contrast image
   print 'starting processSlice...'
   dobpe = of is not None                 

   # Read the file into an array
   print 'reading files...'
   rfarray = np.frombuffer(rf.getvalue(), dtype='uint8') # or use uint16?
   rimg = cv2.imdecode(rfarray, cv2.CV_LOAD_IMAGE_GRAYSCALE)
   rimarray = np.array(rimg)
                         
   if dobpe:
      ofarray = np.frombuffer(of.getvalue(), dtype='uint8') # or use uint16?
      oimg = cv2.imdecode(ofarray, cv2.CV_LOAD_IMAGE_GRAYSCALE)
      oimarray = np.array(oimg)        

      # Not sure if this needs to be implemented; if so, then we need to overwrite the file in S3
      '''if rimarray.shape[0] != oimarray.shape[0]:
        size = rimarray.shape[0], rimarray.shape[1]
        try:
            im = Image.open(ofname)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(ofname, "TIFF")
            oimg = cv2.imread(ofname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            oimarray = np.array(oimg)
        except IOError:
            print >> log, "cannot create thumbnail for '%s'" % ofname'''

   # Make additional arrays
   print 'making additional arrays...'
   maskarray = np.zeros_like(rimarray)
   pretv = np.zeros_like(rimarray)
   imedges = np.zeros_like(rimarray)
   contoursarray = np.zeros((rimarray.shape[0],rimarray.shape[1],3), 'uint8')
   onesarray = np.ones_like(rimarray)   

   # Option for printing entire arrays
   #np.set_printoptions(threshold='nan')
   
   #-----------------------------#
   #        EDGE DETECTION       #
   #-----------------------------#
   print 'edge detection...'
   ebs, edgeblur = lpf(rimarray,30,0,1)
   print 'after lpf...'
   height = rimarray.shape[0]
   ebs[height-145:height-102,10:-10] = 255
   ebs[height-101:,:] = 0
   print 'run canny...'
   edges = cv2.Canny(ebs*2,10,150)
   print 'run gaussian blur...'
   blur = cv2.GaussianBlur(edges,(23,23),0)
   
   #-----------------------------#
   #            LPF              #
   #-----------------------------#
   print 'low pass filtering...'
   gsubpost, gpost = lpf(rimarray, 15, 0, 0.5)
   gpostones = np.add(gpost,onesarray)
   gpostones = (4.8*gpostones/np.amax(gpostones)) + 0.2
   gpostinv = 1/gpostones
   gsub = np.multiply(gpostinv,rimarray)
   gsub = np.clip(gsub*3, 0, 255).astype('uint8')
   
   #------------------------#
   #        DENOISING       #
   #------------------------#
   print 'denoising...'
   pretv = denoise_tv_chambolle(gsub, weight=0.04, multichannel=False)
   gsub = (pretv*255).astype('uint8')

   #---------------------#
   #    GET CONTOURS     #
   #---------------------#
   print 'getting contours...'
   contours, hierarchy = cv2.findContours(blur,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   biggest_contour = []
   for n, contour in enumerate(contours):
      if len(contour) > len(biggest_contour):
         biggest_contour = contour

   # Approximate the breast
   epsilon = 0.007*cv2.arcLength(biggest_contour,True)
   approx = cv2.approxPolyDP(biggest_contour,epsilon,True)
   cv2.fillPoly(maskarray, pts = [biggest_contour], color=cv.RGB(255,255,255))

   # Calculate R/L sidedness using centroid
   M = cv2.moments(maskarray)
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])

   # Plot the center of mass
   cv2.circle(contoursarray,(cx,cy),15,[255,0,255],-1)            

   #---------------------#
   #    SEGMENTATION     #
   #---------------------#
   
   print 'doing segmentation...'

   # Grab a short segment of the big contour near the centroid
   short_segment = biggest_contour[biggest_contour[:,0,0] < (cx + 20)]
   short_segment = short_segment[short_segment[:,0,0] > (cx - 20)]
   short_segment = short_segment[short_segment[:,0,1] < cy]

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
   cv2.drawContours(maskarray,biggest_contour,-1,(0,0,0),SKIN) #
   cv2.drawContours(gsub,biggest_contour,-1,(0,0,0),SKIN) # will later draw the scaled version of sq over this

   #-----------------------------#
   #            SUB              #
   #-----------------------------#
   print 'subtraction...'

   if dobpe:

      rimarray, oimarray = equalize(rimarray, oimarray, dft_cntr)
      print 'before assign simarray...'
      simarray = np.clip(oimarray.astype('int')-rimarray.astype('int'), 0, 255).astype('uint8')
      print 'after assign simarray...'    

      # Draw contours on the subtracted image
      simarray = cv2.bitwise_and(simarray,maskarray)

   #---------------------#
   #      CROPPING       #
   #---------------------#
   print 'cropping...'

   # Crop the arrays based on segmentation; this may have to change with orientation; just switch the colons to the opposite sides
   maskcrop = maskarray[:dft_cntr[1],:]
   gsubcrop = gsub[:dft_cntr[1],:]
   onescrop = onesarray[:dft_cntr[1],:]
   
   if dobpe:
      subcrop = simarray[:dft_cntr[1],:]

   #--------------------------#
   #     FGT SEGMENTATION     #
   #--------------------------#
   print 'fgt segmentation...'
   
   # Get rid of nonzero pixels outside the boundary
   gsubcrop = cv2.bitwise_and(maskcrop, gsubcrop)
   otsu_binary_sum, fgtmask, otsu_binaryinv_sum, fgtmaskinv = motsu(gsubcrop, True)

   # Erode the borders since this otsu was performed on the TV denoised image which is overestimating
   kernel = np.ones((2,2),np.uint8)
   fgtmask = cv2.erode(fgtmask,kernel,iterations = 1)

   #--------------------------#
   #     BPE SEGMENTATION     #
   #--------------------------#
   print 'bpe segmentation...'

   if dobpe:
      bpe = cv2.bitwise_and(subcrop,fgtmask) # this is the bpe mask
      subcrop = subcrop + onescrop
      bpeinv = cv2.bitwise_and(subcrop,fgtmaskinv) # this is the bpe mask inverted
      bpeinvmean = ma.masked_outside(bpeinv,1,255).mean() # use the mean value of subtraction of non-FGT tissue to determine threshold for FGT enhancement
      bpeinvstd = ma.masked_outside(bpeinv,1,255).std() # use the mean value of subtraction of non-FGT tissue to determine threshold for FGT enhancement
      #print 'BPEINV Mean/Std ', bpeinvmean, bpeinvstd
      ret,bpemask = cv2.threshold(bpe,bpeinvmean+bpeinvstd*5,255,cv2.THRESH_BINARY) # threshold out the background by mean of the sub-q fat

   #-------------------------#
   #     ISOLATE BREASTS     #
   #-------------------------#
   print 'isolate breasts...'

   # Original Precontrast #
   pre_r = rimarray[:,:dft_cntr[0]]
   pre_l = rimarray[:,dft_cntr[0]:]

   # Fibroglandular tissue (FGT) mask #
   fgtmask_r = fgtmask[:dft_cntr[1],:dft_cntr[0]]
   fgtmask_l = fgtmask[:dft_cntr[1],dft_cntr[0]:]

   # Breast Mask #
   maskcrop_r = maskcrop[:dft_cntr[1],:dft_cntr[0]]
   maskcrop_l = maskcrop[:dft_cntr[1],dft_cntr[0]:]
    
   if dobpe:

      # Original Postcontrast #
      post_r = oimarray[:,:dft_cntr[0]]
      post_l = oimarray[:,dft_cntr[0]:]

      # Background parenchymal enhancement (BPE) mask #
      bpemask_r = bpemask[:dft_cntr[1],:dft_cntr[0]]
      bpemask_l = bpemask[:dft_cntr[1],dft_cntr[0]:]

   #----------------#
   #     QUANTS     #
   #----------------#
   print 'doing quantifications...'

   # Breast Volume #
   svol_r = maskcrop_r > 0
   svol_r = svol_r.astype(int)
   svol_rsum = svol_r.sum()    

   svol_l = maskcrop_l > 0
   svol_l = svol_l.astype(int)
   svol_lsum = svol_l.sum()
   print 'breast volumes ', svol_rsum, svol_lsum

   # Fibroglandular Tissue (FGT) #
   sfgtmask_r = fgtmask_r > 0
   sfgtmask_r = sfgtmask_r.astype(int)
   sfgtmask_rsum = sfgtmask_r.sum()    

   sfgtmask_l = fgtmask_l > 0
   sfgtmask_l = sfgtmask_l.astype(int)
   sfgtmask_lsum = sfgtmask_l.sum()

   print 'fgt volumes ', sfgtmask_rsum, sfgtmask_lsum

   sfgtrp = 'N/A'
   sfgtlp = 'N/A'
   sfgtbp = 'N/A'

   if svol_rsum+svol_lsum > 0:
      sfgtbp = (sfgtmask_rsum+sfgtmask_lsum)*100//(svol_rsum+svol_lsum)
      if svol_rsum > 0:
         sfgtrp = sfgtmask_rsum*100//svol_rsum
      if svol_lsum > 0:
         sfgtlp = sfgtmask_lsum*100//svol_lsum

   print 'FGTRP,'+str(sfgtrp)+',FGTLP,'+str(sfgtlp)+',FGTBP,'+str(sfgtbp)+',FGTRMS,'+str(sfgtmask_rsum)+',FGTLMS,'+str(sfgtmask_lsum)+',VOLR,'+str(svol_rsum)+',VOLL,'+str(svol_lsum)
    
   # Background Parenchymal Enhancement (BPE) #
   sbpemask_rsum = 0 # these will just be passed as zero if we're only doing BD
   sbpemask_lsum = 0

   if dobpe:
      sbpemask_r = bpemask_r > 0
      sbpemask_r = sbpemask_r.astype(int) 
      sbpemask_rsum = sbpemask_r.sum()

      sbpemask_l = bpemask_l > 0
      sbpemask_l = sbpemask_l.astype(int) 
      sbpemask_lsum = sbpemask_l.sum()

      sbperp = 'N/A'
      sbpelp = 'N/A'
      sbpebp = 'N/A'
    
      if sfgtmask_rsum+sfgtmask_lsum > 0:
         sbpebp = (sbpemask_rsum+sbpemask_lsum)*100//(sfgtmask_rsum+sfgtmask_lsum)   
         if sfgtmask_rsum > 0:
            sbperp = sbpemask_rsum*100//sfgtmask_rsum
         if sfgtmask_lsum > 0:
            sbpelp = sbpemask_lsum*100//sfgtmask_lsum
 
         print 'BPERP,'+str(sbperp)+',BPELP,'+str(sbpelp)+',BPEBP,'+str(sbpebp)+',BPERMS,'+str(sbpemask_rsum)+',BPELMS,'+str(sbpemask_lsum)+',FGTRMS,'+str(sfgtmask_rsum)+',FGTLMS,'+str(sfgtmask_lsum)

   #----------------------------#
   #          PLOTTING          #
   #----------------------------#
   print 'plotting...'

   if dobpe:

      # Plot a 8x8 
      pil_panel1 = makepil(pre_r) 
      pil_panel2 = makepil(pre_l)
      pil_panel3 = makepil(post_r) 
      pil_panel4 = makepil(post_l)
      pil_panel5 = makepil(fgtmask_r) 
      pil_panel6 = makepil(fgtmask_l)
      pil_panel7 = makepil(bpemask_r) 
      pil_panel8 = makepil(bpemask_l)

   else:
      
      # Plot a 4x4 
      print 'before make pils...'
      pil_panel1 = makepil(pre_r) 
      pil_panel2 = makepil(pre_l)
      pil_panel3 = makepil(fgtmask_r) 
      pil_panel4 = makepil(fgtmask_l)
      print 'after make pils...'

   # Get dimensions
   w1,h1 = pil_panel1.size
   w2,h2 = pil_panel2.size
   w3,h3 = pil_panel3.size
   w4,h4 = pil_panel4.size

   if dobpe:
      w5,h5 = pil_panel5.size
      w6,h6 = pil_panel6.size
      w7,h7 = pil_panel7.size
      w8,h8 = pil_panel8.size

   # Paste plots
   BUFF = 10
   dim = (0,0)
   if dobpe:
      dim = (w1+w2+w3+w4+BUFF*3,h1+h5+BUFF*3)
   else:
      dim = (w1+w2+w3+w4+BUFF*3,h1+BUFF*2)
    
   print 'before make backdrop...'
   pil_backdrop = Image.new('RGB', dim, "white")
   print 'after make backdrop...'

   if dobpe:
      pil_backdrop.paste(pil_panel1.resize((w1,h1),Image.ANTIALIAS),(BUFF,BUFF))
      pil_backdrop.paste(pil_panel2.resize((w2,h2),Image.ANTIALIAS),(w1+BUFF,BUFF))
      pil_backdrop.paste(pil_panel3.resize((w3,h3),Image.ANTIALIAS),(w1+w2+BUFF*2,BUFF))
      pil_backdrop.paste(pil_panel4.resize((w4,h4),Image.ANTIALIAS),(w1+w2+w3+BUFF*2,BUFF))
      pil_backdrop.paste(pil_panel5.resize((w5,h5),Image.ANTIALIAS),(BUFF,h1+BUFF*2))
      pil_backdrop.paste(pil_panel6.resize((w6,h6),Image.ANTIALIAS),(w1+BUFF,h1+BUFF*2))
      pil_backdrop.paste(pil_panel7.resize((w7,h7),Image.ANTIALIAS),(w1+w2+BUFF*2,h1+BUFF*2))
      pil_backdrop.paste(pil_panel8.resize((w8,h8),Image.ANTIALIAS),(w1+w2+w3+BUFF*2,h1+BUFF*2))
   else:
      pil_backdrop.paste(pil_panel1.resize((w1,h1),Image.ANTIALIAS),(BUFF,BUFF))
      pil_backdrop.paste(pil_panel2.resize((w2,h2),Image.ANTIALIAS),(w1+BUFF,BUFF))
      pil_backdrop.paste(pil_panel3.resize((w3,h3),Image.ANTIALIAS),(w1+w2+BUFF*2,BUFF))
      pil_backdrop.paste(pil_panel4.resize((w4,h4),Image.ANTIALIAS),(w1+w2+w3+BUFF*2,BUFF))
      
   print 'drawing backdrop...'
   draw = ImageDraw.Draw(pil_backdrop)

   # Add Text

   if dobpe:
      font = ImageFont.truetype(FONT_PATH+'Arial Black.ttf',25)
      draw.text((BUFF*2,BUFF*2),'Pre-contrast',font=font,fill="white")
      draw.text((w1+w2+BUFF*3,BUFF*2),'Post-contrast',font=font,fill="white")
      draw.text((BUFF*2,h1+BUFF*3),'FGT Right '+str(sfgtrp)+'%',font=font,fill="white")
      draw.text((w1+BUFF*4,h1+BUFF*3),'FGT Left '+str(sfgtlp)+'%',font=font,fill="white")
      draw.text((w1+w2+BUFF*3,h1+BUFF*3),'BPE Right '+str(sbperp)+'%',font=font,fill="white")
      draw.text((w1+w2+w3+BUFF*5,h1+BUFF*3),'BPE Left '+str(sbpelp)+'%',font=font,fill="white")
   else:
      font = ImageFont.truetype(FONT_PATH+'Arial Black.ttf',35)
      draw.text((w1+w2+BUFF*5,BUFF*3),'FGT Right '+str(sfgtrp)+'%',font=font,fill="white")
      draw.text((w1+w2+w3+BUFF*5,BUFF*3),'FGT Left '+str(sfgtlp)+'%',font=font,fill="white")

   # Save File
   print 'saving file...'
   imgout = cStringIO.StringIO()
   pil_backdrop.save(imgout, format='png')
   k.set_contents_from_string(imgout.getvalue())
   imgout.close()
   
   print 'results ', svol_rsum, svol_lsum, sfgtmask_rsum, sfgtmask_lsum, sbpemask_rsum, sbpemask_lsum

   return svol_rsum, svol_lsum, sfgtmask_rsum, sfgtmask_lsum, sbpemask_rsum, sbpemask_lsum

def processMammoFileOld(f, k):

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

#--------------------------#
#     HELPER FUNCTIONS     #
#--------------------------#


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

def makepil(a):
   # NOTE: need to copy array b/c of some obscure ubuntu/PIL issue: http://stackoverflow.com/questions/10854903/what-is-causing-dimension-dependent-attributeerror-in-pil-fromarray-function
   copyarr = a.copy() 
   return Image.fromarray(copyarr)

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

# Normalize the pre and post contrast MRI images
def equalize(a,b,dft):

   tmp = np.zeros_like(a)
   ones = np.ones_like(a)   

   # Equalize the pre and post contrast images by measuring the mean value at a filled circle located in the central defect
   if dft:
      cv2.circle(tmp,dft,30,cv.RGB(255,255,255),-1)
      a = a.astype('int')
      np.clip(a+ones, 0, 255, out=a)
      a = a.astype('uint8')
      rfarray = cv2.bitwise_and(tmp,a)
      
      b = b.astype('int')
      np.clip(b+ones, 0, 255, out=b)
      b = b.astype('uint8')
      ofarray = cv2.bitwise_and(tmp,b)

   omean = ma.masked_outside(ofarray,1,255).mean()
   rmean = ma.masked_outside(rfarray,1,255).mean()

   factor = 1

   b = b.astype('int')
   a = a.astype('int')

   print 'checking rmean vs omean...'
   if rmean > omean:
      factor = rmean/omean
      b = b*factor
      np.clip(b, 0, 255, out=b)
   else:
      factor = omean/rmean
      a = a*factor
      np.clip(a, 0, 255, out=a)
      
   return a.astype('uint8'), b.astype('uint8')

def thresh(a, b, max_value, C):
    return max_value if a > b - C else 0

def block_size(size):
    block = np.ones((size, size), dtype='d')
    block[(size - 1 ) / 2, (size - 1 ) / 2] = 0
    return block

def get_number_neighbours(mask,block):
    '''returns number of unmasked neighbours of every element within block'''
    mask = mask / 255.0
    return signal.convolve2d(mask, block, mode='same', boundary='symm')
    #return signal.fftconvolve(mask, block, mode='same')

def masked_adaptive_threshold(image,mask,max_value,size,C):
    '''thresholds only using the unmasked elements'''
    block = block_size(size)
    conv = signal.convolve2d(image, block, mode='same', boundary='symm')

    #conv = signal.fftconvolve(image, block, mode='same')
    mean_conv = conv / get_number_neighbours(mask,block)

    return v_thresh(image, mean_conv, max_value,C)

def motsu(im,iz):
    th = mahotas.otsu(im, ignore_zeros = iz)
    int_out, binary_sum, intinv_out, binaryinv_sum = bintoint(im, th)

    return binary_sum, int_out, binaryinv_sum, intinv_out

def bintoint(im, thresh):
    bool_out = im > thresh
    binary_out = bool_out.astype('uint8')
    binaryinv_out = np.logical_not(bool_out).astype('uint8')
    #opened_out = ndimage.binary_opening(binary_out, structure=np.ones((2,2))).astype('uint8')
    tot = binary_out.sum()
    out = binary_out * 255    

    totinv = binaryinv_out.sum()
    outinv = binaryinv_out * 255    

    return out, tot, outinv, totinv

def lpf(im, s, o, d):
   # Apply Large Gaussian Filter To Cropped Image
   blur = ndimage.gaussian_filter(im, (s,s), order=o)/d
   
   # Apply Subtraction
   crpint = im.astype('int')
   subtracted = np.zeros_like(im)
   np.clip(crpint-blur, 0, 255, out=subtracted) # the choice between 0 and 1 is based on the OTSU calculation, or whether or not to include all fat pixels
   return subtracted, blur

if __name__ == '__main__':
    app.run(debug=True)
