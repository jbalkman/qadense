# General
import os
import re
import StringIO
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file

# Image Processing
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas
import dicom
from matplotlib.figure import Figure

# Image Drawing
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image

# Scientific Image Manipulation
from scipy import ndimage
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filter import rank, threshold_otsu
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # change to relative /var/www/ file

ALLOWED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'dcm', 'dicom']
TIFF_EXTENSIONS = ['tif', 'tiff']
DICOM_EXTENSIONS = ['dcm', 'dicom']
DEBUG = True
FONT_PATH = 'static/fonts/'

@app.route('/')
def hello_world():
   print 'Hello World!'
   return render_template('index.html')

@app.route('/process_serve', methods=['GET'])
def process_serve_img():
   imgfile = request.args.get('imgfile')
   print "Process/Serving Image: "+imgfile
   fnamesplit = imgfile.rsplit('.')
   imgprefix = fnamesplit[0]
   ext = fnamesplit[1]

   # Initialize submitted variables
   a = None
   d = None
   ca  = None
   cv  = None
   s = None
   v = None
   data_encode = None

   try:
      a, d, ca, cv, s, v = processFile(imgfile) # returns density, density category, side, and view
      print a, d, ca, cv, s, v
      result = 1
   except:
      result = 0
   
   try:
      with open(imgprefix+"-out.jpg", "rb") as f: # the imgfile has been resaved as the results from the processing above, so there is no need to change the file
         data = f.read()
         data_encode = data.encode("base64")
         print "Removing output file "+imgprefix+"-out.jpg"
         os.remove(imgprefix+"-out.jpg")
   except:
      result = 0

   # Clean-up upload files so nothing is left on the server
   print "Removing files for "+imgprefix+"."+ext
   try: 
      os.remove(imgprefix+"."+ext)
   except:
      print "Unable to remove files for "+imgprefix+"."+ext

   # Remove histogram file
   try:
      os.remove(imgprefix+'-hist.jpg')
   except:
      print "Unable to remove histogram file"
         
   # Since it was required to convert a dcm file to a tif we need to remove the latter as well
   if 'dcm' in ext:
      try:
         os.remove(imgprefix+'.tif')
      except:
         print "Unable to remove files for "+imgprefix+"."+ext

   return jsonify({"success":result, "imagefile":data_encode, "area_d":a, "volumetric_d":d, "dcat_a":ca, "dcat_v":cv, "side":s, "view":v})

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now = datetime.now()
            ext = file.filename.rsplit('.', 1)[1]
            filename_noext = os.path.join(app.config['UPLOAD_FOLDER'], "%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f")))
            filename_ext = filename_noext+'.'+ext
            file.save(filename_ext)

            return jsonify({"success":True, "file": filename_ext})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def istiff(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in TIFF_EXTENSIONS

def isdicom(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in DICOM_EXTENSIONS

def processFile(f):
   
   # Handle DICOM
   if isdicom(f):
      ds = dicom.read_file(f)
      fname = f.rsplit('.', 1)[0]+'.tif' # make a tiff file under the same name to read from
      pil_dcm = get_dicom_PIL(ds)
      pil_dcm.save(fname)
   else:
      fname = f

   # Set output file names
   fname_out = f.rsplit('.', 1)[0]+"-out.jpg"
   fname_hist = f.rsplit('.', 1)[0]+"-hist.jpg"

   # Open the image file for processing
   print "File to process: "+fname      
   origimg = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
      
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
   hist[0] = hist[1]*0.50
   hist[1] = hist[2]*0.75
   hist[1] = hist[3]*0.75
   plt.subplot(1,2,2),plt.bar(center, hist, align='center', width=width)
   plt.title('Volumetric-based Histogram')
   plt.xlabel("Pixel Value")

   # Save plot as Image and close
   plt.tight_layout()
   plt.savefig(fname_hist)
   plt.close('all')

   # Create PIL images
   pil1 = Image.fromarray(imarray2)
   pil2 = Image.fromarray(contoursarray)
   pil3 = Image.fromarray(maskarray)
   pil4 = Image.fromarray(imcrop_fgt)
   pil5 = Image.open(fname_hist)

   # Pasting images above to a pil background along with text. There's a lot of particular measurements sizing the fonts & pictures so that everything fits.  It's somewhat arbitrary with lots of trial and error, but basically everything is based off the resized width of the first image.  Images needed to be resized down b/c they were too high resolution for canvas.
   rf = 2 # rf = resize factor

   w1,h1 = pil1.size
   pil1_sm = pil1.resize((w1/rf,h1/rf))
   w1_sm,h1_sm = pil1_sm.size
   print "Resize", int(h1_sm*0.9)

   pil2_sm = pil2.resize((w1_sm,h1_sm))
   pil3_sm = pil3.resize((w1_sm,h1_sm))
   pil4_sm = pil4.resize((w1_sm,h1_sm))
   pil5_sm = pil5.resize((w1_sm*2,int(h1_sm*0.8)))

   pil_backdrop = Image.new('RGB', (100+2*w1_sm,3*h1_sm+3*h1_sm/8), "white")

   pil_backdrop.paste(pil1_sm, (0,h1_sm/8))
   pil_backdrop.paste(pil2_sm, (100+w1_sm,h1_sm/8))
   pil_backdrop.paste(pil3_sm, (0,h1_sm/4+h1_sm))
   pil_backdrop.paste(pil4_sm, (100+w1_sm,h1_sm/4+h1_sm))
   pil_backdrop.paste(pil5_sm, (0,3*h1_sm/8+2*h1_sm))

   font = ImageFont.truetype(FONT_PATH+"Arial Black.ttf",w1_sm/20)

   draw = ImageDraw.Draw(pil_backdrop)
   draw.text((0,0),"Original Image",0,font=font)
   draw.text((100+w1_sm,0),"Fibroglandular Tissue",0,font=font)
   draw.text((0,h1_sm+h1_sm/6),"Breast Contouring",0,font=font)
   draw.text((100+w1_sm,h1_sm+h1_sm/6),"Breast Segmentation",0,font=font)

   pil_backdrop.save(fname_out)

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
