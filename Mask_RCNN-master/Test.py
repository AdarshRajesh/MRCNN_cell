import os
import sys
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR) 
import keras
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log
import itertools
import math
import logging
import json
import re
import random
import numpy as np
import tensorflow as tf
import skimage
from skimage.transform import resize
#import tqdm
from tqdm import tqdm_notebook as tqdm
import time
import concurrent.futures
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import imgaug
from imgaug import augmenters as iaa
import cv2
from samples.balloon import balloon
import imageio
from PIL import Image,ImageSequence


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
        
        
def unstack(tiff_image,save_path,save_name):
    print('unstacking :', save_name)
    createFolder(save_path)
    im = Image.open(tiff_image)
    for i, page in tqdm(enumerate(ImageSequence.Iterator(im))):
        page.save(save_path + save_name + str(i).zfill(3)+".tif" )
        
def separate_traps(read_path,save_path):
    createFolder(save_path)
    ids = next(os.walk(read_path))[2]
    IMG_WIDTH = 64
    IMG_HEIGHT = 128
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = read_path + id_
        img = cv2.imread(path,cv2.IMREAD_ANYDEPTH)
        im1 = img[68:196,18:82]
        #im1 = resize(im1, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        im1 = im1.astype(np.uint16)
        im2 = img[68:196,101:165]
        #im2 = resize(im2, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        im2 = im2.astype(np.uint16)
        im3 = img[68:196,183:247]
        #im3 = resize(im3, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        im3 = im3.astype(np.uint16)
        im4 = img[68:196,266:330]
        #im4 = resize(im4, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        im4 = im4.astype(np.uint16)
        im5 = img[68:196,348:412]
        #im5 = resize(im5, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        im5 = im5.astype(np.uint16)
        im6 = img[68:196,430:494]
        #im6 = resize(im6, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        im6 = im6.astype(np.uint16)

        cv2.imwrite(save_path + str(id_)[0:-4] +"_im_1.tif", im1)
        cv2.imwrite(save_path + str(id_)[0:-4] +"_im_2.tif", im2)
        cv2.imwrite(save_path + str(id_)[0:-4] +"_im_3.tif", im3)
        cv2.imwrite(save_path + str(id_)[0:-4] +"_im_4.tif", im4)
        cv2.imwrite(save_path + str(id_)[0:-4] +"_im_5.tif", im5)
        cv2.imwrite(save_path + str(id_)[0:-4] +"_im_6.tif", im6)
        



def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    size = 5
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
    
    
def find_mother_daughter(image_id, old_MA,dataset,model,config):
  mask_md = np.zeros((256,256,2))
  if old_MA == []:
    old_M_Area = 0
  else:
    old_M_Area = old_MA[-1]
  #image_id = random.choice(dataset.image_ids)
  #image_id = 201
  image = dataset.load_image(image_id)
  # Resize
  image, window, scale, padding, _ = utils.resize_image(
      image, 
      min_dim=config.IMAGE_MIN_DIM, 
      max_dim=config.IMAGE_MAX_DIM,
      mode=config.IMAGE_RESIZE_MODE)
  #image, image_meta, gt_class_id, gt_bbox, gt_mask =\
  #    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
  #info = dataset.image_info[image_id]
  #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
  #                                       dataset.image_reference(image_id)))
  #plt.imshow(image/65535)
  # Run object detection
  for i, pix in enumerate(reversed(image[:,65:191,0])): 
    if np.std(pix)>400:
      break
  trapborder = 255 -i
  #print(trapborder)
  results = model.detect([image], verbose=0)

  # Display results
  #ax = get_ax(1)
  r = results[0]
  #all = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
  #                            dataset.class_names, r['scores'], ax=ax,
  #                            title="Predictions")
  m=-1
  d=-1
  M_Area = 0
  D_Area = 0

  #ax = get_ax(1)
  r = results[0]

  rois = r['rois']
  #print(r)
  #print(rois)
  max_ypixel_cnt = 0
  max_i =-1
  for i,region in enumerate(rois):
    #print(region[0])
    ys = region[0]
    ye = region[2]
    #print(ye)
    ypixel_cnt = 0
    for val in range(ys,ye):
      if val>= (trapborder -115) and val<=(trapborder- 45):
        ypixel_cnt+=1

    m_s =r['masks'][:,:,i:i+1]
    M_Areatemp = 0
    for row in m_s[:,:,0]:
      for val in row:
        if val == True:
          M_Areatemp+=1
   # print(i)
   # print(ypixel_cnt)
   # print(M_Areatemp)
   # print(old_M_Area)
    if ypixel_cnt>max_ypixel_cnt and M_Areatemp>0.85*old_M_Area:
      max_i = i
      max_ypixel_cnt = ypixel_cnt


  m = max_i
  #print(m)
  mindis = float('inf')
  #max_i2 =-1
  if m ==-1:
    for i, region in enumerate(rois):
      ys = region[0]

      if ys<(trapborder -115):
        m_s =r['masks'][:,:,i:i+1]
        M_Areatemp = 0
        for row in m_s[:,:,0]:
          for val in row:
            if val == True:
              M_Areatemp+=1
        if (trapborder -115-ys) <mindis and M_Areatemp>0.85*old_M_Area:
          mindis = (trapborder -115)-ys
          max_i =i

    
    m = max_i          
    #print(m)        
    if m ==-1:
      for i,region in enumerate(rois):
        #print(region[0])
        ys = region[0]
        ye = region[2]
        #print(ye)
        ypixel_cnt = 0
        for val in range(ys,ye):
          if val>= (trapborder- 95) and val<=(trapborder- 45):
            ypixel_cnt+=1
        m_s =r['masks'][:,:,i:i+1]
        M_Areatemp = 0
        for row in m_s[:,:,0]:
          for val in row:
            if val == True:
              M_Areatemp+=1

        if ypixel_cnt>max_ypixel_cnt:
          max_i = i
          max_ypixel_cnt = ypixel_cnt

      m = max_i
      #print(m)
          
  if m!=-1:
    mother_image = visualize.display_instances(image, r['rois'][m:m+1], r['masks'][:,:,m:m+1], r['class_ids'][m:m+1], 
                              dataset.class_names, r['scores'],
                              title="Predictions", colors = [(1.0, 0.0, 0.0)])

    mother_segment =r['masks'][:,:,m:m+1]
    for row in mother_segment[:,:,0]:
      for val in row:
        if val == True:
          M_Area+=1

  elif(m==-1):
    mother_image = image
    M_Area = 0
  #log("gt_class_id", gt_class_id) 
  #log("gt_bbox", gt_bbox)
  #log("gt_mask", gt_mask)

  #print('m=',m)
  #print(M_Area)
  #ax = get_ax(1)
  r = results[0]
  if m!= -1:
    MR = rois[m]
    #d_rois = np.delete(rois,(m),axis =0)
    daughter_region  = -1
    MR_up = MR[0]
    MR_low = MR[2]
    NA =-1
    for i, region in enumerate(rois):
      if region[0]>MR[0]and region[1]>MR[1]and region[2]<MR[2]and region[3]<MR[3]:
        NA =i
    min_dif = float('inf')
    flag =0
    for i,region in enumerate(rois):
      if i!=m and i!=NA:
        y_up = region[0]
        if y_up> MR_up:
          dif = y_up - MR_up

          if dif<min_dif:
            min_dif = dif
            daughter_region = i
            flag =1


    if flag==0:
      max_overlap=0
      max_i =-1
      for i,region in enumerate(rois):
         if i!=m and i!=NA:
          y_low = region[2]
          if y_low> MR_up:
            ypixel_overlap = y_low-MR_up
            if ypixel_overlap>max_overlap:
              max_overlap = ypixel_overlap
              max_i =i
              flag = 1
      daughter_region = max_i


    if flag ==0:
      min_distance=float('inf')
      min_i =-1
      for i,region in enumerate(rois):
        if i!=m and i!=NA:
          y_low = region[2]
          if y_low<=MR_up:
            ypixel_distance = MR_up-y_low
            if ypixel_distance<min_distance:
              min_distance = ypixel_distance
              min_i =i
      daughter_region = min_i


    d = daughter_region
    if d!=-1:
      daughter_segment =r['masks'][:,:,d:d+1]
      for row in daughter_segment[:,:,0]:
        for val in row:
          if val == True:
            D_Area+=1
      #daughter_image = visualize.display_instances(image, r['rois'][d:d+1], r['masks'][:,:,d:d+1], r['class_ids'][d:d+1], 
      #                         dataset.class_names, r['scores'], ax=ax,
      #                          title="Predictions")
    #else:
    #  daughter_image = image
  if m==-1 and d==-1:
    moth_daught = image
    ax_m = (0,0)
    ax_d = (0,0)
  elif m!=-1 and d==-1:
    moth_daught = mother_image
    mask_md[:,:,0] = r['masks'][:,:,m]
    ax_d = (0,0)
    ax_m = (rois[m][2]-rois[m][0],rois[m][3]-rois[m][1])
  elif m!=-1 and d!=-1:

    rois_md = np.vstack([rois[m],rois[d]])
    #print(rois)
    #print(rois[m])
    mask_md = np.stack([r['masks'][:,:,m],r['masks'][:,:,d]],axis = -1)
    class_ids_md = np.concatenate([r['class_ids'][m:m+1],r['class_ids'][d:d+1]])
    moth_daught = visualize.display_instances(image, rois_md, mask_md, class_ids_md, 
                               dataset.class_names, r['scores'],
                                title="Predictions", colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0)])

    ax_m = (rois[m][2]-rois[m][0],rois[m][3]-rois[m][1])
    ax_d = (rois[d][2]-rois[d][0],rois[d][3]-rois[d][1])
  #Area_values = {}
  Area_values = [M_Area,D_Area]

  
  return moth_daught, Area_values,m, mask_md,ax_m,ax_d
  
  
def order_test(ids,dataset):
  id_map = {}
  for i in ids:
    name = dataset.image_info[i]['id']
    if len(name) == 19:
      id_map[i] = name[11:-4]+'_'+name[7:10].zfill(3)
    elif len(name) == 18:
      id_map[i]= name[10:-4]+'_'+name[7:9].zfill(3)
    elif len(name) == 17:
      id_map[i]= name[9:-4]+'_'+name[7:8].zfill(3)
  vals = sorted(id_map.values())
  return id_map,vals
  
  
def ordered_id(id_mapping,sorted_vals,n):
  #name = dataset.image_info[n]['id']
  #print(name)
  ordered_ids = sorted_vals.index(id_mapping[n])
  return ordered_ids
  
  
def indvtif(read_path,save_path):
  createFolder(save_path)
  createFolder(save_path+'/trap1/')
  createFolder(save_path+'/trap2/')
  createFolder(save_path+'/trap3/')
  createFolder(save_path+'/trap4/')
  createFolder(save_path+'/trap5/')
  createFolder(save_path+'/trap6/')
  save1 = save_path+'/trap1/'
  save2 = save_path+'/trap2/'
  save3 = save_path+'/trap3/'
  save4 = save_path+'/trap4/'
  save5 = save_path+'/trap5/'
  save6 = save_path+'/trap6/'



  for n in tqdm(range(int(len(ids)/6))):
    im1 = skimage.io.imread(read_path+ids[6*n+0])
    im2 = skimage.io.imread(read_path+ids[6*n+1])
    im3 = skimage.io.imread(read_path+ids[6*n+2])
    im4 = skimage.io.imread(read_path+ids[6*n+3])
    im5 = skimage.io.imread(read_path+ids[6*n+4])
    im6 = skimage.io.imread(read_path+ids[6*n+5])




    cv2.imwrite(save1+ids[n*6], im1)
    cv2.imwrite(save2+ids[n*6], im2)
    cv2.imwrite(save3+ids[n*6], im3)
    cv2.imwrite(save4+ids[n*6], im4) 
    cv2.imwrite(save5+ids[n*6], im5)
    cv2.imwrite(save6+ids[n*6], im6)
    
    
def convert2tif(read_path,save_path):
  createFolder(save_path)
  ids = sorted(next(os.walk(read_path))[2])
  height_old = 128
  width_old = 64
  for n in tqdm(range(int(len(ids)/6))):
      im1 = skimage.io.imread(read_path+ids[6*n+0])
      im2 = skimage.io.imread(read_path+ids[6*n+1])
      im3 = skimage.io.imread(read_path+ids[6*n+2])
      im4 = skimage.io.imread(read_path+ids[6*n+3])
      im5 = skimage.io.imread(read_path+ids[6*n+4])
      im6 = skimage.io.imread(read_path+ids[6*n+5])
      im1 = resize(im1[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
      im2 = resize(im2[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
      im3 = resize(im3[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
      im4= resize(im4[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
      im5 = resize(im5[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
      im6 = resize(im6[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)

      blank_image = np.zeros((256,512,3), np.uint16)
      blank_image[68:196,18:82,:] = im1
      blank_image[68:196,101:165,:] = im2
      blank_image[68:196,183:247,:] = im3
      blank_image[68:196,266:330,:] = im4
      blank_image[68:196,348:412,:] = im5
      blank_image[68:196,430:494,:] = im6

      cv2.imwrite(save_path+ids[n*6][0:10] +'.tif', blank_image)
      

def saveastiff(read_path,save_path,save_name):
    createFolder(save_path)
    ids = sorted(next(os.walk(read_path))[2])
    x = np.array([np.array(skimage.io.imread(read_path + fname)) for fname in tqdm(ids)])
    #print(x)
    imageio.mimwrite(save_path + save_name,x)
    
def resize_all(read_path):
    ids = sorted(next(os.walk(read_path))[2])
    height_old = 128
    width_old = 64
    for n in tqdm(range(len(ids))):
        im1 = skimage.io.imread(read_path+ids[n])
        im1n = resize(im1[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
        im1n = im1n.astype(np.uint16)
        skimage.io.imsave(read_path+ids[n], im1n)
  
def RunAll(read_directory,save_directory,tiffname,totalfiles,BALLOON_WEIGHTS_PATH,DEVICE = '/cpu:0'):
	Files = os.listdir(read_directory)
	for i, file in enumerate(Files):
	    if file[:-4] == tiffname[:-1]:
	        print('starting from', Files[i])
	        start_index = i
	        break
	for file in Files[i:i+totalfiles*3]:
	    if file[-5] == '1':
	        #print(file[:-4]+'/')
	        RunTest(read_directory,save_directory,file[:-4]+'/',BALLOON_WEIGHTS_PATH,DEVICE )

    
def RunTest(read_directory,save_directory,tiffname,BALLOON_WEIGHTS_PATH,DEVICE ):

    SAVE_DIR = save_directory+tiffname +'preds/'
    M_file = save_directory+tiffname+'m_vals.txt'
    mother_file = save_directory+tiffname+ 'mother_areas.txt'
    daughter_file = save_directory+tiffname +'daughter_areas.txt'
    M_mask_dir = save_directory+tiffname+ 'mother_masks'
    D_mask_dir = save_directory+tiffname+ 'daughter_masks'
    TEST_DIR = save_directory+tiffname
    MODEL_DIR = save_directory+'logs/'
    M_axes_file = save_directory+tiffname+'mother_axes'
    D_axes_file = save_directory+tiffname+'daughter_axes'

        #unstack('Z:/Adarsh/20181212/data/xy01c1.tif','Z:/Adarsh/Segmentation_project/GUI/xy01c1/Phase/','xy01c1_')
    unstack(read_directory+tiffname[:-1]+'.tif',save_directory+tiffname+'Phase/',tiffname[:-1]+'_')
    unstack(read_directory+tiffname[:-2]+'2.tif',save_directory+tiffname[:-2]+'2/Phase/',tiffname[:-2]+'2_')
    unstack(read_directory+tiffname[:-2]+'3.tif',save_directory+tiffname[:-2]+'3/Phase/',tiffname[:-2]+'3_')
    print('separating traps...')
    separate_traps(save_directory+tiffname+'Phase/',save_directory+tiffname+'/test/')
    separate_traps(save_directory+tiffname[:-2]+'2/Phase/',save_directory+tiffname[:-2]+'2/intensity/')
    separate_traps(save_directory+tiffname[:-2]+'3/Phase/',save_directory+tiffname[:-2]+'3/intensity/')
    
    
    
    #DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    config = balloon.BalloonConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()	
    TEST_MODE = "inference"
    
        # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                config=config)
    # Load validation dataset
    dataset = balloon.BalloonDataset()
    dataset.load_balloon(TEST_DIR, "test")
    dataset.prepare()
    # Must call before using the dataset


    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


    # Set path to balloon weights file

    # Download file from the Releases page and set its path
    # https://github.com/matterport/Mask_RCNN/releases
    # weights_path = "/path/to/mask_rcnn_balloon.h5"

    # Or, load the last model you trained
    #weights_path = model.find_last()
    weights_path = BALLOON_WEIGHTS_PATH
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    idmap, v = order_test(dataset.image_ids,dataset)
    sorted_ids = [item[0] for item in sorted(idmap.items(), key=lambda x: x[1])]
    
    
    M_vals = []
    createFolder(SAVE_DIR)
    createFolder(M_mask_dir)
    createFolder(D_mask_dir)
    #MD_Areas = {}
    M_Areas = []
    D_Areas = []
    M_Axes = []
    D_Axes = []
    count =0
    MA = []
    DA= []
    MAx = []
    DAx = []
    for n in tqdm((sorted_ids), total=len(sorted_ids)):


      name = dataset.image_info[n]['id']
      #print(name)
      output,AV,M,masks,maxis,daxis = find_mother_daughter(n,MA,dataset,model,config)
      #output,AV = find_mother_daughter(n,MA)
      masks = masks[:,64:192]
      masks = resize(masks, (128,64), mode='constant', preserve_range=True)
      ret,thresh_img = cv2.threshold(masks,0.5,1,cv2.THRESH_BINARY)
      MA.append(AV[0])
      DA.append(AV[1])
      MAx.append(maxis)
      DAx.append(daxis)
      moth_mask = thresh_img[:,:,0]*255
      daught_mask = thresh_img[:,:,1]*255
      if len(name) == 19:
        #MD_Areas[name[0:7]+ name[7:10].zfill(3) + name[10:-4]+'_mask.tif'] = AV
        cv2.imwrite(SAVE_DIR+name[0:7]+ name[7:10].zfill(3) + name[10:-4] +'_mask.tif', output)
        cv2.imwrite(M_mask_dir+'/'+name[0:7]+ name[7:10].zfill(3) + name[10:-4] +'_Mmask.jpg', moth_mask)
        cv2.imwrite(D_mask_dir+'/'+name[0:7]+ name[7:10].zfill(3) + name[10:-4] +'_Dmask.jpg', daught_mask)

      elif len(name) == 18:
        #MD_Areas[name[0:7]+ name[7:9].zfill(3) + name[9:-4]+'_mask.tif'] = AV
        cv2.imwrite(SAVE_DIR+name[0:7] + name[7:9].zfill(3)+name[9:-4]+'_mask.tif', output)
        cv2.imwrite(M_mask_dir+name[0:7] + name[7:9].zfill(3)+name[9:-4]+'_Mmask.jpg', moth_mask)
        cv2.imwrite(D_mask_dir+name[0:7] + name[7:9].zfill(3)+name[9:-4]+'_Dmask.jpg', daught_mask)

      elif len(name) == 17:
        #MD_Areas[name[0:7]+ name[7:8].zfill(3) + name[8:-4]+'_mask.tif'] = AV
        cv2.imwrite(SAVE_DIR+name[0:7]+ name[7:8].zfill(3)+name[8:-4]+'_mask.tif', output)
        cv2.imwrite(M_mask_dir+name[0:7]+ name[7:8].zfill(3)+name[8:-4]+'_Mmask.jpg', moth_mask)
        cv2.imwrite(D_mask_dir+name[0:7]+ name[7:8].zfill(3)+name[8:-4]+'_Dmask.jpg', daught_mask)


      if count == (len(sorted_ids)/6)-1:
        M_Areas.append(MA)
        D_Areas.append(DA)
        M_Axes.append(MAx)
        D_Axes.append(DAx)
        MA = []
        DA = []
        MAx =[]
        DAx =[]
        count = -1

      count +=1
      M_vals.append(M)    
    #print(M_vals)
    np.save(M_axes_file, M_Axes)
    np.save(D_axes_file,D_Axes)
    np.savetxt(M_file, M_vals,fmt = '%i')    
    np.savetxt(mother_file,M_Areas)
    np.savetxt(daughter_file, D_Areas)

    # cell_cycle_counts = find_cell_cycles(daughter_file)
    #filter_cell_cycle_counts(cell_cycle_counts)
    #with open('/content/drive/My Drive/MRCNN_test_results/'+filename[:-1]+'_cell_cycle_counts.txt', "w") as output:
    #  for i in (cell_cycle_counts):
    #   output.write(str(i)+'\n')


    TRAP_SAVE = save_directory+tiffname+'traps/'
    convert2tif(SAVE_DIR,TRAP_SAVE)

    saveastiff(TRAP_SAVE,save_directory,tiffname[:-1]+'_preds.tiff')
    resize_all(save_directory+tiffname+'preds/')

