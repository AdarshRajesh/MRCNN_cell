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
import logging
import json
import re
import random
import math
import numpy as np
import tensorflow as tf
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
import skimage
from skimage.transform import resize
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter.simpledialog
from tkinter.ttk import *

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def find_cell_cycles(file):
  all_cycles = []
  D_Areas = np.loadtxt(file)
  for well in range(len(D_Areas)):
    cycle = []
    DA = D_Areas[well]
    for i in range(len(DA)-1):
      if DA[i+1]<0.8*DA[i]:
        cycle.append(i+2)
    all_cycles.append(cycle)
  return all_cycles
  
def filter_cell_cycle_counts(cycle_counts):
  for well in cycle_counts:
    del_indices =[]
    #print(len(well))
    if len(well)!=0:
        for i in range(len(well)-1):
            if well[i+1]-well[i]<=3:
                del_indices.append(i)
        #print(del_indices)      
    for i in sorted(del_indices, reverse=True):
        del well[i]    
            

def order_test(dataset,ids):
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

def prepare_dataset(directory,tiffname,DEVICE= "/cpu:0"):
  TEST_DIR = directory+tiffname
  # Load validation dataset
  dataset = balloon.BalloonDataset()
  dataset.load_balloon(TEST_DIR, "test")

  # Must call before using the dataset
  dataset.prepare()
  return dataset

  
def get_new_daughter(dataset,model,config,image_id, m, dx,dy):
  flag =0
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
  D_Area =0
  results = model.detect([image], verbose=0)

  # Display results
  #ax = get_ax(1)
  r = results[0]
  #all = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
  #                            dataset.class_names, r['scores'], ax=ax,
  #                            title="Predictions")


  #ax = get_ax(1)
  r = results[0]

  rois = r['rois']

  #print(rois)        
  if m!=-1:
    mother_image = visualize.display_instances(image, r['rois'][m:m+1], r['masks'][:,:,m:m+1], r['class_ids'][m:m+1], 
                              dataset.class_names, r['scores'],
                              title="Predictions", colors = [(1.0, 0.0, 0.0)])


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
    d = -1
    
    dy_con = 256*(dy-68)/128
    dx_con = 64+128*(dx)/64
    #print(dy_con,dx_con)
    for i,region in enumerate(rois):
        #print(region)
        if region[0]<= dy_con and region[2]>=dy_con:
            if region[1]<=dx_con and region[3]>=dx_con:
                daughter_region = i
                flag = 1

    if flag==1:
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
    
  if flag != 1:
    print('not a valid daughter cell')
  if m==-1 and d==-1:
    moth_daught = image
  elif m!=-1 and d==-1:
    moth_daught = mother_image
  elif m!=-1 and d!=-1:

    rois_md = np.vstack([rois[m],rois[d]])
    mask_md = np.stack([r['masks'][:,:,m],r['masks'][:,:,d]],axis = -1)
    class_ids_md = np.concatenate([r['class_ids'][m:m+1],r['class_ids'][d:d+1]])
    moth_daught = visualize.display_instances(image, rois_md, mask_md, class_ids_md, 
                               dataset.class_names, r['scores'],
                                title="Predictions", colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0)])
   
  Area_values = {}
  Area_values = D_Area

  
  return moth_daught,Area_values,(rois[d][2]-rois[d][0],rois[d][3]-rois[d][1])
#Area_values

def ClearMemory(save_directory,tiffname):
    shutil.rmtree(save_directory+tiffname[:-2]+'2')
    shutil.rmtree(save_directory+tiffname[:-2]+'3')
    shutil.rmtree(save_directory+tiffname+'daughter_masks')
    shutil.rmtree(save_directory+tiffname+'mother_masks')	
    shutil.rmtree(save_directory+tiffname+'Phase')	    
    shutil.rmtree(save_directory+tiffname+'preds')	
    shutil.rmtree(save_directory+tiffname+'test')
    shutil.rmtree(save_directory+tiffname+'traps')

def generate_intensity_vals(m_mask,d_mask,channel2,channel3):
    c2 = cv2.imread(channel2,cv2.IMREAD_UNCHANGED)
    c3 = cv2.imread(channel3,cv2.IMREAD_UNCHANGED)    
    m_mask = cv2.imread(m_mask)
    m_mask = m_mask[:,:,1]
    d_mask = cv2.imread(d_mask)
    d_mask = d_mask[:,:,1]
    m_ret,m_thr_img = cv2.threshold(m_mask,20,1,cv2.THRESH_BINARY)
    d_ret,d_thr_img = cv2.threshold(d_mask,20,1,cv2.THRESH_BINARY)
    m_c2 = c2*m_thr_img
    m_c2 = m_c2[m_c2!=0]
    
    d_c2 = c2*d_thr_img
    d_c2 = d_c2[d_c2!=0]
    
    m_c3 = c3*m_thr_img
    m_c3 = m_c3[m_c3!=0]
    
    d_c3 = c3*d_thr_img
    d_c3 = d_c3[d_c3!=0]
    
    mother_intensities = []
    mother_intensities.append(m_c2)
    mother_intensities.append(m_c3)

    daughter_intensities = []
    daughter_intensities.append(d_c2)
    daughter_intensities.append(d_c3)

    return mother_intensities, daughter_intensities



def run_gui(directory,tiffname,dataset,weights_path,DEVICE = "/cpu:0"):
    MODEL_DIR = directory+'logs/'
    #DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    config = balloon.BalloonConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    #config.display()
    TEST_MODE = "inference"

      # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    model.load_weights(weights_path, by_name=True)
    idmap, v = order_test(dataset,dataset.image_ids)
    ordered_id(idmap,v,2)
    sorted_ids = [item[0] for item in sorted(idmap.items(), key=lambda x: x[1])]
    inv_map = {v: k for k, v in idmap.items()}


    class MainWindow:

        #----------------


        def __init__(self, main,directory,tiffname):

            #prepare_dataset(Mod_dir,directory,tiffname,DEVICE)
            #root=Tk()
            #frame=Frame(root,width=300,height=300)
            #frame.grid(row=0,column=0)
            #=Canvas(root,bg='#FFFFFF',width=300,height=300,scrollregion=(0,0,500,500))
            #hbar=Scrollbar(root,orient=HORIZONTAL)
            #hbar.pack(side=BOTTOM,fill=X)
            #hbar.grid(row = 1,column =0,sticky = E+W)
           # hbar.config(command=canvas.xview)

            #canvas.config(width=300,height=300)
            #canvas.config(xscrollcommand=try1)
            #canvas.pack(side=LEFT,expand=True,fill=BOTH)
            #canvas.grid(row = 0,column =0)
            self.directory = directory
            self.tiffname = tiffname
            self.D_Areas = np.loadtxt(self.directory+self.tiffname+'daughter_areas.txt')
            self.D_Axes = np.load(self.directory+self.tiffname+'daughter_axes.npy')
            self.read_path = self.directory+self.tiffname+'preds/'
            self.save_path = self.directory+self.tiffname+'traps/'

            self.canvas = Canvas(main, width=512, height=512)
            self.canvas.grid(row=0, column=0,sticky = 'nw')


            #w = Canvas(root, width=512, height=256)
            #w.pack()
            self.rtlist = []

            #adding the image

            #w.create_image(0, 0, image=img, anchor="nw")
            with open(self.directory+self.tiffname+'m_vals.txt', 'r') as f:
                self.M_vals =[int(x) for x in f.read().splitlines()]

    #instruction on 3 point selection to define grid

            # images
            #self.my_images = []
            #self.my_images.append(PhotoImage(file = 'D:/Segmentation_project/GUI/sample/xy63c1_000.tif'))
            #self.my_images.append(PhotoImage(file = 'D:/Segmentation_project/GUI/sample/xy63c1_004.tif'))
            #self.my_images.append(PhotoImage(file = "ball3.gif"))
            #self.my_image_number = 0
            self.fields = ('Trap1', 'Trap2', 'Trap3', 'Trap4','Trap5', 'Trap6')
            #set first image on canvas
            #self.File = askopenfilename(parent=main, initialdir="Z:/Adarsh/Segmentation_project/GUI/xy01c1/traps",title='Select a folder')
            self.File = self.directory +self.tiffname+'traps/'+self.tiffname[:-1]+'_000.tif'
            #print(self.File)
            self.Foldern = self.directory+self.tiffname+'phase/'
            ids = next(os.walk(self.Foldern))[2]
            self.timepoints = len(ids) 
            self.imnum = self.File[-14::]
            #print(self.imnum)
            self.phase = self.Foldern+self.imnum

            #File = 'D:/Segmentation_project/GUI/sample/xy63c1_000.tif'

            self.original = Image.open(self.File)
            #print(self.original)
            #self.original = self.original.resize((512,256)) #resize image
            self.img = ImageTk.PhotoImage(self.original)
            #print(self.img)
            self.originalp = plt.imread(self.phase)
            self.originalp= (self.originalp/256).astype('uint8')

            self.imgp = ImageTk.PhotoImage(image=Image.fromarray(self.originalp), master = root)


               # cv.create_image(10*i, 10*i, image=photo, anchor='nw')
            self.image_on_canvas1 = self.canvas.create_image(0, 0, anchor = NW, image = self.img)             
            self.image_on_canvas2 = self.canvas.create_image(0, 256, anchor = NW, image = self.imgp)



            #self.image_on_canvas = self.canvas.create_image(0, 0, anchor = NW, image = self.img)
            #self.image_on_canvas = self.canvas.create_image(0, 1, anchor = NW, image = self.img)
            #tkinter.messagebox.showinfo("Instructions", "Click daughter cells in each well: \n")

            #button to change image
            #self.button = Button(main, text="Change", command=self.onButton)
            #self.button.grid(row=2, column=0)
            self.d = Button(main, text="done", command=main.destroy)
            self.d.grid(row = 3, column = 0)
            #self.next = Button(main, text = "next",command = self.nextButton)
            #self.next.grid(row = 2, column =1)
            #self.next = Button(main, text = "back",command = self.backButton)
            #self.next.grid(row = 1, column =1)
            self.ref = Button(main, text = "Generate Data",command = self.create_window)
            self.ref.grid(row = 2, column =1)
            #self.pack()
            #tkinter.messagebox.showinfo("Instructions", "Click daughter cells in each well: \n")
            self.canvas.bind("<Button 1>",self.get6)

            self.progress_bar = Progressbar(orient = 'horizontal', length = 286, mode = 'determinate')
            self.progress_bar.grid(column = 0, row = 2)

            #self.frame=Frame(main,width=300,height=300)
            #self.frame.grid(row= 4 ,column=0)
            self.frame=Frame(main,width=300,height=300)
            self.frame.grid(row=1,column=0)
            self.scrRgn = 500+50*(self.timepoints-1)
            self.num =500/self.scrRgn
            self.den = 50/self.scrRgn
            self.canvas2=Canvas(self.frame,bg='#FFFFFF',width=500,height=10,scrollregion=(0,0,self.scrRgn,10))
            self.hbar=Scrollbar(self.frame,orient=HORIZONTAL)

            #self.canvas2.config(scrollregion=(0, 0, 500, 500))
            self.hbar.grid(row = 1,column =0,sticky = E+W)
            self.hbar.config(command=self.canvas2.xview)
            self.canvas2.config(xscrollcommand=self.call)
            self.canvas2.grid(row=0, column=0,sticky = 'nw')

            #self.Sb = Scrollbar(main, orient = HORIZONTAL)
            #self.Sb.grid(row = 4, column = 0)
            #self.Sb.config(command=self.call)

            self.lbl2 = Label(main, text = "", width =8)
            self.lbl2.grid(row = 2, column =0,sticky = W)
            self.lbl2.configure(font = ("Arial",10))


            #self.RT = Button(main,text = "Remove Traps", command = self.RemTraps)
            #self.RT.grid(row = 4, column =1)

            #self.b = Button(main, text="Refine", command=self.create_window)
            #self.b.grid(row=4,column =1)

     #   def try1(self,val1,val2,val3 = None):
     #       #self.canvas2.xview(val1,val2)
     #       self.hbar.set(val1,val2)
     #       print(int((float(val2)-500/self.scrRgn)/(50/self.scrRgn)))

        def create_window(self):
                self.window = Toplevel(root)
                #RT = Button(self.window,text = "Remove Traps", command = self.RemTraps)
                #RT.grid(row = 0, column =0)

                self.makeform(self.window, self.fields)
                b1 = tk.Button(self.window, text='Confirm',command=self.getvals)
                b1.pack(side=tk.LEFT, padx=5, pady=5)
                #b2 = tk.Button(window, text='Monthly Payment',
                #       command=(lambda e=ents: monthly_payment(e)))
                #b2.pack(side=tk.LEFT, padx=5, pady=5)



        def getvals(self):
            self.Svals={}
            self.Evals ={}
            self.Cvals ={}
            for field in self.fields:
                self.Svals[field] = self.entries1[field].get()
                self.Evals[field] = self.entries2[field].get()
                self.Cvals[field] = self.checks[field].get()
            self.keeplist = list(self.Cvals.values())
            self.rtlist = [i+1 for i, x in enumerate(self.keeplist) if x == 1]
            self.window.destroy()
            self.refreshButton()
            #print(self.Svals)
            #print(self.Evals)
            #print(self.Cvals)


        def makeform(self,main, fields):
            self.entries1 = {}
            self.entries2 ={}
            self.checks = {}
            for field in fields:
                self.checks[field] = IntVar()
                row = tk.Frame(main)
                lab1 = tk.Label(row, width=12, bg = "Yellow", text=field+": ", anchor='w',bd = 2)
                lab2 = tk.Label(row, width = 9, text = "start frame: ", anchor = 'w',bd =2)
                lab3 = tk.Label(row, width = 9, text = "end frame: ", anchor = 'w',bd =2)
                Chk = Checkbutton(row, text="remove", variable=self.checks[field])
                #checks[field] = var1.get()
                ent = tk.Entry(row,width = 5, bd =2)
                ent2 = tk.Entry(row, width = 5 ,bd =2)
                ent.insert(0, "1")
                ent2.insert(0,str(self.timepoints))
                row.pack(side=tk.TOP, 
                         fill=tk.X, 
                         padx=5, 
                         pady=5)
                lab1.pack(side=tk.LEFT)
                lab2.pack(side = tk.LEFT)
                ent.pack(side=tk.LEFT)
                lab3.pack(side = tk.LEFT)
                ent2.pack(side=tk.LEFT)
                Chk.pack(side = tk.LEFT)
                self.entries1[field] = ent
                self.entries2[field] = ent2



        #def RemTraps(self):
            #self.rt = tkinter.simpledialog.askstring("Remove Traps", "Enter Trap Numbers to remove(format :1,2,3)")
            #print(self.rt)
            #self.list = self.rt.split (",")

            #for i in self.list:
            #    self.rtlist.append(int(i))
            #print(self.rtlist)
            #self.cb = Checkbutton(text="Hardcopy")
            #self.cb.grid(row=2, columnspan=2, sticky=W)
            #answerreturn = IntVar()
            #answer = tkinter.simpledialog.askfloat.Checkbutton(variable=answerreturn,text ="1")
            #answer.grid(row=0, column=1)

        def call(self,val1,val2,val3 = None):
            #print(int(val2))
            self.hbar.set(val1,val2)
            #print(float(val2)-float(val1))
            self.scrlval = round((float(val2)-self.num)/self.den)
            #print((float(val2)-500/self.scrRgn)/(50/self.scrRgn))
            #print(self.scrlval)
            #self.filenum = int(self.File[-7:-4])
            #print(self.filenum)
            if self.scrlval>=0:
                self.filenum = self.scrlval
                self.lbl2.configure(text = str(self.filenum)+'/'+str(self.timepoints))
                self.File = self.File[0:-7]+str(self.filenum).zfill(3)+'.tif'
                self.phase = self.phase[0:-7]+str(self.filenum).zfill(3)+'.tif'
                self.original = Image.open(self.File)
                self.originalp = plt.imread(self.phase)
                self.originalp= (self.originalp/256).astype('uint8')
                self.img = ImageTk.PhotoImage(self.original)
                self.imgp = ImageTk.PhotoImage(image=Image.fromarray(self.originalp), master = root)


                self.canvas.itemconfig(self.image_on_canvas1, image = self.img)
                self.canvas.itemconfig(self.image_on_canvas2, image = self.imgp)



      #  def onButton(self):
      #      self.File = askopenfilename(parent=root, initialdir="D:/Segmentation_project/GUI/xy16c1/traps",title='Select an image')
      #      
      #      self.Foldern = "D:/Segmentation_project/GUI/xy16c1/phase/"
      #      self.imnum = self.File[-14::]
      #      self.phase = self.Foldern+self.imnum

      #      #File = 'D:/Segmentation_project/GUI/sample/xy63c1_000.tif'

      #      self.original = Image.open(self.File)
            #self.original = self.original.resize((512,256)) #resize image
      #      self.img  = ImageTk.PhotoImage(self.original)
      #      self.originalp = plt.imread(self.phase)
      #      self.originalp= (self.originalp/256).astype('uint8')

       #     self.imgp = ImageTk.PhotoImage(image=Image.fromarray(self.originalp), master = root)




       #     self.canvas.itemconfig(self.image_on_canvas1, image = self.img)
       #     self.canvas.itemconfig(self.image_on_canvas2, image = self.imgp)
       #     tkinter.messagebox.showinfo("Instructions", "Click daughter cells in each well: \n");;;;;;;;;;;;;;;;;;/////////////

    #--------------------------------------------------------------------;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


        def get6(self,eventextent6):
            global x6,y6
            x6 = eventextent6.x
            y6 = eventextent6.y
            #print(ye)
            print(x6,y6)
            print(self.File)
            self.find_img_dtr(x6,y6,self.File,self.M_vals)
            self.instantrefreshButton()   


        def find_img_dtr(self,x,y,tiffile,mvals):
            #SAVE_DIR = 'D:/Segmentation_project/GUI/xy16c1/xy16c1_indv_preds/'        
            SAVE_DIR = self.directory+self.tiffname+'preds/'
            #print('tiffile=')
            #print(tiffile)
            img_name  = tiffile[-7:-4]
            print(img_name)
            if x in range(18,82):
                img_num = 'im_1'
                x = x-18
            elif x in range(101,165):
                img_num ='im_2'
                x = x- 101
            elif x in range(183,247):
                img_num ='im_3'
                x = x-183
            elif x in range(266,330):
                img_num ='im_4'
                x = x-266
            elif x in range(348,412):
                img_num ='im_5'
                x = x-348
            elif x in range(430,494):
                img_num ='im_6'
                x = x-430
            else:
                print('not a daughter_cell')

            im_id = inv_map[img_num+'_'+img_name]
            M = mvals[sorted_ids.index(im_id)]
            mod_img,da,Dax = get_new_daughter(dataset,model,config,im_id,M, x,y)
            #plt.figure()
            #plt.imshow(mod_img/65535)
            name = dataset.image_info[im_id]['id']
            mod_img = resize(mod_img[:,64:192,:], (128,64), mode='constant', preserve_range=True)
            mod_img = mod_img.astype(np.uint16)
            if len(name) == 19:
                #skimage.io.imsave(SAVE_DIR+name[0:7]+ name[7:10].zfill(3) + name[10:-4] +'_mask.tif', im1n)
                cv2.imwrite(SAVE_DIR+name[0:7]+ name[7:10].zfill(3) + name[10:-4] +'_mask.tif', mod_img)
            elif len(name) == 18:
                #skimage.io.imsave(SAVE_DIR+name[0:7] + name[7:9].zfill(3)+name[9:-4]+'_mask.tif', im1n)
                cv2.imwrite(SAVE_DIR+name[0:7] + name[7:9].zfill(3)+name[9:-4]+'_mask.tif', mod_img)
            elif len(name) == 17:
                #skimage.io.imsave(SAVE_DIR+name[0:7]+ name[7:8].zfill(3)+name[8:-4]+'_mask.tif', im1n)
                cv2.imwrite(SAVE_DIR+name[0:7]+ name[7:8].zfill(3)+name[8:-4]+'_mask.tif', mod_img)
            #TRAP_SAVE = 'D:/Segmentation_project/GUI/xy16c1/traps/'  
            #convert2tif(SAVE_DIR,TRAP_SAVE)
            #plt.figure()
            #plt.imshow(mod_img/65535)
            #saveastiff(TRAP_SAVE,'/content/drive/My Drive/MRCNN_test_results/',filename[:-1]+'_preds.tiff')
            r = int(sorted_ids.index(im_id)/self.timepoints)
            c = sorted_ids.index(im_id)%self.timepoints
            print(r,c)
            self.D_Areas[r,c] = da
            self.D_Axes[r,c] = Dax
            self.modfiles = img_name
            #np.savetxt('D:/Segmentation_project/GUI/xy16c1/xy16c1_daughter_areas.txt',self.D_Areas)
            np.savetxt(self.directory+self.tiffname+'daughter_areas.txt',self.D_Areas)
            np.save(self.directory+self.tiffname+'daughter_axes',self.D_Axes)
        def instantrefreshButton(self):    
            #print(self.modfiles)
            #filenums = [int(x) for x in self.modfiles]
            #print(filenums)
            #read_path = 'D:/Segmentation_project/GUI/xy16c1/xy16c1_indv_preds/'        
            #save_path = 'D:/Segmentation_project/GUI/xy16c1/traps/'  
            #convert2tif(SAVE_DIR,TRAP_SAVE)
            createFolder(self.save_path)
            ids = sorted(next(os.walk(self.read_path))[2])
            #height_old = 104
            #width_old = 52
            #for n in tqdm(range(321)):
            #for n in self.modfiles:
            n = int(self.modfiles)            
            im1 = skimage.io.imread(self.read_path+ids[6*n+0])
            #print(im1.shape)
            im2 = skimage.io.imread(self.read_path+ids[6*n+1])
            im3 = skimage.io.imread(self.read_path+ids[6*n+2])
            im4 = skimage.io.imread(self.read_path+ids[6*n+3])
            im5 = skimage.io.imread(self.read_path+ids[6*n+4])
            im6 = skimage.io.imread(self.read_path+ids[6*n+5])
            #im1 = resize(im1[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
            #im2 = resize(im2[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
            #im3 = resize(im3[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
            #im4=  resize(im4[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
            #im5 = resize(im5[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
            #im6 = resize(im6[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)

            blank_image = np.zeros((256,512,3), np.uint16)
            if not 1 in self.rtlist: 
                blank_image[68:196,18:82,:] = im1
            if not 2 in self.rtlist: 
                blank_image[68:196,101:165,:] = im2
            if not 3 in self.rtlist: 
                blank_image[68:196,183:247,:] = im3
            if not 4 in self.rtlist: 
                blank_image[68:196,266:330,:] = im4
            if not 5 in self.rtlist: 
                blank_image[68:196,348:412,:] = im5
            if not 6 in self.rtlist: 
                blank_image[68:196,430:494,:] = im6

            cv2.imwrite(self.save_path+ids[n*6][0:10] +'.tif', blank_image)



            self.original = Image.open(self.File)
            #self.original = self.original.resize((512,256)) #resize image
            self.img  = ImageTk.PhotoImage(self.original)
            self.canvas.itemconfig(self.image_on_canvas1, image = self.img)
            #tkinter.messagebox.showinfo("Please Wait", "30sec: \n")

        def refreshButton(self):
            #self.create_window()
            #print(self.modfiles)
            #filenums = [int(x) for x in self.modfiles]
            #print(filenums)
            self.progress_bar["maximum"] = self.timepoints
            #read_path = 'D:/Segmentation_project/GUI/xy16c1/xy16c1_indv_preds/'        
            #save_path = 'D:/Segmentation_project/GUI/xy16c1/traps/'  
            #convert2tif(SAVE_DIR,TRAP_SAVE)
            #createFolder(self.save_path)
            C2dir = self.directory+self.tiffname[:-2]+'2/intensity/'
            C3dir = self.directory+self.tiffname[:-2]+'2/intensity/'
            MMaskDir = self.directory+self.tiffname +'mother_masks/'
            DMaskDir = self.directory+self.tiffname +'daughter_masks/'
            ids = sorted(next(os.walk(self.read_path))[2])
            ids2 = sorted(next(os.walk(C2dir))[2])
            ids3 = sorted(next(os.walk(C3dir))[2])
            idsM = sorted(next(os.walk(MMaskDir))[2])
            idsD = sorted(next(os.walk(DMaskDir))[2])
            allintensityM0 = []
            allintensityD0 = []
            allintensityM1 = []
            allintensityD1 = []
            allintensityM2 = []
            allintensityD2 = []
            allintensityM3 = []
            allintensityD3 = []
            allintensityM4 = []
            allintensityD4 = []
            allintensityM5 = []
            allintensityD5 = []
            #height_old = 104
            #width_old = 52
            for n in tqdm(range(self.timepoints)):    
                im1 = skimage.io.imread(self.read_path+ids[6*n+0])
                im2 = skimage.io.imread(self.read_path+ids[6*n+1])
                im3 = skimage.io.imread(self.read_path+ids[6*n+2])
                im4 = skimage.io.imread(self.read_path+ids[6*n+3])
                im5 = skimage.io.imread(self.read_path+ids[6*n+4])
                im6 = skimage.io.imread(self.read_path+ids[6*n+5])
                #im1 = resize(im1[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
                #im2 = resize(im2[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
                #im3 = resize(im3[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
                #im4=  resize(im4[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
                #im5 = resize(im5[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)
                #im6 = resize(im6[:,64:192,:], (height_old, width_old), mode='constant', preserve_range=True)

                blank_image = np.zeros((256,512,3), np.uint16)
                if not 1 in self.rtlist: 
                    if n>= int(self.Svals['Trap1'])-1 and n<=int(self.Evals['Trap1'])-1:
                        blank_image[68:196,18:82,:] = im1
                        C2_0 = C2dir+ids2[6*n+0]
                        C3_0 = C3dir+ids3[6*n+0]
                        maskm_0 = MMaskDir+idsM[6*n+0]
                        maskd_0 = DMaskDir+idsD[6*n+0]
                        m0,d0 =generate_intensity_vals(maskm_0,maskd_0,C2_0,C3_0)
                        allintensityM0.append(m0)
                        allintensityD0.append(d0)
                if not 2 in self.rtlist: 
                    if n>= int(self.Svals['Trap2'])-1 and n<=int(self.Evals['Trap2'])-1:
                        blank_image[68:196,101:165,:] = im2
                        C2_1 = C2dir+ids2[6*n+1]
                        C3_1 = C3dir+ids3[6*n+1]
                        maskm_1 = MMaskDir+idsM[6*n+1]
                        maskd_1 = DMaskDir+idsD[6*n+1]
                        m1,d1 =generate_intensity_vals(maskm_1,maskd_1,C2_1,C3_1)
                        allintensityM1.append(m1)
                        allintensityD1.append(d1)                    
                if not 3 in self.rtlist: 
                    if n>= int(self.Svals['Trap3'])-1 and n<=int(self.Evals['Trap3'])-1:
                        blank_image[68:196,183:247,:] = im3
                        C2_2 = C2dir+ids2[6*n+2]
                        C3_2 = C3dir+ids3[6*n+2]
                        maskm_2 = MMaskDir+idsM[6*n+2]
                        maskd_2 = DMaskDir+idsD[6*n+2]
                        m2,d2 =generate_intensity_vals(maskm_2,maskd_2,C2_2,C3_2)
                        allintensityM2.append(m2)
                        allintensityD2.append(d2)

                if not 4 in self.rtlist:
                    if n>= int(self.Svals['Trap4'])-1 and n<=int(self.Evals['Trap4'])-1:
                        blank_image[68:196,266:330,:] = im4
                        C2_3 = C2dir+ids2[6*n+3]
                        C3_3 = C3dir+ids3[6*n+3]
                        maskm_3 = MMaskDir+idsM[6*n+3]
                        maskd_3 = DMaskDir+idsD[6*n+3]
                        m3,d3 =generate_intensity_vals(maskm_3,maskd_3,C2_3,C3_3)
                        allintensityM3.append(m3)
                        allintensityD3.append(d3)


                if not 5 in self.rtlist: 
                    if n>= int(self.Svals['Trap5'])-1 and n<=int(self.Evals['Trap5'])-1:
                        blank_image[68:196,348:412,:] = im5
                        C2_4 = C2dir+ids2[6*n+4]
                        C3_4 = C3dir+ids3[6*n+4]
                        maskm_4 = MMaskDir+idsM[6*n+4]
                        maskd_4 = DMaskDir+idsD[6*n+4]
                        m4,d4 =generate_intensity_vals(maskm_4,maskd_4,C2_4,C3_4)
                        allintensityM4.append(m4)
                        allintensityD4.append(d4)


                if not 6 in self.rtlist: 
                    if n>= int(self.Svals['Trap6'])-1 and n<=int(self.Evals['Trap6'])-1:
                        blank_image[68:196,430:494,:] = im6
                        C2_5 = C2dir+ids2[6*n+5]
                        C3_5 = C3dir+ids3[6*n+5]
                        maskm_5 = MMaskDir+idsM[6*n+5]
                        maskd_5 = DMaskDir+idsD[6*n+5]
                        m5,d5 =generate_intensity_vals(maskm_5,maskd_5,C2_5,C3_5)
                        allintensityM5.append(m5)
                        allintensityD5.append(d5)


                cv2.imwrite(self.save_path+ids[n*6][0:10] +'.tif', blank_image)  


                self.progress_bar["value"] = n
                self.progress_bar.update()            

            #cell_cycle_counts = find_cell_cycles('D:/Segmentation_project/GUI/xy16c1/xy16c1_daughter_areas.txt')
            cell_cycle_counts = find_cell_cycles(self.directory+self.tiffname+'daughter_areas.txt')
            filter_cell_cycle_counts(cell_cycle_counts)
            for i,field in enumerate(self.fields):
                cell_cycle_counts[i].insert(0,int(self.Evals[field]))
                cell_cycle_counts[i].insert(0,int(self.Svals[field]))




            #print(cell_cycle_counts)
            #np.savetxt('D:/Segmentation_project/GUI/xy16c1/xy16c1_cycle_counts.txt', cell_cycle_counts, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'cell_cycle_counts.txt',cell_cycle_counts, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityM0.txt',allintensityM0, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityM1.txt',allintensityM1, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityM2.txt',allintensityM2, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityM3.txt',allintensityM3, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityM4.txt',allintensityM4, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityM5.txt',allintensityM5, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityD0.txt',allintensityD0, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityD1.txt',allintensityD1, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityD2.txt',allintensityD2, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityD3.txt',allintensityD3, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityD4.txt',allintensityD4, fmt='%s')
            np.savetxt(self.directory+self.tiffname+'IntensityD5.txt',allintensityD5, fmt='%s')

            self.original = Image.open(self.File)
            #self.original = self.original.resize((512,256)) #resize image
            self.img  = ImageTk.PhotoImage(self.original)
            self.canvas.itemconfig(self.image_on_canvas1, image = self.img)
            #tkinter.messagebox.showinfo("Please Wait", "30sec: \n")
            self.progress_bar["value"] = 0
            self.progress_bar.update()            
    root = Tk()
    MainWindow(root,directory,tiffname)
    root.mainloop()
