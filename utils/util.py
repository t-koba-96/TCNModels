
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
import scipy.ndimage.filters as fi
import cv2
import os


def imshape(image):
     image=image/2+0.5
     npimg=image.numpy()
  
     return np.transpose(npimg,(1,2,0))


def imshow(img):
     img=img/2+0.5
     npimg=img.numpy()
     plt.imshow(np.transpose(npimg,(1,2,0)))
     plt.show


def one_hot_2d(labels,class_num):
     y = labels.view(labels.size(0),-1)
     y_onehot = torch.FloatTensor(labels.size(0),class_num)
     y_onehot.zero_()
     y_onehot.scatter_(1, y, 1)
   
     return y_onehot


def one_hot_3d(labels,class_num):
     y = labels.view(labels.size(0),labels.size(1),-1)
     y_onehot = torch.FloatTensor(labels.size(0),labels.size(1),class_num)
     y_onehot.zero_()
     y_onehot.scatter_(2, y, 1)
   
     return y_onehot


def normalize_heatmap(x):
    # choose min (0 or smallest scalar)
     min = x.min()
     max = x.max()
     result = (x-min)/(max-min)
    
     return result

def crop_center(img,cropx,cropy):
     y,x = img.shape
     startx = x//2 - cropx//2
     starty = y//2 - cropy//2    
 
     return img[starty:starty+cropy, startx:startx+cropx]

def delete_line():
     ax = plt.gca() # get current axis
     ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
     ax.tick_params(labelleft="off",left="off") # y軸の削除
     plt.gca().spines['right'].set_visible(False)
     plt.gca().spines['top'].set_visible(False)
     plt.gca().spines['left'].set_visible(False)
     plt.gca().spines['bottom'].set_visible(False)
     ax.set_xticklabels([]) 
    

def make_hand_image(img,l_img,r_img,pose,video_num,f_num,cutout_size):
     fig=plt.figure(figsize=(16,10))
     gs = GridSpec(4,5,left=0.06,right=1.2) 
     gs.update(wspace=0.2)
     gs_1 = GridSpecFromSubplotSpec(nrows=4, ncols=3, subplot_spec=gs[0:4, 0:3])
     fig.add_subplot(gs_1[:, :])
     ax = plt.axes()
     r = patches.Rectangle(xy=(int((pose[0])*224/1080)-87-int(cutout_size*224/540), int((pose[1])*224/1080)-int(cutout_size*224/540)), 
                                 width=int(cutout_size*2*224/540), height=int(cutout_size*2*224/540), ec='#AD13E5',linewidth='4.0', fill=False)
     l = patches.Rectangle(xy=(int((pose[2])*224/1080)-87-int(cutout_size*224/540), int((pose[3])*224/1080)-int(cutout_size*224/540)),
                                 width=int(cutout_size*2*224/540), height=int(cutout_size*2*224/540), ec='#AD13E5',linewidth='4.0', fill=False)
     ax.add_patch(r)
     ax.add_patch(l)
     delete_line()
     plt.imshow(img)
     gs_2 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[1:3, 0:1])
     fig.add_subplot(gs_2[:,:])
     delete_line()
     plt.imshow(l_img)
     gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[1:3, 3:4])
     fig.add_subplot(gs_3[:,:])
     delete_line()
     plt.imshow(r_img)

     # Make the directory if it doesn't exist.
     SAVE_PATH = "../../../demo/images/hand"
     if not os.path.exists(os.path.join(SAVE_PATH,"hand_"+video_num)):
         os.makedirs(os.path.join(SAVE_PATH,"hand_"+video_num))
      
     plt.savefig(os.path.join(SAVE_PATH,"hand_"+video_num,str(f_num).zfill(5)+".png"))
     plt.close()


def show_posemap(testloader,video_num,img_size,gauss_size):
     for i,data in enumerate(testloader):
         images,targets,labels,poses=data
         images=images.view(-1,3,images.size(3),images.size(4))
         poses=poses.view(-1,poses.size(2))     
         f_num=images.size(0)*i
         for x in range(images.size(0)):
             img=imshape(images[x,:,:,:])
             pose=poses[x,:]
             if gauss_size==61:
                make_posemap(img,pose,video_num,img_size,f_num,gauss_size)
             else:     
                make_handmap(img,pose,video_num,img_size,f_num,gauss_size)
             f_num+=1

def make_handmap(img,pose,video_num,img_size,f_num,gauss_size):
     posemap = hand_map(pose,img_size,gauss_size)
     posemap = np.uint8(255*posemap)
     posemap = cv2.applyColorMap(posemap, cv2.COLORMAP_JET)
     posemap = cv2.cvtColor(posemap, cv2.COLOR_BGR2RGB)
     posemap=posemap/255
     s_img = posemap * 1.0 + img
    
     fig=plt.figure(figsize=(16,10))
     gs = GridSpec(4,5,left=0.13,right=0.9) 
     gs.update(wspace=-0.01)
     gs_1 = GridSpecFromSubplotSpec(nrows=4, ncols=3, subplot_spec=gs[0:4, 0:3])
     fig.add_subplot(gs_1[:, :])
     delete_line()
     plt.imshow(s_img)
     gs_2 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[0:2, 3:5])
     fig.add_subplot(gs_2[:,:])
     delete_line()
     plt.imshow(img)
     gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[2:4, 3:5])
     fig.add_subplot(gs_3[:,:])
     delete_line()
     plt.imshow(posemap,cmap='jet')
     plt.clim(0,1)
     plt.colorbar()  
     SAVE_PATH = "../../../demo/images/handmap"
     if not os.path.exists(os.path.join(SAVE_PATH,"handmap_"+video_num)):
         os.makedirs(os.path.join(SAVE_PATH,"handmap_"+video_num))
      
     plt.savefig(os.path.join(SAVE_PATH,"handmap_"+video_num,str(f_num).zfill(5)+".png"))
     plt.close()

def make_posemap(img,pose,video_num,img_size,f_num,gauss_size):
     posemap_all = pose_map(pose,img_size,gauss_size)
     posemaps = []
     for i in range(6):
           posemap = posemap_all[i,:,:]     
           posemap = np.uint8(255*posemap)
           posemap = cv2.applyColorMap(posemap, cv2.COLORMAP_JET)
           posemap = cv2.cvtColor(posemap, cv2.COLOR_BGR2RGB)
           posemap=posemap/255
           posemaps.append(posemap)
     s_img = posemaps[0] * 1.0 + posemaps[1] * 1.0 + posemaps[2] * 1.0 + posemaps[3] * 1.0 + posemaps[4] * 1.0 + posemaps[5] * 1.0 + img
    
     fig=plt.figure(figsize=(16,10))
     gs = GridSpec(6,7,left=0.13,right=0.9) 
     gs.update(wspace=-0.01)
     gs_1 = GridSpecFromSubplotSpec(nrows=4, ncols=3, subplot_spec=gs[0:3, 0:3])
     fig.add_subplot(gs_1[:, :])
     delete_line()
     plt.imshow(img)
     gs_2 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[3:6, 0:3])
     fig.add_subplot(gs_2[:,:])
     delete_line()
     plt.imshow(s_img)
     gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[4:6, 3:5])
     fig.add_subplot(gs_3[:,:])
     delete_line()
     plt.imshow(posemaps[0],cmap='jet')
     plt.clim(0,1)
     plt.colorbar()  
     gs_4 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[4:6, 5:7])
     fig.add_subplot(gs_4[:,:])
     delete_line()
     plt.imshow(posemaps[1],cmap='jet')
     plt.clim(0,1)
     plt.colorbar() 
     gs_5 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[2:4, 3:5])
     fig.add_subplot(gs_5[:,:])
     delete_line()
     plt.imshow(posemaps[2],cmap='jet')
     plt.clim(0,1)
     plt.colorbar()  
     gs_6 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[2:4, 5:7])
     fig.add_subplot(gs_6[:,:])
     delete_line()
     plt.imshow(posemaps[3],cmap='jet')
     plt.clim(0,1)
     plt.colorbar() 
     gs_7 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[0:2, 3:5])
     fig.add_subplot(gs_7[:,:])
     delete_line()
     plt.imshow(posemaps[4],cmap='jet')
     plt.clim(0,1)
     plt.colorbar()  
     gs_8 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[0:2, 5:7])
     fig.add_subplot(gs_8[:,:])
     delete_line()
     plt.imshow(posemaps[5],cmap='jet')
     plt.clim(0,1)
     plt.colorbar() 
     SAVE_PATH = "../../../demo/images/posemap"
     if not os.path.exists(os.path.join(SAVE_PATH,"posemap_"+video_num)):
         os.makedirs(os.path.join(SAVE_PATH,"posemap_"+video_num))
      
     plt.savefig(os.path.join(SAVE_PATH,"posemap_"+video_num,str(f_num).zfill(5)+".png"))
     plt.close()

def hand_map(poses,img_size,gauss_size):
     posemap = torch.zeros(1080,1920)
     poses = poses.type(torch.LongTensor)
     posemap[poses[1]-5:poses[1]+5, poses[0]-5:poses[0]+5]=+1
     posemap[poses[3]-5:poses[3]+5, poses[2]-5:poses[2]+5]=+1
     posemap = posemap.numpy()
     #posemap=fi.gaussian_filter(posemap, gauss_size)
     posemap = cv2.resize(posemap, (int(img_size*1920/1080),img_size))
     posemap = crop_center(posemap,img_size,img_size)
     posemap = cv2.GaussianBlur(posemap,(gauss_size,gauss_size),0)
     posemap = normalize_heatmap(posemap)

     return posemap

def pose_map(poses,img_size,gauss_size):
     posemaps = []
     poses = poses.type(torch.LongTensor)
     for i in range(6):
           posemap = torch.zeros(1080,1920)
           posemap[poses[2*i+1]-5:poses[2*i+1]+5, poses[2*i]-5:poses[2*i]+5]=+1
           posemap = posemap.numpy()
           #posemap=fi.gaussian_filter(posemap, gauss_size)
           posemap = cv2.resize(posemap, (int(img_size*1920/1080),img_size))
           posemap = crop_center(posemap,img_size,img_size)
           posemap = cv2.GaussianBlur(posemap,(gauss_size,gauss_size),0)
           posemap = normalize_heatmap(posemap)
           posemaps.append(posemap)
     posemaps = np.stack(posemaps, 0)

     return posemaps


#masking for mseloss
def masking(labels,class_num):
    mask = torch.ones(labels.size(0),labels.size(1)-1,class_num)
    for i in range(len(labels)):
        for j in range(len(labels[0])-1):   
            if labels[i,j] == 8:
                mask[i,j,6] -= 1
                mask[i,j,8] -= 1
                mask[i,j,9] -= 1
         
            elif labels[i,j] == 10:
                mask[i,j,0] -= 1
                mask[i,j,10] -= 1
        
            else:
                num = labels[i,j]
                mask[i,j,num] -= 1
                mask[i,j,num+1] -= 1
        
    return mask


#masking for outputs
def masking_outputs(outputs,class_num):
    _,labels=torch.max(outputs,dim=2)
    mask = torch.ones(labels.size(0),labels.size(1)-1,class_num)
    for i in range(len(labels)):
        for j in range(len(labels[0])-1):   
            if labels[i,j] == 8:
                mask[i,j,6] -= 1
                mask[i,j,8] -= 1
                mask[i,j,9] -= 1
         
            elif labels[i,j] == 10:
                mask[i,j,0] -= 1
                mask[i,j,10] -= 1
        
            else:
                num = labels[i,j]
                mask[i,j,num] -= 1
                mask[i,j,num+1] -= 1
        
    return mask