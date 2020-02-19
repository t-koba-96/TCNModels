import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
import scipy.ndimage.filters as fi
import cv2
import os



def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)

def imshape(image):
     image=image/2+0.5
     npimg=image.numpy()
  
     return np.transpose(npimg,(1,2,0))

def imshow(img):
     img=img/2+0.5
     npimg=img.numpy()
     plt.imshow(np.transpose(npimg,(1,2,0)))
     plt.show


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
def make_list(n):
     return [[] for i in range(n)]


     plt.gca().spines['right'].set_visible(False)
     plt.gca().spines['top'].set_visible(False)
     plt.gca().spines['left'].set_visible(False)
     plt.gca().spines['bottom'].set_visible(False)
     ax.set_xticklabels([]) 

     
def make_attention_map(img, heatmap,results_dir, f_name, f_num):
    #attention map
    heatmap = heatmap.numpy()
    heatmap = np.average(heatmap,axis=0)
    heatmap = normalize_heatmap(heatmap)
    # 元の画像と同じサイズになるようにヒートマップのサイズを変更
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap=heatmap/255
    # heatmap * image
    s_img = heatmap * 0.7 + img
    #plt
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
    plt.imshow(heatmap,cmap='jet')
    plt.clim(0,1)
    plt.colorbar()

    # make savedir
    savedir = os.path.join(results_dir, f_name)
    if not os.path.exists(savedir):
         os.makedirs(savedir)
      
    plt.savefig(os.path.join(savedir, str(f_num).zfill(6) + ".png"))
    plt.close()