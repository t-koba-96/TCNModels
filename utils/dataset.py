import torch
import torch.utils.data as data
from PIL import Image
from . import util
import os
import math
import functools
import copy
import pandas as pd
import torchvision.transforms as transforms



#datasetlistに各idxにおける
#"video(ビデオのパス)、共通”
#"n_frames(全体のビデオ長）、共通"
#”segment（始まりと終わりのフレーム）、別"
#"frame_indices(要素のフレーム番号)、別"
#を記述
def make_dataset(video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list, sample_duration, step):
    dataset = []

    #動画のフレーム長
    sample = {
        #動画のパス
        'video': video_path_list[0],
        #ラベルのパス
        'label': label_path_list[0]
    }
    
    for num,v_path in enumerate(video_path_list):
        
        n_frames = len(os.listdir(v_path))

        for i in range(0,  (n_frames - sample_duration + 1) , step):
            sample_i = copy.deepcopy(sample)
            sample_i['video'] = v_path
            sample_i['left_cutout'] = left_cutout_path_list[num]
            sample_i['right_cutout'] = right_cutout_path_list[num]
            sample_i['label'] = label_path_list[num]
            sample_i['pose'] = pose_path_list[num]
            sample_i['frame_indices'] = list(range(i, i + sample_duration))
            sample_i['segment'] = torch.IntTensor([num, i, i + sample_duration - 1])
            dataset.append(sample_i)

    
    return dataset



#get_loaderとしてvideo_loaderを利用
def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


#動画内の指定した画像をロード
def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def label_loader(label_path,frame_indices):
    label = []
    for i in frame_indices:
        if os.path.exists(label_path):
            label.append(torch.as_tensor(pd.read_csv(label_path).iat[i,1]))
        else:
            return label

    return label

def pose_loader(pose_path,frame_indices):
    pose = []
    for i in frame_indices:
        if os.path.exists(pose_path):
            pose.append(torch.as_tensor(pd.read_csv(pose_path).iloc[i,0:12]).float())
        else:
            return pose

    return pose


#datasetを作るクラス
class Video(data.Dataset):
    def __init__(self, video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list,image_size,sample_duration,step,
                 class_num,one_hot_label=False,temporal_transform=False,get_loader=get_default_video_loader,cutout_img=False,pose_label=False):

        self.data = make_dataset(video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list, sample_duration, step)

        self.spatial_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])

        self.one_hot_label = one_hot_label
        self.pose_label = pose_label
        self.cutout_img = cutout_img
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

        self.cl_num=class_num

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        #各indexにおけるvideoのパス
        path = self.data[index]['video']

        #各indexにおける要素のリスト
        frame_indices = self.data[index]['frame_indices']

        #clip
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not False:
            clip = [self.spatial_transform(img) for img in clip]
        #(sequence,channel,height,width)
        clip = torch.stack(clip, 0)
        #(channel,sequence,height,width)にしたい場合
        #clip = torch.stack(clip, 0).permute(1,0,2,3)

        #target
        target = self.data[index]['segment']

        #label
        l_path = self.data[index]['label']
        label = label_loader(l_path,frame_indices)
        label = torch.stack(label,0)
        if self.one_hot_label is not False:
            label =  util.one_hot_2d(label,self.cl_num)

        #pose
        if self.pose_label is not False:
            p_path = self.data[index]['pose']
            pose = pose_loader(p_path,frame_indices)
            pose = torch.stack(pose,0)

        if self.cutout_img is not False:
            left_path = self.data[index]['left_cutout']
            left_clip = self.loader(left_path, frame_indices)
            if self.spatial_transform is not False:
                 left_clip = [self.spatial_transform(img) for img in left_clip]
            left_clip = torch.stack(left_clip, 0)
            right_path = self.data[index]['right_cutout']
            right_clip = self.loader(right_path, frame_indices)
            if self.spatial_transform is not False:
                 right_clip = [self.spatial_transform(img) for img in right_clip]
            right_clip = torch.stack(right_clip, 0)

            if self.pose_label is not False:
                 return clip,left_clip,right_clip,target,label,pose

            else:
    
                 return clip,left_clip,right_clip,target,label

        else:
             if self.pose_label is not False:
                 return clip,target,label,pose

             else:
                 return clip,target,label
        

    def __len__(self):
        return len(self.data)
