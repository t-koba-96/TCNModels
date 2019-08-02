import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import datas
import argparse
import yaml

from addict import Dict
from models import network
from utils import util,dataset,loader,loss,train


'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')

    return parser.parse_args()



def main():

     args = get_arguments()
     SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))
     device = torch.device(args.device)
     video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.train_path_list(SETTING.train_video_list)
     test_video_path_list,test_left_cutout_path_list,test_right_cutout_path_list,test_label_path_list,test_pose_path_list = datas.test_path_list(SETTING.test_video_list)
     if SETTING.model == 'Cutout_TCN':
         frameloader = dataset.Video(video_path_list,left_cutout_path_list,right_cutout_path_list,
                                             label_path_list,pose_path_list,
                                             SETTING.image_size,SETTING.clip_length,
                                             SETTING.slide_stride,SETTING.classes,
                                             cutout_img=True,pose_label=True)
         test_frameloader = dataset.Video(test_video_path_list,test_left_cutout_path_list,test_right_cutout_path_list,
                                             test_label_path_list,test_pose_path_list,
                                             SETTING.image_size,SETTING.clip_length,
                                             SETTING.clip_length,SETTING.classes,
                                             cutout_img=True,pose_label=True)

     else:
         frameloader = dataset.Video(video_path_list,left_cutout_path_list,right_cutout_path_list,
                                     label_path_list,pose_path_list,
                                     SETTING.image_size,SETTING.clip_length,
                                     SETTING.slide_stride,SETTING.classes,pose_label=True)
         test_frameloader = dataset.Video(test_video_path_list,test_left_cutout_path_list,test_right_cutout_path_list,
                                          test_label_path_list,test_pose_path_list,
                                          SETTING.image_size,SETTING.clip_length,
                                          SETTING.clip_length,SETTING.classes,pose_label=True)
         
         
     trainloader = torch.utils.data.DataLoader(frameloader,batch_size=SETTING.batch_size,
                                                     shuffle=True,num_workers=SETTING.num_workers)
     testloader = torch.utils.data.DataLoader(test_frameloader,batch_size=SETTING.batch_size,
                                                     shuffle=False,num_workers=SETTING.num_workers)

     #weights = torch.tensor([0.35 , 0.05, 0.04, 0.04, 0.05, 0.35, 0.01, 0.05, 0.02, 0.05, 0.05]).cuda()
     CE_criterion=nn.CrossEntropyLoss()
     MSE_criterion=nn.MSELoss(reduction='none')

     if SETTING.model == 'Twostream_TCN':
         net = network.twostream_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,testloader,net,CE_criterion,MSE_criterion,optimizer,device,SETTING.epoch,SETTING.save_file,two_stream=True)

     elif SETTING.model == 'Cutout_TCN':
         net = network.cutout_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,testloader,net,CE_criterion,MSE_criterion,optimizer,device,SETTING.epoch,SETTING.save_file,cutout_img=True)

     elif SETTING.model == 'Handmap_TCN':
         net = network.handmap_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,testloader,net,CE_criterion,MSE_criterion,optimizer,device,SETTING.epoch,SETTING.save_file,posemap=True)

     elif SETTING.model == 'Posemap_TCN':
         net = network.posemap_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,testloader,net,CE_criterion,MSE_criterion,optimizer,device,SETTING.epoch,SETTING.save_file,two_stream=True,posemap=True)

     else:
         if SETTING.model == 'VGG_TCN':
             net = network.vgg_tcn(SETTING.classes)
         elif SETTING.model == 'Attention_TCN':
             net = network.attention_tcn(SETTING.classes)
         elif SETTING.model == 'Resnet_TCN':
             net = network.resnet_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,testloader,net,CE_criterion,MSE_criterion,optimizer,device,SETTING.epoch,SETTING.save_file)

if __name__ == '__main__':
    main()