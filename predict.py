import torch
import torch.nn as nn

import numpy as np
import datas
import os
import argparse
import yaml

from addict import Dict
from models import network
from utils import dataset,loader,test,util



'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('video' , type=str, help='which video using for test, use alphabet(1=a,2=b,3=c ....)')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')

    return parser.parse_args()



def main():

     args = get_arguments()
     SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))
     device=torch.device(args.device)
     classes=datas.class_list()

     test_video_list=[]
     if args.video == 'a':
         test_video_list.append(1)
     elif args.video == 'b':
         test_video_list.append(2)
     elif args.video == 'c':
         test_video_list.append(3)
     elif args.video == 'd':
         test_video_list.append(4)
     elif args.video == 'e':
         test_video_list.append(5)
     video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.test_path_list(test_video_list)

     if SETTING.model == 'Cutout_TCN':
         frameloader = dataset.Video(video_path_list,left_cutout_path_list,right_cutout_path_list,
                                     label_path_list,pose_path_list,
                                     SETTING.image_size,SETTING.clip_length,
                                     SETTING.clip_length,SETTING.classes,
                                     cutout_img=True,pose_label=True)

     else:
         frameloader = dataset.Video(video_path_list,left_cutout_path_list,right_cutout_path_list,
                                     label_path_list,pose_path_list,
                                     SETTING.image_size,SETTING.clip_length,
                                     SETTING.clip_length,SETTING.classes,pose_label=True)

   
     testloader = torch.utils.data.DataLoader(frameloader,batch_size=SETTING.batch_size,
                                             shuffle=False,num_workers=SETTING.num_workers)

     
     if SETTING.model == 'Twostream_TCN':
         net = network.twostream_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,two_stream=True)
         test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=True)
         
     elif SETTING.model == 'Handmap_TCN':
         net = network.handmap_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,posemap=True)
         test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length,posemap=True)

     elif SETTING.model == 'Posemap_TCN':
         net = network.posemap_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,two_stream=True,posemap=True)
         test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=True,posemap=True)


     elif SETTING.model == 'Cutout_TCN':
         net = network.cutout_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,cutout_img=True)
         test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length,cutout_img=True)
    
     else:
         if SETTING.model == 'VGG_TCN':
             net = network.vgg_tcn(SETTING.classes)
         elif SETTING.model == 'Attention_TCN':
             net = network.attention_tcn(SETTING.classes)
         elif SETTING.model == 'Resnet_TCN':
             net = network.resnet_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file)
         test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length)


if __name__ == '__main__':
    main()