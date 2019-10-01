import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import argparse
import yaml
import random

from addict import Dict
from models import network
from utils import dataset,training,data


'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('mode', type=str, help='train or test')
    parser.add_argument('split', type=str, help='1〜4 (50salads 1〜5)')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')

    return parser.parse_args()



def main():

    args = get_arguments()
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))
    device = torch.device(args.device)
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    sample_rate, train_vid_list, test_vid_list, features_path, gt_path, class_file, weights_dir, results_dir, runs_dir = data.datapath(SETTING.dataset,args.split,SETTING.save_file)
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_ptr = open(class_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    num_classes = len(actions_dict)

    if SETTING.model == "multistage":
        net = network.MultiStageTCN(SETTING.num_stages, SETTING.num_layers, SETTING.num_f_maps, SETTING.features_dim, num_classes)
        net = nn.DataParallel(net)

    elif SETTING.model == "singlestage":
        net = network.SingleStageTCN(SETTING.num_layers, SETTING.num_f_maps, SETTING.features_dim, num_classes)
        net = nn.DataParallel(net)


    trainer = training.Trainer(net,num_classes)

    if args.mode == "train":
        trainloader = dataset.Dataset(num_classes, actions_dict, gt_path, features_path, sample_rate)
        trainloader.read_data(train_vid_list)
        trainer.train(weights_dir, runs_dir, trainloader, SETTING.num_epochs, SETTING.bz, SETTING.lr, device)

    elif args.mode == "test":
        trainer.test(weights_dir, results_dir, features_path, test_vid_list, SETTING.num_epochs, actions_dict, device, sample_rate)


if __name__ == '__main__':
    main()