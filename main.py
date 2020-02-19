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


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('mode', type=str, help='train or test')
    parser.add_argument('split', type=str, help='1〜4 (50salads 1〜5)')
    parser.add_argument('--no_cuda', action = "store_true", help='disable cuda')
    parser.add_argument('--device', type=str, default=0, help='choose device')
    parser.add_argument('--dataparallel', action = "store_true", help='use data parallel')

    return parser.parse_args()



def main():
    # config
    args = get_arguments()
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))
    print(args)
    if len(args.device) > 1: 
        args.device = list (map(str,args.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.device)
    print("using gpu number {}".format(",".join(args.device)))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # get dataset
    sample_rate, train_vid_list, test_vid_list, features_path, gt_path, class_file, weights_dir, results_dir, runs_dir = data.datapath(SETTING.dataset,args.split,args.arg)

    # action label
    file_ptr = open(class_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    num_classes = len(actions_dict)

    # model
    if SETTING.model == "multistage":
        net = network.MultiStageTCN(SETTING, num_classes)
    elif SETTING.model == "singlestage":
        net = network.SingleStageTCN(SETTING, num_classes)
    elif SETTING.model == "attention_multistage":
        net = network.Attention_MultiStageTCN(SETTING, num_classes)

    # dataparallel
    if args.dataparallel:
        print("Using Multiple GPU . . . ")
        net = nn.DataParallel(net)


    # trainer 
    trainer = training.Trainer(net,num_classes)

    # train
    if args.mode == "train":
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        trainloader = dataset.Dataset(num_classes, actions_dict, gt_path, features_path, sample_rate)
        trainloader.read_data(train_vid_list)
        trainer.train(weights_dir, runs_dir, trainloader,args, SETTING, device)

    # eval
    elif args.mode == "test":
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        trainer.test(weights_dir, results_dir, features_path, test_vid_list, args, SETTING.num_epochs, actions_dict, device, sample_rate)


if __name__ == '__main__':
    main()