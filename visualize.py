import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import argparse
import yaml
import random

from addict import Dict
from models import visualize_model , network
from utils import data, util, dataset


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('split', type=str, help='1〜4 (50salads 1〜5)')
    parser.add_argument('--no_cuda', action = "store_true", help='disable cuda')
    parser.add_argument('--device', type=str, default=0, help='choose device')

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
    sample_rate, test_vid_list, features_path, class_file, images_path, weights_dir, results_dir = data.visualizepath(SETTING.dataset,args.split,args.arg)

    # action label
    file_ptr = open(class_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    num_classes = len(actions_dict)

    # model
    if SETTING.model == "attention_multistage":
        test_model = network.Attention_MultiStageTCN(SETTING, num_classes)
    test_model.load_state_dict(torch.load(weights_dir + "/epoch-" + str(SETTING.num_epochs) + ".model"))
    net = visualize_model.Attention_Visualize(SETTING, test_model)

    # visualize
    net.eval()
    with torch.no_grad():
        net.to(device)

        file_ptr = open(test_vid_list, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            print(vid)
            f_name = vid.split('.')[0]
            features = np.load(features_path + f_name + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x = input_x.to(device)
            output = net(input_x)
            output=output.cpu()
            attention=output.detach()
            for f_num in range(attention.size(0)):
               image = dataset.Image_dset(images_path, f_name, f_num)
               img=util.imshape(image)
               heatmap = attention[f_num,:,:,:]
               util.make_attention_map(img, heatmap, results_dir, f_name, f_num)
               print("saved {}.png".format(str(f_num).zfill(6)))




if __name__ == '__main__':
    main()