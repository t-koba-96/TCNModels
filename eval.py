import os
import argparse
import yaml
from utils import data,testing
from addict import Dict


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('split', type=str, help='1〜4 (50salads 1〜5)')

    return parser.parse_args()


def main():
   
    args = get_arguments()
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))

    gt_path, results_dir, test_vid_list = data.evalpath(SETTING.dataset,args.split,SETTING.save_file)

    testing.evaluator(gt_path, results_dir, test_vid_list)


if __name__ == '__main__':
    main()
