import torch


def datapath(dataset,split,save_file):

    if dataset == "50salads":
        sample_rate = 2
    else:
        sample_rate = 1

    train_vid_list = "./data/"+dataset+"/splits/train.split"+split+".bundle"
    test_vid_list = "./data/"+dataset+"/splits/test.split"+split+".bundle"
    features_path = "./data/"+dataset+"/features/"
    gt_path = "./data/"+dataset+"/groundTruth/"

    class_file = "./data/"+dataset+"/mapping.txt"

    weights_dir = "./weights/"+dataset+"/"+save_file+"/split_"+split
    results_dir = "./results/"+dataset+"/"+save_file+"/split_"+split
    runs_dir = "./runs/"+dataset+"/"+save_file+"/split_"+split

    return sample_rate, train_vid_list, test_vid_list, features_path, gt_path, class_file, weights_dir, results_dir, runs_dir


def evalpath(dataset,split,save_file):

    gt_path = "./data/"+dataset+"/groundTruth/"
    results_dir = "./results/"+dataset+"/"+save_file+"/split_"+split+"/"
    scores_dir = "./scores/"+dataset+"/"+save_file
    test_vid_list = "./data/"+dataset+"/splits/test.split"+split+".bundle"

    return gt_path, results_dir, scores_dir, test_vid_list


def visualizepath(dataset,split,save_file):

    if dataset == "50salads":
        sample_rate = 2
    else:
        sample_rate = 1

    test_vid_list = "./data/"+dataset+"/splits/test.split"+split+".bundle"
    features_path = "./data/"+dataset+"/features/"
    images_path = "./data/"+dataset+"/img/"

    class_file = "./data/"+dataset+"/mapping.txt"

    weights_dir = "./weights/"+dataset+"/"+save_file+"/split_"+split
    results_dir = "./results/"+dataset+"/"+save_file+"/split_"+split

    return sample_rate, test_vid_list, features_path, class_file, images_path, weights_dir, results_dir 


