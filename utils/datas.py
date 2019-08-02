import torch


def train_path_list(number):

    video_list=[]
    l_cutout_list=[]
    r_cutout_list=[]
    label_list=[]
    pose_list=[]

    video_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/image_a/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/image_b/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/image_c/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/image_d/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/image_e/train/"]

    left_cutout_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/cutout_image_a/left/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_b/cutout_image_b/left/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_c/cutout_image_c/left/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_d/cutout_image_d/left/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_e/cutout_image_e/left/train/"]
   
    right_cutout_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/cutout_image_a/right/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_b/cutout_image_b/right/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_c/cutout_image_c/right/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_d/cutout_image_d/right/train/",
                           "../../../local/dataset/work_detect/mogi_data/worker_e/cutout_image_e/right/train/"]

    label_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/class_a/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/class_b/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/class_c/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/class_d/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/class_e/train.csv"]

    pose_path_list=[ "../../../local/dataset/work_detect/mogi_data/worker_a/pose_a/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/pose_b/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/pose_c/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/pose_d/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/pose_e/train.csv"]
    
    for data in number:
       video_list.append(video_path_list[data-1])
       l_cutout_list.append(left_cutout_path_list[data-1])
       r_cutout_list.append(right_cutout_path_list[data-1])
       label_list.append(label_path_list[data-1])
       pose_list.append(pose_path_list[data-1])

    return video_list,l_cutout_list,r_cutout_list,label_list,pose_list


def test_path_list(number):

    video_list=[]
    l_cutout_list=[]
    r_cutout_list=[]
    label_list=[]
    pose_list=[]

    video_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/image_a/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/image_b/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/image_c/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/image_d/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/image_e/test/"]

    left_cutout_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/cutout_image_a/left/test/",
                           "../../../local/dataset/work_detect/mogi_data/worker_b/cutout_image_b/left/test/",
                           "../../../local/dataset/work_detect/mogi_data/worker_c/cutout_image_c/left/test/",
                           "../../../local/dataset/work_detect/mogi_data/worker_d/cutout_image_d/left/test/",
                           "../../../local/dataset/work_detect/mogi_data/worker_e/cutout_image_e/left/test/"]

    right_cutout_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/cutout_image_a/right/test/",
                            "../../../local/dataset/work_detect/mogi_data/worker_b/cutout_image_b/right/test/",
                            "../../../local/dataset/work_detect/mogi_data/worker_c/cutout_image_c/right/test/",
                            "../../../local/dataset/work_detect/mogi_data/worker_d/cutout_image_d/right/test/",
                            "../../../local/dataset/work_detect/mogi_data/worker_e/cutout_image_e/right/test/"]

    label_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/class_a/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/class_b/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/class_c/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/class_d/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/class_e/test.csv"]

    pose_path_list=[ "../../../local/dataset/work_detect/mogi_data/worker_a/pose_a/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/pose_b/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/pose_c/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/pose_d/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/pose_e/test.csv"]

    for data in number:
       video_list.append(video_path_list[data-1])
       l_cutout_list.append(left_cutout_path_list[data-1])
       r_cutout_list.append(right_cutout_path_list[data-1])
       label_list.append(label_path_list[data-1])
       pose_list.append(pose_path_list[data-1])

    return video_list,l_cutout_list,r_cutout_list,label_list,pose_list

def class_list():
    class_list=["n","tp","al","ac","ca","gd","as","hp","st","ch","co"]

    return class_list
    