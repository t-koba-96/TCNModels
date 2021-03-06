import numpy as np
import pandas as pd
import os

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        # gtとpredictedの和集合
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        # gtとpredictedの共通部分
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        # GTと同一のlabel部分においてそれぞれIOUを計算
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def evaluator(gt_path, results_dir, scores_dir, test_vid_list, split):
    
    list_of_videos = read_file(test_vid_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = gt_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = results_dir + vid
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
    
    # output results
    
    f_ptr = open(scores_dir + "/" + "split_" + split + ".txt", "w")
            
    print("Acc: %.4f" % (100*float(correct)/total))
    f_ptr.write("Acc: %.4f" % (100*float(correct)/total))
    f_ptr.write("\n")
    print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    f_ptr.write('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    f_ptr.write("\n")

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print ('F1@%0.2f: %.4f' % (overlap[s], f1))
        f_ptr.write('F1@%0.2f: %.4f' % (overlap[s], f1))
        f_ptr.write("\n")

    f_ptr.close()

    return

def demo(gt_path, results_dir, test_vid_list):
    
    list_of_videos = read_file(test_vid_list).split('\n')[:-1]

    num=[]
    Frame=[]
    gt_label=[]
    pr_label=[]

    for vid in list_of_videos:
        gt_file = gt_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = results_dir + vid
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            num.append(i+1)
            Frame.append("%s.png" % str(i).zfill(5))
            gt_label.append(gt_content[i])
            pr_label.append(recog_content[i])

        df = pd.DataFrame({ 'number' : num,
                            'image' : Frame,
                            'gt' : gt_label,
                            'predict' : pr_label
                            })
        f_name = vid.split('/')[-1].split('.')[0]
        df.to_csv(os.path.join(results_dir+"/"+f_name+".csv"))

