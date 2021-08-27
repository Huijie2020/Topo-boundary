import numpy as np
from glob import glob
from PIL import Image
import json
import os
from tqdm import tqdm
import random

source_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_hog_9990patch_0817/1998gt_thr0_pre_from_iter0'
source_json_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_hog_9990patch_0817/1998gt_thr0_pre_from_iter0/metric_ave_4795'

source_pre_dir = os.path.join(source_dir, 'segmentation_overlap')
source_ske_dir = os.path.join(source_dir, 'skeleton_overlap')
source_json = os.path.join(source_dir, '12epoch_256_23976img.json')
tar_com_json = os.path.join(source_dir, '12epoch_256_23976img_combine.json')
tar_select_detail_json = os.path.join(source_json_dir, 'select_4795_detail.json')
tar_select_name_json = os.path.join(source_json_dir, 'select_4795_name.json')

with open(source_json,'r') as jf:
    data = json.load(jf)
jf.close()

overlap_id = data['overlap_id']
mse_all = data['mse']

n_select = int(len(overlap_id) * 0.2)
# progress_bar = tqdm(range(len(overlap_id)+1), ncols=150)

overlap = []
metric_list = []
for i in range(len(overlap_id)):
    file_dic = {}
    file_name = overlap_id[i]
    file_mse = mse_all[i]
    file_dic["file_name"] = file_name
    file_dic["mse"] = file_mse

    seg = Image.open(os.path.join(source_pre_dir, file_name+'.png'))
    seg_nd = np.array(seg)

    ske = Image.open(os.path.join(source_ske_dir, file_name + '.png'))
    ske_nd = np.array(ske)

    if (np.count_nonzero(ske_nd) == 0):
        regu = 0
        file_dic["regu"] = regu

        metric = 10**100
        file_dic["metric"] = metric
    else:
        regu = (seg_nd.sum())/(seg_nd.size)
        file_dic["regu"] = regu
        # print('mse',mse)
        # print('regu',regu)
        metric = file_mse/regu
        file_dic["metric"] = metric

    metric_list.append(metric)
    overlap.append(file_dic)

with open(tar_com_json,'w') as jf:
    json.dump({'overlap':overlap
                },jf)

metric_sort_increase = sorted(metric_list)
thr_select = metric_sort_increase[n_select]

select_bar = tqdm(overlap, ncols=150)
select_detail = []
select_name = []
for idx in select_bar:
    file_name = idx["file_name"]
    file_mse = idx["mse"]
    file_regu = idx['regu']
    file_metrix = idx['metric']
    file_dic = {}

    ske = Image.open(os.path.join(source_ske_dir, file_name + '.png'))
    ske_nd = np.array(ske)

    if (np.count_nonzero(ske_nd)>0) and (file_metrix < thr_select):
        file_dic["file_name"] = file_name
        file_dic["mse"] = file_mse
        file_dic["regu"] = file_regu
        file_dic["metric"] = file_metrix
        select_detail.append(file_dic)
        select_name.append(file_name)

with open(tar_select_detail_json,'w') as jf:
    json.dump({'select_detail':select_detail
                },jf)

with open(tar_select_name_json,'w') as jf:
    json.dump({'select_name':select_name
                },jf)
