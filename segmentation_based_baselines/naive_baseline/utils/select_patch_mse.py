import numpy as np
from glob import glob
from PIL import Image
import json
import os
from tqdm import tqdm
import random

source_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_200of1998_fromiter0_0827/200of1998_thr0_pre_from_iter0'

source_pre_dir = os.path.join(source_dir, 'segmentation_overlap')
source_ske_dir = os.path.join(source_dir, 'skeleton_overlap')
source_json = os.path.join(source_dir, '100epoch_256_20000img.json')
tar_com_json = os.path.join(source_dir, '100epoch_256_20000img_combine.json')
tar_select_detail_json = os.path.join(source_dir, 'select_2000_detail.json')
tar_select_name_json = os.path.join(source_dir, 'select_2000_name.json')

with open(source_json,'r') as jf:
    data = json.load(jf)
jf.close()

overlap_id = data['img_id']
mse_all = data['mse']
ul_all = data['overlap_ul']

n_select = int(len(overlap_id) * 0.1)
# progress_bar = tqdm(range(len(overlap_id)+1), ncols=150)

overlap = []
metric_list = []
for i in range(len(overlap_id)):
    file_dic = {}
    file_name = overlap_id[i]
    file_mse = mse_all[i]
    file_ul = ul_all[i]
    file_dic["file_name"] = file_name
    file_dic["mse"] = file_mse
    file_dic["overlap_ul"] = file_ul

    seg = Image.open(os.path.join(source_pre_dir, file_name+'.png'))
    seg_nd = np.array(seg)
    seg_nd = seg_nd/255

    # ske = Image.open(os.path.join(source_ske_dir, file_name + '.png'))
    # ske_nd = np.array(ske)

    # if (np.count_nonzero(ske_nd) == 0):
    #     regu = 0
    #     file_dic["regu"] = regu
    #
    #     sel_metric = 10**100
    #     file_dic["metric"] = sel_metric
    # else:
    #     regu = ((seg_nd.sum())/(seg_nd.size))**2
    #     file_dic["regu"] = regu
    #
    #     sel_metric = file_mse/regu
    #     file_dic["metric"] = sel_metric

    regu = 0
    file_dic["regu"] = regu
    thr_mask = (seg_nd >= 0.1)
    seg_nd = seg_nd * thr_mask

    if (np.count_nonzero(seg_nd) == 0):
        # sel_metric = 10**100
        sel_metric = 0
        file_dic["metric"] = sel_metric
    else:
        sel_metric = (seg_nd.sum()) / (np.count_nonzero(seg_nd))
        file_dic["metric"] = sel_metric

    metric_list.append(sel_metric)
    overlap.append(file_dic)

with open(tar_com_json,'w') as jf:
    json.dump({'overlap':overlap
                },jf)

metric_sort_decrease = sorted(metric_list, reverse=True)
thr_select = metric_sort_decrease[n_select-1]

select_bar = tqdm(overlap, ncols=150)
select_detail = []
select_name = []
for idx in select_bar:
    file_name = idx["file_name"]
    file_mse = idx["mse"]
    file_regu = idx['regu']
    file_metrix = idx['metric']
    file_ul = idx["overlap_ul"]
    file_dic = {}

    ske = Image.open(os.path.join(source_ske_dir, file_name + '.png'))
    ske_nd = np.array(ske)

    if (file_metrix >= thr_select):
        file_dic["file_name"] = file_name
        file_dic["mse"] = file_mse
        file_dic["regu"] = file_regu
        file_dic["metric"] = file_metrix
        file_dic["overlap_ul"] = file_ul
        select_detail.append(file_dic)
        select_name.append(file_name)

with open(tar_select_detail_json,'w') as jf:
    json.dump({'select_detail':select_detail
                },jf)

with open(tar_select_name_json,'w') as jf:
    json.dump({'select_name':select_name
                },jf)
