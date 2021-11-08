import numpy as np
from glob import glob
from PIL import Image
import json
import os
from tqdm import tqdm
import random



source_dir = '/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/0.01sup_0.99unsup_contrastive_iter0-5/iter2/4451prediction_from_iter1/select_last'
source_img_dir = os.path.join(source_dir, 'select_3560_iter1_iter2') # 1998 segmentation folder
# source_json = '/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1998test(0.01data).json' # load 1998 image name
source_json = os.path.join(source_dir, 'union_two_image_list.json') # load 1998 image name

tar_select_detail_json = os.path.join(source_dir, 'confi_detail_1597.json')
tar_select_img_json = os.path.join(source_dir, 'name_1597from4451.json')


with open(source_json,'r') as jf:
    data = json.load(jf)
jf.close()

img_all = data["union_two_image_list"]
# img_all = data["test"]
n_select = 1597

image_detail = []
metric_list = []

for i in tqdm(range(len(img_all))):
    file_dic = {}
    file_name = img_all[i]
    file_dic["file_name"] = file_name

    seg = Image.open(os.path.join(source_img_dir, file_name + '.png'))
    seg_nd = np.array(seg)
    seg_nd = seg_nd / 255

    # # mask > 0.1
    # thr_mask = (seg_nd >= 0.1)
    # seg_nd = seg_nd * thr_mask

    if (np.count_nonzero(seg_nd) == 0):
        sel_metric = 0.0
        file_dic["confidence"] = sel_metric
    else:
        # sel_metric = (seg_nd.sum()) / (np.count_nonzero(seg_nd))
        sel_metric = (seg_nd.sum()) / (seg_nd.size)
        file_dic["confidence"] = sel_metric

    metric_list.append(sel_metric)
    image_detail.append(file_dic)


metric_sort_decrease = sorted(metric_list, reverse=True)
thr_select = metric_sort_decrease[n_select-1]

select_bar = tqdm(image_detail, ncols=150)
select_detail = []
select_name = []
# select_name_iter = []
other_name = []

for i in range(len(image_detail)):
    file_name = image_detail[i]["file_name"]
    file_confi = image_detail[i]["confidence"]

    file_dic = {}

    if (file_confi >= thr_select):
        file_dic["file_name"] = file_name
        file_dic["confidence"] = file_confi

        select_detail.append(file_dic)
        select_name.append(file_name)
        # file_name_iter = file_name + "_1"
        # select_name_iter.append(file_name_iter)
    else:
        other_name.append(file_name)

with open(tar_select_detail_json,'w') as jf:
    json.dump({'select_image_detail':select_detail
                },jf)

with open(tar_select_img_json,'w') as jf:
    json.dump({'name_1597from4451':select_name
                },jf)




# #
# with open(tar_select_img_json,'r') as jf:
#     data_count = json.load(jf)
# jf.close()
# data_count_num = data_count["name_1597from4451"]



