import numpy as np
from glob import glob
from PIL import Image
import json
import os
from tqdm import tqdm
import random



source_dir = '/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/0.01sup_0.99unsup_contrastive_iter0-5/iter2/4451prediction_from_iter1'
source_img_dir = os.path.join(source_dir, 'segmentation') # 1998 segmentation folder
# source_json = '/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1998test(0.01data).json' # load 1998 image name
source_json = '/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/1_data_split/data_split_deepglobe_45sup_4451unsup_200val_1530test.json' # load 1998 image name

tar_all_detail_json = os.path.join(source_dir, 'confi_detail_4451.json')
tar_select_detail_json = os.path.join(source_dir, 'confi_detail_3560.json')
tar_select_img_json = os.path.join(source_dir, 'name_3560from4451.json')
# tar_select_img_iter_json = os.path.join(source_dir, 'name_1600from1998_iter.json')
tar_other_img_json = os.path.join(source_dir, 'name_891from4451.json')

# tar_all_detail_json = os.path.join(source_dir, 'confi_detail_1917.json')
# tar_select_detail_json = os.path.join(source_dir, 'confi_detail_1534.json')
# tar_select_img_json = os.path.join(source_dir, 'name_1534from1917.json')
# tar_select_img_iter_json = os.path.join(source_dir, 'name_1534from1917_iter.json')
# tar_other_img_json = os.path.join(source_dir, 'name_398from1917.json')

with open(source_json,'r') as jf:
    data = json.load(jf)
jf.close()

img_all = data["train_unsup"]
# img_all = data["test"]
n_select = 3560

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

with open(tar_all_detail_json,'w') as jf:
    json.dump({'train_image_detail':image_detail
                },jf)

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
    json.dump({'name_3560from4451':select_name
                },jf)
#
# with open(tar_select_img_iter_json,'w') as jf:
#     json.dump({'name_1534from1917_iter':select_name_iter
#                 },jf)
#
with open(tar_other_img_json,'w') as jf:
    json.dump({'name_891from4451':other_name
                },jf)


# define list Union function, for the second self-training
def list_union (lst1, lst2):
    return list(set(lst1).union(set(lst2)))

source_iter_pre_json = os.path.join(source_dir, 'union_two_image_list_iter1.json')
# tar_union_two_json = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter2_3560from4451_img_fromiter1_0831/4451gt_thr0_pre_from_iter1/union_two_image_list.json' # new file name json
tar_union_two_json = os.path.join(source_dir, 'union_two_image_list.json') # new file name json

with open(source_iter_pre_json,'r') as jf:
    data_pre = json.load(jf)
jf.close()
name_3560from4451_pre = data_pre["name_3560from4451"]

with open(tar_select_img_json,'r') as jf:
    data_current = json.load(jf)
jf.close()
name_3560from4451_current = data_current["name_3560from4451"]

# union this and pre
union_two_image_list = list_union (name_3560from4451_current, name_3560from4451_pre)
# lenght union_two_image_list
print("union_two_image_list", len(union_two_image_list))
# union_two_image_list 1670 (1% spacenet mse)
# union_two_image_list 3726 (1% deepglobe mse)
# union_two_image_list 1610 (1% spacenet mse)

# spacenet
# union_two_image_list 1652 (1% spacenet contrastive loss iter2)
# union_two_image_list 1698 (1% spacenet contrastive loss iter3)
# union_two_image_list 1716 (1% spacenet contrastive loss iter4)
# union_two_image_list 1808 (1% spacenet contrastive loss iter5)

# deeglobe
# union_two_image_list 3634 (1% spacenet contrastive loss iter2)

# write to json
with open(tar_union_two_json,'w') as jf:
    json.dump({'union_two_image_list':union_two_image_list
                },jf)





# with open(tar_union_two_json,'r') as jf:
#     data_count = json.load(jf)
# jf.close()
# data_count_num = data_count["union_two_image_list"]

