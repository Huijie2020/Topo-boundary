import numpy as np
from glob import glob
from PIL import Image
import json
import os
from tqdm import tqdm
import random

source_ske_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.2_best_thr0_ta_tta_iter/iter1_0628/1998gt_thr0_pre_from_iter0/skeleton'
tar_pos_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.2_best_thr0_ta_tta_iter/iter1_0628/def_pos_neg/pos'
tar_neg_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.2_best_thr0_ta_tta_iter/iter1_0628/def_pos_neg/neg'
tar_total_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.2_best_thr0_ta_tta_iter/iter1_0628/def_pos_neg/total'

source_json = '/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1998test(0.01data).json'
source_train_json = '/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_100val_431test.json'
tar_pos_neg_json = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.2_best_thr0_ta_tta_iter/iter1_0628/def_pos_neg/pos_neg.json'
tar_total_json = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.2_best_thr0_ta_tta_iter/iter1_0628/def_pos_neg/total_crop_score.json'

# def random_crop(ske, size):
#     w, h = ske.size
#     newW, newH = int(w), int(h)
#     assert newW > 0 and newH > 0, 'Scale is too small'
#     ske = ske.resize((newW, newH))
#
#     crop_h = size
#     crop_w = size
#
#     start_x = np.random.randint(1, w - crop_w - 1)
#     start_y = np.random.randint(1, h - crop_h - 1)
#
#     ske_nd = np.array(ske)
#
#     ske_nd = ske_nd[start_y: start_y + crop_h, start_x: start_x + crop_w]
#
#     return ske_nd, start_y, start_x
#
# with open(source_json,'r') as jf:
#     data = json.load(jf)
# jf.close()
# test_img = data['test']
#
# progress_bar = tqdm(test_img, ncols=150)
# total = []
# total_score = []
# for idx in progress_bar:
#     ske_file = os.path.join(source_ske_dir, idx + '.png')
#     ske = Image.open(ske_file)
#
#     crop_ul = []
#     crop_score = []
#     ske_dic = {}
#
#     for i in range(20):
#         ske_nd, start_y, start_x = random_crop(ske, 256)
#         ske_score = np.count_nonzero(ske_nd)
#         crop_ul.append([start_y, start_x])
#         crop_score.append(ske_score)
#         total_score.append(ske_score)
#
#         # ske_i_name = idx + '_' + str(i)
#         # Image.fromarray(ske_nd).save(os.path.join(tar_total_dir, ske_i_name + '.png'))
#
#     ske_dic["file_name"] = idx
#     ske_dic["crop_ul"] = crop_ul
#     ske_dic["crop_score"] = crop_score
#     total.append(ske_dic)
#
# with open(tar_total_json,'w') as jf:
#     json.dump({'total': total[:len(total)],
#                'total_ske_score':total_score
#                }, jf)
# jf.close()





with open(tar_total_json,'r') as jf:
    data = json.load(jf)
jf.close()
total_ske_score = data['total_ske_score']
total = data['total']

n_select = int(len(total_ske_score) * 0.1)
total_ske_score_sort_increase = sorted(total_ske_score)
total_ske_score_sort_decrease = sorted(total_ske_score, reverse=True)
thr_select = total_ske_score_sort_decrease[n_select-1]

progress_bar = tqdm(total, ncols=150)

pos_unsup = []
neg_unsup = []
for ids in progress_bar:
    file_name = ids["file_name"]
    crop_ul = ids["crop_ul"]
    crop_score = ids["crop_score"]

    # pos_crop_ul = []
    # pos_crop_score = []
    # neg_crop_ul = []
    # neg_crop_score = []
    pos_ske_dic = {}
    neg_ske_dic = {}

    for i in range(len(crop_score)):
        if crop_score[i] >= thr_select:
            # pos_crop_ul.append(crop_ul[i])
            # pos_crop_score.append(crop_score[i])
            pos_file_name = file_name
            pos_ske_dic["file_name"] = pos_file_name
            pos_ske_dic["crop_ul"] = crop_ul[i]
            pos_ske_dic["crop_score"] = crop_score[i]
            pos_unsup.append(pos_ske_dic)

        if crop_score[i] == 0:
            # neg_crop_ul.append(crop_ul[i])
            # neg_crop_score.append(crop_score[i])
            neg_file_name = file_name
            neg_ske_dic["file_name"] = neg_file_name
            neg_ske_dic["crop_ul"] = crop_ul[i]
            neg_ske_dic["crop_score"] = crop_score[i]
            neg_unsup.append(neg_ske_dic)


    # if len(pos_crop_score) > 0:
    #     pos_file_name = file_name
    #     pos_ske_dic["file_name"] = pos_file_name
    #     pos_ske_dic["crop_ul"] = pos_crop_ul
    #     pos_ske_dic["crop_score"] = pos_crop_score
    #     pos_unsup.append(pos_ske_dic)
    #
    # if len(neg_crop_score) > 0:
    #     neg_file_name = file_name
    #     neg_ske_dic["file_name"] = neg_file_name
    #     neg_ske_dic["crop_ul"] = neg_crop_ul
    #     neg_ske_dic["crop_score"] = neg_crop_score
    #     neg_unsup.append(neg_ske_dic)


with open(tar_pos_neg_json,'w') as jf:
    json.dump({'pos_unsup': pos_unsup,
               'neg_unsup':neg_unsup
               }, jf)
jf.close()

print('pos_unsup length', len(pos_unsup))
print('neg_unsup length', len(neg_unsup))
# pos_unsup length 1055
# neg_unsup length 1409
# pos_unsup length 3996
# neg_unsup length 8326
