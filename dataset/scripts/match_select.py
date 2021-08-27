import os
from glob import glob
from tqdm import tqdm
import json

name_txt_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_0817/1998gt_thr0_pre_from_iter0/name.txt'
mse_txt_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_0817/1998gt_thr0_pre_from_iter0/mse.txt'

full_json_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_0817/1998gt_thr0_pre_from_iter0/5epoch_256_9990img.json'
select_json_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_0817/1998gt_thr0_pre_from_iter0/select.json'

overlap_source_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_0817/1998gt_thr0_pre_from_iter0/segmentation_overlap'
overlap_tar_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/08032021_unet_baseline_0.11data_99unsup/exp1.3_hog_ta_tta_unsupweight1_lr0.0005_iter/iter1_0817/1998gt_thr0_pre_from_iter0/overlap_select'

name_lines = []
with open(name_txt_dir) as f:
    name_lines = f.read().splitlines()
f.close()

mse_lines = []
with open(mse_txt_dir) as f:
    mse_lines = f.read().splitlines()
f.close()
mse_lines = [float(i) for i in mse_lines]

n_select = 1998*5
# mse_lines_sort = sorted(mse_lines)
# thr_select = mse_lines_sort[n_select-1]

overlap_select_list = []
mse_select_list = []
# for i in range(len(mse_lines)):
#     if mse_lines[i] <= thr_select:
#         overlap_select_list.append(name_lines[i])
#         mse_select_list.append(mse_lines[i])
#         # copy_from = os.path.join(overlap_source_dir, name_lines[i] + ".png")
#         # copy_to = os.path.join(overlap_tar_dir, name_lines[i] + ".png")
#         # os.system('cp {} {}'.format(copy_from, copy_to))
#     else:
#         pass

for i in range(n_select):
    overlap_select_list.append(name_lines[i])
    mse_select_list.append(mse_lines[i])

print('overlap_select_list lenght', len(overlap_select_list))
print('mse_select_list lenght', len(mse_select_list))
print('name_lines lenght', len(name_lines))
print('mse_lines lenght', len(mse_lines))

# with open(select_json_dir, 'w') as jf:
#     json.dump({'overlap_id': overlap_select_list[:len(overlap_select_list)],
#                'mse': mse_select_list[:len(mse_select_list)]
#                }, jf)

with open(full_json_dir, 'w') as jf:
    json.dump({'overlap_id': overlap_select_list[:len(overlap_select_list)],
               'mse': mse_select_list[:len(mse_select_list)]
               }, jf)


