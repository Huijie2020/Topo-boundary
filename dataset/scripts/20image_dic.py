import json
import random
import os

with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_test(train20).json', 'r') as jf:
    data = json.load(jf)
jf.close()

data_test = data["test"]

image_20 = []
for idx in data_test:
    file_name = idx
    file_dic = {}
    file_dic["file_name"] = file_name
    image_20.append(file_dic)

with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/grow_20img_dic.json','w') as jf:
    json.dump({'test':image_20
                },jf)