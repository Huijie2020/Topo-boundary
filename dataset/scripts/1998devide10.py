import json
import random
import os

tar_file_name = '/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split'
def split_dataset():
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1998test(0.01data).json','r') as jf:
        data = json.load(jf)

    data_1998 = data["test"]
    random.shuffle(data_1998)

    for i in range(10):
        save_json_name = 'split1998to10_iter'+str(i+1)+'.json'
        save_json_dir = os.path.join(tar_file_name,save_json_name)
        test_i = data_1998[(200*i):(200*(i+1))]
        print('save_json_name:', save_json_name)
        print('json len:', len(test_i))
        with open(save_json_dir, 'w') as jf:
            json.dump({'train_sup': [],
                       'train_unsup': [],
                       'valid': [],
                       'test': test_i,
                       'pretrain': []
                       }, jf)


split_dataset()

