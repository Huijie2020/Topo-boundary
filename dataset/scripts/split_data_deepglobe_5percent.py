import json
import random
import os
import shutil
from tqdm import tqdm

# source_dir = '/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/5_data_split'
# scr_json = os.path.join(source_dir, '4496train_200val_1530test.json')

def split_data():
    with open(scr_json, 'r') as jf:
        data = json.load(jf)
    jf.close()

    train = data['train_sup']
    valid = data['valid']
    test = data['test']

    random.shuffle(train)
    train_sup = train[:225]
    train_unsup = train[225:]

    print('length of train_sup data {}'.format(len(train_sup)))
    print('length of train_unsup data {}'.format(len(train_unsup)))
    print('length of valid data {}'.format(len(valid)))
    print('length of test data {}'.format(len(test)))

    with open(os.path.join(source_dir, 'data_split_deepglobe_225sup_4271unsup_200val_1530test.json'),'w') as jf:
        json.dump({'train_sup': train_sup,
                   'train_unsup': train_unsup,
                   'valid': valid,
                   'test': test
                    },jf)
    jf.close()

def split_data_repeat():
    with open(os.path.join(source_dir, 'data_split_deepglobe_225sup_4271unsup_200val_1530test.json'), 'r') as jf:
        data = json.load(jf)
    jf.close()

    train_sup = data['train_sup']
    train_unsup = data['train_unsup']
    valid = data['valid']
    test = data['test']

    train_sup_new = []
    for i in range(40):
        random.shuffle(train_sup)
        for i in range(len(train_sup)):
            train_sup_new.append(train_sup[i])

    print('length of train_sup data {}'.format(len(train_sup_new)))
    print('length of train_unsup data {}'.format(len(train_unsup)))
    print('length of valid data {}'.format(len(valid)))
    print('length of test data {}'.format(len(test)))

    with open(os.path.join(source_dir, 'data_split_deepglobe_9000sup_225sup_4271unsup_200val_1530test.json'),'w') as jf:
        json.dump({'train_sup': train_sup_new,
                   'train_unsup': train_unsup,
                   'valid': valid,
                   'test': test
                    },jf)
    jf.close()


source_dir = '/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/5_data_split'
scr_json = os.path.join(source_dir, 'data_split_deepglobe_225sup_4271unsup_200val_1530test.json')
def testdata():
    with open(scr_json, 'r') as jf:
        data = json.load(jf)
    jf.close()

    train_unsup = data['train_unsup']
    print('\n train_unsup lenght:', len(train_unsup))

    with open(os.path.join(source_dir, 'data_split_4271test.json'),'w') as jf:
        json.dump({'train_sup': [],
                   'train_unsup': [],
                   'valid': [],
                   'test': train_unsup
                    },jf)
    jf.close()

    for idx in tqdm(train_unsup):
        scr_img_path = os.path.join(source_dir, 'segmentation', idx+'.png')
        des_img_path = os.path.join(source_dir, '4271prediction_from_iter0', idx+'.png')
        shutil.move(scr_img_path, des_img_path)

testdata()
