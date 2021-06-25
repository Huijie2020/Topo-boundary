import json
import random

def split_dataset():
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_100val_431test.json','r') as jf:
    #with open('./data_split.json','r') as jf:
        data = json.load(jf)
    train_len = round(len(data['train'])*1)
    valid_len = len(data['valid'])
    # valid_len_w = round(valid_len*1)
    test_len = len(data['test'])
    # pretrain_len = round(len(data['pretrain'])/10)
    pretrain_len = round(len(data['pretrain']))
    # train list
    #all_images = data['train'] + data['valid'] + data['test'] + data['pretrain']
    #random.shuffle(all_images)

    train_img = data['train']
    valid_img = data['valid']
    test_img = data['test']
    pretrain_img = data['pretrain']

    #random.shuffle(train_img)
    #random.shuffle(pretrain_img)

   # with open('./data_split_previous.json','w') as jf:
   #     json.dump(data,jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_train_for_test.json','w') as jf:
    # with open('./scripts/data_split_100val.json','w') as jf:
    #     json.dump({'train':train_img[:train_len],
    #                 'valid':valid_img[:100],
    #                 'test':test_img[100:test_len],
    #                 'pretrain':pretrain_img[:pretrain_len]
    #                 },jf)
        json.dump({'train': [],
                   'valid':[],
                   'test': train_img[:train_len],
                   'pretrain': pretrain_img[:pretrain_len]
                   }, jf)
    # random.shuffle(valid_img)
    # with open('./scripts/data_split_w_val.json','w') as jf:
    #     json.dump({'train':train_img[:train_len],
    #                 'valid':valid_img[:valid_len_w],
    #                 'test':test_img[:test_len],
    #                 'pretrain':pretrain_img[:pretrain_len]
    #                 },jf)
    print("train_len", train_len)
    print("valid_len",valid_len)
    print('"valid_len', valid_len)
    print("test_len", test_len)
    print("pretrain_len", pretrain_len)

split_dataset()





# def split_dataset():
#     with open('./data_split.json','r') as jf:
#         data = json.load(jf)
#     train_len = len(data['train'])
#     valid_len = len(data['valid'])
#     test_len = len(data['test'])
#     # train list
#     all_images = data['train'] + data['valid'] + data['test'] + data['pretrain']
#     random.shuffle(all_images)
#     with open('./data_split_previous.json','w') as jf:
#         json.dump(data,jf)
#     with open('./scripts/data_split.json','w') as jf:
#         json.dump({'train':all_images[:train_len],
#                     'valid':all_images[train_len:train_len+valid_len],
#                     'test':all_images[train_len+valid_len:train_len+valid_len+test_len],
#                     'pretrain':all_images[train_len+valid_len+test_len:]
#                     },jf)
#
# split_dataset()