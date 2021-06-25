# show_pkl.py

import pickle

# path = '000147_00.pickle'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
path = '/mnt/git/Topo-boundary/conn_experiment/expriment/06132021_unet_baseline_100data_24batch_0.01lr/records/test/graph/RGB-PanSharpen_AOI_2_Vegas_img56.pickle'

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))