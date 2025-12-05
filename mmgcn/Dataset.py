import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# def data_load(dataset, has_v=True, has_a=True, has_t=True):
#     dir_str = './Data/' + dataset
#     train_edge = np.load(dir_str+'/train.npy', allow_pickle=True)
#     user_item_dict = np.load(dir_str+'/user_item_dict.npy', allow_pickle=True).item()

#     if dataset == 'movielens':
#         num_user = 330
#         num_item = 21651
#         v_feat = np.load(dir_str+'/FeatureVideo_normal.npy', allow_pickle=True) if has_v else None
#         a_feat = np.load(dir_str+'/FeatureAudio_avg_normal.npy', allow_pickle=True) if has_a else None
#         t_feat = np.load(dir_str+'/FeatureText_stl_normal.npy', allow_pickle=True) if has_t else None
#         v_feat = torch.tensor(v_feat, dtype=torch.float).cuda() if has_v else None
#         a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() if has_a else None
#         t_feat = torch.tensor(t_feat, dtype=torch.float).cuda() if has_t else None
#     elif dataset == 'Tiktok':
#         num_user = 36656
#         num_item = 76085
#         if has_v:
#             v_feat = torch.load(dir_str+'/feat_v.pt')
#             v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
#         else:
#             v_feat = None

#         if has_a:
#             a_feat = torch.load(dir_str+'/feat_a.pt')
#             a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() 
#         else:
#             a_feat = None
        
#         t_feat = torch.load(dir_str+'/feat_t.pt') if has_t else None
#     elif dataset == 'Kwai':
#         num_user = 7010
#         num_item = 86483
#         v_feat = torch.load(dir_str+'/feat_v.pt')
#         v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
#         a_feat = t_feat = None

#     return num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat
def data_load(dataset, has_v=True, has_a=True, has_t=True):
    """
    Generic loader for MMGCN data in:

      mmgcn/Data/<dataset>/

    Expects:
      train.npy              [N_edges, 2] (user, global_item)
      user_item_dict.npy     dict[user] -> set(global_item)
      FeatureVideo_normal.npy        (optional)
      FeatureAudio_avg_normal.npy    (optional)
      FeatureText_stl_normal.npy     (optional)
    """
    # dir_str = os.path.join('./Data', dataset)
    if os.path.isdir(dataset):
        dir_str = dataset
    else:
        dir_str = os.path.join('./Data', dataset)

    train_edge = np.load(os.path.join(dir_str, 'train.npy'), allow_pickle=True)
    user_item_dict = np.load(
        os.path.join(dir_str, 'user_item_dict.npy'),
        allow_pickle=True
    ).item()

    train_edge = np.asarray(train_edge, dtype=np.int64)

    # infer num_user and item count from edges first
    max_user = int(train_edge[:, 0].max())
    max_node = int(train_edge[:, 1].max())

    num_user = max_user + 1
    num_item_from_edge = max_node + 1 - num_user

    v_feat = a_feat = t_feat = None
    num_item_from_feat = None

    # visual
    v_path = os.path.join(dir_str, 'FeatureVideo_normal.npy')
    if has_v and os.path.exists(v_path):
        v_arr = np.load(v_path, allow_pickle=True)
        v_feat = torch.tensor(v_arr, dtype=torch.float).cuda()
        num_item_from_feat = v_arr.shape[0]

    # audio (probably unused for MovieLens-modern)
    a_path = os.path.join(dir_str, 'FeatureAudio_avg_normal.npy')
    if has_a and os.path.exists(a_path):
        a_arr = np.load(a_path, allow_pickle=True)
        a_feat = torch.tensor(a_arr, dtype=torch.float).cuda()
        if num_item_from_feat is None:
            num_item_from_feat = a_arr.shape[0]

    # text
    t_path = os.path.join(dir_str, 'FeatureText_stl_normal.npy')
    if has_t and os.path.exists(t_path):
        t_arr = np.load(t_path, allow_pickle=True)
        t_feat = torch.tensor(t_arr, dtype=torch.float).cuda()
        if num_item_from_feat is None:
            num_item_from_feat = t_arr.shape[0]

    # Prefer the feature-based item count when available so that
    # v_feat / a_feat / t_feat and id_embedding all agree.
    if num_item_from_feat is not None:
        if num_item_from_feat != num_item_from_edge:
            print(
                f"[WARN] num_item from edges ({num_item_from_edge}) != "
                f"num_item from features ({num_item_from_feat}); "
                f"using feature count."
            )
        num_item = num_item_from_feat
    else:
        num_item = num_item_from_edge

    return num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        # self.all_set = set(range(num_user, num_user+num_item))
        self.all_items = list(range(num_user, num_user + num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.choice(self.all_items)
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item])
