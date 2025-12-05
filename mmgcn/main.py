import argparse
import os
import time
import numpy as np
import torch
from Dataset import TrainingDataset, data_load
from Model_MMGCN import Net
from torch.utils.data import DataLoader
from Train import train
from Full_vt import full_vt
# from torch.utils.tensorboard import SummaryWriter
###############################248###########################################

# python main.py --data_path movielens --save_file mlm_mmgcn --dim_E 64 --l_r 3e-4 --weight_decay 3e-5 --batch_size 1024 --num_epoch 1000 --topK 10 --has_v True --has_t True --has_a False > mmgcn_baseline.txt
# python main.py --data_path ml-modern-gts --save_file mlm_mmgcn_gts --dim_E 64 --l_r 3e-4 --weight_decay 3e-5 --batch_size 1024 --num_epoch 20 --topK 10 --has_v True --has_t True --has_a False --dropout 0.4 --PATH_weight_save ./Data/ml-modern-gts/mmgcn_ml-modern-gts.pt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')

    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation Mode.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')

    parser.add_argument('--has_entropy_loss', default='False', help='Has Cross Entropy loss.')
    parser.add_argument('--has_weight_loss', default='False', help='Has Weight Loss.')
    parser.add_argument('--has_v', default='True', help='Has Visual Features.')
    parser.add_argument('--has_a', default='True', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')

    args = parser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path
    save_file = args.save_file

    learning_rate = args.l_r
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    # num_routing = args.num_routing
    topK = args.topK
    prefix = args.prefix
    aggr_mode = args.aggr_mode

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False
    has_entropy_loss = True if args.has_entropy_loss == 'True' else False
    has_weight_loss = True if args.has_weight_loss == 'True' else False
    dim_E = args.dim_E
    writer = None#SummaryWriter()
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, weight_decay), 'w') as save_file:
    #     save_file.write('---------------------------------lr: {0} \t Weight_decay:{1} ---------------------------------\r\n'.format(l_r, weight_decay))
    ##########################################################################################################################################
    print('Data loading ...')

    # num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(data_path)

    # v_feat = torch.tensor(v_feat, dtype=torch.float).cuda() if has_v else None
    # a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() if has_a else None
    # t_feat = torch.tensor(t_feat, dtype=torch.float).cuda() if has_t else None
    num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(
    data_path,
    has_v=has_v,
    has_a=has_a,
    has_t=has_t,
)

    # data_load already returns CUDA tensors (or None) based on has_v/has_a/has_t.
    if not has_v:
        v_feat = None
    if not has_a:
        a_feat = None
    if not has_t:
        t_feat = None

    train_dataset = TrainingDataset(num_user, num_item, user_item_dict, train_edge)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    val_data = np.load('./Data/'+data_path+'/val_full.npy', allow_pickle=True)
    test_data = np.load('./Data/'+data_path+'/test_full.npy', allow_pickle=True)
    print('Data has been loaded.')
    ##########################################################################################################################################
    model = Net(v_feat, a_feat, t_feat, None, train_edge, batch_size, num_user, num_item, 'mean', 'False', 2, True, user_item_dict, weight_decay, dim_E).cuda()
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    val_max_recall = 0.0
    num_decreases = 0 
    for epoch in range(num_epoch):
        loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, writer)
        if torch.isnan(loss):
            with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} is Nan\r\n'.format(learning_rate, weight_decay))
            break
        torch.cuda.empty_cache()

        val_precision, val_recall, val_ndcg = full_vt(epoch, model, val_data, 'Val', writer)
        test_precision, test_recall, test_ndcg = full_vt(epoch, model, test_data, 'Test', writer)

        if val_recall > val_max_recall:
            val_max_recall = val_recall
            max_precision = test_precision
            max_recall = test_recall
            max_NDCG = test_ndcg
            num_decreases = 0

            if args.PATH_weight_save is not None:
                torch.save(model.state_dict(), args.PATH_weight_save)
        else:
            if num_decreases > 20:
                with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} =====> Precision:{2} \t Recall:{3} \t NDCG:{4}\r\n'.
                                    format(learning_rate, weight_decay, max_precision, max_recall, max_NDCG))
                break
            else:
                num_decreases += 1
