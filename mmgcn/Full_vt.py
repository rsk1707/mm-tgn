from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np

def full_vt(epoch, model, data, prefix, writer=None):   
    print(prefix+' start...')
    model.eval()

    with no_grad():
        # here topk is your K for Hits@K and MRR; ndcg is fixed at 10
        hits, ndcg_score, mrr = model.full_accuracy(data, topk=10, ndcg_k=10)

        print('---------------------------------{0}-th {4} Hits@{5}:{1:.4f} NDCG@10:{2:.4f} MRR@{5}:{3:.4f}---------------------------------'
            .format(epoch, hits, ndcg_score, mrr, prefix, 10))

        if writer is not None:
            writer.add_scalar(prefix + '_Hits@10', hits, epoch)
            writer.add_scalar(prefix + '_NDCG@10', ndcg_score, epoch)
            writer.add_scalar(prefix + '_MRR@10', mrr, epoch)

            writer.add_histogram(prefix+'_visual_distribution', model.v_rep, epoch)
            writer.add_histogram(prefix+'_acoustic_distribution', model.a_rep, epoch)
            writer.add_histogram(prefix+'_textual_distribution', model.t_rep, epoch)
            
            writer.add_histogram(prefix+'_user_visual_distribution', model.user_preferences[:,:44], epoch)
            writer.add_histogram(prefix+'_user_acoustic_distribution', model.user_preferences[:, 44:-44], epoch)
            writer.add_histogram(prefix+'_user_textual_distribution', model.user_preferences[:, -44:], epoch)

            writer.add_embedding(model.v_rep)
            writer.add_embedding(model.a_rep)
            writer.add_embedding(model.t_rep)
            
        return hits, ndcg_score, mrr



