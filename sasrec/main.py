import os
import time
import argparse
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
import numpy as np

# python main.py --dataset ml-modern-gts --train_dir ml-modern-gts_runs --batch_size 128 --lr 0.001 --maxlen 50 --hidden_units 64 --num_blocks 2 --num_epochs 200 --num_heads 1 --dropout_rate 0.5 --l2_emb 0.0 > sasrec_mm_results_200epoch.txt

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--mm_item_init', default=None,
                    help='Path to multimodal item embedding init .npy '
                         '(shape: [itemnum+1, hidden_units]). If None, use '
                         'standard random initialization.')

args = parser.parse_args()

out_dir = args.dataset + '_' + args.train_dir
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join(
        [str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]
    ))

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset

# Python 3: make sure this is an int (Python 2 used int division implicitly)
num_batch = len(user_train) // args.batch_size

cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(out_dir, 'log.txt'), 'w')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum,
                      batch_size=args.batch_size,
                      maxlen=args.maxlen,
                      n_workers=3)

model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(max_to_keep=1)

if args.mm_item_init is not None:
    print('Loading multimodal item embeddings from %s' % args.mm_item_init)
    mm_init = np.load(args.mm_item_init)

    # mm_init is expected to be [num_items_all+1, dim]
    # We may have fewer items actually used (itemnum) than in the .npy.
    if mm_init.shape[1] != args.hidden_units:
        raise ValueError('mm_item_init dim mismatch: got %d, expected hidden_units=%d'
                         % (mm_init.shape[1], args.hidden_units))

    if mm_init.shape[0] < itemnum + 1:
        raise ValueError('mm_item_init has only %d rows, but need at least itemnum+1 = %d'
                         % (mm_init.shape[0], itemnum + 1))

    # Slice just the rows we need: 0..itemnum
    mm_init_used = mm_init[:itemnum + 1, :]

    # Name of the item embedding table created in Model/embedding():
    # with tf.variable_scope("SASRec/input_embeddings"), variable "lookup_table"
    emb_var = tf.get_default_graph().get_tensor_by_name(
        "SASRec/input_embeddings/lookup_table:0")

    sess.run(tf.assign(emb_var, mm_init_used))
    print('Multimodal item embeddings loaded into SASRec item embedding table.')

T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):
        # print(epoch)
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run(
                [model.auc, model.loss, model.train_op],
                {
                    model.u: u,
                    model.input_seq: seq,
                    model.pos: pos,
                    model.neg: neg,
                    model.is_training: True
                }
            )

        if epoch % 20 == 0:
            print(epoch)
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end=' ')
            t_test_ndcg, t_test_hr, t_test_mrr = evaluate(model, dataset, args, sess)
            t_valid_ndcg, t_valid_hr, t_valid_mrr = evaluate_valid(model, dataset, args, sess)
            print('')
            
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f), '
                'test (NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f)' %
                (epoch, T,
                t_valid_ndcg, t_valid_hr, t_valid_mrr,
                t_test_ndcg,  t_test_hr,  t_test_mrr))

            f.write(str((t_valid_ndcg, t_valid_hr, t_valid_mrr)) + ' ' + str((t_test_ndcg, t_test_hr, t_test_mrr)) + '\n')
            f.flush()
            t0 = time.time()
except Exception:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
ckpt_path = os.path.join(args.train_dir, "sasrec.ckpt")
saver.save(sess, ckpt_path)
print("Saved SASRec checkpoint to", ckpt_path)
print("Done")