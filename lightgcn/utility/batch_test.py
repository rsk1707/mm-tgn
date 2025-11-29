'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from utility.parser import parse_args
from utility.load_data import *
from evaluate_loo import eval_score_matrix_loo
import multiprocessing
import numpy as np

cores = multiprocessing.cpu_count() // 2

args = parse_args()

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size


def test(sess, model, users_to_test, drop_flag=False, train_set_flag=0):
    """
    Evaluate model on given users using leave-one-out/top-K metrics.

    Returns dict with:
      result['hit']  -> hit@K (a.k.a. recall@K in LOO setting)
      result['ndcg'] -> ndcg@K
      result['mrr']  -> mrr@K
    """
    top_show = np.sort(model.Ks)
    max_top = max(top_show)

    result = {
        'hit':  np.zeros(len(model.Ks)),
        'ndcg': np.zeros(len(model.Ks)),
        'mrr':  np.zeros(len(model.Ks)),
    }

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    all_result = []
    item_batch = range(ITEM_NUM)

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start:end]
        if len(user_batch) == 0:
            continue

        if not drop_flag:
            rate_batch = sess.run(
                model.batch_ratings,
                {model.users: user_batch,
                 model.pos_items: item_batch}
            )
        else:
            rate_batch = sess.run(
                model.batch_ratings,
                {model.users: user_batch,
                 model.pos_items: item_batch,
                 model.node_dropout: [0.] * len(eval(args.layer_size)),
                 model.mess_dropout: [0.] * len(eval(args.layer_size))}
            )

        rate_batch = np.array(rate_batch)  # (B, N)

        test_items = []
        if train_set_flag == 0:
            # eval on held-out items
            for user in user_batch:
                test_items.append(data_generator.test_set[user])

            # mask training items
            for idx, user in enumerate(user_batch):
                train_items_off = data_generator.train_items[user]
                rate_batch[idx][train_items_off] = -np.inf
        else:
            # eval on training items
            for user in user_batch:
                test_items.append(data_generator.train_items[user])

        # (B, 3 * max_top)  -> [hit@1..K, ndcg@1..K, mrr@1..K]
        batch_result = eval_score_matrix_loo(rate_batch, test_items, max_top)
        count += len(batch_result)
        all_result.append(batch_result)

    assert count == n_test_users, f"Expected {n_test_users} users, got {count}"

    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)   # shape (3 * max_top,)
    final_result = np.reshape(final_result, newshape=[3, max_top])

    # keep only the Ks you requested via args.Ks
    final_result = final_result[:, top_show - 1]         # (3, len(Ks))
    final_result = np.reshape(final_result, [3, len(top_show)])

    # row 0: hit@K
    # row 1: ndcg@K
    # row 2: mrr@K
    result['hit']  += final_result[0]
    result['ndcg'] += final_result[1]
    result['mrr']  += final_result[2]

    return result