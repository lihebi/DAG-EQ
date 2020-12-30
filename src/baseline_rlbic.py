import time
import pickle
import os
import datetime
import math
import numpy as np
import h5py
from tqdm import tqdm

import tensorflow as tf

import networkx as nx

from data_loader.dataset_read_data import DataGenerator

from models import Actor
from rewards import get_Reward

from helpers.config_graph import get_config, print_config
from helpers.dir_utils import create_dir
from helpers.log_helper import LogHelper
from helpers.tf_utils import set_seed
from helpers.analyze_utils import convert_graph_int_to_adj_mat, graph_prunned_by_coef, \
                                  count_accuracy, graph_prunned_by_coef_2nd
from helpers.cam_with_pruning_cam import pruning_cam
from helpers.lambda_utils import BIC_lambdas


# python main.py  --max_length 12 \
#                 --data_size 5000 \
#                 --score_type BIC \
#                 --reg_type LR \
#                 --read_data  \
#                 --transpose \
#                 --lambda_flag_default \
#                 --nb_epoch 20000 \
#                 --input_dimension 64 \
#                 --lambda_iter_num 1000

config, _ = get_config()

config.max_length = 10
config.lambda_iter_num = 1000
config.lambda_flag_default = True
config.nb_epoch = 20000
config.transpose = True
# ???
config.input_dimension = 64


def test():
    raw_x.shape
    npx = raw_x[0].transpose()
    npy = raw_y[0]
    try:
        rlbic(10, npx, npy)
    except KeyboardInterrupt as e:
        exit(1)
    
def rlbic(d, npx, npy, lambda_iter_num=1000, nb_epoch=20000):
    # reset graph
    tf.reset_default_graph()
    # FIXME set the mx_length at the beginning. This might be used even in RLBIC
    #
    # but I'm not sure if the RLBIC code is already initialized with above settings
    config.max_length = d
    config.lambda_iter_num = lambda_iter_num
    config.nb_epoch = nb_epoch

    training_set = DataGenerator(npx, npy, False, True)
    score_type = 'BIC'
    # FIXME what is this?
    reg_type = 'LR'
    sl, su, strue = BIC_lambdas(training_set.inputdata,
                                None, None,
                                training_set.true_graph.T,
                                reg_type, score_type)

    lambda1 = 0
    lambda1_upper = 5
    lambda1_update_add = 1
    lambda2 = 1/(10**(np.round(config.max_length/3)))
    lambda2_upper = 0.01
    lambda2_update_mul = 10
    lambda_iter_num = config.lambda_iter_num

    actor = Actor(config)
    callreward = get_Reward(actor.batch_size, config.max_length,
                            actor.input_dimension, training_set.inputdata,
                            sl, su, lambda1_upper, score_type, reg_type,
                            config.l1_graph_reg, False)

    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())
        rewards_avg_baseline = []
        rewards_batches = []
        reward_max_per_batch = []
        
        lambda1s = []
        lambda2s = []
        
        graphss = []
        probsss = []
        max_rewards = []
        max_reward = float('-inf')
        image_count = 0
        
        accuracy_res = []
        accuracy_res_pruned = []
        
        max_reward_score_cyc = (lambda1_upper+1, 0)

        result = None
        print('Starting training.')
        for i in tqdm(range(1, config.nb_epoch + 1)):
            # print('Start training for {}-th epoch'.format(i))
            input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
            graphs_feed = sess.run(actor.graphs, feed_dict={actor.input_: input_batch})
            reward_feed = callreward.cal_rewards(graphs_feed, lambda1, lambda2)

            # max reward, max reward per batch
            max_reward = -callreward.update_scores([max_reward_score_cyc], lambda1, lambda2)[0]
            max_reward_batch = float('inf')
            max_reward_batch_score_cyc = (0, 0)

            for reward_, score_, cyc_ in reward_feed:
                if reward_ < max_reward_batch:
                    max_reward_batch = reward_
                    max_reward_batch_score_cyc = (score_, cyc_)
                        
            max_reward_batch = -max_reward_batch

            if max_reward < max_reward_batch:
                max_reward = max_reward_batch
                max_reward_score_cyc = max_reward_batch_score_cyc

            # for average reward per batch
            reward_batch_score_cyc = np.mean(reward_feed[:,1:], axis=0)
            # print('Finish calculating reward for current batch of graph')

            # Get feed dict
            feed = {actor.input_: input_batch, actor.reward_: -reward_feed[:,0], actor.graphs_:graphs_feed}

            summary, base_op, score_test, probs, graph_batch, \
                reward_batch, reward_avg_baseline, train_step1, train_step2 = sess.run([actor.merged, actor.base_op,
                actor.test_scores, actor.log_softmax, actor.graph_batch, actor.reward_batch, actor.avg_baseline, actor.train_step1,
                actor.train_step2], feed_dict=feed)

            # print('Finish updating actor and critic network using reward calculated')
            lambda1s.append(lambda1)
            lambda2s.append(lambda2)

            rewards_avg_baseline.append(reward_avg_baseline)
            rewards_batches.append(reward_batch_score_cyc)
            reward_max_per_batch.append(max_reward_batch_score_cyc)

            graphss.append(graph_batch)
            probsss.append(probs)
            max_rewards.append(max_reward_score_cyc)

            if i == 1 or i % 500 == 0:
                print('[iter {}] reward_batch: {}, max_reward: {}, max_reward_batch: {}'
                      .format(i, reward_batch, max_reward, max_reward_batch))
            # update lambda1, lamda2
            if (i+1) % lambda_iter_num == 0:
                ls_kv = callreward.update_all_scores(lambda1, lambda2)
                # np.save('{}/solvd_dict_epoch_{}.npy'.format(config.graph_dir, i), np.array(ls_kv))
                max_rewards_re = callreward.update_scores(max_rewards, lambda1, lambda2)
                rewards_batches_re = callreward.update_scores(rewards_batches, lambda1, lambda2)
                reward_max_per_batch_re = callreward.update_scores(reward_max_per_batch, lambda1, lambda2)

                graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

                if cyc_min < 1e-5:
                    lambda1_upper = score_min
                lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
                lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
                    
                graph_batch = convert_graph_int_to_adj_mat(graph_int)

                if reg_type == 'LR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, training_set.inputdata))
                elif reg_type == 'QR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))
                elif reg_type == 'GPR':
                    # The R codes of CAM pruning operates the graph form that (i,j)=1 indicates i-th node-> j-th node
                    # so we need to do a tranpose on the input graph and another tranpose on the output graph
                    graph_batch_pruned = np.transpose(pruning_cam(training_set.inputdata, np.array(graph_batch).T))

                # estimate accuracy
                # FIXME graph_batch or graph_batch_pruned
                # FIXME .T should be adj mat?
                acc_est = count_accuracy(training_set.true_graph, graph_batch.T)
                acc_est2 = count_accuracy(training_set.true_graph, graph_batch_pruned.T)

                # TODO return the adj matrix
                fdr, tpr, fpr, shd, nnz = acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], acc_est['shd'], \
                                          acc_est['pred_size']
                fdr2, tpr2, fpr2, shd2, nnz2 = acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], acc_est2['shd'], \
                                               acc_est2['pred_size']
                    
                accuracy_res.append((fdr, tpr, fpr, shd, nnz))
                accuracy_res_pruned.append((fdr2, tpr2, fpr2, shd2, nnz2))

                print('fdr:', fdr, 'tpr:', tpr, 'fpr:', fpr, 'shd:', shd)
                result = acc_est
        return graph_batch.T
