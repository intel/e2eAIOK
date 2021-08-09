import pickle
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *

import argparse

from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

import horovod.tensorflow as hvd
#hvd.init()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train',
                    help="mode, train or test")
parser.add_argument("--model", type=str, default='DIEN', help="model")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_type", type=str, default='FP32',
                    help="data type: FP32 or FP16")
parser.add_argument("--num_accelerators", type=int, default=1,
                    help="number of accelerators used for training")
parser.add_argument("--embedding_device", type=str, default='cpu',
                    help="synthetic input embedding layer reside on gpu or cpu")
parser.add_argument("--num-intra-threads", type=int,
                    dest="num_intra_threads", default=None, help="num-intra-threads")
parser.add_argument("--num-inter-threads", type=int,
                    dest="num_inter_threads", default=None, help="num-inter-threads")

args = parser.parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0
#TARGET_AUC = 0.6
TARGET_AUC = 0.83
current_auc = 0.0
lower_than_current_cnt = 0

#TOTAL_TRAIN_SIZE = 512000
TOTAL_TRAIN_SIZE = 5120000
#TOTAL_TRAIN_SIZE = 51200


def prepare_data(input, target, maxlen=None, return_neg=False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    if args.data_type == 'FP32':
        data_type = 'float32'
    elif args.data_type == 'FP16':
        data_type = 'float16'
    else:
        raise ValueError("Invalid model data type: %s" % args.data_type)
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype(data_type)
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)


def eval(sess, test_data, model, model_path, test_prepared = None):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    eval_time = 0
    prepare_time = 0

    sample_freq = 70000
    options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    start_prepare_time = time.time()
    prepared_data = []
    if test_prepared:
        prepared_data = test_prepared
        for data in test_prepared:
            nums += 1
            sys.stdout.flush()
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = data
            end_prepare_time = time.time()
            # print("begin evaluation")
            start_time = time.time()
    
            if nums % sample_freq == 0:
                prob, loss, acc, aux_loss = model.calculate(sess,
                                                            [uids, mids, cats, mid_his, cat_his, mid_mask,
                                                                target, sl, noclk_mids, noclk_cats],
                                                            timeline=True, options=options, run_metadata=run_metadata)
            else:
                prob, loss, acc, aux_loss = model.calculate(sess,
                                                            [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
            end_time = time.time()
            # print("evaluation time of one batch: %.3f" % (end_time - start_time))
            # print("end evaluation")
            eval_time += (end_time - start_time)
            prepare_time += (end_prepare_time - start_prepare_time)
            loss_sum += loss
            aux_loss_sum = aux_loss
            accuracy_sum += acc
            prob_1 = prob[:, 0].tolist()
            target_1 = target[:, 0].tolist()
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])
            # print("nums: ", nums)
            # break
            start_prepare_time = time.time()
    else:
        for src, tgt in test_data:
            nums += 1
            sys.stdout.flush()
            data = prepare_data(src, tgt, return_neg=True)
            prepared_data.append(data)
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = data
            end_prepare_time = time.time()
            # print("begin evaluation")
            start_time = time.time()
    
            if nums % sample_freq == 0:
                prob, loss, acc, aux_loss = model.calculate(sess,
                                                            [uids, mids, cats, mid_his, cat_his, mid_mask,
                                                                target, sl, noclk_mids, noclk_cats],
                                                            timeline=True, options=options, run_metadata=run_metadata)
            else:
                prob, loss, acc, aux_loss = model.calculate(sess,
                                                            [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
            end_time = time.time()
            # print("evaluation time of one batch: %.3f" % (end_time - start_time))
            # print("end evaluation")
            eval_time += (end_time - start_time)
            prepare_time += (end_prepare_time - start_prepare_time)
            loss_sum += loss
            aux_loss_sum = aux_loss
            accuracy_sum += acc
            prob_1 = prob[:, 0].tolist()
            target_1 = target[:, 0].tolist()
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])
            # print("nums: ", nums)
            # break
            start_prepare_time = time.time()

    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
    # with open('dien_1core_1inst_timeline_keras-in115_skx6248_fp32_0513_step1_ckpt.json', 'w') as f:
    #     f.write(chrome_trace)
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        if args.mode == 'train':
            model.save(sess, model_path)

    global current_auc
    global lower_than_current_cnt
    if test_auc >= current_auc:
        current_auc = test_auc
    else:
        print("current auc is %.4f and test auc is %.4f" % (current_auc, test_auc))
        lower_than_current_cnt += 1

    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, prepare_time, nums, prepared_data


def train_synthetic(
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        data_type='FP32',
        seed=2,
        n_uid=543060,
        n_mid=100000 * 300,
        n_cat=1601,
        embedding_device='gpu'
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    synthetic_input = True

    sess_config = tf.compat.v1.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False)
    with tf.compat.v1.Session(config=sess_config) as sess:
        # parameters needs to put in config file

        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type=data_type,
                              synthetic_input=synthetic_input, batch_size=batch_size, max_length=maxlen, device=embedding_device)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat,
                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat,
                                   EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat,
                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type,
                                                    synthetic_input=synthetic_input, batch_size=batch_size, max_length=maxlen, device=embedding_device)
        else:
            print("Invalid model_type : %s", model_type)
            return

        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sys.stdout.flush()

        iter = 0
        train_size = 0
        approximate_accelerator_time = 0

        for itr in range(1):
            for i in range(500):
                start_time = time.time()
                _, _, _ = model.train_synthetic_input(sess)
                end_time = time.time()
                # print("training time of one batch: %.3f" % (end_time - start_time))
                one_iter_time = end_time - start_time
                approximate_accelerator_time += one_iter_time
                iter += 1
                sys.stdout.flush()
                if (iter % 100) == 0:
                    print('iter: %d ----> speed: %.4f  QPS' %
                          (iter, 1.0 * batch_size / one_iter_time))

        print("Total recommendations: %d" % (iter * batch_size))
        print("Approximate accelerator time in seconds is %.3f" %
              approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" %
              (float(iter * batch_size)/float(approximate_accelerator_time)))


def train(
        train_file="local_train_splitByUser",
        test_file="local_test_splitByUser",
        uid_voc="uid_voc.pkl",
        mid_voc="mid_voc.pkl",
        cat_voc="cat_voc.pkl",
        batch_size=128,
        maxlen=100,
        test_iter=500,
        save_iter=500,
        model_type='DNN',
        data_type='FP32',
    seed=2,
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

    session_config = tf.compat.v1.ConfigProto()
    if args.num_intra_threads and args.num_inter_threads:
        session_config.intra_op_parallelism_threads = args.num_intra_threads
        session_config.inter_op_parallelism_threads = args.num_inter_threads

    #hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    session_start_time = time.time()
    #with tf.compat.v1.train.MonitoredTrainingSession(config=session_config, hooks=hooks) as sess:
    with tf.compat.v1.Session(config=session_config) as sess:
        sess.run(hvd.broadcast_global_variables(0))
        # with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(
            train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
        test_data = DataIterator(
            test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        # Number of uid = 543060, mid = 367983, cat = 1601 for Amazon dataset
        print("Number of uid = %i, mid = %i, cat = %i" % (n_uid, n_mid, n_cat))
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type=data_type,
                              batch_size=batch_size, max_length=maxlen)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat,
                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat,
                                   EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat,
                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type,
                                                    batch_size=batch_size, max_length=maxlen, device='cpu')
        else:
            print("Invalid model_type : %s", model_type)
            return
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print("global variable dtype: ", var.dtype)
        #     if var.dtype == 'float32_ref':
        #         print("global variable: ", var)
        # model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sys.stdout.flush()
        #print('test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- num_iters: %d' % eval(sess, test_data, model, best_model_path))
        sys.stdout.flush()

        # output_node_names = ["dien/fcn/Metrics/add",
        #                     "dien/fcn/Metrics/Mean_1",
        #                     "dien/aux_loss/Mean",
        #                     "dien/fcn/Metrics/Adam"]

        # print(output_node_names)
        # graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        # frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess,
        #     sess.graph_def,
        #     output_node_names)
        # tf.io.write_graph(frozen_graph_def, '/home2/yunfeima/tmp/dien', 'constant_sub_node_adam_train.pb',as_text=False)
        # exit(0)

        iter = 0
        lr = 0.001
        train_size = 0
        approximate_accelerator_time = 0

        session_end_time = time.time()
        session_init_elapse_time = session_end_time - session_start_time

        data_load_elapse_time = 0
        save_elapse_time = 0
        test_elapse_time = 0
        train_elapse_time = 0
        test_prepare_time = 0
        test_prepared = None
        for itr in range(1):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.

            sample_freq = 20000
            options = tf.compat.v1.RunOptions(
                trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            nums = 0
            elapsed_time_records = []

            total_data = []

            data_load_start_time = time.time()
            for src, tgt in train_data:
                nums += 1
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
                    src, tgt, maxlen, return_neg=True)
                data_load_end_time = time.time()
                data_load_elapse_time += (data_load_end_time -
                                          data_load_start_time)

                total_data.append(
                    [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
                data_load_start_time = time.time()

            # prepared_filename = "dien_data_trained_prepared_batch_{}.pickle".format(batch_size)
            # with open(prepared_filename, 'wb') as f:
            #     pickle.dump(total_data, f)

            # prepared_filename = "dien_data_trained_prepared_batch_{}.pickle".format(batch_size)
            # with open(prepared_filename, 'rb') as f:
            #     total_data = pickle.load(f)

            nums = 0
            for i in range(len(total_data)):
                nums += 1
                # print("iter", nums)

                # if nums == sample_freq + 2:
                #     break

                # if nums == 400:
                #     break

                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = tuple(
                    total_data[i])

                start_time = time.time()
                try:
                    if nums == sample_freq:
                        loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats],
                                                          timeline_flag=True, options=options, run_metadata=run_metadata, step=nums)
                    else:
                        loss, acc, aux_loss = model.train(
                            sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats])
                    loss_sum += loss
                    accuracy_sum += acc
                    aux_loss_sum += aux_loss
                except:
                    pass
                end_time = time.time()

                # print("step:", nums)
                # print(cat_his.shape)
                # print(noclk_mids.shape)

                # print("training time of one batch: %.3f" % (end_time - start_time))
                approximate_accelerator_time += end_time - start_time
                train_elapse_time += (end_time - start_time)
                elapsed_time_records.append(end_time - start_time)

                iter += 1
                train_size += batch_size
                sys.stdout.flush()
                test_auc = 0.0
                if (iter % test_iter) == 0:
                    # print("train_size: %d" % train_size)
                    # print("approximate_accelerator_time: %.3f" % approximate_accelerator_time)
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' %
                          (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    try:
                        test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, prepare_time, nums, test_prepared = eval(
                            sess, test_data, model, best_model_path, test_prepared)
                        print(' test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- num_iters: %d' %
                              (test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, nums))
                        test_elapse_time += eval_time
                        test_prepare_time += prepare_time
                    except:
                        pass

                    # delete test every 100 iterations no need in training time
                    # print(' test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- num_iters: %d' % eval(sess, test_data, model, best_model_path))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0 and hvd.rank() == 0:
                    # if (iter % 10000) == 0:
                    print('save model iter: %d' % (iter))
                    save_start_time = time.time()
                    model.save(sess, model_path+"--"+str(iter))
                    save_end_time = time.time()
                    save_elapse_time += (save_end_time - save_start_time)
                if train_size >= TOTAL_TRAIN_SIZE:
                    break
                if current_auc >= TARGET_AUC:
                    break
                if lower_than_current_cnt >= 2:
                    break

            # with open('./times/24core_1inst_train_timeline_g210_8260_nosparse_adam_disable_noaddn_batch128_newunsortedsum_inter4_intra6_omp6_0615.txt', 'w') as wf:
            # with open('./times/dien_24core_1inst_train_timeline_8260_fp32_step{}_im0617allmklinput_tanhfusion_inter1_intra24_omp24_0701.txt'.format(nums), 'w') as wf:
            # with open('./times/dien_train_timeline_fp32_step{}.txt'.format(nums), 'w') as wf:
            #   for time_per_iter in elapsed_time_records:
            #       wf.write(str(time_per_iter) + '\n')

            print("iteration: ", nums)

            lr *= 0.5
            if train_size >= TOTAL_TRAIN_SIZE:
                break
        print("iter: %d" % iter)
        print("Total recommendations: %d" % TOTAL_TRAIN_SIZE)
        print("Approximate accelerator time in seconds is %.3f" %
              approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" %
              (float(TOTAL_TRAIN_SIZE)/float(approximate_accelerator_time)))
        print("process time breakdown in seconds are session_init %.3f, train_prepare %.3f, train %.3f, test_prepare %.3f, test %.3f, model save %.3f" %
              (session_init_elapse_time, data_load_elapse_time, train_elapse_time, test_prepare_time, test_elapse_time, save_elapse_time))


def test(
        train_file="local_train_splitByUser",
        test_file="local_test_splitByUser",
        uid_voc="uid_voc.pkl",
        mid_voc="mid_voc.pkl",
        cat_voc="cat_voc.pkl",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        data_type='FP32',
    seed=2
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_best_model_trained/ckpt_noshuff" + model_type + str(seed)
    print(model_path)

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

    # with tf.io.gfile.GFile("/home2/yunfeima/tmp/dien/frozen_graph.pb", "rb") as f:
    #     graph_def = tf.compat.v1.GraphDef()
    #     graph_def.ParseFromString(f.read())
    # with tf.Graph().as_default() as graph:
    #     tf.import_graph_def(graph_def, name='')
    #     # pass

    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    #         intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as sess:
    sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    with tf.compat.v1.Session(config=sess_config) as sess:
        train_data = DataIterator(
            train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        test_data = DataIterator(
            test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat,
                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat,
                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat,
                                   EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat,
                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type, device='cpu')
        else:
            print("Invalid model_type : %s", model_type)
            return
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print("global variable: ", var)
        if data_type == 'FP32':
            model.restore(sess, model_path)

            # output_node_names = ["dien/fcn/add_6",
            #                     "dien/fcn/Metrics/add",
            #                     "dien/fcn/Metrics/Mean_1",
            #                     "dien/aux_loss/Mean"]
            # output_node_names = ["dien/fcn/add_6",
            #                     "dien/fcn/Metrics/Mean_1"]

            # print(output_node_names)
            # graph_def = tf.compat.v1.get_default_graph().as_graph_def()
            # frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            #     sess,
            #     sess.graph_def,
            #     output_node_names)
            # tf.io.write_graph(frozen_graph_def, '/home2/yunfeima/tmp/dien', 'constant_sub_node_fixed_reshape.pb',as_text=False)
            # exit(0)

        if data_type == 'FP16':
            fp32_variables = [var_name for var_name,
                              _ in tf.contrib.framework.list_variables(model_path)]
            #print("fp32_variables: ", fp32_variables)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            for variable in tf.global_variables():
                #print("variable: ", variable)
                if variable.op.name in fp32_variables:
                    var = tf.contrib.framework.load_variable(
                        model_path, variable.op.name)
                    # print("var: ", var)
                    # print("var.dtype: ", var.dtype)
                    if(variable.dtype == 'float16_ref'):
                        tf.add_to_collection(
                            'assignOps', variable.assign(tf.cast(var, tf.float16)))
                        # print("var value: ", sess.run(tf.cast(var, tf.float16)))
                    else:
                        tf.add_to_collection('assignOps', variable.assign(var))
                else:
                    raise ValueError(
                        "Variable %s is missing from checkpoint!" % variable.op.name)
            sess.run(tf.get_collection('assignOps'))
            # for variable in sess.run(tf.get_collection('assignOps')):
            #     print("after load checkpoint: ", variable)
        # for variable in tf.global_variables():
        #     print("after load checkpoint: ", sess.run(variable))
        approximate_accelerator_time = 0
        test_elapse_time = 0
        prepare_elapse_time = 0
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time, num_iters, test_prepared = eval(
            sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        test_elapse_time += eval_time
        prepare_elapse_time += prepare_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- prepare_time: %.3f' %
              (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time, num_iters, test_prepared = eval(
            sess, test_data, model, model_path, test_prepared)
        approximate_accelerator_time += eval_time
        test_elapse_time += eval_time
        prepare_elapse_time += prepare_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- prepare_time: %.3f' %
              (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time, num_iters, test_prepared = eval(
            sess, test_data, model, model_path, test_prepared)
        approximate_accelerator_time += eval_time
        test_elapse_time += eval_time
        prepare_elapse_time += prepare_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- prepare_time: %.3f' %
              (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time, num_iters, test_prepared = eval(
            sess, test_data, model, model_pat, test_prepared)
        approximate_accelerator_time += eval_time
        test_elapse_time += eval_time
        prepare_elapse_time += prepare_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- prepare_time: %.3f' %
              (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time, num_iters, test_prepared = eval(
            sess, test_data, model, model_path, test_prepared)
        approximate_accelerator_time += eval_time
        test_elapse_time += eval_time
        prepare_elapse_time += prepare_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- prepare_time: %.3f' %
              (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, prepare_time))
        print("Total recommendations: %d" % (num_iters*batch_size))
        print("Approximate accelerator time in seconds is %.3f" %
              approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" %
              (float(5*num_iters*batch_size)/float(approximate_accelerator_time)))
        print("Process time breakdown, prepare data took %.3f and test took %.3f, avg is prepare %.3f, test %.3f" % (
            prepare_elapse_time, test_elapse_time, prepare_elapse_time/5, test_elapse_time/5,))


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_control_flow_v2()
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_resource_variables()
    tf.compat.v1.disable_tensor_equality()
    tf.compat.v1.disable_v2_tensorshape()
    SEED = args.seed
    if tf.__version__[0] == '1':
        tf.compat.v1.set_random_seed(SEED)
    elif tf.__version__[0] == '2':
        tf.random.set_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if args.mode == 'train':
        train(model_type=args.model, seed=SEED,
              batch_size=args.batch_size, data_type=args.data_type)
    elif args.mode == 'test':
        test(model_type=args.model, seed=SEED,
             batch_size=args.batch_size, data_type=args.data_type)
    elif args.mode == 'synthetic':
        train_synthetic(model_type=args.model, seed=SEED, batch_size=args.batch_size,
                        data_type=args.data_type, embedding_device=args.embedding_device
                        )
    else:
        print('do nothing...')
