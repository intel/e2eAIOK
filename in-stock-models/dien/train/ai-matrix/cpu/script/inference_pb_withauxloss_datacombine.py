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

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="mode, train or test")
parser.add_argument("--model", type=str, default='DIEN', help="model")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_type", type=str, default='FP32', help="data type: FP32 or FP16")
parser.add_argument("--num_accelerators", type=int, default=1, help="number of accelerators used for training")
parser.add_argument("--embedding_device", type=str, default='gpu', help="synthetic input embedding layer reside on gpu or cpu")
parser.add_argument("--pb_path", type=str, default='', help="path for frozen pb")
parser.add_argument("--num-intra-threads", type=int, dest="num_intra_threads", default=None, help="num-intra-threads")
parser.add_argument("--num-inter-threads", type=int, dest="num_inter_threads", default=None, help="num-inter-threads")
args = parser.parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

TOTAL_TRAIN_SIZE = 512000
#TOTAL_TRAIN_SIZE = 16000


def prepare_data(input, target, maxlen = None, return_neg = False):
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
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
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

def calculate(sess, test_data, input_tensor, output_tensor):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    eval_time = 0

    sample_freq = 10000
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    # my_profiler = model_analyzer.Profiler(graph=sess.graph)

    elapsed_time_records = []
    for src, tgt in test_data:
        nums += 1

        # if nums == 22:
        #     break

        prepare_start = time.time()
        sys.stdout.flush()
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        feed_data  = [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats]
        prepare_end = time.time()
        # print("prepare time", prepare_end - prepare_start)

        # print("steps: ", nums)
 
        # print("uids:",uids.shape)
        # print("mids:",mids.shape)
        # print("cats:",cats.shape)
        # print("mid_his:",mid_his.shape)
        # print("cat_his:",cat_his.shape)
        # print("mid_mask:",mid_mask.shape)
        # print("target:", target.shape)
        # print("sl:",sl.shape)
        # print("noclk_mids:",noclk_mids.shape)
        # print("noclk_cats:",noclk_cats.shape)
        
        # print("sl data: ", sl)

        # print("begin evaluation")
        start_time = time.time()

        if nums % sample_freq == 0:
        # if nums == 700:
            prob, loss, acc, aux_loss = sess.run(output_tensor, 
                options=options, 
                run_metadata=run_metadata, 
                feed_dict=dict(zip(input_tensor, feed_data)))
            # my_profiler.add_step(step=nums, run_meta=run_metadata)
        else:
            prob, loss, acc, aux_loss = sess.run(output_tensor, 
                feed_dict=dict(zip(input_tensor, feed_data)))
        end_time = time.time()
        # print("evaluation time of one batch: %.3f" % (end_time - start_time))
        # print("end evaluation")
        eval_time += end_time - start_time
        elapsed_time_records.append(end_time - start_time)
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        # print("nums: ", nums)
        # break
        
        if nums % sample_freq == 0:
        # if nums == 700:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('./timeline/dien_1core_1inst_timeline_g115_skx6248_fp32_step{}_0515_newset_pb_2.json'.format(nums), 'w') as f:
                f.write(chrome_trace)

    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums

    # with open('./times/1core_1inst_g115_time_record-0515-2.txt', 'w') as wf:
    #     for time_per_iter in elapsed_time_records:
    #         wf.write(str(time_per_iter) + '\n')

    # profile_op_builder = option_builder.ProfileOptionBuilder()
    # profile_op_builder.select(['micros', 'occurrence'])
    # profile_op_builder.order_by('micros')
    # profile_op_builder.with_max_depth(4)
    # my_profiler.profile_graph(profile_op_builder.build())   
    
    # test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, nums = [1] * 6
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, nums


def inference(pb_path,      
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        data_type = 'FP32',
	    seed = 2):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    print(pb_path)

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    
    input_layers = ["Inputs/uid_batch_ph",
                    "Inputs/mid_batch_ph",
                    "Inputs/cat_batch_ph",
                    "Inputs/mid_his_batch_ph",  
                    "Inputs/cat_his_batch_ph",
                    "Inputs/mask",
                    "Inputs/target_ph",
                    "Inputs/seq_len_ph",
                    "Inputs/noclk_mid_batch_ph",
                    "Inputs/noclk_cat_batch_ph"]
    input_tensor = [graph.get_tensor_by_name(x + ":0") for x in input_layers]
    output_layers = ["dien/fcn/add_6", 
                    "dien/fcn/Metrics/add", 
                    "dien/fcn/Metrics/Mean_1",
                    "dien/aux_loss/Mean"]
    output_tensor = [graph.get_tensor_by_name(x + ":0") for x in output_layers]

    # gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    # with tf.compat.v1.Session(graph=graph,config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:

    session_config = tf.compat.v1.ConfigProto()
    if args.num_intra_threads and args.num_inter_threads:
        session_config.intra_op_parallelism_threads = args.num_intra_threads
        session_config.inter_op_parallelism_threads = args.num_inter_threads
    with tf.compat.v1.Session(graph=graph, config=session_config) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()

        approximate_accelerator_time = 0
        
        niters = 1
        for i in range(niters):
            test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = calculate(sess, test_data, input_tensor, output_tensor)
            approximate_accelerator_time += eval_time
            print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        print("num_iters ", num_iters)
        print("batch_size ", batch_size)
        print("niters ", niters)
        print("Total recommendations: %d" % (num_iters*batch_size))
        print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" % (float(niters*num_iters*batch_size)/float(approximate_accelerator_time)))

if __name__ == '__main__':
    SEED = args.seed
    if tf.__version__[0] == '1':
        tf.compat.v1.set_random_seed(SEED)  
    elif tf.__version__[0] == '2':
        tf.random.set_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    # if args.mode == 'train':
    #     train(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
    # elif args.mode == 'test':
    #     test(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
    # elif args.mode == 'synthetic':
    #     train_synthetic(model_type=args.model, seed=SEED, batch_size=args.batch_size, 
    #     data_type=args.data_type, embedding_device = args.embedding_device
    #     ) 
    # else:
    #     print('do nothing...')
    pb_path = args.pb_path
    inference(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type, pb_path=pb_path)

