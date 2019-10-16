"""
script developing 2d complex SPN
for synthetic sine data
"""
import numpy as np
import logging
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('../../2dgaussian/SPFlow/src/')
from spn.algorithms.Inference import log_likelihood


current_time=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
logging.basicConfig(filename='/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev/spn_2d_'+current_time+'.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def learn_whittle_spn_1d(train_data, n_RV, n_min_slice=2000, init_scope=None):
    from spn.structure.Base import Context
    from spn.structure.leaves.parametric.Parametric import Gaussian
    from spn.algorithms.LearningWrappers import learn_parametric
    # learn spn
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # l_rfft=None --> 1d gaussian node, is_pair does not work
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=0.4,
                            initial_scope=init_scope, cpus=4, l_rfft=None, is_pair=False)
    if ARGS.data_type==1:
        save_path = './dev/sine/wspn1d_'+str(n_min_slice)+'/'
    else:
        save_path = './dev/mnist/wspn1d_'+str(n_min_slice)+'/'
    check_path(save_path)
    f = open(save_path+'wspn_1d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_1d(n_min_slice):
    if ARGS.data_type==1:
        save_path = './dev/sine/wspn1d_'+str(n_min_slice)+'/'
    else:
        save_path = './dev/mnist/wspn1d_'+str(n_min_slice)+'/'
    f = open(save_path+'wspn_1d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()
    from spn.algorithms.Statistics import get_structure_stats
    print(get_structure_stats(spn))

    return spn


def learn_whittle_spn_2d(train_data, n_RV, n_min_slice, init_scope=None):

    from spn.structure.Base import Context
    from spn.structure.leaves.parametric.Parametric import MultivariateGaussian
    from spn.structure.leaves.parametric.Parametric import Gaussian
    from spn.algorithms.LearningWrappers import learn_parametric

    # learn spn
    # ds_context = Context(parametric_types=[MultivariateGaussian] * n_RV).add_domains(train_data)
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # need to pair RVs
    # need flag for 2d?
    if ARGS.data_type==2:
        l_rfft = 8
    else:
        l_rfft = 17
    # l_rfft!=None --> 2d/pair gaussian node, is_pair=False --> 2d gaussian, diagonal covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, initial_scope=init_scope,
                            cpus=4, l_rfft=l_rfft, is_pair=False)
    if ARGS.data_type==1:
        save_path = './dev/sine/wspn2d_'+str(n_min_slice)+'/'
    elif ARGS.data_type==2:
        save_path = './dev/mnist/wspn2d_'+str(n_min_slice)+'/'
    elif ARGS.data_type==3:
        save_path = './dev/SP/wspn2d_' + str(n_min_slice) + '/'
    else:
        save_path = './dev/stock/wspn2d_' + str(n_min_slice) + '/'
    check_path(save_path)
    f = open(save_path+'wspn_2d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_2d(n_min_slice):
    if ARGS.data_type==1:
        save_path = './dev/sine/wspn2d_'+str(n_min_slice)+'/'
    elif ARGS.data_type==2:
        save_path = './dev/mnist/wspn2d_'+str(n_min_slice)+'/'
    else:
        save_path = './dev/SP/wspn2d_'+str(n_min_slice)+'/'
    f = open(save_path+'wspn_2d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()
    from spn.algorithms.Statistics import get_structure_stats
    print(get_structure_stats(spn))

    return spn


def learn_whittle_spn_pair(train_data, n_RV, n_min_slice, init_scope=None):

    from spn.structure.Base import Context
    from spn.structure.leaves.parametric.Parametric import MultivariateGaussian
    from spn.structure.leaves.parametric.Parametric import Gaussian
    from spn.algorithms.LearningWrappers import learn_parametric

    # learn spn
    ds_context = Context(parametric_types=[MultivariateGaussian] * n_RV).add_domains(train_data)
    # ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # need to pair RVs
    # need flag for 2d?
    if ARGS.data_type==1:
        l_rfft = 17
    else:
        l_rfft = 8
    # l_rfft!=None --> 2d/pair gaussian node, is_pair=True --> pairwise gaussian, full covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, initial_scope=init_scope,
                            cpus=4, l_rfft=l_rfft, is_pair=True)
    if ARGS.data_type==1:
        save_path = './dev/sine/wspn_pair_'+str(n_min_slice)+'/'
    else:
        save_path = './dev/mnist/wspn_pair_'+str(n_min_slice)+'/'
    check_path(save_path)
    f = open(save_path+'wspn_pair.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_pair(n_min_slice):
    if ARGS.data_type==1:
        save_path = './dev/sine/wspn_pair_'+str(n_min_slice)+'/'
    else:
        save_path = './dev/mnist/wspn_pair_'+str(n_min_slice)+'/'
    f = open(save_path+'wspn_pair.pkl', 'rb')
    spn = pickle.load(f)
    f.close()
    from spn.algorithms.Statistics import get_structure_stats
    print(get_structure_stats(spn))

    return spn


def data_to_2d(data, p, L):
    # transfer data from 1d to 2d
    h, w = data.shape
    l = L//2+1
    data1 = data.reshape(h * p, -1)
    data1_r = data1[:, 0:l].reshape(h * p, l, 1)
    data1_i = data1[:, l:].reshape(h * p, l, 1)
    data2 = np.concatenate([data1_r, data1_i], 2)
    data2 = data2.reshape(h, -1, 2)

    return data2


def check_path(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--wspn_type', type=int, default=2,
                        help='Type of wspn, 1-1d, 2-2d, 3-pair')
    parser.add_argument('--train_type', type=int, default=2,
                        help='Type of train, 1-train, 2-test')
    parser.add_argument('--n_min_slice', type=int, default=1100,
                        help='minimum size of slice.')
    parser.add_argument('--data_type', type=int, default=2,
                        help='Type of data, 1-sine, 2-mnist, 3-S&P')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Threshold of splitting features')

    ARGS, unparsed = parser.parse_known_args()

    start_time = time.time()
    np.random.seed(2019052799)

    if ARGS.data_type==1:
        print('loading sine data')
        data_train = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/train_sine.dat',
                                 dtype=np.float64).reshape(-1, 204)
        data_pos = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/test_sine_positive.dat',
                                 dtype=np.float64).reshape(-1, 204)
        data_neg = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/test_sine_negative.dat',
                                 dtype=np.float64).reshape(-1, 204)
        n_RV = 204  # number of RVs
        p = 6  # dim
        L = 32  # TS length
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
        # init_scope = np.delete(scope_list, np.where(scope_list % 34 == 33))
    elif ARGS.data_type==2:
        print('loading mnist data')
        data_train = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/train_mnist.dat',
                                 dtype=np.float64).reshape(-1, 224)
        data_pos = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/test_mnist_positive.dat',
                                 dtype=np.float64).reshape(-1, 224)
        data_neg = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/test_mnist_negative.dat',
                                 dtype=np.float64).reshape(-1, 224)
        n_RV = 224  # number of RVs
        p = 14  # dim
        L = 14  # TS length
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 16 == 8))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 16 == 15)))
        # data_train = data_train[0:100, :]
    elif ARGS.data_type==3:
        print('loading S&P data')
        data_train = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/train_SP.dat',
                                 dtype=np.float64).reshape(-1, 374)
        data_pos = data_train.copy()
        data_neg = data_train.copy()
        n_RV = 374  # number of RVs
        p = 11  # dim
        L = 32  # TS length
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
    else:
        sys.exit()
    print('data done')

    if ARGS.wspn_type==1:
        # train/load wspn 1d
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type==1:
            wspn = learn_whittle_spn_1d(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type==2:
            wspn = load_whittle_spn_1d(n_min_slice)
            #f = open('./big_mixture_spn_yu_sine_EM_optimized_70_EM_iter', 'rb')
            #wspn = pickle.load(f)
            #f.close()
            # calculate LL
            ll_train = np.median(log_likelihood(wspn, data_train))
            print('LL_train=', ll_train)
            ll_pos = np.median(log_likelihood(wspn, data_pos))
            print('LL_positive=', ll_pos)
            ll_neg = np.median(log_likelihood(wspn, data_neg))
            print('LL_negative=', ll_neg)
    elif ARGS.wspn_type==2:
        # train/load wspn 2d
        n_min_slice = ARGS.n_min_slice
        # data_train_2 = data_to_2d(data_train, p, L)
        if ARGS.train_type == 1:
            # data_train = data_train[0:1000, 0:34]
            # n_RV = 34
            wspn = learn_whittle_spn_2d(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type==2:
            wspn = load_whittle_spn_2d(n_min_slice)
            # calculate LL
            ll_train = np.median(log_likelihood(wspn, data_train))
            print('LL_train=', ll_train)
            ll_pos = np.median(log_likelihood(wspn, data_pos))
            print('LL_positive=', ll_pos)
            ll_neg = np.median(log_likelihood(wspn, data_neg))
            print('LL_negative=', ll_neg)
    elif ARGS.wspn_type==3:
        # train/load wspn pair
        n_min_slice = ARGS.n_min_slice
        # data_train_2 = data_to_2d(data_train, p, L)
        if ARGS.train_type == 1:
            # data_train = data_train[0:1000, 0:34]
            # n_RV = 34
            wspn = learn_whittle_spn_pair(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type==2:
            wspn = load_whittle_spn_pair(n_min_slice)
            # calculate LL
            ll_train = np.median(log_likelihood(wspn, data_train))
            print('LL_train=', ll_train)
            ll_pos = np.median(log_likelihood(wspn, data_pos))
            print('LL_positive=', ll_pos)
            ll_neg = np.median(log_likelihood(wspn, data_neg))
            print('LL_negative=', ll_neg)





