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
from spn.algorithms.Statistics import get_structure_stats
from spn.structure.Base import Context
from spn.algorithms.LearningWrappers import learn_parametric


# current_time=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
# logging.basicConfig(filename='/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev/whittle_spn_'+current_time+'.log', filemode='w', level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


def get_save_path(ARGS):
    if ARGS.wspn_type == 1:
        key = 'wspn1d'
    elif ARGS.wspn_type == 2:
        key = 'wspn2d'
    elif ARGS.wspn_type == 3:
        key = 'wspn_pair'
    else:
        print('input spn type error')
        sys.exit()

    if ARGS.data_type==1:
        data = 'sine/'
    elif ARGS.data_type==2:
        data = 'mnist/'
    elif ARGS.data_type==3:
        data = 'SP/'
    elif ARGS.data_type==4:
        data = 'stock/'
    elif ARGS.data_type==5:
        data = 'billiards/'
    else:
        print('input data type error')
        sys.exit()

    save_path = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev_aies/' + data + key + '_' + str(ARGS.n_min_slice) + '_' + str(ARGS.threshold) + '/'

    return save_path


def get_l_rfft(ARGS):
    if ARGS.data_type==1:
        l_rfft = 17
    elif ARGS.data_type==2:
        l_rfft = 8
    elif ARGS.data_type==3:
        l_rfft = 17
    elif ARGS.data_type==4:
        l_rfft = 17
    elif ARGS.data_type==5:
        l_rfft = 51
    else:
        print('input l_rfft error')
        sys.exit()

    return l_rfft


def learn_whittle_spn_1d(train_data, n_RV, n_min_slice=2000, init_scope=None):
    from spn.structure.leaves.parametric.Parametric import Gaussian

    # learn spn
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # l_rfft=None --> 1d gaussian node, is_pair does not work
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=ARGS.threshold,
                            initial_scope=init_scope, cpus=4, l_rfft=None, is_pair=False)
    save_path = get_save_path(ARGS)
    check_path(save_path)
    f = open(save_path + 'wspn_1d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_1d(ARGS):

    save_path = get_save_path(ARGS)
    f = open(save_path+'wspn_1d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    logger.info(log_msg)
    return spn


def learn_whittle_spn_2d(train_data, n_RV, n_min_slice, init_scope=None):
    from spn.structure.leaves.parametric.Parametric import Gaussian

    # learn spn
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # need to pair RVs
    # need flag for 2d?
    l_rfft = get_l_rfft(ARGS)
    # l_rfft!=None --> 2d/pair gaussian node, is_pair=False --> 2d gaussian, diagonal covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=ARGS.threshold,
                            initial_scope=init_scope, cpus=4, l_rfft=l_rfft, is_pair=False)
    save_path = get_save_path(ARGS)
    check_path(save_path)
    f = open(save_path+'wspn_2d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_2d(ARGS):
    save_path = get_save_path(ARGS)
    f = open(save_path+'wspn_2d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    logger.info(log_msg)

    return spn


def learn_whittle_spn_pair(train_data, n_RV, n_min_slice, init_scope=None):

    from spn.structure.leaves.parametric.Parametric import MultivariateGaussian

    # learn spn
    ds_context = Context(parametric_types=[MultivariateGaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # need to pair RVs
    # need flag for 2d?
    l_rfft = get_l_rfft(ARGS)
    # l_rfft!=None --> 2d/pair gaussian node, is_pair=True --> pairwise gaussian, full covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=ARGS.threshold,
                            initial_scope=init_scope, cpus=4, l_rfft=l_rfft, is_pair=True)
    save_path = get_save_path(ARGS)
    check_path(save_path)
    f = open(save_path+'wspn_pair.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_pair(ARGS):
    save_path = get_save_path(ARGS)
    f = open(save_path+'wspn_pair.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    logger.info(log_msg)

    return spn


def load_whittle_spn_res(ARGS):
    # load res-spn, need to be modified when model changed
    log_msg = 'Have you set the latest model path?'
    print(log_msg)
    logger.info(log_msg)

    rspn_path = 'ventola/em_optimized_fuse_spn_yu_mnist'
    f = open(rspn_path, 'rb')
    rspn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(rspn)
    print(log_msg)
    logger.info(log_msg)

    return rspn


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


def load_data_for_wspn(ARGS):
    if ARGS.data_type==1:
        log_msg = 'loading sine data'
        print(log_msg)
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
    elif ARGS.data_type==2:
        log_msg = 'loading mnist data'
        print(log_msg)
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
        log_msg = 'loading S&P data'
        print(log_msg)
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
    elif ARGS.data_type==4:
        log_msg = 'loading Stock data'
        print(log_msg)
        data_train = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/train_stock.dat',
                                 dtype=np.float64).reshape(-1, 578)
        data_pos = data_train.copy()
        data_neg = data_train.copy()
        n_RV = 578  # number of RVs
        p = 17  # dim
        L = 32  # TS length
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
    elif ARGS.data_type==5:
        log_msg = 'loading Billiards data'
        print(log_msg)
        data_path = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/datasets/billiards_data/'
        # Load training data
        data = pickle.load(open(data_path+'billiards_train.pkl', 'rb'))
        # extract data and do DTFT
        positions = data['y']
        positions = positions[..., :2]
        data_rfft = np.fft.rfft(positions, axis=1)
        d_r = data_rfft.real
        d_i = data_rfft.imag
        data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
        data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
        data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
        data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
        data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
        data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
        # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
        data_train = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

        # Load test data
        data = pickle.load(open(data_path+'billiards_test.pkl', 'rb'))
        # extract data and do DTFT
        positions = data['y']
        positions = positions[..., :2]
        data_rfft = np.fft.rfft(positions, axis=1)
        d_r = data_rfft.real
        d_i = data_rfft.imag
        data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
        data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
        data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
        data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
        data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
        data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
        # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
        data_pos = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

        # Load outlier data
        data = pickle.load(open(data_path+'billiards_test_2.pkl', 'rb'))
        # extract data and do DTFT
        positions = data['y']
        positions = positions[..., :2]
        data_rfft = np.fft.rfft(positions, axis=1)
        d_r = data_rfft.real
        d_i = data_rfft.imag
        data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
        data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
        data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
        data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
        data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
        data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
        # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
        data_neg = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

        n_RV = 612  # number of RVs
        p = 6  # dim
        L = 100  # TS length
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 102 == 51))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 102 == 101)))
    else:
        print('input data error')
        sys.exit()
    print('data done')
    logger.info(log_msg)
    logger.info('data done')
    return data_train, data_pos, data_neg, n_RV, p, L, init_scope


def check_path(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def calc_ll(wspn, data_train, data_pos, data_neg):
    # calculate LL
    log_msg = 'Log-likelihood calculating...'
    print(log_msg)
    logger.info(log_msg)

    ll_train = log_likelihood(wspn, data_train)
    ll_pos = log_likelihood(wspn, data_pos)
    ll_neg = log_likelihood(wspn, data_neg)
    log_msg = '---------median-----------'
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_train=' + str(np.median(ll_train))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_positive=' + str(np.median(ll_pos))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_negative=' + str(np.median(ll_neg))
    print(log_msg)
    logger.info(log_msg)
    log_msg = '--------- mean -----------'
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_train=' + str(np.mean(ll_train))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_positive=' + str(np.mean(ll_pos))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_negative=' + str(np.mean(ll_neg))
    print(log_msg)
    logger.info(log_msg)


def init_log(ARGS):
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # Creating log file
    path_base = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev/'
    if ARGS.train_type == 1:
        file_base = 'train_wspn_' + str(ARGS.wspn_type) + '_on_data' + str(ARGS.data_type) + '_'
    elif ARGS.train_type == 2:
        file_base = 'test_wspn' + str(ARGS.wspn_type) + '_on_data' + str(ARGS.data_type) + '_'
    else:
        file_base = 'error'
    logging.basicConfig(
        filename=path_base + file_base + current_time + '.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    return logger


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--wspn_type', type=int, default=1,
                        help='Type of wspn, 1-1d, 2-2d, 3-pair, 4-res-spn')
    parser.add_argument('--train_type', type=int, default=1,
                        help='Type of train, 1-train, 2-test')
    parser.add_argument('--n_min_slice', type=int, default=2100,
                        help='minimum size of slice.')
    parser.add_argument('--data_type', type=int, default=1,
                        help='Type of data, 1-sine, 2-mnist, 3-S&P, 4-stock, 5-billiards')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Threshold of splitting features')

    ARGS, unparsed = parser.parse_known_args()

    # init logger
    logger = init_log(ARGS)
    log_msg = '\n--wspn_type=' + str(ARGS.wspn_type) + \
              '\n--train_type=' + str(ARGS.train_type) + \
              '\n--n_min_slice=' + str(ARGS.n_min_slice) + \
              '\n--data_type=' + str(ARGS.data_type) + \
              '\n--threshold=' + str(ARGS.threshold)
    print(log_msg)
    logger.info(log_msg)
    start_time = time.time()
    np.random.seed(2019052799)

    # load data and data_info
    data_train, data_pos, data_neg, n_RV, p, L, init_scope = load_data_for_wspn(ARGS)

    if ARGS.wspn_type==1:
        # train/load wspn 1d
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type==1:
            log_msg = 'Train WSPN 1d'
            logger.info(log_msg)
            wspn = learn_whittle_spn_1d(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type==2:
            log_msg = 'Test WSPN 1d'
            logger.info(log_msg)
            wspn = load_whittle_spn_1d(ARGS)
            calc_ll(wspn, data_train, data_pos, data_neg)

    elif ARGS.wspn_type==2:
        # train/load wspn 2d
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type == 1:
            log_msg = 'Train WSPN pair'
            logger.info(log_msg)
            wspn = learn_whittle_spn_2d(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type==2:
            log_msg = 'Test WSPN pair'
            logger.info(log_msg)
            wspn = load_whittle_spn_2d(ARGS)
            # calculate LL
            calc_ll(wspn, data_train, data_pos, data_neg)

    elif ARGS.wspn_type==3:
        # train/load wspn pair
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type == 1:
            log_msg = 'Train WSPN 2d'
            logger.info(log_msg)
            wspn = learn_whittle_spn_pair(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type==2:
            log_msg = 'Test WSPN 2d'
            logger.info(log_msg)
            wspn = load_whittle_spn_pair(ARGS)
            # calculate LL
            calc_ll(wspn, data_train, data_pos, data_neg)

    elif ARGS.wspn_type==4:
        # train/load W-res-spn
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type == 1:
            log_msg = 'Cannot train here'
            logger.info(log_msg)
            sys.exit()
        elif ARGS.train_type==2:
            log_msg = 'Test Res-SPN'
            logger.info(log_msg)
            wspn = load_whittle_spn_res(ARGS)
            # calculate LL
            calc_ll(wspn, data_train, data_pos, data_neg)


    log_msg = 'Running time: ' + str((time.time() - start_time)/60.0) + 'minutes'
    logger.info(log_msg)



