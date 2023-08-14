import numpy as np
import torch
import pandas as pd
import os
import argparse
import shutil
import random
from monai.utils import set_determinism 
import torch.nn as nn
from scipy.stats import norm

def init():
    parser = argparse.ArgumentParser("BrainAgeEstimation")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--lr_s", type=float, default=0.001, help="init learning rate"  # 5e-6
    )
    parser.add_argument("--wd_s", type=float, default=0.0001, help="weight decay") # 0.001
    parser.add_argument('--GPU_num', help="Input the amount of GPU you need", default=1, type=int)
    parser.add_argument('--GPU_no', help="Input the No of GPU you want", default='0', type=str)
    parser.add_argument("--n_epochs", type=int, default=200, help="num of training epochs")
    parser.add_argument('--model_depth', help="Input your resnet depth", default=34, type=int)
    parser.add_argument("--n_exps", type=int, default=1, help="num of independent experiments")
    parser.add_argument("--note", type=str, default="", help="note for the training")
    parser.add_argument("--loss_func", type=str, default="MSE", help="type for loss, MSE")
    parser.add_argument("--pretrain_3D", type=bool, default=False, help="Input if you need a pre-trained 3D model")
    args = parser.parse_args()
    return args, parser

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_options(path, args, parser, phase):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    opt = args
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    mkdirs(path)
    file_name = os.path.join(path, '{}_opt.txt'.format(phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def print_networks(net, name):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')

    num_params = 0
    trainable_params = 0
    for param in net.parameters():
        num_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(net)
    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    print('[Network %s] Total number of trainable parameters : %.3f M' % (name, trainable_params / 1e6))
    print('-----------------------------------------------')

def seed_torch(seed=123):
    '''
    fix the seed
    '''
    random.seed(seed)
    np.random.seed(seed) 
    set_determinism(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.backends.cudnn.deterministic = True 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # True could improve effiency of running, but will leads to different results

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6

def calc_MAE(data_MAE, path, n_exps):
    """
    record the experimental results in 5 indepedent experiments
    """
    n = n_exps
    data_MAE[n, :] = np.mean(data_MAE[:n, :], axis=0)
    data_MAE[n+1, :] = np.std(data_MAE[:n, :], axis=0)

    df_MAE = pd.DataFrame(data_MAE,
                      index=['exp_1', 'mean', 'std'], # index=['exp_1', 'exp_2', 'exp_3', 'exp_4', 'exp_5', 'mean', 'std'],
                      columns=['Val_s', 'In_test_s']
                     )
    df_MAE = df_MAE.apply(lambda x:round(x, 3))
    df_MAE_path = os.path.join(path, 'MAE_val.csv')
    df_MAE.to_csv(df_MAE_path, index=True)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    script_path = os.path.join(path, 'scripts')
    if not os.path.exists(script_path):
        os.mkdir(script_path)
    for script in scripts_to_save:
        dst_file = os.path.join(path, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)


def load_pretrain_3d(model, model_path_3d):
    """
    Load the pre-trained 3d model
    """
    state_dict = model.state_dict()

    pre_s_dict = torch.load(model_path_3d)

    pre_s_new = {}

    for k, v in state_dict.items():
        if k.startswith("encoder3d."):
            k1 = k[10:]
            pre_s_new[k] = pre_s_dict[k1] 

    state_dict.update(pre_s_new)

    model.load_state_dict(state_dict)
    return model


"""
Accurate brain age prediction with lightweight deep neural networks
Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith 
Medical Image Analysis (2021); 
doi: https://doi.org/10.1016/j.media.2020.101871
"""

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        

def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = data.ndim
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    return loss
