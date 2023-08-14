import time
from models.threedim_3view_GAF_MSA_share_encoder import Threedim_3view_GAF_CNN
import utils
from data import ExCustomDataset_threedim_3view_GAF, data_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import glob
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import os
import sys
import glob
import platform
import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
def main():

    # initialization
    args, parser = utils.init()

    pathpar = '/public/bme/home/zhaonan/brain_age_est/codes/BRAIN_AGE_ESTIMATION_CONFERENCE' # path to BRAIN_AGE_ESTIMATION
    data_csv = "/public_bme/share/sMRI/csv/subinfo_HCs_8_DK106_new.csv" # path to data
    outpath =os.path.join(pathpar, 'OUTPUT')
    if not os.path.exists(outpath):
        os.mkdir(outpath)   

    args.parsave = "threedim_3view_GAF-20230811-153250"
    args.parsave = os.path.join(outpath, args.parsave)

    testpath = os.path.join(args.parsave, "TEST")
    if not os.path.exists(testpath):
        os.makedirs(testpath)

    utils.print_options(args.parsave, args, parser, "test")

    # cuda
    assert args.GPU_num <= torch.cuda.device_count(), 'GPU exceed the maximum num'

    if torch.cuda.is_available():
        if args.GPU_no:
            device = torch.device("cuda:"+args.GPU_no[0])
        else:
            device = torch.device("cuda:0")
    else:
        logging.info("no gpu device available")
        device = torch.device('cpu')

    data_MAE = np.zeros((3, 2))

    utils.seed_torch(seed=3407) # inside or outside, control net

    for exp in range(0, args.n_exps):
        
        print("******** Test on exp %d ********" % (exp+1)) 
        
        args.save = os.path.join(args.parsave, "exp_"+str(exp))

        path_weights = os.path.join(args.save, "WEIGHTS")

        # data
        num_workers = 0 if platform.system() == "Windows" else 8

        df = pd.read_csv(data_csv)
        dataset = ExCustomDataset_threedim_3view_GAF(df=df, transforms=False)

        _, _, in_test_sampler, _, _, _, n_in_tests, _ = data_split(dataset)

        in_test_queue = DataLoader(dataset, batch_size=args.batch_size, sampler=in_test_sampler,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        model_s = Threedim_3view_GAF_CNN(in_channels=1).to(device)

        logging.info("sMRI, param size = %.3f MB", utils.count_parameters_in_MB(model_s))
    
        model_path_s = os.path.join(path_weights, 'model_s_best_weight.pkl')
        model_s.load_state_dict(torch.load(model_path_s))
        model_s.to(device)

        if device.type == 'cuda' and args.GPU_num > 1:
            if args.GPU_no:
                assert len(args.GPU_no) == args.GPU_num
                model_s = nn.DataParallel(model_s, [int(each) for each in args.GPU_no])
            else:
                model_s = nn.DataParallel(model_s, list(range(args.GPU_num)))

        in_test_mae_s = preds_store(model_s, in_test_queue, n_in_tests, stage="best_in_test", device=device, path_figs=testpath, exp=exp)

        data_MAE[exp][:2] = 0.0, in_test_mae_s
                            
        print("******** Test Finished on Exp %d ********" % (exp+1))

    utils.calc_MAE(data_MAE, args.parsave, args.n_exps)

def preds_store(model_s, queue, n_datas, stage="val", device=None, path_figs=None, exp=0):
    """
    Save the age prediction.
    """
    preds_s = []
    idxs = []
    filenames = []
    sites = []
    sexs = []
    ages = []
    maes = 0.0
    model_s.eval()
    with torch.no_grad():
        for data in tqdm(queue):
            sMRI, axial, sagittal, coronal, GAF, labels, sex, idx, filename, site = data
            sMRI = sMRI.to(device)
            axial = axial.to(device)
            sagittal = sagittal.to(device)
            coronal = coronal.to(device)
            GAF = GAF.to(device)
            labels = labels.unsqueeze(1).to(device)

            output_s = model_s(sMRI, axial, sagittal, coronal, GAF)

            maes += torch.sum(torch.abs(labels - output_s)).item()

            preds_s.extend(output_s.squeeze(1).tolist())

            idxs.extend(idx.tolist())
            filenames.extend(filename)
            sites.extend(site.tolist())
            sexs.extend(sex.tolist())
            ages.extend(labels.squeeze(1).tolist())

    mae_s = maes / n_datas
    preds_s = np.around(preds_s, decimals=3)
    ages = np.around(ages, decimals=3)
    diffs = np.around(preds_s - ages, decimals=2)
    df = pd.DataFrame({'index': idxs, 'filename': filenames, 'preds_s': preds_s,
                       'age': ages, 'diff': diffs, 'site': sites, 'sex': sexs})  # 1->male, 0-> female

    save_path = os.path.join(
        path_figs, 'age_prediction_{}_exp_{}.csv'.format(stage, str(exp)))

    df.to_csv(save_path, index=False)

    p_value = pearsonr(ages, preds_s)
    r = round(p_value[0], 3)
    rmse = np.sqrt(mean_squared_error(ages, preds_s))
    rmse = round(rmse, 2)

    x = ages[:, np.newaxis]
    y = preds_s[:, np.newaxis]
    model = LinearRegression()
    model.fit(x, y)
    preds_lr = model.predict(x)

    font = {'family': 'Times New Roman',
            'color':  'black',
            'size': 12
            }
    leg_font = {'family': 'Times New Roman',
                'size': 10
                }
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()
    ax.scatter(ages, preds_s, c='darkgray', edgecolors='dimgray')
    v_min = ages.min()
    v_max = ages.max()
    ax.plot(np.linspace(v_min, v_max, 100), np.linspace(v_min, v_max, 100),
            linestyle='dashed', c='lightgray', label='Matched Prediction')
    ax.plot(ages, preds_lr, linestyle='dashed',
            c='darkgray', label='Predicted LR')
    # my_ticks = np.arange(0, 101, 10) #原始数据有13个点，故此处为设置从0开始，间隔为1
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim([-5, 105])
    ax.set_ylim([-5, 105])
    ax.set_xlabel("Chronological Age (years)", fontdict=font)
    ax.set_ylabel('Brain-Predicted Age (years)', fontdict=font)
    plt.legend(loc='lower right', prop=leg_font, frameon=False)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig(os.path.join(path_figs, "exp_%d_%s_MAE_%.2f_r_%.3f_RMSE_%.2f" % (
        exp, stage, mae_s, r, rmse) + ".png"))

    return mae_s

if __name__ == '__main__':
    main()
