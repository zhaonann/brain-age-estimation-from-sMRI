import sys
# sys.path.append('/public/bme/home/zhaonan/brain_age_est/codes/BRAIN_AGE_ESTIMATION_CONFERENCE/OUTPUT/threedim_3view_GAF-20230805-110245/exp_0/scripts')
from models.threedim_3view_GAF_MSA_share_encoder import Threedim_3view_GAF_CNN
import utils
from data import ExCustomDataset_threedim_3view_GAF_BDs
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import platform
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

Diag = {1:'ASD', 2: 'AD', 3:'SVD', 4:'MCI', 5:'ADHD'} # subinfo_BDs_dive.csv

def one_disease(disease=1):
    
    # initialization
    args, parser = utils.init()

    pathpar = '/public/bme/home/zhaonan/brain_age_est/codes/BRAIN_AGE_ESTIMATION_CONFERENCE' # path to BRAIN_AGE_ESTIMATION
    data_csv = "/public_bme/share/sMRI/csv/subinfo_BDs_dive_DK106_new.csv" # path to data
    outpath =os.path.join(pathpar, 'OUTPUT') 
    if not os.path.exists(outpath):
        os.mkdir(outpath)   

    args.parsave = "threedim_3view_GAF-20230805-110245"
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
        device = torch.device('cpu')

    data_MAE = np.zeros((3, 2))

    utils.seed_torch(seed=3407) 

    for exp in range(0, args.n_exps):
        seed = 3407
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print("******** Test on exp %d ********" % (exp+1)) 
        
        args.save = os.path.join(args.parsave, "exp_"+str(exp))

        path_weights = os.path.join(args.save, "WEIGHTS")
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)

        # data
        num_workers = 0 if platform.system() == "Windows" else 8

        df = pd.read_csv(data_csv)

        dataset = ExCustomDataset_threedim_3view_GAF_BDs(df=df, disease=disease, transforms=False)
        n_in_tests = len(dataset)

        in_test_queue = DataLoader(dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=num_workers, pin_memory=True)

        # model
        model = Threedim_3view_GAF_CNN(in_channels=1).to(device) 

        model_path = os.path.join(path_weights, "model_s_best_weight.pkl")
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        if device.type == 'cuda' and args.GPU_num > 1:
            if args.GPU_no:
                assert len(args.GPU_no) == args.GPU_num
                model = nn.DataParallel(model, [int(each) for each in args.GPU_no])
            else:
                model = nn.DataParallel(model, [int(each) for each in args.GPU_no])
                    
        in_test_loss = preds_store(model, device, testpath, disease, exp, in_test_queue, n_in_tests, stage="in_test")

        data_MAE[exp][:2] = 0.0, in_test_loss
                            
        print("******** Test Finished on Exp %d! ********" % (exp+1))

    utils.calc_MAE(data_MAE, testpath, args.n_exps)

def preds_store(model, device, testpath, disease, exp, queue, n_datas, stage="val"):
    """
    Save the age prediction.
    """
    preds = []
    idxs = []
    filenames = []
    sites = []
    sexs = []
    ages = []
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in tqdm(queue):
            sMRI, axial, sagittal, coronal, GAF, labels, sex, idx, filename, site = data
            sMRI = sMRI.to(device)
            axial = axial.to(device)
            sagittal = sagittal.to(device)
            coronal = coronal.to(device)
            GAF = GAF.to(device)
            labels = labels.unsqueeze(1).to(device)

            output = model(sMRI, axial, sagittal, coronal, GAF)

            running_loss += torch.sum(torch.abs(labels - output))
            if labels.numel() == 1:
                preds.extend(output.squeeze(dim=0).tolist())
                ages.extend(labels.squeeze(dim=0).tolist())
            else:
                preds.extend(output.squeeze().tolist())
                ages.extend(labels.squeeze().tolist())

            idxs.extend(idx.tolist())
            filenames.extend(filename)
            sites.extend(site.tolist())
            sexs.extend(sex.tolist())

    loss = (running_loss/n_datas).item()
    preds = np.around(preds, decimals=3)
    ages = np.around(ages, decimals=3)
    diffs = np.around(preds - ages, decimals=2)
    df = pd.DataFrame({'index': idxs, 'filename': filenames, 'preds_s': preds, 
                    'age': ages, 'diff': diffs, 'site': sites, 'sex': sexs}) # 1->male, 0-> female

    save_path = os.path.join(testpath, '{}_sMRI_age_prediction_{}_exp_{}.csv'.format(Diag[disease], stage, str(exp)))
    df.to_csv(save_path, index=False)

    mae = mean_absolute_error(ages, preds)
    p_value = pearsonr(ages, preds)
    r = round(p_value[0], 3)

    rmse = np.sqrt(mean_squared_error(ages, preds))
    rmse = round(rmse, 2)
    x = ages[:, np.newaxis]
    y = preds[:, np.newaxis]
    model1 = LinearRegression()
    model1.fit(x, y)
    preds_lr = model1.predict(x)

    font = {'family': 'Times New Roman',
            'color':  'black',
            'size': 12
            }
    leg_font = {'family': 'Times New Roman',
            'size': 10
            }
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()
    ax.scatter(ages, preds, c='darkgray', edgecolors='dimgray')
    v_min = ages.min()
    v_max = ages.max()
    ax.plot(np.linspace(v_min, v_max, 100), np.linspace(v_min, v_max, 100), linestyle='dashed', c='lightgray', label='Matched Prediction')
    ax.plot(ages, preds_lr, linestyle='dashed', c='darkgray', label='Predicted LR')
    plt.xticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.yticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim([-5, 105])
    ax.set_ylim([-5, 105])
    ax.set_xlabel("Chronological Age (years)", fontdict=font)
    ax.set_ylabel('Brain-Predicted Age (years)', fontdict=font)
    plt.legend(loc='lower right', prop=leg_font, frameon=False)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig(os.path.join(testpath, "%s_%d_sMRI_BEST_TEST_MAE_%.2f_r_%.3f_RMSE_%.2f" %(Diag[disease], n_datas, mae, r, rmse)+ ".png"))

    return loss

if __name__ == '__main__':
    num_of_disease = 5
    for i in range(1, num_of_disease+1):
        one_disease(disease=i)
