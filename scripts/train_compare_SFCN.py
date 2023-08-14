"""
Accurate brain age prediction with lightweight deep neural networks
Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith 
Medical Image Analysis (2021); 
doi: https://doi.org/10.1016/j.media.2020.101871
"""
import time
from models.compare_SFCN_model import SFCN
import utils
from data import ExCustomDataset, data_split
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
from sklearn.metrics import mean_squared_error, r2_score

def main():
    
    # initialization
    args, parser = utils.init()

    pathpar = '/public/bme/home/zhaonan/brain_age_est/codes/BRAIN_AGE_ESTIMATION_CONFERENCE' # path to BRAIN_AGE_ESTIMATION
    data_csv = "/public_bme/share/sMRI/csv/subinfo_HCs_8_DK106_new.csv"

    outpath =os.path.join(pathpar, 'OUTPUT')
    if not os.path.exists(outpath):
        os.mkdir(outpath)   

    args.note = ""
    args.parsave = "{}-{}".format('SFCN', time.strftime("%Y%m%d-%H%M%S")) # spectral graph conv
    args.parsave = os.path.join(outpath, args.parsave)
    if not os.path.exists(args.parsave):
        os.makedirs(args.parsave)

    utils.print_options(args.parsave, args, parser, "train")
    # cuda
    if torch.cuda.is_available():
        if args.GPU_no:
            device = torch.device("cuda:"+args.GPU_no[0])
        else:
            device = torch.device("cuda:0")
    else:
        logging.info("no gpu device available")
        device = torch.device('cpu')

    data_MAE = np.zeros((3, 2))

    utils.seed_torch(seed=3407)

    for exp in range(0, args.n_exps):

        print("******** Training on exp %d ********" % (exp+1)) 
        
        args.save = os.path.join(args.parsave, "exp_"+str(exp))
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        pathpar1 = os.path.abspath(os.getcwd())
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "models", "compare_SFCN_model.py")))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "train_compare_SFCN.py")))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "utils.py")))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "data.py")))

        path_logging = os.path.join(args.save, 'logging')
        if not os.path.exists(path_logging):
            os.makedirs(path_logging)

        log_format = "%(asctime)s %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        fh = logging.FileHandler(os.path.join(path_logging, "log.txt"))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        dash_writer = SummaryWriter(os.path.join(args.save, "TENSORBOARD"))

        Layout = {}
        Layout["RESULT" ] = {
            "Train_Loss": ["Multiline", ["sMRI"]],
            "Val_Loss": ["Multiline", ["sMRI"]],
            "Train_MAE": ["Multiline", ["sMRI"]],
            "Val_MAE": ["Multiline", ["sMRI"]],
        }
        dash_writer.add_custom_scalars(Layout)

        path_weights = os.path.join(args.save, "WEIGHTS")
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)

        path_figs = os.path.join(args.save, "FIGURES_CSV")
        if not os.path.exists(path_figs):
            os.makedirs(path_figs)

        num_workers = 0 if platform.system() == "Windows" else 8
        
        df = pd.read_csv(data_csv)
        dataset = ExCustomDataset(df=df, transforms=True)

        train_sampler, val_sampler, in_test_sampler,  _, n_trains, n_vals, n_in_tests, _ = data_split(dataset)

        train_queue = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                shuffle=False,  num_workers=num_workers, pin_memory=True)
        
        dataset = ExCustomDataset(df=df, transforms=False)
        val_queue = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        in_test_queue = DataLoader(dataset, batch_size=args.batch_size, sampler=in_test_sampler,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        # model      
        model_s = SFCN(output_dim=90).to(device) # our_channels = n_classes, include bg

        logging.info("sMRI, param size = %.3f MB", utils.count_parameters_in_MB(model_s))
        utils.print_networks(model_s, "sMRI")

        if device.type == 'cuda' and args.GPU_num > 1:
            if args.GPU_no:
                assert len(args.GPU_no) == args.GPU_num
                model_s = nn.DataParallel(model_s, [int(each) for each in args.GPU_no])
            else:
                model_s = nn.DataParallel(model_s, list(range(args.GPU_num)))

        optimizer_s = optim.AdamW(model_s.parameters(), lr=args.lr_s, weight_decay=args.wd_s)
        scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, float(args.n_epochs))

        losses = {"train_loss_sMRI": [], "val_mae_sMRI": []}
        maes = {"train_mae_sMRI": [], "val_mae_sMRI": []}
        val_opt = {"Epoch": 0, "Opt_val": 100.00, "Opt_train": 0.0}

        for epoch in range(1, args.n_epochs+1):

            epoch_start_time = time.time()

            loss_dict = {"sMRI": []}
            mae_dict = {"sMRI": 0.0}

            if epoch > 1:
                scheduler_s.step()

            bin_range = [6, 96]
            bin_step = 1
            sigma = 1

            model_s.train()
            for data in train_queue:
                sMRI, labels = data[:2]
                sMRI = sMRI.to(device)
                labels = labels.unsqueeze(1)
                optimizer_s.zero_grad()

                output_s = model_s(sMRI)

                y, bc = utils.num2vect(labels, bin_range, bin_step, sigma)
                y = torch.tensor(y, dtype=torch.float32)     

                loss_s = utils.my_KLDivLoss(output_s, y.to(device))

                loss_dict['sMRI'].append(loss_s.cpu().item())
                loss_s.backward()

                optimizer_s.step()
                prob = np.exp(output_s.detach().cpu().numpy())

                pred = prob@np.expand_dims(bc, axis=1)
                mae_dict['sMRI'] += torch.sum(torch.abs(labels - pred)).item()

            loss_dict['sMRI'] = np.average(loss_dict['sMRI'])
            mae_dict['sMRI'] = (mae_dict['sMRI']/n_trains)

            losses['train_loss_sMRI'].append(loss_dict['sMRI'])

            maes['train_mae_sMRI'].append(mae_dict['sMRI'])

            dash_writer.add_scalars("RESULT/Train_Loss", loss_dict, epoch)  
            dash_writer.add_scalars("RESULT/Train_MAE", mae_dict, epoch)  

            train_loss = loss_dict['sMRI']
            train_mae = mae_dict['sMRI']

            loss_dict = {"sMRI": []}
            mae_dict = {"sMRI": 0.0}

            model_s.eval()
            with torch.no_grad():   
                for data in val_queue:
                    sMRI, labels = data[:2]
                    sMRI = sMRI.to(device)
                    labels = labels.unsqueeze(1)

                    output_s = model_s(sMRI)

                    y, bc = utils.num2vect(labels, bin_range, bin_step, sigma)
                    y = torch.tensor(y, dtype=torch.float32)

                    loss_s = utils.my_KLDivLoss(output_s, y.to(device))

                    loss_dict['sMRI'].append(loss_s.cpu().item())

                    prob = np.exp(output_s.detach().cpu().numpy())
                    pred = prob@np.expand_dims(bc, axis=1)

                    mae_dict['sMRI'] += torch.sum(torch.abs(labels - pred)).item()

            loss_dict['sMRI'] = np.average(loss_dict['sMRI'])
            mae_dict['sMRI'] = (mae_dict['sMRI']/n_vals)

            if mae_dict['sMRI'] <= val_opt["Opt_val"]: # record the best
                val_opt["Opt_val"] = mae_dict['sMRI']
                val_opt["Opt_train"] = train_mae
                val_opt["Epoch"] = epoch
                model_path_s = os.path.join(path_weights, 'model_s_best_weight.pkl')
                torch.save(model_s.state_dict(), model_path_s)

            losses['val_mae_sMRI'].append(loss_dict['sMRI'])

            maes['val_mae_sMRI'].append(mae_dict['sMRI'])

            dash_writer.add_scalars("RESULT/Val_Loss", loss_dict, epoch)  
            dash_writer.add_scalars("RESULT/Val_MAE", mae_dict, epoch)  

            logging.info("******* Epoch %d, Train Loss %.2f, Val Loss %.2f, Train MAE %.2f, Val MAE %.2f *******", epoch, train_loss, loss_dict['sMRI'], train_mae, mae_dict['sMRI'])

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs, time.time() - epoch_start_time))
 
        # model save
        model_path_s = os.path.join(path_weights, 'model_s_epoch_%d.pkl'%args.n_epochs)
        torch.save(model_s.state_dict(), model_path_s)

        val_mae_s = preds_store(model_s, val_queue, n_vals, stage="last_val", device=device, path_figs=path_figs, exp=exp)
        in_test_mae_s = preds_store(model_s, in_test_queue, n_in_tests, stage="last_in_test_last", device=device, path_figs=path_figs, exp=exp)

        logging.info("******* Again Last Epoch %d,  Val MAE %.2f, In Test MAE %.2f *******",
                     args.n_epochs, val_mae_s, in_test_mae_s)

        model_path_s = os.path.join(path_weights, 'model_s_best_weight.pkl')
        model_s.load_state_dict(torch.load(model_path_s))

        in_test_mae_s = preds_store(model_s, in_test_queue, n_in_tests, stage="last_in_test_best", device=device, path_figs=path_figs, exp=exp)

        logging.info("******* Best Epoch %d, Train MAE %.2f, Val MAE %.2f, In Test MAE %.2f *******", 
                     val_opt["Epoch"], val_opt["Opt_train"], val_opt["Opt_val"], in_test_mae_s)

        data_MAE[exp][:2] = val_opt["Opt_val"], in_test_mae_s

        print('**** Exp %d Finished Training  ****' % exp)

        logging.shutdown()
        
        plt.figure(figsize=(20, 10))
        plt.plot(losses['train_loss_sMRI'], 'r--', label='train/sMRI')
        plt.plot(losses['val_mae_sMRI'], 'g--', label='val/sMRI')
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss of exp {}'.format(str(exp)))
        plt.savefig(os.path.join(path_figs, "exp_%d_loss_history"%exp + ".png"))
    
        plt.figure(figsize=(20, 10))
        plt.plot(maes['train_mae_sMRI'], 'r--', label='train/sMRI')
        plt.plot(maes['val_mae_sMRI'], 'g--', label='val/sMRI')
        plt.xlabel("Epoch")
        plt.ylabel('MAE')
        plt.legend()
        plt.title('MAE of exp {}'.format(str(exp)))
        plt.savefig(os.path.join(path_figs, "exp_%d_mae_history"%exp + ".png"))

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

    bin_range = [6, 96]
    bin_step = 1
    sigma = 1

    model_s.eval()
    with torch.no_grad():
        for data in tqdm(queue):
            sMRI, labels, sex, idx, filename, site = data
            sMRI = sMRI.to(device)
            labels = labels.unsqueeze(1)

            output_s = model_s(sMRI)

            y, bc = utils.num2vect(labels, bin_range, bin_step, sigma)

            prob = np.exp(output_s.detach().cpu().numpy())
            pred = prob@np.expand_dims(bc, axis=1)

            maes += torch.sum(torch.abs(labels - pred)).item()

            preds_s.extend(pred.squeeze(1).tolist())

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
