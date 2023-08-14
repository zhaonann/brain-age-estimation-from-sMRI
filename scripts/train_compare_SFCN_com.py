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
    # subinfo_HCs_PET_CENTER.csv
    outpath =os.path.join(pathpar, 'OUTPUT')
    if not os.path.exists(outpath):
        os.mkdir(outpath)   

    args.note = "half unet, No lr policy, fix_seed, sMRI, 100 epoch, lr_init 0.001, Adam  wd=0.001, Cosine /1 epoch, 1/2 MSE debug"
    # args.note = "RECAP_GUO-Get 3.012 50 epoch, lr_init 0.001, epoch>30, lr = 0.1*org, redefine optimizer"
    # args.parsave = "{}-{}".format('3D-RESNET-%d'%args.model_depth, time.strftime("%Y%m%d-%H%M%S")) # spectral graph conv
    args.parsave = "{}-{}".format('Unet3D-Single-FULL', time.strftime("%Y%m%d-%H%M%S")) # spectral graph conv
    args.parsave = os.path.join(outpath, args.parsave)
    if not os.path.exists(args.parsave):
        os.makedirs(args.parsave)

    utils.print_options(args.parsave, args, parser, "train")
    # cuda
    # assert args.GPU_num <= torch.cuda.device_count(), 'GPU exceed the maximum num'
    if torch.cuda.is_available():
        if args.GPU_no:
            device = torch.device("cuda:"+args.GPU_no[0])
        else:
            device = torch.device("cuda:0")
    else:
        logging.info("no gpu device available")
        device = torch.device('cpu')

    data_MAE = np.zeros((7, 9))
    utils.seed_torch(seed=3407) # inside or outside, control net

    for exp in range(0, args.n_exps):

        print("******** Training on exp %d ********" % (exp+1)) 
        
        args.save = os.path.join(args.parsave, "exp_"+str(exp))
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        pathpar1 = os.path.abspath(os.getcwd())
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "models", "compare_SFCN.py")))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "cpmpare_SFCN_conference_com.py")))
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
            "Train_Loss": ["Multiline", ["sMRI", "fMRI", "mean"]],
            "Val_Loss": ["Multiline", ["sMRI", "fMRI", "mean"]],
            "Train_MAE": ["Multiline", ["sMRI", "fMRI", "mean"]],
            "Val_MAE": ["Multiline", ["sMRI", "fMRI", "mean"]],
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

        logging.info("sMRI, param size = %.3f MB", utils.count_parameters_in_MB(model_s))
        # logging.info("fMRI, param size = %fMB", utils.count_parameters_in_MB(model_f))
        
        # 8-16-32-64 double conv  555876
        # model_path_Unet3D = "/public/bme/home/zhaonan/brain_age_est/codes/OUTPUT/3D-UNET-32-64-128-256-single-conv-SEG-20230507-122427/exp_0/WEIGHTS/model_s_epoch_100.pkl" 
        # 16-32-64-128 Single-Conv 556474
        # RIGID_path
        # model_path_Unet3D ="/public/bme/home/zhaonan/brain_age_est/codes/OUTPUT/3D-UNET-32-64-128-256-single-conv-SEG-20230516-212353/exp_0/WEIGHTS/model_s_best_weight.pkl"
        # Affine path
        model_path_Unet3D = "/public/bme/home/zhaonan/brain_age_est/codes/OUTPUT/3D-UNET-32-64-128-256-single-conv-SEG-20230508-190337/exp_0/WEIGHTS/model_s_epoch_100.pkl" 
        # 16-32-64-128-256 Single-Conv Hang 556531
        # model_path_Unet3D = "/public/bme/home/zhaonan/brain_age_est/codes/OUTPUT/3D-UNET-32-64-128-256-single-conv-SEG-20230508-210917/exp_0/WEIGHTS/model_s_epoch_68.pkl"
        # model_s = utils.load_pretrain_unet(model_s, model_path_Unet3D)
        utils.print_networks(model_s, "sMRI")

        if device.type == 'cuda' and args.GPU_num > 1:
            if args.GPU_no:
                assert len(args.GPU_no) == args.GPU_num
                model_s = nn.DataParallel(model_s, [int(each) for each in args.GPU_no])
                # model_f = nn.DataParallel(model_f, [int(each) for each in args.GPU_no])
            else:
                model_s = nn.DataParallel(model_s, list(range(args.GPU_num)))
                # model_f = nn.DataParallel(model_f, [int(each) for each in args.GPU_no])

        optimizer_s = optim.AdamW(model_s.parameters(), lr=args.lr_s, weight_decay=args.wd_s)
        # optimizer_s = optim.SGD(model_s.parameters(), lr=args.lr_s, weight_decay=args.wd_s)
        # scheduler_s = optim.lr_scheduler.ExponentialLR(optimizer_s, gamma=0.5) # lr = 0.001
        scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, float(args.n_epochs))

        # loss func
        if args.loss_func == "adaptive_MAE":
            loss_func_s = utils.adaptive_MAE
            # loss_func_f = utils.adaptive_MAE
        elif args.loss_func == "adaptive_MSE":
            loss_func_s = utils.adaptive_MSE
            # loss_func_f = utils.adaptive_MSE
        elif args.loss_func == "MAE":
            loss_func_s =  nn.L1Loss() # nn.L1Loss MSELoss()
            # loss_func_f =  nn.L1Loss() # nn.L1Loss MSELoss()
        elif args.loss_func == "MSE":
            loss_func_s =  nn.MSELoss()
            # loss_func_f =  nn.MSELoss()     


        losses = {"train_loss_sMRI": [], "train_loss_fMRI": [], "val_mae_sMRI": [], "val_loss_fMRI": []}
        maes = {"train_mae_sMRI": [], "train_mae_fMRI": [], "val_mae_sMRI": [], "val_mae_fMRI": []}
        val_opt = {"Epoch": 0, "Opt_val": 100.00, "Opt_train": 0.0}
        lr = args.lr_s
        for epoch in range(1, args.n_epochs+1):

            epoch_start_time = time.time()
            loss_dict = {"sMRI": [], "fMRI": 0.0, "mean": 0.0}
            mae_dict = {"sMRI": 0.0, "fMRI": 0.0, "mean": 0.0}

            # if epoch != 0 and epoch % 10 == 0:
            if epoch > 1:
                scheduler_s.step()
            # if epoch !=0 and epoch % 30 == 0:
            #     lr = lr * 0.3
            #     optimizer_s = optim.SGD(model_s.parameters(), lr=lr, weight_decay=args.wd_s)


            # if args.warm_up and epoch <= args.warm_up_epochs:
            #     lr_s = (args.lr_s - 0)/args.warm_up_epochs * epoch
            #     # lr_f=0.001
            #     optimizer_s = optim.AdamW(model_s.parameters(), lr=lr_s, weight_decay=args.wd_s)
            bin_range = [6, 96]
            bin_step = 1
            sigma = 1

            model_s.train()
            # model_f.train()
            for data in train_queue:
                sMRI, labels = data[:2]
                sMRI = sMRI.to(device, non_blocking=True)
                # fMRI = fMRI.to(device, non_blocking=True)
                labels = labels.unsqueeze(1)
                optimizer_s.zero_grad()
                # optimizer_f.zero_grad()

                y, bc = utils.num2vect(labels, bin_range, bin_step, sigma)
                y = torch.tensor(y, dtype=torch.float32)

                output_s = model_s(sMRI)
                # output_f = model_f(fMRI)
                
                loss_s = utils.my_KLDivLoss(output_s, y.to(device))
                # loss_s = loss_func_s(output_s, labels)
                # loss_s = 0.5*loss_s
                # loss_f = loss_func_f(output_f, labels)

                loss_dict['sMRI'].append(loss_s.cpu().item())
                # loss_dict['fMRI'].append(loss_f.item())
                loss_s.backward()
                # loss_f.backward()

                optimizer_s.step()
                # optimizer_f.step()
                prob = np.exp(output_s.detach().cpu().numpy())
                # print("prob shape: ", prob.shape, flush=True)
                # print("bc shape: ", bc.shape, flush=True)

                pred = prob@np.expand_dims(bc, axis=1)
                mae_dict['sMRI'] += torch.sum(torch.abs(labels - pred)).item()
                # mae_dict['fMRI'] += torch.sum(torch.abs(labels - output_f))

            # if args.warm_up and epoch > args.warm_up_epochs: # warm up and finish
            #     scheduler_s.step()
            # elif not args.warm_up: # no warm up 
            #     scheduler_s.step()


            loss_dict['sMRI'] = np.average(loss_dict['sMRI'])
            # loss_dict['fMRI'] = np.average(loss_dict['fMRI'])
            # loss_dict['mean'] = (loss_dict['sMRI'] + loss_dict['fMRI']) / 2
            mae_dict['sMRI'] = (mae_dict['sMRI']/n_trains)

            # mae_dict['fMRI'] = (mae_dict['fMRI']/n_trains).item()
            # mae_dict['mean'] = (mae_dict['sMRI'] + mae_dict['fMRI']) / 2

            losses['train_loss_sMRI'].append(loss_dict['sMRI'])
            # losses['train_loss_fMRI'].append(loss_dict['fMRI'])

            maes['train_mae_sMRI'].append(mae_dict['sMRI'])
            # maes['train_mae_fMRI'].append(mae_dict['fMRI'])

            dash_writer.add_scalars("RESULT/Train_Loss", loss_dict, epoch)  
            dash_writer.add_scalars("RESULT/Train_MAE", mae_dict, epoch)  

            # logging.info("******* Epoch %d, Train Loss sMRI %.2f, MAE %.2f *******", epoch, loss_dict['sMRI'],  mae_dict['sMRI'])

            train_loss = loss_dict['sMRI']
            train_mae = mae_dict['sMRI']

            loss_dict = {"sMRI": [], "fMRI": 0, "mean": 0.0}
            mae_dict = {"sMRI": 0.0, "fMRI": 0.0, "mean": 0.0}

            model_s.eval()
            # model_f.eval()
            with torch.no_grad():   
                for data in val_queue:
                    sMRI, labels = data[:2]
                    sMRI = sMRI.to(device)
                    # fMRI = fMRI.to(device, non_blocking=True)
                    labels = labels.unsqueeze(1)

                    output_s = model_s(sMRI)
                    # output_f = model_f(fMRI)

                    y, bc = utils.num2vect(labels, bin_range, bin_step, sigma)
                    y = torch.tensor(y, dtype=torch.float32)
                    
                    loss_s = utils.my_KLDivLoss(output_s, y.to(device))

                    loss_dict['sMRI'].append(loss_s.cpu().item())
                    # loss_dict['fMRI'].append(loss_f.item())
                    prob = np.exp(output_s.detach().cpu().numpy())
                    pred = prob@np.expand_dims(bc, axis=1)
                    mae_dict['sMRI'] += torch.sum(torch.abs(labels - pred)).item()
                    # mae_dict['sMRI'] += torch.sum(torch.abs(labels - output_s)).item()
                    # mae_dict['fMRI'] += torch.sum(torch.abs(labels - output_f))

            loss_dict['sMRI'] = np.average(loss_dict['sMRI'])
            # loss_dict['fMRI'] = np.average(loss_dict['fMRI'])
            # loss_dict['mean'] = (loss_dict['sMRI'] + loss_dict['fMRI']) / 2
            mae_dict['sMRI'] = (mae_dict['sMRI']/n_vals)

            # mae_dict['fMRI'] = (mae_dict['fMRI']/n_vals).item()
            # mae_dict['mean'] = (mae_dict['sMRI'] + mae_dict['fMRI']) / 2
            if mae_dict['sMRI'] <= val_opt["Opt_val"]: # record the best
                val_opt["Opt_val"] = mae_dict['sMRI']
                val_opt["Opt_train"] = train_mae
                val_opt["Epoch"] = epoch
                model_path_s = os.path.join(path_weights, 'model_s_best_weight.pkl')
                torch.save(model_s.state_dict(), model_path_s)

            losses['val_mae_sMRI'].append(loss_dict['sMRI'])
            # losses['val_loss_fMRI'].append(loss_dict['fMRI'])

            maes['val_mae_sMRI'].append(mae_dict['sMRI'])
            # maes['val_mae_fMRI'].append(mae_dict['fMRI'])

            dash_writer.add_scalars("RESULT/Val_Loss", loss_dict, epoch)  
            dash_writer.add_scalars("RESULT/Val_MAE", mae_dict, epoch)  

            # logging.info("******* Epoch %d, Val Loss sMRI %.2f, MAE %.2f *******", epoch,  loss_dict['sMRI'], mae_dict['sMRI'])
            # logging.info("******* Epoch %d, Val MAE sMRI %.3f, fMRI %.3f *******", epoch,  mae_dict['sMRI'], mae_dict['fMRI'])
            logging.info("******* Epoch %d, Train Loss %.2f, Val Loss %.2f, Train MAE %.2f, Val MAE %.2f *******", epoch, train_loss, loss_dict['sMRI'], train_mae, mae_dict['sMRI'])

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs, time.time() - epoch_start_time))
 

        # model save
        model_path_s = os.path.join(path_weights, 'model_s_epoch_%d.pkl'%args.n_epochs)
        torch.save(model_s.state_dict(), model_path_s)

        plt.figure(figsize=(20, 10))
        plt.plot(losses['train_loss_sMRI'], 'r--', label='train/sMRI')
        # plt.plot(losses['train_loss_fMRI'], 'b--', label='train/fMRI')
        plt.plot(losses['val_mae_sMRI'], 'g--', label='val/sMRI')
        # plt.plot(losses['val_loss_fMRI'], 'y--', label='val/fMRI')
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss of exp {}'.format(str(exp)))
        plt.savefig(os.path.join(path_figs, "exp_%d_loss_history"%exp + ".png"))
    
        plt.figure(figsize=(20, 10))
        plt.plot(maes['train_mae_sMRI'], 'r--', label='train/sMRI')
        # plt.plot(maes['train_mae_fMRI'], 'b--', label='train/fMRI')
        plt.plot(maes['val_mae_sMRI'], 'g--', label='val/sMRI')
        # plt.plot(maes['val_mae_fMRI'], 'y--', label='val/fMRI')
        plt.xlabel("Epoch")
        plt.ylabel('MAE')
        plt.legend()
        plt.title('MAE of exp {}'.format(str(exp)))
        plt.savefig(os.path.join(path_figs, "exp_%d_mae_history"%exp + ".png"))


        val_mae_s, val_loss_f = preds_store(model_s, val_queue, n_vals, stage="val", device=device, path_figs=path_figs, exp=exp)
        in_test_mae_s,  in_test_loss_f = preds_store(model_s, in_test_queue, n_in_tests, stage="in_test_last", device=device, path_figs=path_figs, exp=exp)

        logging.info("*******Again Last Epoch %d,  Val MAE %.2f, In Test MAE %.2f *******",
                     args.n_epochs, val_mae_s, in_test_mae_s)
   
        data_MAE[exp][:6] = val_mae_s, val_loss_f, (val_mae_s + val_loss_f)/2, \
                            in_test_mae_s, in_test_loss_f, 0.0, 

        model_path_s = os.path.join(path_weights, 'model_s_best_weight.pkl')
        model_s.load_state_dict(torch.load(model_path_s))

        in_test_mae_s,  in_test_loss_f = preds_store(model_s, in_test_queue, n_in_tests, stage="in_test_best", device=device, path_figs=path_figs, exp=exp)

        logging.info("******* Best Epoch %d, Train MAE %.2f, Val MAE %.2f, In Test MAE %.2f *******", 
                     val_opt["Epoch"], val_opt["Opt_train"], val_opt["Opt_val"], in_test_mae_s)

        print('**** Exp %d Finished Training  ****' % exp)

        logging.shutdown()
        
    utils.calc_MAE(data_MAE, args.parsave, args.n_exps)


def preds_store(model_s, queue, n_datas, stage="val", device=None, path_figs=None, exp=0):
    """
    Save the age prediction.
    """
    preds_s = []
    # preds_f = []
    # com_preds = []
    idxs = []
    filenames = []
    sites = []
    genders = []
    ages = []
    maes = 0.0
    # running_loss_f = 0.0
    model_s.eval()
    # model_f.eval()

    bin_range = [6, 96]
    bin_step = 1
    sigma = 1
    with torch.no_grad():
        for data in tqdm(queue):
            sMRI, labels, gender, idx, filename, site = data
            sMRI = sMRI.to(device)
            # fMRI = fMRI.to(device, non_blocking=True)
            labels = labels.unsqueeze(1)

            output_s = model_s(sMRI)

            y, bc = utils.num2vect(labels, bin_range, bin_step, sigma)

            prob = np.exp(output_s.detach().cpu().numpy())
            pred = prob@np.expand_dims(bc, axis=1)
            maes += torch.sum(torch.abs(labels - pred)).item()

            preds_s.extend(pred.squeeze(1).tolist())
            # preds_f.extend(output_f.squeeze().tolist())

            idxs.extend(idx.tolist())
            filenames.extend(filename)
            sites.extend(site.tolist())
            genders.extend(gender.tolist())
            ages.extend(labels.squeeze(1).tolist())

    mae_s = maes / n_datas
    loss_f = 0.0  # (running_loss_f/n_datas).item()
    # com_preds = (np.array(preds_s) + np.array(preds_f)) / 2
    # com_preds = np.around(com_preds, decimals=3)
    preds_s = np.around(preds_s, decimals=3)
    # preds_f = np.around(preds_f, decimals=3)
    ages = np.around(ages, decimals=3)
    diffs = np.around(preds_s - ages, decimals=2)
    # pdb.set_trace()
    df = pd.DataFrame({'index': idxs, 'filename': filenames, 'preds_s': preds_s,
                       'age': ages, 'diff': diffs, 'site': sites, 'sex': genders})  # 1->male, 0-> female

    save_path = os.path.join(
        path_figs, 'age_prediction_{}_exp_{}.csv'.format(stage, str(exp)))
    df.to_csv(save_path, index=False)

    p_value = pearsonr(ages, preds_s)
    r = round(p_value[0], 2)
    rmse = np.sqrt(mean_squared_error(ages, preds_s))
    rmse = round(rmse, 2)
    r2 = r2_score(ages, preds_s)
    r2 = round(r2, 2)
    x = ages[:, np.newaxis]
    y = preds_s[:, np.newaxis]
    model = LinearRegression()
    model.fit(x, y)
    preds_lr = model.predict(x)
    R2 = model.score(x, y)
    coef = model.coef_
    intercept = model.intercept_

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
    plt.savefig(os.path.join(path_figs, "exp_%d_%s_MAE_%.2f_r_%.2f_r2_%.2f_RMSE_%.2f_R2_%.2f_coef_%.2f_intercept_%.2f" % (
        exp, stage, mae_s, r, r2, rmse, R2, coef, intercept) + ".eps"))

    return mae_s, loss_f


if __name__ == '__main__':
    main()
