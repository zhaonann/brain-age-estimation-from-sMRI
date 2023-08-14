import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick

"""
 { 'tab:blue': '#1f77b4',
      'tab:orange': '#ff7f0e',
      'tab:green': '#2ca02c',
      'tab:red': '#d62728',
      'tab:purple': '#9467bd',
      'tab:brown': '#8c564b',
      'tab:pink': '#e377c2',
      'tab:gray': '#7f7f7f',
      'tab:olive': '#bcbd22',
      'tab:cyan': '#17becf'}
"""
sites = ['ABIDE', 'RENJI', 'PET_CENTER', 'ADNI', 'OASIS', 'ADHD', 'CoRR', 'CBMFM'] # 1-8
site_dict = {1:'ABIDE', 2: 'RENJI', 3: 'PET_CENTER', 4: 'ADNI', 5: 'OASIS', 6: 'ADHD', 7:'CoRR', 8:'CBMFM'}
HCs_comp = {1: 6, 2:1, 3:2, 4:[4, 3], 5: [4, 5]}
Diag = {1:'ADHD', 2: 'ASD', 3:'VCI', 4:'MCI', 5:'AD', 6:'HC'} # subinfo_BDs_dive.csv

def HC_comp(disease=0):
    root_path = "/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636406_best_ours/FIGURES_CSV"
    test_csv =root_path + '/' + 'age_prediction_best_in_test_exp_0.csv'
    val_csv = root_path + '/' + 'age_prediction_val_exp_0.csv'
    train_csv = root_path + '/' + 'HCs_sMRI_age_prediction_train_exp_0.csv'
    if disease <4:
        if disease == 3:
            idx = HCs_comp[disease]
            df1 = pd.read_csv(val_csv)
            df2 = pd.read_csv(test_csv)
            df3 = pd.read_csv(train_csv)
            pred1 = df1.loc[df1['site'] == idx].copy() 
            pred2 = df2.loc[df2['site'] == idx].copy() 
            pred3 = df3.loc[df3['site'] == idx].copy()
            diff1 = pred1['diff'].to_numpy()
            diff2 = pred2['diff'].to_numpy()
            diff3 = pred3['diff'].to_numpy()

            diff= np.concatenate((diff1, diff2, diff3))
            print(len(diff))
            BAG = np.sum(diff)/len(diff)
            # print(MAE)
            print("Comp_HCs: {}, MAE {:.2f} ".format(Diag[disease], BAG))
            return diff
   
        else:
            idx = HCs_comp[disease]
            df1 = pd.read_csv(val_csv)
            df2 = pd.read_csv(test_csv)
            pred1 = df1.loc[df1['site'] == idx].copy() 
            pred2 = df2.loc[df2['site'] == idx].copy() 
            diff1 = pred1['diff'].to_numpy()
            diff2 = pred2['diff'].to_numpy()

            diff= np.concatenate((diff1, diff2))
            print(len(diff))
            BAG = np.sum(diff)/len(diff)
            # print(MAE)
            print("Comp_HCs: {}, MAE {:.2f} ".format(Diag[disease], BAG))
            return diff
    elif disease >= 4: # 4
        idx = HCs_comp[disease]
        df1 = pd.read_csv(val_csv)
        df2 = pd.read_csv(test_csv)

        pred1 = df1.loc[df1['site'] == idx[0]].copy() 
        pred2 = df2.loc[df2['site'] == idx[0]].copy() 
        pred3 = df1.loc[df1['site'] == idx[1]].copy() 
        pred4 = df2.loc[df2['site'] == idx[1]].copy() 
        diff1 = pred1['diff'].to_numpy()
        diff2 = pred2['diff'].to_numpy()
        diff3 = pred3['diff'].to_numpy()
        diff4 = pred4['diff'].to_numpy()
        diff = np.concatenate((diff1, diff2, diff3, diff4))
        print(len(diff))
        BAG = np.sum(diff)/len(diff)
        # print(MAE)
        print("Comp_HCs: {}, brain age gap {:.2f} ".format(Diag[disease], BAG))
        return diff       


def func():
    """
    obtain the brain age gap distribution
    """
    inpath = '/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636406_best_ours/TEST'
    outpath = '/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/visualization/analysis'

    font = {'family': 'Times New Roman',
            'color':  'black',
            'size': 16
            }
    leg_font = {'family': 'Times New Roman',
            'size': 16
            }
    plt.rc('font', family='Times New Roman')
    # fig, ax = plt.subplots()
    fig, axs = plt.subplots(nrows=5, ncols=1)


    # file = ["Autism_age_prediction_in_test_exp_0.csv","AD_age_prediction_in_test_exp_0.csv", 
    #         "VCI_age_prediction_in_test_exp_0.csv", "MCI-AD_age_prediction_in_test_exp_0.csv", "ADHD_age_prediction_in_test_exp_0.csv"]
    # Diag = {1:'Autism', 2: 'AD', 3:'VCI', 4:'MCI', 5:'ADHD'} # subinfo_BDs_dive.csv
    file = ["ADHD_sMRI_age_prediction_in_test_exp_0.csv", "Autism_sMRI_age_prediction_in_test_exp_0.csv", 
            "SVD_sMRI_age_prediction_in_test_exp_0.csv","MCI_sMRI_age_prediction_in_test_exp_0.csv", "AD_sMRI_age_prediction_in_test_exp_0.csv"]
    Diag = {1:'ADHD', 2: 'ASD', 3:'SVD', 4:'MCI', 5:'AD', 6:'HC'} # subinfo_BDs_dive.csv
    colors = ['g', 'c', 'm', '#e377c2', '#9467bd', 'lightgray']
    colors_l = ['limegreen', 'darkslategrey' , 'hotpink', 'deeppink',  'indigo'] # 

    for idx, f in enumerate(file):
        csv_file = inpath + '/' + f
        df = pd.read_csv(csv_file)

        diff = df['diff'].to_numpy()
        BAG = np.sum(diff)/df.shape[0]
        print("Disease: {}, BAG {:.2f} ".format(Diag[idx+1], BAG))

        alpha1 = 0.8
        alpha2 = 0.5
        color_h = 'lightgray'
        if idx == 4:
            diff_HC_comp = HC_comp(idx+1)
            axs[idx].hist(diff, bins=100, range=(-15, 15),  color=colors[idx], label=Diag[idx+1], zorder=2, alpha=alpha2, density=True)
            axs[idx].hist(diff_HC_comp,  bins=100, range=(-15, 15), color='gray', label='HCs', density=True, alpha=alpha1)   
            axs[idx].legend( bbox_to_anchor=(-0.015, 1.2), loc='upper left', prop=leg_font, frameon=False, handleheight=0.6, handlelength=2, labelspacing=0.2, handletextpad=0.2)
            axs[idx].axvline(x=np.mean(diff), color=colors_l[idx], linewidth=2)
            axs[idx].axvline(x=np.mean(diff_HC_comp), color=color_h, linewidth=2) # gainsboro

        else:
            axs[idx].hist(diff,  bins=100, range=(-15, 15), color=colors[idx], label=Diag[idx+1], zorder=2, density=True, alpha=alpha2)
            diff_HC_comp = HC_comp(idx+1)

            axs[idx].hist(diff_HC_comp, bins=100, range=(-15, 15), color='gray', label='HCs', density=True, alpha=alpha1)   

            axs[idx].legend(bbox_to_anchor=(-0.015, 1.2), loc="upper left", prop=leg_font, frameon=False, handleheight=0.6, handlelength=2, labelspacing=0.2, handletextpad=0.2)
            axs[idx].tick_params(axis='y',labelsize=15)
            axs[idx].axvline(x=np.mean(diff), color=colors_l[idx], linewidth=2)
            axs[idx].axvline(x=np.mean(diff_HC_comp), color=color_h, linewidth=2) # gainsbora

            axs[idx].set_xlim([-15, 15])
            axs[idx].xaxis.set_major_locator(MultipleLocator(40))
            axs[idx].axes.xaxis.set_ticklabels([])
            axs[idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

        axs[idx].tick_params(axis='y',labelsize=15)
        if idx ==4:
            axs[idx].set_xlim([-15, 15])
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            axs[idx].set_xlabel("Brain Age Gap (years)", fontdict=font, x=0.5, labelpad=1)
            axs[idx].set_ylabel("Probability", fontdict=font, y=3.0, rotation='vertical')
            
    fig.tight_layout()
    plt.savefig(os.path.join(outpath, "BDs_BAG_ours_train_val_test_part.pdf"))
    plt.show()
    plt.close()

if __name__ == '__main__':
    func()
