import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def read_file(file, idx):
    df = pd.read_csv(file)
    df = df.sort_values('diff')
    L = df.shape[0]
    df['New_idx'] = np.arange(df.shape[0])
    df = df.set_index('New_idx')

    age_gap = df['diff']
    age_gap = np.sort(age_gap)
    return df, age_gap

def cal_measures(df=None):
    ages = df['age']
    preds_s = df['preds_s']
    diff = df['diff']
    MAE = np.sum(np.abs(diff))/df.shape[0]
   
    p_value = pearsonr(ages, preds_s)
    r = round(p_value[0], 3)
    rmse = np.sqrt(mean_squared_error(ages, preds_s))
    rmse = round(rmse, 2)
    r2 = r2_score(ages, preds_s)
    r2 = round(r2, 2)
    return MAE, rmse, r

def plotfigure(dfs):
    """
    show the predictions on HC for methods of ours and SFCN
    """
    outpath = '/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/analysis'

    colors = ['c', '#e377c2']
    font = {'family': 'Times New Roman',
            'color':  'black',
            'size': 16
            }
    leg_font = {'family': 'Times New Roman',
            'size': 16
            }
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()

    df1 =dfs[0] # Ours
    df2 = dfs[1] # SFCN
    ax.scatter(df2['age'], df2['preds_s'], c=colors[0], edgecolors='dimgray', label='SFCN', alpha=0.5)

    ax.scatter(df1['age'], df1['preds_s'], c=colors[1], edgecolors='dimgray', label='Ours', alpha=0.5)

    v_min = df1['age'].min()
    v_max = df1['age'].max()

    ax.plot(np.linspace(v_min, v_max, 100), np.linspace(v_min, v_max, 100),
            linestyle='dashed', c='lightgray', label='Matched Prediction')

    plt.xticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.yticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlim([-5, 105])
    ax.set_ylim([-5, 105])
    ax.set_xlabel("Chronological Age (years)", fontdict=font)
    ax.set_ylabel('Brain-Predicted Age (years)', fontdict=font)
    plt.legend(loc='upper left', prop=leg_font, frameon=False, handleheight=0.7, handlelength=2, labelspacing=0.4, handletextpad=0.2)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig(os.path.join(outpath, "test_HCs_ours_SFCN" + ".pdf"))
    plt.show()
    plt.close()
    print("FINISH TEST")

def func():
    ours = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636406_best_ours/FIGURES_CSV/age_prediction_best_in_test_exp_0.csv"
    SFCN = '/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636625_SFCN/FIGURES_CSV/age_prediction_in_test_best_exp_0.csv' 
    res18 = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636405_res18/FIGURES_CSV/age_prediction_in_test_best_exp_0.csv"
    res34 = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/635864_res34/FIGURES_CSV/age_prediction_in_test_best_exp_0.csv"
    base_3d = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/635871_3d/FIGURES_CSV/age_prediction_in_test_best_exp_0.csv"
    base_3d_GAF = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636569_GAF/FIGURES_CSV/age_prediction_best_in_test_exp_0.csv"
    base_3d_3view = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636284_3d_3view_bad/FIGURES_CSV/age_prediction_best_in_test_exp_0.csv"
    GAF = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/MEI_636579_GAF/FIGURES_CSV/age_prediction_best_in_test_exp_0.csv"
    view3 = "/home/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636569_2d/FIGURES_CSV/age_prediction_best_in_test_exp_0.csv"

    total = 11
    files = [ours, SFCN, res18, res34, base_3d, base_3d_GAF, base_3d_3view, GAF, view3]
    dfs = [0]*total
    age_gaps = [0]*total
    MAEs = [0]*total
    RMSEs = [0]*total
    r = [0]*total

    Method = {1: "Ours", 2: "SFCN", 3:"ResNet-18", 4: "ResNet-34", 5: "3D", 6:"3D_GAF", 7:"3D-3view", 8:"GAF", 9:"3view"}
    for i in range(total):
        dfs[i], age_gaps[i] = read_file(file=files[i], idx=i)
        MAEs[i], RMSEs[i], r[i] = cal_measures(dfs[i])
        print("Method: {}, MAE: {:.2f}, RMSE: {:.2f}, r:{:.3f} ".format(Method[i+1], MAEs[i], RMSEs[i], r[i]))

    plotfigure(dfs)


if __name__ == '__main__':
    func()
