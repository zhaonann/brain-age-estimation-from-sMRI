

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def Age_distribution_HC():
    in_file = '/home/zhaonan/XXXXX/BrainAgeEst/6-site/subinfo_8_DK106_new.csv'
    outpath = '/home/zhaonan/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/dist_figure'
    ### 1=Male 0=Female
    # dict_cor = ['ABIDE', 'ADHD-200', 'ADNI', 'OASIS', 'CoRR', 'RENJI', 'HUASHAN', 'CBMFM'] # 1-8
    dict_cor = ['ABIDE', 'ADHD-200', 'ADNI', 'OASIS', 'CoRR', r'Institute$_1$',  r'Institute$_2$', r'Institute$_3$'] # 1-8

    df = pd.read_csv(in_file)
    print(df.columns)
    L = df.shape[0]

    df = df.loc[(df['disease'] == 0)].copy()    
    df['site'] = df['site'] + 10

    df["site"].loc[df["site"]==11] = 1   
    df["site"].loc[df["site"]==12] = 6      
    df["site"].loc[df["site"]==13] = 7     
    df["site"].loc[df["site"]==14] = 3  
    df["site"].loc[df["site"]==15] = 4  
    df["site"].loc[df["site"]==16] = 2   
    df["site"].loc[df["site"]==17] = 5  
    df["site"].loc[df["site"]==18] = 8   

    df1 = df.loc[(df['site'] == 1)].copy()    
    df2 = df.loc[(df['site'] == 2)].copy()    
    df3 = df.loc[(df['site'] == 3)].copy()    
    df4 = df.loc[(df['site'] == 4)].copy()    
    df5 = df.loc[(df['site'] == 5)].copy()    
    df6 = df.loc[(df['site'] == 6)].copy()    
    df7 = df.loc[(df['site'] == 7)].copy()    
    df8 = df.loc[(df['site'] == 8)].copy()    
    com_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
    com_df = com_df[['age', 'site']].copy()
    No = com_df.shape[0]
    com_df['No'] = np.arange(No)
    com_df = com_df.set_index('No')
    print(com_df)
    plt.rc('font', family='Times New Roman')

    fig, ax1 = plt.subplots()
    font = {'color':  'black',
            'size': 16
            }
    font_l = {
        'size': 16,
        'style': 'normal'
            }

    way="stack"
    hue_order = [1, 2, 3, 4, 5, 6, 7, 8]
    sp1 = sns.histplot(data=com_df, x='age', hue='site', hue_order=hue_order, binwidth=2, shrink=.8, palette='deep', multiple=way,  edgecolor ='none', ax=ax1, alpha=0.8)

    plt.xlabel("Chronological Age (years)", fontdict=font)
    plt.ylabel("Number of Subjects", fontdict=font)
    plt.xlim([0, 100])
    plt.xticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0, 250])
    plt.yticks([0, 50, 100, 150, 200, 250])

    legend = ax1.get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax1.legend(handles, dict_cor, prop=font_l, frameon=False, columnspacing=0.4, handleheight=0.7, handlelength=2, labelspacing=0.4, ncol=2, handletextpad=0.2)

    fig.tight_layout()
    plt.savefig(os.path.join(outpath, 'HCs_Age_Distribution_8_site_{}_anony.pdf'.format(way)))
    plt.show()
    plt.close()

def Age_distribution_BD():
    in_file = '/home/zhaonan/XXXXX/BrainAgeEst/6-site/subinfo_8_DK106_new.csv'
    outpath = '/home/zhaonan/XXXXX/BrainAgeEst/MICCAI-WORKSHOP/dist_figure'
    ### 1=Male 0=Female
    dict_cor = ['ADHD',  'ASD', 'SVD', 'MCI', 'AD'] # 1-8


    df = pd.read_csv(in_file)


    print(df.columns)
    L = df.shape[0]

    df = df.loc[(df['disease'] != 0)].copy()    
    df['disease'] = df['disease'] + 10

    df["disease"].loc[df["disease"]==11] = 2  
    df["disease"].loc[df["disease"]==12] = 5 
    df["disease"].loc[df["disease"]==13] = 3  
    df["disease"].loc[df["disease"]==14] = 4  
    df["disease"].loc[df["disease"]==15] = 1  

    df1 = df.loc[(df['disease'] == 1)].copy()    
    df2 = df.loc[(df['disease'] == 2)].copy()    
    df3 = df.loc[(df['disease'] == 3)].copy()    
    df4 = df.loc[(df['disease'] == 4)].copy()    
    df5 = df.loc[(df['disease'] == 5)].copy()    

    com_df = pd.concat([df1, df2, df3, df4, df5])
    com_df = com_df[['age', 'disease']].copy()
    No = com_df.shape[0]
    com_df['No'] = np.arange(No)
    com_df = com_df.set_index('No')
    print(com_df)
    plt.rc('font', family='Times New Roman')

    fig, ax1 = plt.subplots()
    font = {'color':  'black',
            'size': 16
            }
    font_l = {
        'size': 16,
        'style': 'normal'
            }

    way="stack"
    hue_order = [1, 2, 3, 4, 5]
    sp1 = sns.histplot(data=com_df, x='age', hue='disease', hue_order=hue_order, binwidth=2, shrink=.8, palette='deep', multiple=way,  edgecolor ='none', ax=ax1, alpha=0.8)

    plt.xlabel("Chronological Age (years)", fontdict=font)
    plt.ylabel("Number of Subjects", fontdict=font)
    plt.xlim([0, 100])
    plt.xticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.ylim([0, 250])
    plt.yticks([0, 50, 100, 150, 200, 250])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    legend = ax1.get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax1.legend(handles, dict_cor, prop=font_l, bbox_to_anchor=(1.025, 1.025), loc='upper right', frameon=False, columnspacing=0.4, handleheight=0.7, handlelength=2, labelspacing=0.4, handletextpad=0.2)

    fig.tight_layout()
    plt.savefig(os.path.join(outpath, 'BDs_Age_Distribution_8_site_{}.pdf'.format(way)))
    plt.show()
    plt.close()



if __name__ == "__main__":
    Age_distribution_HC()
    Age_distribution_BD()
