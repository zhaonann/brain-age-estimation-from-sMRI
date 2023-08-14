import nibabel as nib
import os
import numpy as np
from joblib import Parallel, delayed
import time

tissues = ["CSF", "GM", "WM"]
values = [1, 2, 3]

template = '/public/bme/home/XXXXX/brain_age_est/data_proc/Template/MNI152_T1_1mm_Brain.nii.gz' # fixed

def normalize_neg1_pos1(img):
    min_v = np.min(img)
    max_v = np.max(img)
    ratio = 1.0 * (1 - (-1)) / (max_v - min_v)
    img = 1.0 * (img - min_v) * ratio - 1
    return img


def affine_reg(parpath_T1, parpath_mask, outpath, sub):
    
    txt1 = '/public/bme/home/zhaonan/brain_age_est/data_proc/affine_CBMFM_log.txt'
    txt2 = '/public/bme/home/zhaonan/brain_age_est/data_proc/mask_affine_CBMFM_log.txt'

    outpath = "/public_bme/share/sMRI"
    site = 'CBMFM'

    T1_skull = os.path.join(parpath_T1, site, sub, 'T1_n4_skullstrip.nii.gz')

    sMRI_mask = os.path.join(parpath_mask, site, sub, 'T1_tissue_mask.nii.gz')
    fixImg = template
    movImg = sMRI_mask
    path_img = os.path.join(outpath, 'n4', site, sub)
    path_mask = os.path.join(outpath, 'mask_reg', site, sub)
    if not os.path.exists(path_img):
        os.makedirs(path_img)

    if not os.path.exists(path_mask):
        os.makedirs(path_mask)

    deforField = os.path.join(path_img, 'affine_7.mat')
    outRegMask = os.path.join(path_mask, 'T1_tissue_mask_reg.nii.gz')
    outReg = os.path.join(path_img, 'T1_n4_skullstrip_reg.nii.gz')

    cmd_str1 = "flirt -in {} -ref {} -out {} -omat {} -dof 7 >> {}".format(T1_skull, fixImg, outReg, deforField, txt1)
    cmd_str2 = "flirt -in {} -ref {} -out {} -applyxfm -init {} -interp nearestneighbour >> {}".format(movImg, fixImg, outRegMask, deforField, txt2)

    os.system(cmd_str1)
    os.system(cmd_str2)

    # normalize registrated image to -1 to 1
    T1_org = nib.load(outReg)
    img_affine = T1_org.affine
    img_hdr = T1_org.header
    data_org = np.asarray(T1_org.dataobj) # or sMRI.get_fdata()
    data_norm = normalize_neg1_pos1(data_org)
    T1_new = nib.Nifti1Image(data_norm, img_affine, img_hdr)
    new_path = os.path.join(path_img, 'T1_n4_skullstrip_reg_norm.nii.gz')
    nib.save(T1_new, new_path)

def get_path():
    sMRI = "/public_bme/share/sMRI"
    outpath0 = "/public_bme/share/sMRI/n4/"
    
    site = 'CBMFM'
    outpath = outpath0 + '/' + site
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    path0 = sMRI + '/' + 'n4' + '/' + site
    for sub in os.listdir(path0):
        parpath_T1 = sMRI + '/' + 'n4' 
        parpath_mask = sMRI + '/' + 'seg'
        yield (parpath_T1, parpath_mask, outpath, sub)

if __name__ == '__main__':

    print("Start!")
    os.system('date')
    start_time = time.time()
    
    # affine registration with dof 7, registrating T1 and its mask to MNI 152 space
    Parallel(n_jobs=-2)(delayed(affine_reg)(para1, para2, para3, para4) for para1, para2, para3, para4 in get_path())

    print("Finished!")
    os.system('date')
    print('End of Parallel, Time Taken: %d sec' % ( time.time() - start_time))
