import os
import ants
import multiprocessing as mp
import time
import shutil

def bias_correction(subpath):
    """
    n4 bias correction for each subject
    """
    outpath_bias = "/home/XXXXX/disk/CoRR_T1_bias"
    site = "NKI_1"
    sub = os.path.basename(subpath)
    T1_path = subpath + '/' + 'session_1' + '/' + "anat_1" + '/' + 'anat.nii.gz'
    org = ants.image_read(T1_path)
    n4 = ants.n4_bias_field_correction(org)
    outfile = "bias_correction.nii.gz"
    dest_dir_bias = os.path.join(outpath_bias, site, sub)
    if not os.path.exists(dest_dir_bias):
        os.makedirs(dest_dir_bias)
    ants.image_write(n4, dest_dir_bias + '/' + outfile)


def multiproc():
    srcpath = "/home/XXXXX/disk/CoRR_T1"
    site = "NKI_1" # BMB_1, IBATRT, NYU_1, UWM, NYU_2
    srcpath1 = os.path.join(srcpath, site)
    paths = []
    for sub in os.listdir(srcpath1):
        path = srcpath + '/' + site + '/' + sub # path to subject. /path/ADNI/sub1
        paths.append(path) 

    with mp.Pool(8) as pool:  
        pool.map(bias_correction, paths)  
        pool.close()  
        pool.join()

if __name__ == "__main__":

    start_time = time.time()
    multiproc()
    print('Time Taken: %d sec' % ( time.time() - start_time))