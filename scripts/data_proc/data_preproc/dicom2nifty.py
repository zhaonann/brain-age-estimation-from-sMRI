import os

def transfer():
    """
    DICOM to NifTI
    """
    sour_dir_dcm = "/home/zhaonan/disk/ADNI/"
    dest_dir_nii = "/home/zhaonan/disk/ADNI_T1Img/"

    if not os.path.exists(dest_dir_nii):
        os.makedirs(dest_dir_nii)

    for idx in os.listdir(sour_dir_dcm):
        os.system("mkdir {}".format(dest_dir_nii + idx))
        cmd_str = "/home/zhaonan/mricron/Resources/dcm2niix -p y -x y -z y -o {} {}".format(dest_dir_nii + idx, sour_dir_dcm + idx)
        os.system(cmd_str)

if __name__ == "__main__":
    transfer()
