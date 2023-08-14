import os
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import torch
import torchvision.transforms.functional as TF


def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

def show_one_slice(data_path, filename, crop=False):
    new_path = "/home/zhaonan/ZHAONAN/BrainAgeEst/6-site/showcase/AFM0459_crop"
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    imgs = sitk.ReadImage(data_path)  # read MRI images of .mhd format, [width, height, depth]
    imgs = sitk.GetArrayFromImage(imgs)  # indices are z, y, x, ndarray
    if crop:
        imgs = crop_center(imgs, (160, 192, 160))

    z, y, x = imgs.shape # 
    img_ax = imgs.transpose(0,1,2) # z x y 
    idx= int(z/2)
    img = img_ax[idx]
    axial = np.rot90(img, 2)  # rotate 180 <left> anti-clockwise # imgs[idx][::-1, ::-1]

    img_sa = imgs.transpose(2,0,1) # x y z
    x, z, y = img_sa.shape # 
    idx= int(x/2)
    sagittal = img_sa[idx+5]
    sagittal = np.rot90(sagittal, 2)  

    img_co = imgs.transpose(1,0,2) # x y z
    y, z, x = img_co.shape # 
    idx= int(y/2)
    coronal = img_co[idx]
    coronal = np.rot90(coronal, 2) 

    print(z)
    print(x)
    print(y)

    Fig = plt.figure()
    plt.imshow(axial, 'gray') 
    plt.xticks([])
    plt.yticks([])
    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, '{}_One_Axial_{}.eps'.format(filename, crop)))
    plt.show()
    plt.close()

    Fig = plt.figure()
    plt.imshow(sagittal, 'gray') 
    plt.xticks([])
    plt.yticks([])
    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, '{}_One_Sagittal_{}.eps'.format(filename,crop)))
    plt.show()
    plt.close()

    Fig = plt.figure()
    plt.imshow(coronal, 'gray') 
    plt.xticks([])
    plt.yticks([])
    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, '{}_One_Coronal_{}.eps'.format(filename,crop)))
    plt.show()
    plt.close()

def show_single_subject(data_path, filename, crop=False):
    new_path = "/home/zhaonan/ZHAONAN/BrainAgeEst/6-site/showcase/crop"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    # Axial
    Fig = plt.figure("Axial")
    Fig.set_facecolor('green')

    imgs = sitk.ReadImage(data_path)  # read MRI images of .mhd format, [width, height, depth]
    imgs = sitk.GetArrayFromImage(imgs)  # indices are z, y, x, ndarray
    if crop:
        imgs = crop_center(imgs, (160, 192, 160))
    z, y, x = imgs.shape
    idx= int(z/2)
    print(z)
    print(y)
    print(x)
    for j, img in enumerate(imgs):
        plt.subplot(19, 10,  j+1)
        plt.imshow(img[::-1, ::-1], 'gray')
        plt.xticks([])
        plt.yticks([])

    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, '{}_Axial_{}.eps'.format(filename, crop)))
    plt.show()
    plt.close()

    # Sagittal
    Fig = plt.figure("Sagittal")
    Fig.set_facecolor('green')


    imgs = sitk.ReadImage(data_path)  # read MRI images of .mhd format, [width, height, depth]
    imgs = sitk.GetArrayFromImage(imgs)  # indices are z, y, x, ndarray
    z, y, x = imgs.shape
    img_sa = imgs.transpose(2,0,1) # x y z
    x, z, y = img_sa.shape # 
    for j, img in enumerate(img_sa):
        plt.subplot(19, 10,  j+1)
        plt.imshow(img[::-1, ::-1], 'gray')
        plt.xticks([])
        plt.yticks([])

    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, '{}_Sagittal_{}.eps'.format(filename, crop)))
    plt.show()
    plt.close()

    # Coronal
    Fig = plt.figure("Coronal")
    Fig.set_facecolor('green')

    imgs = sitk.ReadImage(data_path)  # read MRI images of .mhd format, [width, height, depth]
    imgs = sitk.GetArrayFromImage(imgs)  # indices are z, y, x, ndarray
    img_co = imgs.transpose(1,0,2) # x y z
    y, z, x = img_co.shape # 
    for j, img in enumerate(img_co):
        plt.subplot(22, 10,  j+1)
        plt.imshow(img[::-1, ::-1], 'gray')
        plt.xticks([])
        plt.yticks([])

    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, '{}_Coronal_{}.eps'.format(filename, crop)))
    plt.show()
    plt.close()

def show_three_slices(data_path, filename):
    new_path = "/home/zhaonan/ZHAONAN/BrainAgeEst/6-site/showcase/AFM0459_crop"
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    Fig = plt.figure()

    Fig.set_facecolor('white')

    imgs = nib.load(data_path)
    imgs = np.array(imgs.dataobj) # or sMRI.get_fdata()

    imgs = crop_center(imgs, (160, 192, 160))
    data_sMRI = torch.tensor(imgs, dtype=torch.float32) # x, y, z

    z, y, x = imgs.shape # 
    print(z,y,x) # 182 218 182

    start = 44 
    end = 113
    img_ax = data_sMRI.permute(2, 1, 0) # z, y, x, axial
    axial1 = img_ax[start] # 3, 218, 182
    axial2 = img_ax[int((start+end)/2)] # 3, 218, 182
    axial3 = img_ax[end] # 3, 218, 182

    plt.subplot(1, 3, 1)
    plt.imshow(TF.vflip(axial1), 'gray') 
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(TF.vflip(axial2), 'gray') 
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(TF.vflip(axial3), 'gray') 
    plt.xticks([])
    plt.yticks([])

    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, 'Nib_{}_axial_3_crop.eps'.format(filename)))
    plt.show()
    plt.close()

    start = 45
    end = 114
    img_ax = data_sMRI.permute(0, 2, 1) # x, z, y, sagittal
    axial1 = img_ax[start] # 3, 218, 182
    axial2 = img_ax[int((start+end)/2)] # 3, 218, 182
    axial3 = img_ax[end] # 3, 218, 182

    plt.subplot(1, 3, 1)
    plt.imshow(TF.vflip(axial1), 'gray') 
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(TF.vflip(axial2), 'gray') 
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(TF.vflip(axial3), 'gray') 
    plt.xticks([])
    plt.yticks([])


    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, 'Nib_{}_sagittal_3_crop.eps'.format(filename)))
    plt.show()
    plt.close()

    start = 55 
    end = 134
    img_ax = data_sMRI.permute(1, 2, 0) # y, z, x, coronal
    axial1 = img_ax[start] # 3, 218, 182
    axial2 = img_ax[int((start+end)/2)] # 3, 218, 182
    axial3 = img_ax[end] # 3, 218, 182

    plt.subplot(1, 3, 1)
    plt.imshow(TF.vflip(axial1), 'gray') 
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(TF.vflip(axial2), 'gray') 
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(TF.vflip(axial3), 'gray') 
    plt.xticks([])
    plt.yticks([])


    Fig.tight_layout()
    plt.savefig(os.path.join(new_path, 'Nib_{}_coronal_3_crop.eps'.format(filename)))
    plt.show()
    plt.close()


def show_one_axial(data_path, filename, age):
    """
    show aging pattern from 10-90
    """
    new_path = "/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/AGING_PATTERN/OUT_BASAL_GANGLIA"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    Fig = plt.figure("{}".format(age))
    Fig.set_facecolor('white')

    imgs = sitk.ReadImage(data_path)  # read MRI images of .mhd format, [width, height, depth]
    imgs = sitk.GetArrayFromImage(imgs)  # indices are z, y, x, ndarray


    z, y, x = imgs.shape # 
    img_ax = imgs.transpose(0,1,2) # z x y 
    if str(age) == '20':
        idx = 83 - 1
    else:
        idx = 88 - 1
    
    img = img_ax[idx]
    axial = np.rot90(img, 2)  # rotate 180 <left> anti-clockwise # imgs[idx][::-1, ::-1]
    plt.imshow(axial, 'gray') 

    print(z)
    print(x)
    print(y)
    plt.xticks([])
    plt.yticks([])
    Fig.tight_layout()

    plt.savefig(os.path.join(new_path, '{}_Axial_{}.eps'.format(filename, age)), bbox_inches='tight')
    plt.show()
    plt.close()


def show_10_Rigid():
    root="/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/AGING_PATTERN/Age_10_90"
    age_list = [10, 20, 30, 40, 50, 60, 70, 80, 90,]
    for age in age_list:
        sec_path = root + '/' + str(age)
        for file in os.listdir(sec_path):
            filename = file
            third_path = sec_path + '/' + filename
            data_path = third_path + '/' + "Warped_norm.nii.gz"
            show_one_axial(data_path, filename, age)

def architecture_show_case():
    data_path =  '/home/zhaonan/ZHAONAN/BrainAgeEst/6-site/AFM0459/T1_n4_skullstrip_reg_norm_0_1.nii'
    filename = "AFM0459" # HC, HUASHAN, age 70, sex=1 male   
    show_one_slice(data_path=data_path, filename=filename, crop=True)

if __name__ == "__main__":

    show_10_Rigid()

