from __future__ import (absolute_import, division, print_function, unicode_literals)


import numpy as np
import tomopy
import pyxas
from copy import deepcopy
from scipy.ndimage import shift, center_of_mass
from pystackreg import StackReg
from pyxas.image_util import dftregistration

def align_img(img_ref, img, align_flag=1):
    img1_fft = np.fft.fft2(img_ref)
    img2_fft = np.fft.fft2(img)
    output = dftregistration(img1_fft, img2_fft, 10)
    row_shift = output[2]
    col_shift = output[3]
    if align_flag:
        img_shift = shift(img, [row_shift, col_shift], mode='constant', cval=0, order=1)
        return img_shift, row_shift, col_shift
    else:
        return row_shift, col_shift 


def align_img_stackreg(img_ref, img):
    sr = StackReg(StackReg.TRANSLATION)
    tmat = sr.register(img_ref, img)
    img_ali = sr.transform(img)
    row_shift = -tmat[1, 2]
    col_shift = -tmat[0, 2]
    return img_ali, row_shift, col_shift


def align_img_stack(img, img_mask=None, select_image_index=None):
    img_align = deepcopy(img)
    n = img_align.shape[0]
    if img_mask.any()==None:
        img_mask = deepcopy(img)
    if select_image_index==None:
        for i in range(1, n):     
            img_mask[i], r, c = align_img(img_mask[i-1], img_mask[i], align_flag=1)
            img_align[i] = shift(img_align[i], [r, c], mode='constant', cval=0)
            print('aligning #{0}, rshift:{1:3.2f}, cshift:{2:3.2f}'.format(i,r,c))
    else:
        print('align image stack refereced with imgage[{}]'.format(select_image_index))
        for i in range(n):
            _, r, c = align_img(img_mask[select_image_index], img_mask[i], align_flag=1)
            img_align[i] = shift(img_align[i], [r, c], mode='constant', cval=0)
            print('aligning #{0}, rshift:{1:3.2f}, cshift:{2:3.2f}'.format(i,r,c))            
    return img_align


def align_img_stack_stackreg(img, img_mask=None, select_image_index=None):
    img_align = deepcopy(img)
    n = img_align.shape[0]
    if not len(img_mask):
        img_mask = deepcopy(img)
    if select_image_index == None:
        for i in range(1, n):
            img_mask[i], r, c = align_img_stackreg(img_mask[i - 1], img_mask[i])
            img_align[i] = shift(img_align[i], [r, c], mode='constant', cval=0)
            print('aligning #{0}, rshift:{1:3.2f}, cshift:{2:3.2f}'.format(i, r, c))
    else:
        print('align image stack refereced with imgage[{}]'.format(select_image_index))
        for i in range(n):
            img_mask[i], r, c = align_img_stackreg(img_mask[select_image_index], img_mask[i])
            img_align[i] = shift(img_align[i], [r, c], mode='constant', cval=0)
            print('aligning #{0}, rshift:{1:3.2f}, cshift:{2:3.2f}'.format(i, r, c))
    return img_align


def align_two_img_stack(img_ref, img):
    s = img_ref.shape
    img_ali = deepcopy(img)
    for i in range(s[0]):
        img_ali[i],_, _ = align_img(img_ref[i], img[i])
    return img_ali



def move_3D_to_center(img, circle_mask_ratio):
    from scipy.ndimage import center_of_mass
    img0 = img
    s = np.array(img0.shape)/2
    img0 = tomopy.circ_mask(img0, axis=0, ratio=circle_mask_ratio, val=0)
    cm = np.array(center_of_mass(img0))
    shift_matrix = list(s - cm)
    img_cen = pyxas.shift(img, shift_matrix, order=0)
    return img_cen



def align_3D_fine(img_ref, img1, circle_mask_ratio=1, sli_select=0, row_select=0, test_range=[-30, 30], sli_shift_guess=0, row_shift_guess=0, col_shift_guess=0, cen_mass_flag=0, ali_direction=[1,1,1]):
    '''
    ali_direction = [1,1,1] -> shift [sli, row, col] if it is "1"
    '''
    
    from scipy.ndimage import center_of_mass
    time_s = time.time()
    img_tmp = img_ref.copy()
    img_ref_crop = tomopy.circ_mask(img_tmp, axis=0, ratio=circle_mask_ratio, val=0)
    
    img_tmp = img1.copy()
    img_raw_crop = tomopy.circ_mask(img_tmp, axis=0, ratio=circle_mask_ratio, val=0)
    if sli_shift_guess != 0 or row_shift_guess != 0 or col_shift_guess != 0:
        img_raw_crop= shift(img_raw_crop, [sli_shift_guess, row_shift_guess, col_shift_guess], order=0)

    if sli_select == 0 or sli_select >= img_ref_crop.shape[0]:
        sli_select = int(img_ref_crop.shape[0]/2.0)

    if row_select == 0 or row_select >= img_ref_crop.shape[1]:
        row_select = int(img_ref_crop.shape[1]/2.0)

    if cen_mass_flag:
        prj_ref = np.sum(img_ref_crop, axis=1)
        sli_select = int(center_of_mass(prj_ref)[0])
        prj_ref = np.sum(img_ref_crop, axis=0)
        row_select = int(center_of_mass(prj_ref)[0])
    print(f'aligning using sli = {sli_select}, row = {row_select}')

    # align height first (sli)
    if ali_direction[0] == 1:
        print('aligning height ...')
        t1 = np.squeeze(img_ref_crop[:, row_select])
        t1 = t1/np.mean(t1)
        t1_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(t1)))
                
        rang = np.arange(test_range[0], test_range[1])
        corr_max = []
        for j in rang + row_select:
            t2 = np.squeeze(img_raw_crop[:, j])
            t2 = t2/np.mean(t2)
            t2_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(t2)))
            tmp = np.fft.ifft2(t1_fft * np.conj(t2_fft))  
            corr_max.append(np.max(tmp))      
        _, idmax = idxmax(np.abs(corr_max))
        row_shft = -rang[int(idmax)]
        t2 = np.squeeze(img_raw_crop[:, row_select])
        sli_shft, cshft = pyxas.align_img(t1, t2, align_flag=0)
        img_raw_crop = shift(img_raw_crop, [sli_shft, 0, 0], order=1)

    # align row and col
    print('aligning row and col ...')
    t1 = img_ref_crop[sli_select]
    t1 = t1/np.mean(t1)
    t1_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(t1)))
    t2 = img_raw_crop[sli_select]
    rshft, cshft = pyxas.align_img(t1, t2, align_flag=0)

    if ali_direction[1] == 0:
        rshft = 0
    if ali_direction[2] == 0:
        cshft = 0
    img_ali= shift(img_raw_crop, [0, rshft, cshft], order=1)

    shift_matrix = [sli_shft, rshft, cshft]
    print(f'sli_shift: {sli_shft: 04.1f},   rshft: {rshft: 04.1f},   cshft: {cshft: 04.1f}')
    print(f'time elaped: {time.time() - time_s:4.2f} sec')
    return img_ali, shift_matrix



def align_3D_coarse_axes(img_ref, img1, circle_mask_ratio=0.6, axes=0, shift_flag=1):
    '''
    Aligning the reconstructed tomo with assigned 3D reconstruction along given axis. It will project the 3D data along given axis to find the shifts

    Inputs:
    -----------
    ref: 3D array

    data: 3D array need to align

    axis: int
        along which axis to project the 3D reconstruction to find image shifts 
        0, or 1, or 2
    
    Output:
    ----------------
    aligned tomo, shfit_matrix

    '''
    img_tmp = img_ref.copy()
    img_ref_crop = tomopy.circ_mask(img_tmp, axis=0, ratio=circle_mask_ratio, val=0)    
    s = img_ref_crop.shape
    stack_range = [int(s[0]*(0.5-circle_mask_ratio/2)), int(s[0]*(0.5+circle_mask_ratio/2))]
    prj0 = np.sum(img_ref_crop[stack_range[0]:stack_range[1]], axis=axes)

    img_tmp = img1.copy()    
    img_raw_crop = tomopy.circ_mask(img_tmp, axis=0, ratio=circle_mask_ratio, val=0)
    prj1 = np.sum(img_raw_crop[stack_range[0]:stack_range[1]], axis=axes)
    
    sr = StackReg(StackReg.TRANSLATION)
    tmat = sr.register(prj0, prj1)
    r = -tmat[1, 2]
    c = -tmat[0, 2]  
    #print(f'rshift = {r}, cshift = {c}')
    if axes == 0:
        shift_matrix = np.array([0, r, c])
    elif axes == 1:
        shift_matrix = np.array([r, 0, c])
    elif axes == 2:
        shift_matrix = np.array([r, c, 0])
    else:
        shift_matrix = np.array([0, 0, 0])
    if shift_flag:
        img_ali = pyxas.shift(img1, shift_matrix, order=0)
        return img_ali, shift_matrix    
    else:
        return shift_matrix


def align_3D_coarse(img_ref, img1, circle_mask_ratio=1, method='other'):
    '''
    method: 'center_mass' 
            else: aligning projection
    '''
    if method == 'center_mass':
        
        img0_crop = img_ref
        img1_crop = img1
        img0_crop = tomopy.circ_mask(img0_crop, axis=0, ratio=circle_mask_ratio, val=0)
        img1_crop = tomopy.circ_mask(img1_crop, axis=0, ratio=circle_mask_ratio, val=0)
        cm0 = np.array(center_of_mass(img0_crop))
        cm1 = np.array(center_of_mass(img1_crop))
        shift_matrix = cm1 - cm0
    else:    
        shift_matrix0 = pyxas.align_3D_coarse_axes(img_ref, img1, circle_mask_ratio=circle_mask_ratio, axes=0, shift_flag=0)
        shift_matrix1 = pyxas.align_3D_coarse_axes(img_ref, img1, circle_mask_ratio=circle_mask_ratio, axes=1, shift_flag=0)
        shift_matrix2 = pyxas.align_3D_coarse_axes(img_ref, img1, circle_mask_ratio=circle_mask_ratio, axes=2, shift_flag=0)
        shift_matrix = (shift_matrix0 + shift_matrix1 + shift_matrix2) / 2.0
    print(f'shifts: {shift_matrix}')
    img_ali = pyxas.shift(img1, shift_matrix, order=0)
    return img_ali, shift_matrix





def align_3D_tomo_file(file_path='.', ref_index=-1, binning=1, circle_mask_ratio=0.9, file_prefix='recon', file_type='.h5'):
    file_path = os.path.abspath(file_path)
    files_recon = pyxas.retrieve_file_type(file_path, file_start=file_prefix, file_type=file_type)

    num_file = len(files_recon)
    res = pyxas.get_img_from_hdf_file(files_recon[ref_index], 'img', 'scan_id', 'X_eng')
    img_ref = res['img']
    scan_id = int(res['scan_id'])
    X_eng = float(res['X_eng'])
    s = img_ref.shape
    if binning > 1:
        img_ref = pyxas.bin_image(img_ref, binning)
    img_ref = pyxas.move_3D_to_center(img_ref, circle_mask_ratio=circle_mask_ratio)
    fn_save = f'{file_path}/ali_recon_{scan_id}_bin_{binning}.h5'
    pyxas.save_hdf_file(fn_save, 'img', img_ref, 'scan_id', scan_id, 'X_eng', X_eng)

    time_start = time.time()
    for i in range(num_file):
        fn = files_recon[i]
        fn_short = fn.split('/')[-1]
        if fn == files_recon[ref_index]:
            continue
        print(f'#{i+1}/{num_file}  aligning {fn_short} ...')
        res = pyxas.get_img_from_hdf_file(fn, 'img', 'scan_id', 'X_eng')
        img1 = res['img']
        scan_id = int(res['scan_id'])
        X_eng = float(res['X_eng'])
        if binning > 1:
            img1 = pyxas.bin_image(img1, binning)
        img1, shift_matrix = pyxas.align_3D_coarse(img_ref, img1, circle_mask_ratio=circle_mask_ratio, method='other')
        img1_ali, shift_matrix = pyxas.align_3D_fine(img_ref, img1, circle_mask_ratio=circle_mask_ratio, sli_select=0, row_select=0, test_range=[-30, 30], sli_shift_guess=0, row_shift_guess=0, col_shift_guess=0, cen_mass_flag=1)
        fn_save = f'{file_path}/ali_recon_{scan_id}_bin_{binning}.h5'  
        print(f'saving aligned file: {fn_save.split("/")[-1]}\n')
        pyxas.save_hdf_file(fn_save, 'img', img1_ali, 'scan_id', scan_id, 'X_eng', X_eng)   
        print(f'time elasped: {time.time() - time_start:05.1f}\n')
        
        

