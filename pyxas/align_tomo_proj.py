import pyxas
import numpy as np
import time
import h5py

def align_tomo_proj_serial(file_path, file_prefix='fly', file_type='.h5', ref_index=-1):
    files_scan = pyxas.retrieve_file_type(file_path, file_prefix, file_type)
    n = len(files_scan)
    if ref_index == -1:
        ref_index = len(files_scan)-1
    fn_ref = files_scan[ref_index]
    res_ref = pyxas.get_img_from_hdf_file(fn_ref, 'angle', 'img_tomo', 'img_bkg_avg', 'img_dark_avg', 'scan_id', 'X_eng')
    fn_current = fn_ref.split('/')[:-1]
    fn_current = '/'.join(tmp for tmp in fn_current)
    fn_short = fn_ref.split('/')[-1]
    fn_save = f'{fn_current}/ali_{fn_short}'

    ref_img = res_ref['img_tomo']
    ref_angle = res_ref['angle']
    r_shift = np.zeros(len(ref_angle))
    c_shift = r_shift.copy()
    pyxas.save_hdf_file(fn_save, 'angle', res_ref['angle'], 'img_tomo', res_ref['img_tomo'], 'img_bkg_avg', res_ref['img_bkg_avg'], 'img_dark_avg', res_ref['img_dark_avg'], 'scan_id', res_ref['scan_id'], 'X_eng', res_ref['X_eng'], 'r_shift', r_shift, 'c_shift', c_shift)

    for i in range(len(files_scan)):
        fn = files_scan[i]
        if i == ref_index:
            continue
        time_s = time.time()
        res = pyxas.get_img_from_hdf_file(fn, 'angle', 'img_tomo', 'img_bkg_avg', 'img_dark_avg', 'scan_id', 'X_eng')
        res_angle = res['angle']
        prj_ali = res['img_tomo'].copy()
        r_shift = np.zeros(len(res_angle))
        c_shift = r_shift.copy()
        for j in range(len(res_angle)):
            angle_id = pyxas.find_nearest(ref_angle, res_angle[j])
            img_ref = res_ref['img_tomo'][angle_id]
            prj_ali[j], r_shift[j], c_shift[j] = pyxas.align_img_stackreg(img_ref, prj_ali[j], align_flag=1)
            print(f'{fn.split("/")[-1]} proj #{j}: r_shift = {r_shift[j]:3.1f}, c_shift = {c_shift[j]:3.1f}')
        
        fn_short = fn.split('/')[-1]
        fn_save = f'{fn_current}/ali_{fn_short}'

        pyxas.save_hdf_file(fn_save, 'angle', res['angle'], 'img_tomo', np.array(prj_ali, dtype=np.float32), 'img_bkg_avg', res['img_bkg_avg'], 'img_dark_avg', res['img_dark_avg'], 'scan_id', res['scan_id'], 'X_eng', res['X_eng'], 'r_shift', r_shift, 'c_shift', c_shift)
        print(f'{fn_save.split("/")[-1]} saved,  time elaped: {time.time() - time_s:4.2f} sec')
        


def align_tomo_prj_mpi_sub(current_angle_index, fn_ref, fn_target, img_dark, img_bkg):    

    f_ref = h5py.File(fn_ref, 'r')
    f_target = h5py.File(fn_target, 'r')

    ref_angle = np.array(f_ref['angle'])
    target_angle = np.array(f_target['angle']) 
    angle_id = pyxas.find_nearest(ref_angle, target_angle[int(current_angle_index)])

    img_ref = np.array(f_ref['img_tomo'][angle_id])
    img = np.array(f_target['img_tomo'][int(current_angle_index)])

    r, c = pyxas.align_img_stackreg(img_ref, img, align_flag=0)
    tmp = np.squeeze((img - img_dark) / (img_bkg - img_dark))    
    res = {}
    res['img_ali'] = shift(tmp, [r, c], mode='constant', cval=0, order=1)
    res['r'] = r
    res['c'] = c
    print(f'proj# {current_angle_index}: r_shift = {r:3.1f},  c_shift = {c:3.1f}')
    f_ref.close()
    f_target.close()
    return res
    

def align_two_tomo_prj_mpi(fn_ref, fn_target, num_cpu=20):
    from multiprocessing import Pool, cpu_count
    from functools import partial
    res_ref = pyxas.get_img_from_hdf_file(fn_ref, 'angle', 'img_tomo', 'img_bkg_avg', 'img_dark_avg', 'scan_id', 'X_eng')
    fn_current = fn_ref.split('/')[:-1]
    if not len(fn_current):
        fn_current = '.'
    fn_current = '/'.join(tmp for tmp in fn_current)
    fn_short = fn_ref.split('/')[-1]
    fn_save = f'{fn_current}/ali_{fn_short}'

    ref_img = res_ref['img_tomo']
    ref_angle = res_ref['angle']

    res = pyxas.get_img_from_hdf_file(fn_target, 'angle', 'img_tomo', 'img_bkg_avg', 'img_dark_avg', 'scan_id', 'X_eng')
    img_bkg = res['img_bkg_avg']
    img_dark = res['img_dark_avg']
    target_angle = res['angle']
    fn_target_short = fn_target.split('/')[-1]
    fn_save = f'{fn_current}/ali_norm_{fn_target_short}'
    current_angle_index = np.arange(len(target_angle))

    time_s = time.time()    
    pool = Pool(num_cpu)
    res_mpi = pool.map(partial(align_tomo_prj_mpi_sub, fn_ref=fn_ref, fn_target=fn_target, img_dark=img_dark, img_bkg=img_bkg), current_angle_index)
    pool.close()
    print(f'aligning {fn_target_short} using {time.time() - time_s:4.1f} sec')

    img_ali_norm = np.zeros(res['img_tomo'].shape)
    r_shift = np.zeros(len(ref_angle))
    c_shift = r_shift.copy()
    
    for i in range(len(target_angle)):
        r_shift[i] = res_mpi[i]['r']
        c_shift[i] = res_mpi[i]['c']
        img_ali_norm[i] = res_mpi[i]['img_ali']
    pyxas.save_hdf_file(fn_save, 'angle', res['angle'], 'img_tomo', np.array(img_ali_norm, dtype=np.float32), 'scan_id', res['scan_id'], 'X_eng', res['X_eng'], 'r_shift', r_shift, 'c_shift', c_shift)
    


def align_tomo_prj_mpi(file_path, file_prefix='fly', file_type='.h5', ref_index=-1, num_cpu=8):
    files_scan = pyxas.retrieve_file_type(file_path, file_prefix, file_type)
    n = len(files_scan)
    if ref_index == -1:
        ref_index = len(files_scan)-1
    fn_ref = files_scan[ref_index]
    for i in range(n):
        fn_target = files_scan[i]
        if i == ref_index:
            continue
        print(f'aligning #{i}/{n}   {fn_ref.split("/")[-1]} ')
        align_two_tomo_prj_mpi(fn_ref, fn_target, num_cpu)



def batch_recon(file_path, file_prefix='fly', file_type='.h5', rot_cen=[], binning=1, sli=[], block_list=[], txm_normed_flag=0, read_full_memory=1):
    files_scan = pyxas.retrieve_file_type(file_path, file_prefix, file_type)
    n = len(files_scan)
    if not type(rot_cen) is list:
        rot_cen = [rot_cen] * n
    for i in range(n):
        time_s = time.time()
        fn = files_scan[i]
        print(f'recon {i+1}/{n}: {fn.split("/")[-1]}')
        recon(fn, rot_cen[i], sli=sli, binning=binning, txm_normed_flag=txm_normed_flag, block_list=block_list, read_full_memory=read_full_memory)
        print(f'************* take {time.time()-time_s:4.1f} sec ***************\n')











