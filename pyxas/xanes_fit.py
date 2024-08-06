from __future__ import (absolute_import, division, print_function, unicode_literals)

import pyxas

from pyxas.xanes_util import *
from pyxas.misc import *
from pyxas.colormix import *


def load_xanes_ref_file(*args):
    num_ref = len(args)
    spectrum_ref = {}
    for i in range(num_ref):
        spectrum_ref[f'ref{i}'] = np.loadtxt(args[i])
    return spectrum_ref


def check_if_need_align(img_xanes, fit_param):
    align_flag = fit_param['align_flag']
    roi_ratio = fit_param['roi_ratio']
    align_ref_index = fit_param['align_ref_index']
    img = img_xanes.copy()
    s = img.shape
    if roi_ratio >= 1:
        img_mask = None
    else:
        rs, re = 1 - roi_ratio, roi_ratio
        img_mask = img_xanes[:, int(s[1] * rs):int(s[1] * re), int(s[2] * rs):int(s[2] * re)]
    if align_ref_index == -1:
        align_ref_index = img.shape[0] - 1
    if align_flag == 1:  # using stackreg method
        print('aligning image stack ...')
        img = pyxas.align_img_stack_stackreg(img_xanes, img_mask, select_image_index=align_ref_index)
    elif align_flag == 2:
        print('aligning image stack ...')
        img = pyxas.align_img_stack(img_xanes, img_mask, select_image_index=align_ref_index)
    else:
        img = img_xanes
    if align_flag:
        for i in range(s[0]):
            t = img[i]
            t[t<=0] = np.median(t)
            img[i] = t
    return img


def fit_2D_xanes_using_param(img_xanes, xanes_eng, fit_param, spectrum_ref):
    res = {}
    img_thickness = None
    #img_xanes_norm = img_xanes.copy()
    try:
        sigma = fit_param['fit_iter_sigma']
    except:
        sigma = 0.05

    idx = np.argsort(xanes_eng)
    xanes_eng = xanes_eng[idx]
    img_xanes = img_xanes[idx]
    norm_txm_flag = fit_param['norm_txm_flag']
    pre_edge = fit_param['pre_edge']
    post_edge = fit_param['post_edge']

    learning_rate = fit_param['fit_iter_learning_rate']
    n_iter = int(fit_param['fit_iter_num'])
    bkg_polynomial_order = fit_param['fit_bkg_poly']

    b = fit_param['bin']
    if b > 1:
        s = img_xanes.shape
        ss = [s[0], s[1] // b * b, s[2] // b * b]
        img_xanes = img_xanes[:, :ss[1], :ss[2]]
        img_xanes = pyxas.bin_ndarray(img_xanes, (s[0], s[1] // b, s[2] // b), 'mean')
    '''
    regulation_flag = fit_param['regulation_flag']
    if regulation_flag:
        regulation_designed_max = fit_param['regulation_designed_max']
        regulation_gamma = fit_param['regulation_gamma']
    '''
    mask_xanes_flag = fit_param['mask_xanes_flag']

    n_comp = fit_param['n_comp'] if mask_xanes_flag else 1

    # optional: aligning xanes_image_stack
    img_xanes_norm = check_if_need_align(img_xanes, fit_param)

    # normalize by (-log)
    if norm_txm_flag:
        img_xanes_norm = pyxas.norm_txm(img_xanes_norm)

    # mask components using kmean_mask
    if mask_xanes_flag:
        n_comp = int(np.max([n_comp, 2]))
        # img_xanes_norm_smooth = pyxas.medfilt(img_xanes_norm, 3)
        mask_comp, _ = kmean_mask(img_xanes_norm, n_comp=n_comp)
    else:
        mask_comp = np.ones([1, img_xanes_norm.shape[1], img_xanes_norm.shape[2]])
        n_comp = 1

    # normalize edge
    norm_edge_flag = fit_param['norm_edge_flag']
    norm_edge_method = fit_param['norm_edge_method']
    xanes_eng, img_xanes_norm = pyxas.check_eng_image_order(xanes_eng, img_xanes_norm)
    if norm_edge_flag:
        if norm_edge_method == 'old':
            img_xanes_norm, img_thickness = pyxas.normalize_2D_xanes_old(img_xanes_norm, xanes_eng, pre_edge, post_edge,
                                                                     pre_edge_only_flag=0)
        else:
            img_xanes_norm, img_thickness = pyxas.normalize_2D_xanes2(img_xanes_norm, xanes_eng, pre_edge, post_edge,
                                                                      pre_edge_only_flag=0)
    else:
        img_thickness = None
    # regularize xanes image
    '''
    if regulation_flag:
        img_xanes_norm = pyxas.normalize_2D_xanes_regulation(img_xanes_norm, xanes_eng, pre_edge, post_edge,
                                                             regulation_designed_max, regulation_gamma)
    '''
    fit_method = fit_param['fit_method']
    fit_eng = fit_param['fit_eng']
    eng_s = pyxas.find_nearest(xanes_eng, fit_eng[0])
    eng_e = pyxas.find_nearest(xanes_eng, fit_eng[-1])
    if eng_e == len(xanes_eng) - 1:
        eng_e += 1
    tmp = np.ones(len(xanes_eng[eng_s:eng_e]))
    for i in range(len(spectrum_ref)):
        tmp = np.array(xanes_eng[eng_s: eng_e] >= spectrum_ref[f'ref{i}'][0, 0]) * np.array(
        xanes_eng[eng_s: eng_e] <= spectrum_ref[f'ref{i}'][-1, 0]) * tmp
    fit_eng_range = np.arange(eng_s, eng_e)[np.bool8(tmp)]
    # fitting
    if fit_method == 'basic':
        fit_coef, fit_cost, X, Y_hat, fit_offset, var, eng_interp, Y_interp = pyxas.fit_2D_xanes_basic(img_xanes_norm[fit_eng_range],
                                                                            xanes_eng[fit_eng_range],
                                                                            spectrum_ref,
                                                                            bkg_polynomial_order)
    # fit_method == 'admm'
    else:
        '''
        fit_coef, fit_cost, X, Y_hat, fit_offset, var, eng_interp, Y_interp = pyxas.fit_2D_xanes_admm(img_xanes_norm[fit_eng_range],
                                                                           xanes_eng[fit_eng_range],
                                                                           spectrum_ref,
                                                                           learning_rate,
                                                                           n_iter,
                                                                           bounds=[0, 1e10],
                                                                           bkg_polynomial_order=bkg_polynomial_order)
        '''
        method = 'nl'
        fit_coef, fit_cost, X, y_fit, fit_offset, var, x_interp, y_interp = pyxas.fit_2D_xanes_admm_denoise(img_xanes_norm[fit_eng_range],
                                                                                                      xanes_eng[fit_eng_range],
                                                                                                      spectrum_ref,
                                                                                                      learning_rate,
                                                                                                      n_iter,
                                                                                                      [0, 1e10],
                                                                                                      bkg_polynomial_order,
                                                                                                      method,
                                                                                                      sigma)
    fit_coef_sum = np.sum(fit_coef, axis=0, keepdims=True)
    fit_coef_norm = pyxas.rm_abnormal(fit_coef / fit_coef_sum)
    fit_coef_norm[fit_coef_norm > 1] = 1
    fit_coef_norm[fit_coef_norm < 0] = 0

    if img_thickness is None:
        img_thickness = fit_coef_sum

    res['xanes_fit_thickness'] = img_thickness
    res['xanes_2d_fit'] = fit_coef
    res['xanes_2d_fit_norm'] = fit_coef_norm
    res['xanes_fit_cost'] = fit_cost
    res['img_xanes_norm'] = img_xanes_norm
    res['xanes_2d_fit_offset'] = fit_offset[-1]
    res['fit_variance'] = var
    res['n_comp'] = n_comp
    for i in range(n_comp):
        res[f'mask_{i}'] = mask_comp[i]
    return res


def fit_2D_xanes_single_file(files_scan, xanes_eng, fit_param, spectrum_ref, file_save_path):

    time_start = time.time()
    files_scan_short = files_scan.split("/")[-1]
    print(f'fitting {files_scan_short} ...')
    num_channel = len(spectrum_ref)
    thresh_thick = fit_param['fit_mask_thickness_threshold'] # thickness < thick_thresh will be 0
    thresh_cost = fit_param['fit_mask_cost_threshold'] # fit_error > cost_thresh will be 0

    true_file_type = fit_param['file_type']
    if 'tif' in true_file_type:
        img_xanes = pyxas.get_img_from_tif_file(files_scan)
    elif 'h5' in true_file_type:
        hdf_attr = fit_param['hdf_attr']
        img_xanes = pyxas.get_img_from_hdf_file(files_scan, hdf_attr)[hdf_attr]
    else:
        raise Exception('Un-supported file type.')
    img_xanes[np.isnan(img_xanes)] = 0
    res = fit_2D_xanes_using_param(img_xanes, xanes_eng, fit_param, spectrum_ref)
    mask = pyxas.fit_xanes2D_generate_mask(res['xanes_fit_thickness'], res['xanes_fit_cost'], thresh_cost, thresh_thick)
    res['mask'] = mask
    color = fit_param['color'] if len(fit_param['color']) else 'r, g, b, c, y'

    save_xanes_fitting_image(res, file_save_path, files_scan, color)
    print(f'{files_scan_short} taking time: {time.time() - time_start:05.1f}\n')
    tk = list(res.keys())
    for k in tk:
        del res[k]
    del res, img_xanes, mask

    #return res


def fit_2D_xanes_series_file(files_scan, xanes_eng, fit_param, spectrum_ref, file_save_path):
    n = len(files_scan)
    #res = {}
    for i in trange(n):
        #res[i] = fit_2D_xanes_single_file(files_scan[i], xanes_eng, fit_param, spectrum_ref, file_save_path)
        fit_2D_xanes_single_file(files_scan[i], xanes_eng, fit_param, spectrum_ref, file_save_path)
    #return res


def compile_img_files(file_path, prefix, postfix):
    try:
        files = pyxas.retrieve_file_type(file_path, prefix, postfix)
        n = len(files)
        if n > 0:
            img = io.imread(files[0])
            s = img.shape
            if len(s) == 3:
                n_comp = s[0]
            else:
                n_comp = 1
            img_comb = np.zeros([n_comp, n, s[-2], s[-1]])
            for i in trange(n):
                fn = files[i]
                img = io.imread(fn)
                if n_comp > 1:
                    for j in range(n_comp):
                        img_comb[j, i] = img[j]
                else:
                    img_comb[0, i] = img
        else:
            print(f'no image exist in {file_path}')
    except Exception as err:
        img_comb = None
        print(err)
    return img_comb



def combine_xanes_fit_results(file_path_fit, sub_folder, mask_folder='', name=''):
    '''
    If mask_folder is exist, it will compile the mask:
    e.g., mask_folder='fitting_mask/mask_0'
    '''
    try:
        if len(mask_folder):
            file_path_mask = f'{file_path_fit}/{mask_folder}'
            mask = compile_img_files(file_path_mask, 'mask', 'tiff')
            mask = mask[0]
        else:
            mask = 1
    except Exception as err:
        print(f'Error in reading mask')
        raise Exception(str(err))

    try:
        file_path_coef = f'{file_path_fit}/{sub_folder}'
        fit_coef = compile_img_files(file_path_coef, 'fit', 'tiff')
        n_comp = fit_coef.shape[0]
        for j in range(n_comp):
            if len(name):
                fsave = f'{file_path_fit}/fit_coef_{name}_ref_{j}.tiff'
                fsave_masked = f'{file_path_fit}/fit_coef_{name}_ref_{j}_masked.tiff'
            else:
                fsave = f'{file_path_fit}/fit_coef_ref_{j}.tiff'
                fsave_masked = f'{file_path_fit}/fit_coef_ref_{j}_masked.tiff'
            io.imsave(fsave, fit_coef[j].astype(np.float32))
            io.imsave(fsave_masked, (fit_coef[j]*mask).astype(np.float32))
            print(fsave + ' saved')
            print(fsave_masked + ' saved')
    except Exception as err:
        raise Exception(str(err))


def save_xanes_fitting_image(res, file_save_path, fn, color='r,g,b'):
    file_save_colormix1 = f'{file_save_path}/colormix_concentration'
    file_save_colormix2 = f'{file_save_path}/colormix_ratio'
    file_save_fit_cost = f'{file_save_path}/fitting_cost'
    file_save_thickness = f'{file_save_path}/fitting_thickness'    
    file_save_mask = f'{file_save_path}/fitting_mask'
    file_save_fit = f'{file_save_path}/fitting_result'
    file_save_fit_norm = f'{file_save_path}/fitting_result_norm'
    file_save_comb = f'{file_save_path}/comb_jpg'


    create_directory(file_save_path)
    create_directory(file_save_colormix1)
    create_directory(file_save_colormix2)
    create_directory(file_save_fit_cost)
    create_directory(file_save_thickness)
    create_directory(file_save_mask)
    create_directory(file_save_fit)    
    create_directory(file_save_fit_norm)
    create_directory(file_save_comb)
    create_directory(f'{file_save_mask}/mask')   
    for n in range(res['n_comp']):
        create_directory(f'{file_save_mask}/mask_{n}')

    fn = fn.split('/')[-1].split('.')[0]
    sli_id = fn.split('_')[-1] # slice id: string
    try:
        sli_id = f'{int(sli_id):04d}'
    except:
        pass

    # save to jpg image
    tmp_concentration = res['xanes_2d_fit_norm'] * res['xanes_fit_thickness']
    tmp_ratio = res['xanes_2d_fit_norm']
    tmp_f_concentration = pyxas.medfilt(tmp_concentration, 3)
    #tmp_f_ratio = pyxas.medfilt(tmp_ratio, 3)
    img_color_concentration = pyxas.colormix(tmp_concentration, color=color, clim=[0, np.max(tmp_f_concentration)])
    img_color_ratio = pyxas.colormix(tmp_ratio, color=color, clim=[0, 1])
    # mask1 = np.expand_dims(np.squeeze(res['mask_0']), axis=2)
    mask1 = np.expand_dims(np.squeeze(res['mask']), axis=2)
    mask1 = np.repeat(mask1, 4, axis=2)
    mask2 = res['mask_0']
    mask2 = np.expand_dims(np.squeeze(mask2), axis=2)
    mask2 = np.repeat(mask2, 4, axis=2)

    img_color_concentration *= mask1 * mask2
    img_color_concentration[img_color_concentration<0] = 0
    if np.max(img_color_concentration) == 0:
        img_color_concentration[0,0,0] = 1
    fn_save = f'{file_save_colormix1}/colormix_concentration_{fn}.jpg'
    pyxas.toimage(img_color_concentration[:,:, :3], cmin=0, cmax=1).save(fn_save)

    img_color_ratio *= mask1 * mask2
    img_color_ratio[img_color_ratio<0] = 0
    if np.max(img_color_ratio) == 0:
        img_color_ratio[0,0,0] = 1
    fn_save = f'{file_save_colormix2}/colormix_ratio_{fn}.jpg'
    pyxas.toimage(img_color_ratio[:, :, :3], cmin=0, cmax=1).save(fn_save)

    # save fitting cost
    fn_save = f'{file_save_fit_cost}/fitting_cost_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_fit_cost'], dtype=np.float32))
    # save fitting thickness
    fn_save = f'{file_save_thickness}/fitting_thickness_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_fit_thickness'], dtype=np.float32))
    # save fitting mask
    fn_save = f'{file_save_mask}/mask/mask_{fn}.tiff'
    io.imsave(fn_save, np.array(np.squeeze(res['mask']), dtype=np.int16))
    for n in range(res['n_comp']):
        fn_save = f'{file_save_mask}/mask_{n}/mask{0}_{fn}.tiff'
        io.imsave(fn_save, np.array(res[f'mask_{n}'], dtype=np.int16))
    # save fitting results
    fn_save = f'{file_save_fit}/fit_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_2d_fit'], dtype=np.float32))
    fn_save = f'{file_save_fit_norm}/fit_norm_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_2d_fit_norm'], dtype=np.float32))
    # save combined RGB
    file_jpg_save = file_save_comb + f'/fig_{fn}.jpg'
    plot_fitting_results(res, color, file_jpg_save, display_flag=0, save_flag=1)
    del tmp_concentration, tmp_ratio, tmp_f_concentration
    del img_color_concentration, img_color_ratio, mask1, mask2

def fit_2D_xanes_file_mpi(file_path, file_prefix, fit_param, xanes_eng, spectrum_ref, file_range=[]):
    '''
    batch processing xanes given xanes files
    '''
    from multiprocessing import Pool, cpu_count
    from functools import partial
    num_cpu = fit_param['num_cpu']
    max_num_cpu = round(cpu_count() * 0.8)
    if num_cpu == 0 or num_cpu > max_num_cpu:
        num_cpu = max_num_cpu
    print(f'fiting xanes using {num_cpu:2d} CPUs')

    time_start = time.time()
    file_save_path = f'{file_path}/fitted_xanes'
    create_directory(file_save_path)

    file_type = fit_param['file_type']
    files_scan = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    try:
        true_file_type = fit_param['file_type']
    except:
        true_file_type = 'tiff'
    if true_file_type == 'tiff' or true_file_type == 'tif':
        tmp = pyxas.get_img_from_tif_file(files_scan[0])
    elif true_file_type == 'h5' or files_scan[0].split('.')[-1] == 'h5':
        hdf_attr = fit_param['hdf_attr']
        tmp = pyxas.get_img_from_hdf_file(files_scan[0], hdf_attr)[hdf_attr]
    else:
        raise Exception('un-supported file type')

    s = tmp.shape
    num_file = len(files_scan)
    fs, fe = 0, num_file
    if len(file_range):
        fs = int(np.max([0, np.min(file_range)]))
        fe = int(np.min([np.max(file_range), num_file]))
    files_scan = files_scan[fs:fe]
    num_file = len(files_scan)
    print(f'processing {files_scan[0].split("/")[-1]}\n ... to {files_scan[-1].split("/")[-1]}')

    num_channel = len(spectrum_ref)
    try:
        n_comp = int(fit_param['n_comp']) if fit_param['mask_xanes_flag'] else 1 # num of mask
    except:
        n_comp = 1

    if num_cpu == 1:
        fit_2D_xanes_series_file(files_scan, xanes_eng, fit_param, spectrum_ref, file_save_path)

    else:
        pool = Pool(num_cpu)
        #res = pool.map(partial(pyxas.fit_2D_xanes_file_mpi_sub, fit_param=fit_param, xanes_eng=xanes_eng, spectrum_ref=spectrum_ref, file_save_path=file_save_path), files_scan)
        pool.map(
            partial(fit_2D_xanes_single_file, xanes_eng=xanes_eng,
                    fit_param=fit_param,  spectrum_ref=spectrum_ref,
                    file_save_path=file_save_path), files_scan)
        #pool.join()
        pool.close()
     # group fitting coef.
    print('group fitting coefficients seperately ... ')
    combine_xanes_fit_results(file_save_path, 'fitting_result',
                              mask_folder='fitting_mask/mask_0', name='', )
    combine_xanes_fit_results(file_save_path, 'fitting_result_norm',
                              mask_folder='fitting_mask/mask_0', name='norm')
    print(f'total time elapsed: {time.time()-time_start:6.1f}')


def plot_fitting_results(res, color='', file_save_path='', display_flag=1, save_flag=0):
    '''
    res = {'xanes_fit_thickness', 'xanes_2d_fit', 'xanes_2d_fit_norm',
    'xanes_fit_cost', 'img_xanes_norm', 'xanes_2d_fit_offset', 'n_comp',
    'mask_0', 'mask'}
    '''

    fit_img_norm = res['xanes_2d_fit_norm']
    thickness = np.squeeze(res['xanes_fit_thickness'])

    mask_thresh = np.squeeze(res['mask'])
    mask_cluster = np.squeeze(res[f'mask_0'])
    mask = mask_thresh * mask_cluster

    color = convert_color_string(color)
    color_vec = convert_rgb_vector(color)
    img_colormix = convert_rgb_img(fit_img_norm*mask, color_vec)
    newcmp = create_binary_color_cmp(color)
    n = fit_img_norm.shape[0]
    plt.figure(figsize=(12, 9))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(fit_img_norm[i]*mask, cmap='Spectral_r', clim=[0,1])
        plt.colorbar()
        plt.axis('off')
        plt.title(f'normalized coef {i}')
    plt.subplot(2, n, n+1)
    plt.imshow(img_colormix, cmap=newcmp)
    plt.colorbar()
    plt.axis('off')
    plt.title(f'colormix using {color[:n]}')
    plt.subplot(2, n, n+2)
    plt.imshow(thickness, cmap='bone')
    plt.axis('off')
    plt.title('thickness')
    plt.colorbar()
    if display_flag:
        plt.show()
    if save_flag:
        if not len(file_save_path):
            file_save_path = 'fig.jpg'
        plt.savefig(file_save_path)
        print(f'{file_save_path} saved')
        plt.suptitle(f'{file_save_path} saved')


def fit_xanes2D_generate_mask(img_thickness, xanes_fit_cost, thresh_cost=0.1, thresh_thick=0):
    mask1 = np.squeeze((xanes_fit_cost < thresh_cost).astype(int))
    mask2 = np.squeeze((img_thickness > thresh_thick).astype(int))
    mask = mask1 * mask2
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, 2).astype(int)
    mask = ndimage.binary_fill_holes(mask, structure=struct).astype(mask.dtype)
    mask = pyxas.medfilt(mask, 5)
    mask = mask.reshape([1, mask.shape[0], mask.shape[1]])
    return mask


def assemble_xanes_slice_from_tomo_mpi_sub(sli, file_path, files_recon, attr_img='img', attr_eng='X_eng', align_flag=0,
                                           align_ref_index=-1, align_roi_ratio=0.8, roi=[], ali_sli=[],
                                           align_algorithm='stackreg', align_method='translation',
                                           flag_save_2d_xanes=1, flag_mask=1):
    from PIL import Image
    time_s = time.time()
    num_file = len(files_recon)

    file_type = files_recon[0].split('.')[-1]
    if 'tif' in file_type:
        img_tmp = io.imread(files_recon[0])
        s = img_tmp.shape
        num_slice = s[0]
        s = (s[1], s[2])
    elif 'h5' in file_type:
        f = h5py.File(files_recon[0], 'r')
        num_slice = len(f[attr_img])
        s = np.array(list(f[attr_img][0])).shape
        f.close()
    else:
        print('un-recognized file type')
        return 0

    img_xanes = np.zeros([num_file, s[0], s[1]])
    xanes_eng = np.zeros(num_file)
    mask = np.ones([s[0], s[1]])
    res = {}    
    print(f'processing slice: {sli}')
    for j in trange(num_file):
        fn = files_recon[j]
        if 'tif' in file_type:
            #tmp = io.imread(fn)[sli]
            dataset = Image.open(fn)
            dataset.seek(sli)
            tmp = np.array(dataset)
            tmp_eng = 0
        elif 'h5' in file_type:
            f = h5py.File(fn, 'r')
            tmp = np.array(f[attr_img][sli])
            try:
                tmp_eng = np.array(f[attr_eng])
            except:
                tmp_eng = 0
            f.close()
        else:
            raise Exception('un-supported file type')
        img_xanes[j] = tmp
        xanes_eng[j] = tmp_eng
    # align xanes stack
    if align_flag:
        if len(ali_sli) == 0:
            ali_sli = [0, num_slice-1]        
        ali_sli = np.array(ali_sli)
        if sli >= np.min(ali_sli) and sli <= np.max(ali_sli):
            if align_roi_ratio >= 1:
                img_mask = img_xanes
            else:
                rs, re = 1-align_roi_ratio, align_roi_ratio
                img_mask = img_xanes[:, int(s[0]*rs):int(s[0]*re), int(s[1]*rs):int(s[1]*re)]
            if len(roi) == 4:
                roi_rs = max(roi[0], 0)
                roi_re = min(roi[1], s[0])
                roi_cs = max(roi[2], 0)
                roi_ce = min(roi[3], s[1])
                img_mask = img_mask[:, roi_rs:roi_re, roi_cs:roi_ce]
            if align_ref_index == -1:
                align_ref_index = img_xanes.shape[0] - 1
            if align_algorithm == 'stackreg':
                img_xanes = pyxas.align_img_stack_stackreg(img_xanes, img_mask, select_image_index=align_ref_index,
                                                           print_flag=0, method=align_method)
            else:
                img_xanes = pyxas.align_img_stack(img_xanes, img_mask, select_image_index=align_ref_index, print_flag=0)
            # apply mask        
            if flag_mask:
                try:
                    mask = kmean_mask(img_xanes)
                except:
                    pass
    if flag_save_2d_xanes:
        io.imsave(f'{file_path}/xanes_assemble/xanes_2D_slice_{sli:04d}.tiff', np.array(img_xanes, dtype=np.float32))
        print(f'xanes_2D_slice_{sli:04d}.tiff saved, using time: {time.time()-time_s:4.1f}sec\n')
    res['mask'] = mask
    return res


def assemble_xanes_slice_from_tomo_mpi(file_path='.', file_prefix='ali_recon', file_type='.h5', attr_img='img',
                                       attr_eng='X_eng', sli=[], align_flag=0, align_ref_index=-1, align_roi_ratio=0.8,
                                       roi=[], ali_sli=[], align_algorithm='stackreg', align_method='translation',
                                       flag_save_2d_xanes=1,flag_mask=1, num_cpu=0):

    # if flag_mask = 1, it will calculate the mask from img_xanes

    from multiprocessing import Pool, cpu_count
    from functools import partial
    max_num_cpu = round(cpu_count() * 0.8)
    if not num_cpu or num_cpu > max_num_cpu:
        num_cpu = round(cpu_count() * 0.8)
    print(f'assembling slice using {num_cpu:2d} CPUs')

    file_path = os.path.abspath(file_path)
    files_recon = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    num_file = len(files_recon)
        
    xanes_assemble_dir = f'{file_path}/xanes_assemble'
    if not os.path.exists(xanes_assemble_dir):
        os.makedirs(xanes_assemble_dir)

    if 'tif' in file_type:
        img_tmp = io.imread(files_recon[0])
        s = img_tmp[0].shape
        num_slice = len(img_tmp)
    elif 'h5' in file_type:
        f = h5py.File(files_recon[0], 'r')
        num_slice = len(f['img'])
        s = np.array(f['img'][0]).shape
        f.close()
    else:
        print('unrecognized file type')
        return 0

    if len(sli) == 0:
        sli=[0, num_slice]
    if len(sli) == 1:
        sli = [sli[0], sli[0]+1]
    sli = np.arange(sli[0], sli[1])

    mask_3D = np.ones([num_slice, s[0], s[1]])
    time_s = time.time()
    if num_cpu == 1:
        for i in trange(len(sli)):
            res = assemble_xanes_slice_from_tomo_mpi_sub(sli[i], file_path, files_recon, attr_img=attr_img, attr_eng=attr_eng,
                                                   align_flag=align_flag, align_ref_index=align_ref_index, align_roi_ratio=align_roi_ratio,
                                                   roi=roi, ali_sli=ali_sli, align_algorithm=align_algorithm, align_method='translation',
                                                   flag_save_2d_xanes=flag_save_2d_xanes, flag_mask=flag_mask)
            mask_3D[i] = res['mask']
    else:
        pool = Pool(num_cpu)
        res = pool.map(partial(pyxas.assemble_xanes_slice_from_tomo_mpi_sub,
                               file_path=file_path, files_recon=files_recon,
                               attr_img=attr_img, attr_eng=attr_eng, align_flag=align_flag,
                               align_ref_index=align_ref_index, align_roi_ratio=align_roi_ratio,
                               roi=roi, ali_sli=ali_sli, align_algorithm=align_algorithm, align_method='translation',
                               flag_save_2d_xanes=flag_save_2d_xanes, flag_mask=flag_mask), sli)
        pool.join()
        pool.close()
        for i in range(len(files_recon)):
            mask_3D[i] = res[i]['mask']

    io.imsave(f'{xanes_assemble_dir}/mask3D.tiff', np.array(mask_3D, dtype=np.float32))
    print(f'total time for assembling xanes is: {time.time()-time_s:4.1f}')


def assemble_xanes_slice_from_tomo(file_path='.', file_prefix='ali_recon', file_type='.h5', attr_img='img', attr_eng='XEng', sli=[], align_flag=0, align_ref_index=-1, align_roi_ratio=0.8, ali_sli=[], align_algorithm='stackreg', flag_save_2d_xanes=1, flag_mask=0, return_flag=0):
#    Re-assemble the "ALIGNED" 3D xanes tomography into 2D xanes for each slices

    file_path = os.path.abspath(file_path)
    files_recon = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    num_file = len(files_recon)

    if 'tif' in file_type:
        img_tmp = io.imread(files_recon[0])
        s = img_tmp[0].shape
        num_slice = len(img_tmp)
    elif 'h5' in file_type:
        f = h5py.File(files_recon[0], 'r')
        num_slice = len(f['img'])
        s = np.array(f['img'][0]).shape
        f.close()
    else:
        print('unrecognized file type')
        return 0

    xanes_assemble_dir = f'{file_path}/xanes_assemble'
    if not os.path.exists(xanes_assemble_dir):
        os.makedirs(xanes_assemble_dir)    
    if len(sli) == 0:
        sli=[0, num_slice]
    if len(sli) == 1:
        sli = [sli[0], sli[0]+1]
    sli = np.arange(sli[0], sli[1])
    img_xanes = np.zeros([num_file, s[0], s[1]])
    mask3D = np.ones([num_slice, s[0], s[1]])
    xanes_eng = np.zeros(num_file)
    for i in range(sli[0], sli[1]):
        print(f'processing slice: {i}/({sli[0]}: {sli[-1]})')
        
        for j in range(num_file):
            if 'tif' in file_type:
                tmp = io.imread(files_recon[j])[i] 
                tmp_eng = 0
            else:                
                f = h5py.File(files_recon[j], 'r')
                tmp = np.array(f[attr_img][i])
                try:
                    tmp_eng = np.array(f[attr_eng])
                except:
                    tmp_eng = 0
                f.close()
            img_xanes[j] = tmp
            xanes_eng[j] = tmp_eng
        # align xanes stack
        if align_flag:            
            s = img_xanes.shape
            if len(ali_sli) == 0:
                ali_sli = [0, num_slice-1]
            ali_sli = np.array(ali_sli)
            if i >= np.min(ali_sli) and i <= np.max(ali_sli):
                if align_roi_ratio >= 1:
                    img_mask = None
                else:
                    rs, re = 1-align_roi_ratio, align_roi_ratio
                    img_mask = img_xanes[:, int(s[1]*rs):int(s[1]*re), int(s[2]*rs):int(s[2]*re)]

                if align_ref_index == -1:
                    align_ref_index = img_xanes.shape[0] - 1
                if align_algorithm == 'stackreg':
                    img_xanes = pyxas.align_img_stack_stackreg(img_xanes, img_mask, select_image_index=align_ref_index) 
                else:
                    img_xanes = pyxas.align_img_stack(img_xanes, img_mask, select_image_index=align_ref_index) 
                # apply mask        
                if flag_mask:
                    try:
                        mask = kmean_mask(img_xanes)
                    #img_xanes *= mask
                        mask3D[i] = mask
                    except:
                        pass
        if flag_save_2d_xanes:
            io.imsave(f'{file_path}/xanes_assemble/xanes_2D_slice_{i:04d}.tiff', np.array(img_xanes, dtype=np.float32))
    io.imsave(f'{file_path}/xanes_assemble/mask3D.tiff', np.array(mask3D, dtype=np.float32))
    if return_flag:
        return img_xanes, xanes_eng


def assemble_xanes_from_files(files_recon, sli=[], align_flag=0, align_ref_index=-1, align_roi_ratio=0.8, ali_sli=[], align_algorithm='stackreg', flag_save_2d_xanes=1, flag_mask=0, return_flag=0, attr_img='img', attr_eng='X_eng'):
    num_file = len(files_recon)    
    file_type = files_recon[0].split('.')[-1]
    if 'tif' in file_type:
        img_tmp = io.imread(files_recon[0])
        s = img_tmp[0].shape
        num_slice = s[0]
    elif 'h5' in file_type:
        f = h5py.File(files_recon[0], 'r')
        num_slice = len(f['img'])
        s = np.array(f['img'][0]).shape
        f.close()
    else:
        print('unrecognized file type')
        return 0

    file_path = os.getcwd()
    xanes_assemble_dir = f'{file_path}/xanes_assemble'
    if not os.path.exists(xanes_assemble_dir):
        os.makedirs(xanes_assemble_dir)    
    if len(sli) == 0:
        sli=[0, num_slice]
    if len(sli) == 1:
        sli = [sli[0], sli[0]+1]
        xanes_assemble_dir = f'{file_path}/xanes_assemble'

    sli = np.arange(sli[0], sli[1])
    img_xanes = np.zeros([num_file, s[0], s[1]])
    mask3D = np.ones([num_slice, s[0], s[1]])
    xanes_eng = np.zeros(num_file)
    for i in sli:
        print(f'processing slice: {i}/({sli[0]}: {sli[-1]})')
        
        for j in range(num_file):
            if 'tif' in file_type:
                tmp = io.imread(files_recon[j])[i] 
                tmp_eng = 0
            else:                
                f = h5py.File(files_recon[j], 'r')
                tmp = np.array(f[attr_img][i])
                try:
                    tmp_eng = np.array(f[attr_eng])
                except:
                    tmp_eng = 0
                f.close()
            img_xanes[j] = tmp
            xanes_eng[j] = tmp_eng
        # align xanes stack
        if align_flag:            
            s = img_xanes.shape
            if len(ali_sli) == 0:
                ali_sli = [0, num_slice-1]
            ali_sli = np.array(ali_sli)
            if i >= np.min(ali_sli) and i <= np.max(ali_sli):
                if align_roi_ratio >= 1:
                    img_mask = None
                else:
                    rs, re = 1-align_roi_ratio, align_roi_ratio
                    img_mask = img_xanes[:, int(s[1]*rs):int(s[1]*re), int(s[2]*rs):int(s[2]*re)]
                if align_ref_index == -1:
                    align_ref_index = img_xanes.shape[0] - 1
                if align_algorithm == 'stackreg':
                    img_xanes = pyxas.align_img_stack_stackreg(img_xanes, img_mask, select_image_index=align_ref_index) 
                else:
                    img_xanes = pyxas.align_img_stack(img_xanes, img_mask, select_image_index=align_ref_index) 
                # apply mask        
                if flag_mask:
                    try:
                        mask = kmean_mask(img_xanes)
                    #img_xanes *= mask
                        mask3D[i] = mask
                    except:
                        pass
        if flag_save_2d_xanes:
            io.imsave(f'{file_path}/xanes_assemble/xanes_2D_slice_{i:03d}.tiff', np.array(img_xanes, dtype=np.float32))
    io.imsave(f'{file_path}/xanes_assemble/mask3D.tiff', np.array(mask3D, dtype=np.float32))
    if return_flag:
        return img_xanes, xanes_eng


def fit_2D_multi_elem_thick(img_xanes, xanes_eng, elem, eng_exclude,
                            bkg_polynomial_order, method, admm_iter, admm_rate, admm_sigma=0.1):
    s = img_xanes.shape
    n_elem = len(elem)
    X, A, A_all, x_eng, Y, Y_fit, Y_diff, x_eng_all, Y_all, Y_fit_all = fit_multi_element_mu(img_xanes,
                                                                                             xanes_eng,
                                                                                             elem,
                                                                                             eng_exclude,
                                                                                             bkg_polynomial_order,
                                                                                             method,
                                                                                             admm_iter,
                                                                                             admm_rate,
                                                                                             admm_sigma)

    thickness = X[:n_elem].reshape((n_elem, s[1], s[2]))
    y_diff = np.sum(Y_diff, axis=0).reshape((s[1], s[2]))
    y_fit_err = np.abs(y_diff)
    res = {}
    res['X'] = X
    res['A'] = A
    res['A_all'] = A_all
    res['x_eng'] = x_eng
    res['Y'] = Y
    res['Y_fit'] = Y_fit
    res['x_eng_all'] = x_eng_all
    res['Y_all'] = Y_all
    res['Y_fit_all'] = Y_fit_all

    res['y_diff_sum'] = y_diff
    res['thickness'] = thickness
    res['y_fit_err'] = y_fit_err
    return res
    #return thickness, y_fit_err

