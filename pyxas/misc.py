from __future__ import (absolute_import, division, print_function, unicode_literals)

import pyxas
import os
import numpy as np
import h5py
import time
from pyxas.image_util import *
from scipy import ndimage
from pystackreg import StackReg
from skimage import io


def create_directory(fn):
    try:
        if not os.path.exists(fn):
            os.makedirs(fn)
            print(f'creat {fn}')
        #else:
            #print(f'{fn} exists')
    except:
        print(f'fails to create {fn}')


def save_xanes_fit_param_file(fit_param, fn='xanes_fit_param.csv'):
    import csv
    if len(fn) == 0:
        fn = 'xanes_fit_param.csv'

    w = csv.writer(open(fn, 'w'))
    for key, val in fit_param.items():
        w.writerow([key, val])
    print(f'{fn} is saved\n')



def string_to_list(string):
    if string == '[]':
        b = []
    elif string[0] == '[' and string[-1] == ']':
        a = string[1:-1].split(',')
        b = [float(a[0]), float(a[1])]
    else:
        try:
            b = np.float32(string)
            if b - np.round(b) == 0:
                b = int(b)
        except:
            b = string
    return b

        
   

def load_xanes_fit_param_file(fn='xanes_fit_param.csv', num_items=0):
    import csv
    fit_param = {}
    items = "\n'pre_edge'\n'post_edge'\n'fit_eng'\n'norm_txm_flag'\n'fit_pre_edge_flag'\n'fit_post_edge_flag'\n'align_flag'\n'align_ref_index' \n'align_flag'\n'align_ref_index'\n'roi_ratio'\n'fit_iter_flag'\n'fit_iter_learning_rate'\n'fit_iter_num'\n'fit_iter_bound'\n'regulation_flag'\n'regulation_designed_max'\n'regulation_gamma'\n'fit_mask_thickness_threshold'\n'fit_mask_cost_threshold'"
    with open(fn, 'r') as csvfile:
        f = csv.reader(csvfile)
        for row in f:
            fit_param[row[0]] = pyxas.string_to_list(row[1])
    if num_items:
        assert len(fit_param) == num_items, print(f'missing items in fit_param, which should consist of: {items}')
    return fit_param  



def load_xanes_image_file(fn_image, fn_eng, h5_attri_xanes='img_xanes', h5_attri_eng='XEng'):

    #fn = '/home/mingyuan/Work/xanes_fit/demo/Ni_4.4V_xanes_6679_aligned.tiff'
    #fn_eng = '/home/mingyuan/Work/xanes_fit/demo/Ni_eng_scan_6679.txt'

    try:
        if fn_image[-5:] == '.tiff' or fn_image[-4:] == '.tif':
            img_xanes = np.array(io.imread(fn_image))
            xanes_eng = np.loadtxt(fn_eng)
        elif fn_image[-3:] == '.h5':
            f = h5py.File(fn_image, 'r')
            img_xanes = np.array(f[h5_attri_xanes])
            try:
                xanes_eng = np.array(f[h5_attri_eng])
            except:
                xanes_eng = np.loadtxt(fn_eng)
            finally:
                f.close()
        else:
            img_xanes, xanes_eng = [], []
    except:
        img_xanes, xanes_eng = [], []
    finally:
        return img_xanes, xanes_eng



def retrieve_file_type(file_path='/home/mingyuan/Work/scan8538_8593', file_prefix='fly', file_type='.h5'):
    import os
    path = os.path.abspath(file_path)
    files = sorted(os.listdir(file_path))
    files_filted = []   
    n_type = len(file_type)
    n_start = len(file_prefix)
    for f in files:
        if f[-n_type:] == file_type and f[:n_start] == file_prefix:
            f = f'{path}/{f}' 
            files_filted.append(f)
    return files_filted


def retrieve_norm_tomo_image(fn, index=[0], binning=1, sli=[]):
    f = h5py.File(fn, 'r')
    eng_tomo = float(np.array(f['X_eng']))
    theta = np.array(f['angle'])
    if index == -1:
        index = range(len(theta))
    tmp_img = np.array(f['img_tomo'][index])
    if len(tmp_img.shape) == 2:
        tmp_img.reshape([1, tmp_img.shape[0], tmp_img.shape[1]])
    tmp_bkg = np.array(f['img_bkg_avg'])
    tmp_dark = np.array(f['img_dark_avg'])
    img_tomo = (tmp_img - tmp_dark) / (tmp_bkg - tmp_dark)
    img_tomo[np.isnan(img_tomo)] = 0
    img_tomo[np.isinf(img_tomo)] = 0  
    scan_id = int(np.array(f['scan_id']))
    f.close()
    s0 = img_tomo.shape
    if len(sli) == 0:
        sli = [0, s0[1]]
    img_tomo = img_tomo[:, sli[0]:sli[1], :]
    try:
        s = img_tomo.shape
        img_tomo = bin_ndarray(img_tomo, new_shape=(s[0], int(s[1]/binning), int(s[2]/binning)))
    except:
        print(f'binning {binning} fails, will return unbinned image')
        pass
    return img_tomo, eng_tomo, theta, scan_id



def retrieve_image_from_file(file_path='/home/mingyuan/Work/scan8538_8593/align', file_prefix='recon', file_type='.h5', index=0):
    files = retrieve_file_type(file_path, file_prefix, file_type)        
    img = []
    for fn in files:
        f = h5py.File(fn,'r')
        tmp = np.squeeze(np.array(f['img'][index]))
        img.append(tmp)
    img = np.array(img)
    return img


def get_1st_image_from_tomo_scan(file_path='/home/mingyuan/Work/scan8538_8593'):
    files = retrieve_file_type(file_path, file_prefix='fly', file_type='.h5')
    index = 0
    eng = []
    img0 = []
    theta =[]
    for fn in files:
        if fn[-3:] == '.h5':
            print(f'reading file {fn}')
            img_tomo, eng_tomo, theta_norm = retrieve_norm_tomo_image(fn, index=0)
            img0.append(img_tomo)
            eng.append(eng_tomo)
            theta.append(theta_norm)
    img0 = np.squeeze(np.array(img0))
    return img0, eng, theta


def retrieve_shift_list(fn_ref, fn_need_to_align, region=0.8, save_flag=1):
    img_ref = io.imread(fn_ref)
    img_need = io.imread(fn_need_to_align)
    
    img_ref = -np.log(img_ref)
    img_ref[np.isnan(img_ref)] = 0
    img_ref[np.isinf(img_ref)] = 0
    
    img_need = -np.log(img_need)
    img_need[np.isnan(img_need)] = 0
    img_need[np.isinf(img_need)] = 0

    s = img_ref.shape
    xs = int((1 - region)/2 * s[1])
    xe = int(xs + s[1] * region)
    img_ref = img_ref[:, xs:xe, xs:xe]
    img_need = img_need[:, xs:xe, xs:xe]
    row_shift = []
    col_shift = []
    img_tmp = []
    num = len(img_ref)
    for i in range(num):
        print(f'processing {i}')
        tmp, r, c = align_img(img_ref[i], img_need[i])
        img_tmp.append(tmp)
        row_shift.append(r)
        col_shift.append(c)
    img_tmp = np.array(img_tmp)
    if save_flag:
        io.imsave('check_align.tiff', img_tmp.astype(np.float32))
        print('check_align.tiff has been saved')
    return row_shift, col_shift


def get_eng_from_file(file_path, file_prefix='fly', file_type='.h5'):
    import pandas as pd
    files = pyxas.retrieve_file_type(file_path, file_prefix, file_type)
    eng_list = []
    scan_id_list =[]
    for fn in files:
        f = h5py.File(fn, 'r')
        eng_list.append(float(np.array(f['X_eng'])))
        scan_id_list.append(float(np.array(f['scan_id'])))
        f.close()
    res = {'scan_id_list': scan_id_list, 'eng_list':eng_list}
    df = pd.DataFrame(res)
    return df



def get_img_from_tif_file(fn):
    from skimage import io
    if fn[-4:] == 'tiff' or fn[-3:] == 'tif':
        img = io.imread(fn)
        s = img.shape
        if s[2] == 3 or s[2] == 4:
            img = np.transpose(img, (2, 0, 1))
    return img



def get_img_from_hdf_file(fn, *attr_args):
    f = h5py.File(fn, 'r')
    n = len(attr_args)
    res = {}
    for i in range(n):
        try:
            res[f'{attr_args[i]}'] = np.array(f[attr_args[i]])
        except:
            res[f'{attr_args[i]}'] = str([attr_args[i]])
    f.close()
    return res


def save_hdf_file(fn, *args):
    n = len(args)
    assert n%2 == 0, 'even number of args only'
    n = int(n/2)
    j = 0
    with h5py.File(fn, 'w') as hf:
        for i in range(n):
            j = int(2*i)
            tmp = args[j+1]
            hf.create_dataset(args[j], data=tmp)


def bin_image(img, binning):
    s = img.shape
    n = len(s)
    new_shape = [int(i/binning) for i in s ]
    img_bin = pyxas.bin_ndarray(img, new_shape = new_shape)
    return img_bin



def retrieve_h5_xanes_energy(file_path='.', file_prefix='recon', file_type='.h5', attr_eng='X_eng'):    
    file_path = os.path.abspath(file_path)
    files_list = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)

    num_file = len(files_list)
    eng_list = np.zeros(num_file)
    for i in range(num_file):
        f = h5py.File(files_list[i], 'r')
        eng_list[i] = float(np.array(f[attr_eng]))
    np.savetxt('eng_list.txt', eng_list)
    return eng_list



def rgb(img_r, img_g=[], img_b=[], norm_flag=1, filter_size=3, circle_mask_ratio=0.8):
    '''
    compose RGB image
    '''
    from scipy.signal import medfilt
    assert len(img_r.shape) == 2, '2D image only'
    s = img_r.shape
    if len(img_g) == 0:
        img_g = np.zeros(s)
    if len(img_b) == 0:
        img_b = np.zeros(s)
    if filter_size >= 2:
        r = medfilt(img_r, int(filter_size))
        g = medfilt(img_g, int(filter_size))
        b = medfilt(img_b, int(filter_size))
    img_rgb = np.zeros([s[0], s[1], 3])
    img_rgb[:,:,0] = r
    img_rgb[:,:,1] = g
    img_rgb[:,:,2] = b
    img_rgb = pyxas.circ_mask(img_rgb, axis=2, ratio=circle_mask_ratio)
    if norm_flag:
        for i in range(3):
            t = np.max(img_rgb[:,:,i])
            img_rgb[:,:,i] /= t
    plt.figure()
    plt.imshow(img_rgb) 

    


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        self._indx_txt = ax.set_title(' ', loc='center')
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        #self.ax.set_ylabel('slice %s' % self.ind)
        self._indx_txt.set_text(f"frame {self.ind + 1} of {self.slices}")
        self.im.axes.figure.canvas.draw()


def image_scrubber(data, *, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    tracker = IndexTracker(ax, data)
    # monkey patch the tracker onto the figure to keep it alive
    fig._tracker = tracker
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return tracker




   
#####  obsolete function ###

def fit_xanes2D_align_tomo_proj(file_path='.', files_scan=[], binning=2, ref_index=-1, ref_rot_cen=-1, block_list=[], sli=[], ratio=0.8, file_prefix='fly', file_type='.h5'):
    '''
    Aligning the tomo-scan projections with assigned scan file, and generate 3D reconstruction.

    Inputs:
    -----------
    file_path: str
        Directory contains all "fly_scans"
    binning: int
        binning of reconstruction
    ref_index: int
        index of "fly_scans" which is assigned as reference projections
        this fly_scan should has has good reconstruction quality and fare full list of rotation angles
        if -1: use the last scan (sorted by alphabetic file name)
    ref_rot_cen: float
        rotation center for the referenced "fly_scan"
        if -1: find rotation center using cross-corelationship at angle-0 and angle-180  
    block_list: list
        indexes of bad projection
        e.g., list(np.arange(380,550)
    ratio: float: (0 < ratio < 1)
        faction of projection image to be use to align projections
        e.g., 0.6        
    file_prefix: str
        prefix of the "fly_scan"
        e.g., 'fly'
    file_type: str
        e.g. '.h5'
    
    Output:
    ----------------
    None, will save aligned 3D reconstruction in folder of "{file_path}/ali_recon"
    '''
    file_path = os.path.abspath(file_path)
    binning = int(binning)
    if len(files_scan) == 0:
        files_scan = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    num_files = len(files_scan)
    #    block_list=list(np.arange(380,550))    
    fn_ref = files_scan[ref_index]
    f_ref = h5py.File(fn_ref, 'r')
    img_ref, _, angle_ref = pyxas.retrieve_norm_tomo_image(fn_ref, index=ref_index, binning=binning)
    theta_ref = angle_ref / 180 * np.pi
    f_ref.close()
    if ref_rot_cen == -1:
        rot_cen = find_rot(fn_ref) / binning
    else:
        rot_cen = ref_rot_cen / binning
    sr = StackReg(StackReg.TRANSLATION)

    new_dir = f'{file_path}/ali_recon'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for i in range(num_files):
        fn = files_scan[i]
        f1 = h5py.File(fn, 'r')
        scan_id = np.array(f1['scan_id'])
        angle1 = np.array(f1['angle'])
        theta1 = angle1 / 180 * np.pi
        f1.close()
        num_angle = len(angle1)
        s = time.time()
        img1_ali, eng1, rshft, cshft = pyxas.align_proj_sub(fn_ref, angle_ref, fn, angle1, binning, ratio=ratio, sli=sli, ali_method='stackreg')
        print(f'#{i}/{num_files}, time elapsed: {time.time()-s}\n')

        img1_ali = pyxas.norm_txm(img1_ali)
        rec = pyxas.recon_sub(img1_ali, theta1, rot_cen, block_list)
        rec = pyxas.circ_mask(rec, axis=0, ratio=ratio, val=0)
        print('saving files...\n')
        
        
        fn_save = f'{new_dir}/ali_recon_{scan_id}.h5'
        with h5py.File(fn_save, 'w') as hf:
            hf.create_dataset('img', data = rec.astype(np.float32))
            hf.create_dataset('scan_id', data = scan_id)
            hf.create_dataset('XEng', data = eng1)
            hf.create_dataset('angle', data = angle1)
        print(f'{fn_save} saved\n')




def fit_xanes2D_align_tomo_recon_along_axis(file_path='.', ref_index=-1, circle_mask_ratio=0.6, file_prefix='ali', file_type='.h5', axes=[0]):
    '''
    Aligning the reconstructed tomo with assigned 3D reconstruction along given axis. It will project the 3D data along given axis to find the shifts

    Inputs:
    -----------
    file_path: str
        Directory contains all "fly_scans"
    binning: int
        binning of reconstruction
    ref_index: int
        index of "fly_scans" which is assigned as reference projections
        this fly_scan should has has good reconstruction quality and fare full list of rotation angles
        if -1: use the last scan (sorted by alphabetic file name)
    file_prefix: str
        prefix of the "fly_scan"
        e.g., 'fly'
    file_type: str
        e.g. '.h5'
    axis: int
        along which axis to project the 3D reconstruction to find image shifts 
        0, or 1, or 2
    
    Output:
    ----------------
    None, will save aligned 3D reconstruction in folder of "{file_path}/align_axis_{axis}"
    '''
    file_path = os.path.abspath(file_path)
    files_recon = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    f = h5py.File(files_recon[ref_index], 'r')
    rec0 = np.array(f['img'])
    #rec0 = pyxas.circ_mask(rec0, axis=0, ratio=0.8, val=0)    
    f.close()
    s = rec0.shape
    stack_range = [int(s[0]*(0.5-ratio/2)), int(s[0]*(0.5+ratio/2))]
    #prj0 = np.sum(rec0[stack_range[0]:stack_range[1]], axis=axis)
    num_files = len(files_recon)    
    sr = StackReg(StackReg.TRANSLATION)
    for j in range(len(files_recon)):
        fn = files_recon[j]
        f = h5py.File(fn, 'r')
        rec1 = np.array(f['img'])
        scan_id = np.array(f['scan_id'])
        eng1 = np.array(f['XEng'])
        angle1 = np.array(f['angle'])
        f.close()
        tmp = rec1.copy() 
        for ax in axes:
            tmp, shift_matrix = align_3D_cause_axes(rec0, tmp, circle_mask_ratio, ax)
            print(f'along axes = {ax}, shift by: {shift_matrix}')
        new_dir = f'{file_path}/align_axis_{axis}'       
        fn_save = f'{new_dir}/fn'        
        with h5py.File(fn_save, 'w') as hf:
            hf.create_dataset('img', data = tmp)
            hf.create_dataset('scan_id', data = scan_id)
            hf.create_dataset('XEng', data = eng1)
            hf.create_dataset('angle', data = angle1)
            hf.create_dataset('shift', data = np.array([r, c]))
            hf.create_dataset('axis', data = axis)
        print(f'{fn_save} saved \n')
    np.savetxt(f'{new_dir}/rshft_axis_{axis}.txt', rshft)
    np.savetxt(f'{new_dir}/cshft_axis_{axis}.txt', cshft)  
    del tmp     
                        


def align_proj_sub(fn_ref, angle_ref, fn, angle1, binning=2, ratio=0.6, block_list=[], sli=[], ali_method='stackreg'):
    # ali_method can be 'stackreg' or 'cross_corr'
    num_angle = len(angle1)
    theta1 = angle1/180.*np.pi
    img1_ali = []
    rshft = []
    cshft = []
    img1_all, eng1, _ = pyxas.retrieve_norm_tomo_image(fn, index=-1, binning=binning, sli=sli)
    # img1_all = tomopy.prep.stripe.remove_stripe_fw(img1_all, level=5, wname='db5', sigma=1, pad=True)
    for j in range(num_angle):
        #sr = StackReg(StackReg.RIGID_BODY)
        sr = StackReg(StackReg.TRANSLATION)
        if not j%20:
            print(f'process file {fn}, angle #{j}/{num_angle}')
        ang_index = int(pyxas.find_nearest(angle_ref, angle1[j]))
        img0, _, _ = pyxas.retrieve_norm_tomo_image(fn_ref, index=ang_index, binning=binning, sli=sli)
        img0 = np.squeeze(img0)
        s = img0.shape
        ratio_s = 0.5 - ratio / 2
        ratio_e = 0.5 + ratio /2
        img0 = img0[int(s[0]*ratio_s) : int(s[0]*ratio_e), int(s[1]*ratio_s) : int(s[1]*ratio_e)]
        #img1, eng1, _ = pyxas.retrieve_norm_tomo_image(fn, index=j, binning=binning, sli=sli)
        img1 = np.squeeze(img1_all[j])
        tmp = img1[int(s[0]*ratio_s) : int(s[0]*ratio_e), int(s[1]*ratio_s) : int(s[1]*ratio_e)]
        
        if ali_method == 'stackreg':
            tmat = sr.register(img0, tmp)
            tmp = sr.transform(img1)
            rs, cs = -tmat[1, 2], -tmat[0, 2]
        else:
            _,rs,cs = pyxas.align_img(img0, tmp)
            tmp = pyxas.shift(img1, [rs, cs], order=0)

        img1_ali.append(tmp)
        rshft.append(rs)
        cshft.append(cs)
    '''
    allow_list = list(set(np.arange(len(angle1))) - set(block_list))
    x = angle1/180.0*np.pi
    x = x[allow_list]
    y = np.array(cshft)[allow_list]
    zero_angle_pos = pyxas.find_nearest(theta1, 0)
    y0 = y[zero_angle_pos]
    y_scale = np.max(np.abs(y-y0))
    y = (y - y0)/y_scale
    r_fit, phi_fit= fit_cos(x, y, num_iter=6000, learning_rate=0.005)
    r_fit *= y_scale
    Y_fit = eval_fit_cos(r_fit, phi_fit, theta1)+y0

    r_shift_mean = np.mean(np.array(rshft)[allow_list])
    for j in range(num_angle):
        if not j%20:
            print(f'process file {fn}, angle #{j}/{num_angle}')
        tmp_mat=np.array([[1,0,-Y_fit[j]],[0,1,-r_shift_mean], [0,0,1]])
        img1, eng1, _ = pyxas.retrieve_norm_tomo_image(fn, index=j, binning=binning)
        img1 = np.squeeze(img1)
        tmp = sr.transform(img1, tmp_mat)
        img1_ali.append(tmp)
    '''

    img1_ali = np.array(img1_ali)
    return img1_ali, eng1, rshft, cshft

#### end of obsolete function ###
