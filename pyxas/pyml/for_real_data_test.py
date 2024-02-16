from skimage import io
import numpy as np
import pyxas
import torch.optim as optim
import os
import random
from tqdm import trange
import torch
from .model_lib import *
import xraylib
from .util import *
from .fit_xanes import *
from .train_lib import *

def prepare_N85_Co():
    fn_root = '/data/xanes_bkg_denoise/test_experiment/sample8_N85'
    img_Co = io.imread(fn_root + '/aligned_29874.tiff')
    s = img_Co.shape
    eng = np.loadtxt(fn_root + '/exafs_Co.txt')
    # select 6 pre-edge image, 10 post-edge image
    id_post = pyxas.find_nearest(eng, 7.9)
    id_pre = pyxas.find_nearest(eng, 7.7)


    for n in trange(128):
        idx = [0] * 16
        for i in range(6):
            idx[i] = np.random.randint(0, id_pre)
        for i in range(6, 16):
            idx[i] = np.random.randint(id_post, s[0]-1)
        idx = np.sort(idx)
        img_t = img_Co[idx]
        img_t = img_t[:, 30:1024+30, 30:1024+30]
        img_t = pyxas.bin_ndarray(img_t, (16, 256, 256))
        img_t[img_t<=0] = 1
        img_log = -np.log(img_t)
        img_log = pyxas.rm_abnormal(img_log)


        eng_t = eng[idx]
        fsave = f'{fn_root}/img_product/img_product_{n:04d}.tiff'
        fsave_log = f'{fn_root}/img_log/img_log_{n:04d}.tiff'
        fsave_eng = f'{fn_root}/eng_list/eng_list_{n:04d}_Co.txt'
        io.imsave(fsave, img_t.astype(np.float32))
        io.imsave(fsave_log, img_log.astype(np.float32))
        
        np.savetxt(fsave_eng, eng_t)

def process_real_data():
    from tqdm import trange
    import bm3d
    device = torch.device('cpu')
    model_bkg = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_bkg_load_path = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_tmp/tmp_1499.pth'
    model_bkg.load_state_dict(torch.load(model_bkg_load_path))

    model_bkg2 = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_bkg2_load_path = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_bkg_bkg_tmp/bkg_bkg_tmp_170.pth'
    model_bkg2.load_state_dict(torch.load(model_bkg2_load_path))

    model_bkg_img = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_bkg_img_load_path = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_tmp_img/tmp_bkg_img_0199.pth'
    model_bkg_img.load_state_dict(torch.load(model_bkg_img_load_path))

    fn_root = '/data/xanes_bkg_denoise/test_experiment/sample4_LCO_460'
    fn = fn_root + '/xanes_13689_pos01.tiff'
    #fn = '/data/xanes_bkg_denoise/IMG_256_stack/ali_256.tiff'
    img_stack = io.imread(fn) # (16, 256, 256)
    img_stack = img_stack[:, 10:266, 10:266]
    s = img_stack.shape
    img_output = np.zeros(s)
    img_output_bm3d = np.zeros(s)
    img_bkg = np.zeros(s)
 
    for i in trange(s[0]):
        img = img_stack[i]
        p = img.copy()
        for j in range(1):
            p = check_image_fitting(model_bkg, p, device, 0)
            #p[p>0.98] = 0.8
        #img_denoise = check_image_fitting(model_bkg_img, img/p, device, 0)
        img_output[i] = img/p
        #img_output_bm3d[i] = bm3d.bm3d(img_output[i], sigma_psd=0.01, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        img_bkg[i] = p

def prepare_production_training_data(f_root, x_eng, img_xanes, elem='Co', edge=[7.7, 7.85], n_sample=100):
    if f_root[-1] == '/':
        f_root = f_root[:-1]

    mk_directory(f_root + '/img_gt_stack')
    mk_directory(f_root + '/img_blur_stack')
    mk_directory(f_root + '/img_eng_list')
    mk_directory(f_root + '/model_tmp')
    mk_directory(f_root + '/model_output')

    if len(edge) < 2:
        abs_edge = xraylib.EdgeEnergy(xraylib.SymbolToAtomicNumber(elem), xraylib.K_SHELL)
        edge = [abs_edge-0.03, abs_edge+0.1] 

    n = len(x_eng)
    id1 = find_nearest(x_eng, edge[0])
    id2 = find_nearest(x_eng, edge[1])
    for i in trange(n_sample):
        n1 = random.randint(6,8)
        n2 = 16 - n1
        idx = np.sort(random.sample(range(id1), n1))
        idy = np.sort(random.sample(range(id2, n), n2))
        id_comb = list(idx) + list(idy)
        eng_list = x_eng[id_comb]
        img_list = img_xanes[id_comb] 

        fsave_gt = f_root + f'/img_gt_stack/img_gt_stack_{i:04d}.tiff' 
        fsave_bl = f_root + f'/img_blur_stack/img_blur_stack_{i:04d}.tiff' 
        fsave_en = f_root + f'/img_eng_list/eng_list_{i:04d}_{elem}.txt'
        io.imsave(fsave_gt, img_list.astype(np.float32))
        io.imsave(fsave_bl, img_list.astype(np.float32))
        np.savetxt(fsave_en, eng_list)


def main_train_production_single_elem(f_root, img_raw, x_eng, elem, thickness_elem=None, ratio=0.99,
                                        mask=None, n_train=50, n_epoch=30, device='cuda:0', save_flag=True):
     
    #device = torch.device('cuda:2')
    loss_r = {}
    loss_r['mse_fit_img'] = 1           # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"
    loss_r['tv_bkg'] = 6e-5

    model_prod = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_bkg_load_path = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_tmp/sorted/tmp_1499.pth'
    model_prod.load_state_dict(torch.load(model_bkg_load_path))
    mse_criterion = nn.MSELoss()

    lr = 0.00005
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value':[], 'rate':''}

    #elem = 'Co'

    blur_dir = f_root + '/img_blur_stack'
    gt_dir = f_root + '/img_gt_stack'
    eng_dir = f_root + '/img_eng_list'
    trans_gt, trans_blur = None, None

    ############### calculate thickness 
    if mask is None:
        mask = 1

    img_all = img_raw[:, np.newaxis]
    img_all = torch.tensor(img_all).to(device)
    s = img_all.shape
    x_eng = torch.tensor(x_eng).to(device)
    if thickness_elem is None:
        thickness_elem = cal_thickness(elem, x_eng, img_all, order=[-3, 0], rho=None, take_log=True, device=device)
        thickness_elem = torch.tensor(thickness_elem * mask) 
    thickness = {}
    thickness[elem] = thickness_elem
    mask = torch.tensor(mask)
    ##################### end thickness

    #n_train = 50
    train_loader, valid_loader = get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)
   
    for epoch in range(n_epoch):
        #loss_summary_train = train_1_branch_bkg_production_NCM(train_loader, loss_r, thickness, device, ratio)
        loss_summary_train = train_1_branch_bkg_production(train_loader, loss_r, thickness, model_prod, device, ratio)
        
        if (epoch+1) % 5 == 0:
            print('\nupdate thickness\n')
            thickness_elem = update_thickness_elem(elem, img_all, x_eng, model_prod, device, n_iter=1, gaussian_filter=2)
            thickness[elem] = thickness_elem * mask
        img_output, img_bkg = apply_model_to_stack(img_raw, model_prod, device, 1, gaussian_filter=3)
        
        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(f'epoch #{epoch}')
        print(txt_t)

        if save_flag:
            io.imsave(f_root + f'/model_output/output_{epoch:03d}.tiff', img_output.astype(np.float32))
            io.imsave(f_root + f'/model_output/bkg_{epoch:03d}.tiff', img_bkg.astype(np.float32))
            
            ftmp = f_root + f'/model_tmp/tmp_{epoch:04d}.pth'
            torch.save(model_prod.state_dict(), ftmp)
            with open(f_root + f'/model_tmp/h_loss.json', 'w') as f:
                json.dump(h_loss_train, f)




def mk_directory(fpath):
    try:
        os.makedirs(fpath)
    except:
        pass


def example_train_prod():
    f_root = '/data/quant_xanes/N77_NMC_comb/2023Q2/ML_train/train_Ni'
    img_raw = io.imread('Ni_pos1_crop_300eng.tiff')
    x_eng = np.loadtxt('Ni_comb.txt')
    elem = 'Ni'
    thickness_elem=None
    ratio=0.99
    device = 'cuda:0'
    mask=None
    n_train=50
    n_epoch=30

    main_train_production_single_elem(f_root, img_raw, x_eng, elem, thickness_elem, ratio,
                                        mask, n_train, n_epoch, device)