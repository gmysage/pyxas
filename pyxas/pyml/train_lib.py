import torch
from tqdm import tqdm
from .fit_xanes import *
from .loss_lib import *
from torch.nn import MSELoss
import torch.optim as optim

def transform_gt(img):
    img_log = -np.log(img)
    img_log[np.isnan(img_log)] = 0
    img_log[np.isinf(img_log)] = 0
    return img_log

def transform_blur(img):
    img_log = -np.log(img)
    img_log[np.isnan(img_log)] = 0
    img_log[np.isinf(img_log)] = 0
    return img_log
    
def train_1_branch_bkg(model_gen, dataloader, loss_r, vgg19, device='cuda:1', lr=1e-4, train_fit=False, take_log=True):
    
    #global vgg19
    mse_criterion = MSELoss()
    model_gen.train()
    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(0.5, 0.999))

    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0
    
    ########################################
    # need to change when change to new dataloader (e.g., a image stack)
    #batch_size = dataloader.batch_size
    batch_size = 1
    #####################################

    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/batch_size)):
        image_data = data[0].to(device) # (1, 16, 256, 256)
        label = data[1].to(device) # blurred bkg
        x_eng = data[2][0].to(device)
        elem = data[3][0] # e.g, 'Ni'
        img_stack_size = len(x_eng)

        s0 = image_data.size()       
        s = (s0[1], s0[0], s0[2], s0[3])
        
        image_data = image_data.reshape(s)   # (16, 1, 256, 256)
        label = label.reshape(s) 

        if take_log:
            real_img = image_data / label        
        else:
            real_img = image_data - label
        
        output_bkg = model_gen(image_data) # background image

        if take_log:
            output_img = image_data / output_bkg
            output_img[output_img>1.5] = 1
        else:
            output_img = image_data - output_bkg
                  
        # additional loss (for img) to update generator
        fit_para_dn, y_fit_dn = fit_element_xraylib_barn(elem, x_eng, output_img, order=[-3,0], rho=None, take_log=take_log, device=device)
        fit_para_gt, y_fit_gt = fit_element_xraylib_barn(elem, x_eng, real_img, order=[-3,0], rho=None, take_log=take_log, device=device)
        
        fit_dn_coeff = fit_para_dn[0].reshape((1, 1, s[-2], s[-1])).type(torch.float32)
        fit_gt_coeff = fit_para_gt[0].reshape((1, 1, s[-2], s[-1])).type(torch.float32)
    
        y_fit_reshape = y_fit_dn.reshape(s).type(torch.float32)
        if take_log:
            y_fit_reshape = torch.exp(-y_fit_reshape) 
            #y_fit_reshape[y_fit_reshape > 1.5] = 1       

        # identity loss
        mse_identity_bkg = mse_criterion(output_bkg, label)
        mse_identity_img = mse_criterion(output_img, real_img)
        loss_value['mse_identity_bkg'] = mse_identity_bkg
        #loss_value['mse_identity_img'] = mse_identity_img
        loss_value['mse_fit_coef'] = mse_criterion(fit_dn_coeff, fit_gt_coeff)

        # self-consistant of fitting results
        a = y_fit_reshape
        b = output_img
        loss_value['mse_fit_self_consist'] = mse_criterion(a, b)

        # r_vgg: identity feature loss
        loss_value['vgg_identity'] = vgg_loss(output_bkg, label, vgg19, device=device)
        
        # r_vgg_fit:
        loss_value['vgg_fit'] = vgg_loss(y_fit_reshape, real_img, vgg19, device=device)

        # r_vgg_1st_last: (also available in production)
        '''
        # this does not work well, so delete it
        loss_value['vgg_1st_last'] = 0.0
        im1 = output_img[0] 
        im2 = output_img[-1] 
        im1 = im1[None]
        im2 = im2[None]
        im1 = im1 / torch.mean(im1)
        im2 = im2 / torch.mean(im2)
        loss_value['vgg_1st_last'] += vgg_loss(im1, im2, vgg19, device=device)
        '''
        loss_value['l1_identity'] = l1_loss(output_bkg, label)

        total_loss_gen = 0.0
        total_loss_gen += loss_value['mse_identity_bkg']
        #total_loss_gen += loss_value['mse_identity_img']
        '''
        for k in keys:
            if 'mse_identity' in k:
                continue
            if train_fit:
                total_loss_gen += loss_value[k] * loss_r[k]
        '''
        for k in keys:
            if 'mse_identity' in k:
                continue
            if loss_r[k] > 0:
                total_loss_gen += loss_value[k] * loss_r[k]

        model_gen.zero_grad()
        total_loss_gen.backward()
        opt_gen.step()
        
        for k in keys:
            if loss_r[k] > 0:
                running_loss[k] += loss_value[k].item() * loss_r[k]
            else:
                running_loss[k] += loss_value[k].item()
        
        # calculate batch psnr (once every `batch_size` iterations)
        #batch_psnr = psnr(label, output_img)
        # should be:
        batch_psnr = psnr(label, output_bkg)
        running_psnr += batch_psnr
    loss_summary = {}
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr/int(len(dataloader.dataset)/batch_size)

    return loss_summary


def train_1_branch_bkg_production(dataloader, loss_r, thickness_dict, model_prod, device='cuda:1', lr=1e-4, ratio=0.5):
    mse_criterion = MSELoss()
    model_prod.train()
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)
    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_psnr_raw = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0    

    ########################################
    # need to change when change to new dataloader (e.g., a image stack)
    #batch_size = dataloader.batch_size
    batch_size = 1
    #####################################

    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/batch_size)):
        image_data = data[0].to(device) # (1, 16, 256, 256)
        image_data[image_data==0] = 1
        label = data[1].to(device) # blurred bkg
        x_eng = data[2][0].to(device)
        elem = data[3][0] # e.g, 'Ni'
        
        thickness = thickness_dict[elem]
        
        s0 = image_data.size()       
        s = (s0[1], s0[0], s0[2], s0[3])
        
        image_data = image_data.reshape(s)   # (16, 1, 256, 256)
        output_bkg = model_prod(image_data)
        output_img = image_data / output_bkg
        output_img[output_img>1.5] = 1
        
        fit_para_dn, y_fit_dn = fit_element_xraylib_barn_fix_thickness(elem, x_eng, output_img, thickness, order=[-3,0], rho=None, device=device)
        y_fit_reshape = y_fit_dn.reshape(s).type(torch.float32)
        y_fit_reshape = torch.exp(-y_fit_reshape) 
        #y_fit_reshape[y_fit_reshape > 1.5] = 1 

        mse_fit_img1 = mse_criterion(y_fit_reshape, output_img)
        mse_fit_img2 = mse_criterion(y_fit_reshape, image_data)
        loss_value['mse_fit_img'] = ratio * mse_fit_img1 + (1-ratio) * mse_fit_img2
        
         # TV loss added on 11/20/2022
        loss_value['tv_bkg'] = tv_loss(output_bkg) * loss_r['tv_bkg']
        
        total_loss_gen = 0.0
        for k in keys:
            total_loss_gen += loss_value[k] * loss_r[k]

        model_prod.zero_grad()
        total_loss_gen.backward()
        opt_prod.step()

        for k in keys:
            if loss_r[k] > 0:
                running_loss[k] += loss_value[k].item() * loss_r[k]
            else:
                running_loss[k] += loss_value[k].item()   
        batch_psnr = psnr(output_img, y_fit_reshape)
        batch_psnr_raw = psnr(output_img, image_data)
        running_psnr += batch_psnr 
        running_psnr_raw += batch_psnr_raw 

    loss_summary = {}
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr/int(len(dataloader.dataset)/batch_size)
    loss_summary['psnr_raw'] = running_psnr_raw/int(len(dataloader.dataset)/batch_size)
    return loss_summary


def train_xanes_bkg_production(dataloader, loss_r, thickness_dict, model_prod, lr,
                               device='cuda:1', ratio=0.5):
    mse_criterion = MSELoss()
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)
    model_prod.train()

    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_psnr_raw = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0

    batch_size = 1

    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / batch_size)):
        image_data = data[0].to(device)  # (1, 16, 256, 256)
        image_data[image_data == 0] = 1
        #label = data[1].to(device)  # blurred bkg
        x_eng = data[2][0].to(device)
        elem = data[3][0]  # e.g, 'Ni'

        thickness = thickness_dict[elem]
        mask = thickness_dict['mask']

        s0 = image_data.size()
        s = (s0[1], s0[0], s0[2], s0[3])

        image_data = image_data.reshape(s)  # (16, 1, 256, 256)
        output_bkg = model_prod(image_data)
        output_img = image_data / output_bkg
        output_img[output_img > 1.5] = 1

        fit_para_dn, y_fit_dn = fit_element_xraylib_barn_fix_thickness(elem, x_eng, output_img, thickness,
                                                                       order=[1, 0], rho=None, device=device)
        y_fit_reshape = y_fit_dn.reshape(s).type(torch.float32)
        y_fit_reshape = torch.exp(-y_fit_reshape)

        mse_fit_img1 = mse_criterion(y_fit_reshape, output_img)
        mse_fit_img2 = mse_criterion(y_fit_reshape, image_data)
        loss_value['mse_fit_img'] = ratio * mse_fit_img1 + (1 - ratio) * mse_fit_img2

        # TV loss added on 11/20/2022
        loss_value['tv_bkg'] = tv_loss(output_bkg) * loss_r['tv_bkg']
        loss_value['ssim_img'] = 1 - ssim_loss(output_img, y_fit_reshape)

        total_loss_gen = 0.0
        for k in keys:
            total_loss_gen += loss_value[k] * loss_r[k]

        model_prod.zero_grad()
        total_loss_gen.backward()
        opt_prod.step()

        for k in keys:
            if loss_r[k] > 0:
                running_loss[k] += loss_value[k].item() * loss_r[k]
            else:
                running_loss[k] += loss_value[k].item()
        batch_psnr = psnr(output_img, y_fit_reshape)
        batch_psnr_raw = psnr(output_img, image_data)
        running_psnr += batch_psnr
        running_psnr_raw += batch_psnr_raw

    loss_summary = {}
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr / int(len(dataloader.dataset) / batch_size)
    loss_summary['psnr_raw'] = running_psnr_raw / int(len(dataloader.dataset) / batch_size)
    return loss_summary, model_prod


def train_xanes_bkg_production_with_reference(dataloader, loss_r, model_prod, lr, spectrum_ref, mask,
                                              ratio=0.99, device='cuda:1'):
    mse_criterion = MSELoss()
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)
    model_prod.train()

    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0

    batch_size = 1

    mask = mask.to(device)
    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / batch_size)):
        image_data = data[0].to(device)  # (1, 48, 256, 256)
        image_data[image_data == 0] = 1
        x_eng = data[2][0].to(device)
        #elem = data[3][0]  # e.g, 'Ni'

        s0 = image_data.size()
        s = (s0[1], s0[0], s0[2], s0[3])

        image_data = image_data.reshape(s)  # (16, 1, 256, 256)
        output_bkg = model_prod(image_data)
        output_img = image_data / output_bkg
        output_img[output_img > 1.5] = 1

        fit_para_dn, y_fit_dn = fit_element_with_reference(x_eng, output_img, spectrum_ref, order=[1, 0],
                                                           take_log=True, device=device)
        y_fit_reshape = y_fit_dn.reshape(s).type(torch.float32)
        y_fit_reshape = torch.exp(-y_fit_reshape)*mask

        mse_fit_img1 = mse_criterion(y_fit_reshape, output_img)
        mse_fit_img2 = mse_criterion(y_fit_reshape, image_data)
        loss_value['mse_fit_img'] = ratio * mse_fit_img1 + (1 - ratio) * mse_fit_img2

        # TV loss added on 11/20/2022
        loss_value['tv_bkg'] = tv_loss(output_bkg) * loss_r['tv_bkg']
        loss_value['ssim_img'] = 1 - ssim_loss(output_img, y_fit_reshape)

        total_loss_gen = 0.0
        for k in keys:
            total_loss_gen += loss_value[k] * loss_r[k]

        model_prod.zero_grad()
        total_loss_gen.backward()
        opt_prod.step()

        for k in keys:
            if loss_r[k] > 0:
                running_loss[k] += loss_value[k].item() * loss_r[k]
            else:
                running_loss[k] += loss_value[k].item()
        batch_psnr = psnr(output_img, y_fit_reshape)
        running_psnr += batch_psnr

    loss_summary = {}
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr / int(len(dataloader.dataset) / batch_size)
    return loss_summary, model_prod


def train_production_with_fitted_param(dataloader, loss_r, model_prod, lr, spectrum_ref, fit_param_X,
                                       order, mask, device='cuda:1'):
    mse_criterion = MSELoss()
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)
    model_prod.train()

    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0

    batch_size = 1

    mask = mask.to(device)

    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / batch_size)):
        image_data = data[0].to(device)  # (1, 48, 256, 256)
        image_data[image_data == 0] = 1
        x_eng = data[2][0].to(device)
        A = compose_matrix_A_with_reference(x_eng, spectrum_ref, order, device)
        fitted_img = torch.matmul(A, fit_param_X)

        s0 = image_data.size()
        s = (s0[1], s0[0], s0[2], s0[3])
        fitted_img = fitted_img.reshape(s) * mask
        fitted_img = torch.exp(-fitted_img)
        #fitted_img = fitted_img.float()


        image_data = image_data.reshape(s)  # (16, 1, 256, 256)
        output_bkg = model_prod(image_data)
        output_img = image_data / output_bkg
        #output_img[output_img > 1.5] = 1
        #output_img = output_img.double()

        for k in keys:
            if k == 'mse_fit_img':
                loss_value['mse_fit_img'] = mse_criterion(output_img, fitted_img)
            elif k == 'tv_bkg':
                loss_value['tv_bkg'] = tv_loss(output_bkg)
            elif k == 'ssim_img':
                loss_value['ssim_img'] = 1 - ssim_loss(output_img, fitted_img)

        total_loss_gen = 0.0
        for k in keys:
            total_loss_gen += loss_value[k] * loss_r[k]

        model_prod.zero_grad()
        total_loss_gen.backward()
        opt_prod.step()

        for k in keys:
            if loss_r[k] > 0:
                running_loss[k] += loss_value[k].item() * loss_r[k]
            else:
                running_loss[k] += loss_value[k].item()
        batch_psnr = psnr(output_img, fitted_img)
        running_psnr += batch_psnr

    loss_summary = {}
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr / int(len(dataloader.dataset) / batch_size)
    return loss_summary, model_prod


def validate(model, dataloader, loss_r, vgg19, device='cuda:1', take_log=True):
    model.eval()
    mse_criterion = nn.MSELoss()
    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0

    batch_size = 1
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/batch_size)):
            image_data = data[0].to(device)  # (1, 16, 256, 256)
            label = data[1].to(device)  # blurred bkg
            x_eng = data[2][0].to(device)
            elem = data[3][0]  # e.g, 'Ni'
            img_stack_size = len(x_eng)

            s0 = image_data.size()
            s = (s0[1], s0[0], s0[2], s0[3])

            image_data = image_data.reshape(s)  # (16, 1, 256, 256)
            label = label.reshape(s)

            if take_log:
                real_img = image_data / label
            else:
                real_img = image_data - label

            output_bkg = model(image_data)  # background image

            if take_log:
                output_img = image_data / output_bkg
                output_img[output_img > 1.5] = 1
            else:
                output_img = image_data - output_bkg

            # additional loss (for img) to update generator
            fit_para_dn, y_fit_dn = fit_element_xraylib_barn(elem, x_eng, output_img, order=[-3, 0], rho=None,
                                                             take_log=take_log, device=device)
            fit_para_gt, y_fit_gt = fit_element_xraylib_barn(elem, x_eng, real_img, order=[-3, 0], rho=None,
                                                             take_log=take_log, device=device)

            fit_dn_coeff = fit_para_dn[0].reshape((1, 1, s[-2], s[-1])).type(torch.float32)
            fit_gt_coeff = fit_para_gt[0].reshape((1, 1, s[-2], s[-1])).type(torch.float32)

            y_fit_reshape = y_fit_dn.reshape(s).type(torch.float32)
            if take_log:
                y_fit_reshape = torch.exp(-y_fit_reshape)
                # y_fit_reshape[y_fit_reshape > 1.5] = 1

            # identity loss
            loss_value['mse_identity_bkg'] = mse_criterion(output_bkg, label).detach()
            loss_value['mse_fit_coef'] = mse_criterion(fit_dn_coeff, fit_gt_coeff).detach()

            # self-consistant of fitting results
            a = y_fit_reshape
            b = output_img
            loss_value['mse_fit_self_consist'] = mse_criterion(a, b).detach()
            # r_vgg: identity feature loss
            loss_value['vgg_identity'] = vgg_loss(output_bkg, label, vgg19, device=device).detach()

            # r_vgg_fit:
            loss_value['vgg_fit'] = vgg_loss(y_fit_reshape, real_img, vgg19, device=device).detach()

            loss_value['l1_identity'] = l1_loss(output_bkg, label).detach()

            total_loss_gen = 0.0
            total_loss_gen += loss_value['mse_identity_bkg']

            for k in keys:
                if loss_r[k] > 0:
                    running_loss[k] += loss_value[k].item() * loss_r[k]
                else:
                    running_loss[k] += loss_value[k].item()

            batch_psnr = psnr(label, output_bkg)
            running_psnr += batch_psnr

    loss_summary = {}
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr / int(len(dataloader.dataset) / batch_size)
    return loss_summary


def train_xanes_3D_production(dataloader, loss_r, thickness_dict, model_prod, opt_prod, lr,
                              device='cuda:1', mask=None, order=[-3, 0]):
    # the image data is from xanes_slice after 3D reconstruction.
    # no need to take -log()

    mse_criterion = MSELoss()

    model_prod.train()

    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_psnr_raw = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0

    batch_size = 1
    if mask is None:
        mask = 1
    if not type(mask) is torch.Tensor:
        mask = torch.tensor(mask).to(device)

    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / batch_size)):
        image_data = data[0].to(device)  # (1, 16, 256, 256)
        x_eng = data[2][0].to(device)
        elem = data[3][0]  # e.g, 'Ni'
        thickness = thickness_dict[elem]

        s0 = image_data.size()
        s = (s0[1], s0[0], s0[2], s0[3])
        image_data = image_data.reshape(s)  # (16, 1, 256, 256)

        ################################
        # scale individual image
        if torch.sum(mask) == 1:
            mask = torch.ones((s[2], s[3])).to(device)
        scale_forward = torch.ones(s[0]).to(device)

        mask_sum = torch.sum(mask)
        for i in range(s[0]):
             scale_forward[i] = torch.sum(image_data[i] * mask) / mask_sum
             image_data[i] /= scale_forward[i]
        # end of scale individual image
        ################################

        #######################
        ### adapt R2R method from:
        ### "https://github.com/PangTongyao/Recorrupted-to-Recorrupted-Unsupervised-Deep-Learning-for-Image-Denoising/blob/main/train_AWGN.py"
        input_train = torch.zeros(s)
        target_train = torch.zeros(s)
        eps = 0.1
        alpha = 0.5
        sz = image_data[0].size()
        for i in range(s[0]):
            std = torch.std(image_data[i])
            pert = eps * torch.FloatTensor(sz).normal_(mean=0, std=std).to(device)
            target_train[i] = image_data[i] + alpha * pert
            input_train[i] = image_data[i] - pert / alpha
        input_train = input_train.to(device)
        target_train = target_train.to(device)
        ### end R2R
        #########################

        output_img = model_prod(input_train)
        loss_value['mse_r2r'] = mse_criterion(output_img, target_train)


        #############################
        ### rescale output_img back to xanes_spectrum
        output_img_clone = output_img.clone()
        for i in range(s[0]):
            output_img_clone[i] = output_img[i] * scale_forward[i]
        #############################

        fit_para_dn, y_fit_dn = fit_element_xraylib_barn_fix_thickness(elem, x_eng, output_img_clone, thickness,
                                                                             order=order, rho=None, take_log=False,
                                                                             device=device)
        y_fit_reshape = y_fit_dn.reshape(s).type(torch.float32)
        y_fit_reshape = y_fit_reshape * mask

        loss_value['mse_fit_img'] = mse_criterion(y_fit_reshape, output_img_clone)

        # TV loss added on 11/20/2022
        loss_value['tv_img'] = tv_loss(output_img)
        loss_value['ssim_img'] = 1 - ssim_loss(output_img, y_fit_reshape)

        # L1 loss
        loss_value['l1_img'] = l1_loss(output_img, y_fit_reshape)

        total_loss_gen = 0.0
        for k in keys:
            if loss_r[k] > 0:
                total_loss_gen += loss_value[k] * loss_r[k]

        model_prod.zero_grad()
        opt_prod.zero_grad()

        total_loss_gen.backward()
        opt_prod.step()

        for k in keys:
            if loss_r[k] > 0:
                running_loss[k] += loss_value[k].item() * loss_r[k]
            else:
                running_loss[k] += loss_value[k].item()
        batch_psnr = psnr(output_img_clone, y_fit_reshape)
        batch_psnr_raw = psnr(output_img, image_data)
        running_psnr += batch_psnr
        running_psnr_raw += batch_psnr_raw

    loss_summary = {}
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr / int(len(dataloader.dataset) / batch_size)
    loss_summary['psnr_raw'] = running_psnr_raw / int(len(dataloader.dataset) / batch_size)
    return loss_summary, model_prod

def ML_xanes_default_param(n_epoch=100, n_train=50, lr=2e-4,
                           order=[0, 1], loss_mse_r2r=1, loss_mse_fit_img=1e2,
                           loss_tv=0, loss_ssim=0, loss_l1=0):
    loss_r = {}
    loss_r['mse_r2r'] = loss_mse_r2r
    loss_r['mse_fit_img'] = loss_mse_fit_img
    loss_r['tv_img'] = loss_tv  # 1e-3
    loss_r['ssim_img'] = loss_ssim
    loss_r['l1_img'] = loss_l1
    param = {}
    param['loss_r'] = loss_r
    param['n_train'] = n_train
    param['lr'] = lr
    param['n_epoch'] = n_epoch
    param['order'] = order
    return param