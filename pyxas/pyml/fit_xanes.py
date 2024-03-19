import xraylib
import torch
import numpy as np
from .util import *
from scipy.interpolate import InterpolatedUnivariateSpline

def fit_element_xraylib_barn(elem, x_eng, image_xanes, order=[-3, 0], rho=None, take_log=True, device='cuda:1'):
    # y = mu*t1 + mu_bkg *t2
    # mu_bkg = -a*E + b
    # assume a line profile for background attenuation

    if rho is None:
        rho = 1
    # flatten the image
    try:
        s = image_xanes.size()
    except:
        s = image_xanes.shape
    y_spec = torch.reshape(image_xanes, (s[0], s[2]*s[3]))
    
    x_eng = x_eng.double()
    y_spec = y_spec.double()
    
    if take_log:
        y_spec = -torch.log(y_spec)
    y_spec[torch.isnan(y_spec)] = 0.0  
    y_spec[torch.isinf(y_spec)] = 0.0  
    num = len(x_eng)
    cs = torch.tensor((), dtype=torch.float64, device=device)
    cs = cs.new_zeros((num))
    if len(y_spec.size()) == 1:
        Y = torch.reshape(y_spec, (num, 1))
    else:
        Y = torch.clone(y_spec)
        
    for i in range(num):
        cs[i] = xraylib.CSb_Total(xraylib.SymbolToAtomicNumber(elem), x_eng[i].item())
    mu = cs * rho
    
    n_order = len(order) 
    
    A = torch.tensor((), dtype=torch.float64, device=device)
    A = A.new_ones((num, n_order+1))
    
    A[:, 0] = mu
    
    for i in range(n_order):
        A[:, i+1] = x_eng ** (order[i])
    
    AT = A.t()
    ATA = torch.matmul(AT,A)
    ATA_inv = torch.linalg.inv(ATA, out=None)
    X = torch.matmul(torch.matmul(ATA_inv, AT), Y)

    Y_fit = torch.matmul(A, X)
    
    return X, Y_fit


####################
def interp_spec1(spectrum_ref, xanes_eng):
    if torch.is_tensor(xanes_eng):
        eng = xanes_eng.cpu().numpy()
    else:
        eng = xanes_eng.copy()
    n_ref = len(spectrum_ref)
    n_eng = len(xanes_eng)
    ref_spec_interp = np.zeros((n_eng, n_ref))
    for i in range(n_ref):
        f = InterpolatedUnivariateSpline(spectrum_ref[f'ref{i}'][:, 0], spectrum_ref[f'ref{i}'][:, 1], k=3)
        ref_spec_interp[:, i] = f(eng)
    return ref_spec_interp


def fit_element_with_reference(x_eng, image_xanes, spectrum_ref, order=[1, 0], take_log=True, device='cuda:1'):
    # x_eng: numpy
    # image_xanes: numpy or torch.tensor
    # spectrumF_ref: dictionary, numpy


    # flatten the image
    if not torch.is_tensor(image_xanes):
        image_xanes = torch.from_numpy(image_xanes).float().to(device)
    try:
        s = image_xanes.size() # e.g., (60, 1, 256, 256)
    except:
        s = image_xanes.shape

    y_spec = torch.reshape(image_xanes, (s[0], s[-2] * s[-1]))

    if not torch.is_tensor(x_eng):
        x_eng = torch.from_numpy(x_eng).float()
    x_eng = x_eng.float().to(device)
    #x_eng = x_eng.double()
    #y_spec = y_spec.double()

    if take_log:
        y_spec = -torch.log(y_spec)
    y_spec[torch.isnan(y_spec)] = 0.0
    y_spec[torch.isinf(y_spec)] = 0.0
    num = len(x_eng)

    if len(y_spec.size()) == 1:
        Y = torch.reshape(y_spec, (num, 1))
    else:
        Y = torch.clone(y_spec)

    '''
    ref_spec = interp_spec1(spectrum_ref, x_eng)
    n_spec = ref_spec.shape[1]
    n_order = len(order)

    A = torch.tensor((), dtype=torch.float64, device=device)
    A = A.new_ones((num, n_order + n_spec))

    A[:, :n_spec] = torch.tensor(ref_spec, device=device)

    for i in range(n_order):
        A[:, i + n_spec] = x_eng ** (order[i])
    '''
    A = compose_matrix_A_with_reference(x_eng, spectrum_ref, order, device)
    AT = A.t()
    ATA = torch.matmul(AT, A)
    ATA_inv = torch.linalg.inv(ATA, out=None)
    X = torch.matmul(torch.matmul(ATA_inv, AT), Y)

    Y_fit = torch.matmul(A, X)

    return X, Y_fit


def compose_matrix_A_with_reference(x_eng, spectrum_ref, order, device):
    if not torch.is_tensor(x_eng):
        x_eng = torch.from_numpy(x_eng)
    x_eng = x_eng.to(device)
    ref_spec = interp_spec1(spectrum_ref, x_eng)
    n_spec = ref_spec.shape[1]
    n_order = len(order)
    num = len(x_eng)
    A = torch.tensor((), dtype=torch.float32, device=device)
    A = A.new_ones((num, n_order + n_spec))

    A[:, :n_spec] = torch.tensor(ref_spec, device=device)

    for i in range(n_order):
        A[:, i + n_spec] = x_eng ** (order[i])
    return A

####################
def fit_element_xraylib_barn_fix_thickness(elem, x_eng, image_xanes, thickness, order=[-3, 0], rho=None, take_log=True, device='cuda:1'):
    # y = mu*t1 + mu_bkg *t2
    # mu_bkg = -a*E + b
    # assume a line profile for background attenuation

    if rho is None:
        rho = 1
    # flatten the image
    try:
        s = image_xanes.size()
    except:
        s = image_xanes.shape
    if take_log:
        image_xanes_log = -torch.log(image_xanes)
        y_spec = torch.reshape(image_xanes_log, (s[0], s[2]*s[3]))
    else:
        y_spec = torch.reshape(image_xanes, (s[0], s[2]*s[3]))

    thick = torch.reshape(thickness, (1, s[2]*s[3]))

    x_eng = x_eng.double()
    y_spec = y_spec.double()
    y_spec[torch.isnan(y_spec)] = 0.0  
    y_spec[torch.isinf(y_spec)] = 0.0  
    num = len(x_eng)
    cs = torch.tensor((), dtype=torch.float64, device=device)
    cs = cs.new_zeros((num))
    if len(y_spec.size()) == 1:
        Y = torch.reshape(y_spec, (num, 1))
    else:
        Y = torch.clone(y_spec)

    Y_all = torch.clone(Y)

    for i in range(num):
        cs[i] = xraylib.CSb_Total(xraylib.SymbolToAtomicNumber(elem), x_eng[i].item())
        Y[i] = Y[i] - cs[i] * rho * thick
    mu = cs * rho
    
    n_order = len(order) 
    
    A_full = torch.tensor((), dtype=torch.float64, device=device)
    A_full = A_full.new_ones((num, n_order+1))
    A_full[:, 0] = mu

    A = torch.tensor((), dtype=torch.float64, device=device)
    A = A.new_ones((num, n_order))
    for i in range(n_order):
        A_full[:, i+1] = x_eng ** (order[i])
        A[:, i] = x_eng ** (order[i])
    
    AT = A.t()
    ATA = torch.matmul(AT,A)
    ATA_inv = torch.linalg.inv(ATA, out=None)
    X = torch.matmul(torch.matmul(ATA_inv, AT), Y)

    X_comb = torch.tensor((), dtype=torch.float64, device=device)
    X_comb = X_comb.new_ones((n_order+1, s[2]*s[3]))
    X_comb[0] = thick
    X_comb[1:] = X

    Y_fit = torch.matmul(A_full, X_comb)
    
    return X_comb, Y_fit


def cal_thickness(elem, x_eng, img_xanes, order=[-3, 0], rho=None, take_log=True, device='cuda:1'):
    edge = xraylib.EdgeEnergy(xraylib.SymbolToAtomicNumber(elem), xraylib.K_SHELL)
    eng_s = edge - 0.02
    eng_e = edge + 0.1
    s = img_xanes.shape
    if type(x_eng) is torch.Tensor:
        id1 = find_nearest(x_eng.cpu().numpy(), eng_s)
        id2 = find_nearest(x_eng.cpu().numpy(), eng_e)
    else:
        id1 = find_nearest(x_eng, eng_s)
        id2 = find_nearest(x_eng, eng_e)
        x_eng = torch.tensor(x_eng, device=device)
    l1 = list(np.arange(id1))
    l2 = list(np.arange(id2,len(x_eng)))
    l = l1 + l2

    X, Y_fit =  fit_element_xraylib_barn(elem, x_eng[l], img_xanes[l], order=order, rho=rho, take_log=take_log, device=device)
    thickness = X[0].reshape((s[-2], s[-1])).detach()
    thickness[thickness<0] = 0
    thickness = thickness.to(device)
    return thickness

def update_thickness_elem(elem, img, x_eng, model, device, n_iter=3, gaussian_filter=1, take_log=True, order=[-3, 0]):
    img_output, _ = apply_model_to_stack(img, model, device, n_iter)
    if len(img_output.shape) == 3:
        img_output = img_output[:, np.newaxis]
    img_output = torch.tensor(img_output).to(device)
    thickness_tmp = cal_thickness(elem, x_eng, img_output, order=order, rho=None, take_log=take_log, device=device)

    return thickness_tmp

