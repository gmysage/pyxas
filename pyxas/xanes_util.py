from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
from pyxas.lsq_fit import lsq_fit_iter, lsq_fit_iter2, coordinate_descent_lasso, admm_iter
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from copy import deepcopy
from numpy.polynomial.polynomial import polyfit, polyval
from pyxas.image_util import rm_abnormal, bin_ndarray, img_smooth



def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()


def fit_curve(x_raw, y_raw, x_fit, deg=1):
    coef1 = polyfit(x_raw, y_raw, deg)
    y_fit = polyval(x_fit, coef1)
    return y_fit

def L(x, x0, gamma):
    '''
     Return Lorentzian line shape at x with HWHM gamma
    '''

    y = 1/np.pi * (0.5*gamma) / ((x-x0)**2 + (0.5*gamma)**2)
    y_max = 2/np.pi/gamma

    return y, y_max


def load_xanes_ref(*args):
    '''
    load reference spectrum, use it as:    ref = load_xanes_ref(Ni, Ni2, Ni3)
    each spectrum is two-column array, containing: energy(1st column) and absortion(2nd column)

    It returns a dictionary, which can be used as: spectrum_ref['ref0'], spectrum_ref['ref1'] ....
    '''

    num_ref = len(args)
    assert num_ref >1, "num of reference should larger than 1"
    spectrum_ref = {}    
    for i in range(num_ref):
        spectrum_ref[f'ref{i}'] = args[i]
    return spectrum_ref


def norm_txm(img):
    tmp = deepcopy(img)
    tmp = -np.log(tmp)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    tmp[tmp<0] = 0
    return tmp



def fit_2D_xanes_non_iter(img_xanes, eng, spectrum_ref):
    '''
    Solve equation of Ax=b, where:

    Inputs:
    ----------
    A: reference spectrum (2-colume array: xray_energy vs. absorption_spectrum)
    X: fitted coefficient of each ref spectrum
    b: experimental 2D XANES data 

    Outputs:
    ----------
    fit_coef: the 'x' in the equation 'Ax=b': fitted coefficient of each ref spectrum
    cost: cost between fitted spectrum and raw data
    '''

    '''
    # b = Ax 
    
    num_ref = len(spectrum_ref)
    spec_interp = {}
    comp = {}   
    A = [] 
    s = img_xanes.shape
    for i in range(num_ref):
        tmp = interp1d(spectrum_ref[f'ref{i}'][:,0], spectrum_ref[f'ref{i}'][:,1], kind='cubic')
        A.append(tmp(eng).reshape(1, len(eng)))
        spec_interp[f'ref{i}'] = tmp(eng).reshape(1, len(eng))
        comp[f'A{i}'] = spec_interp[f'ref{i}'].reshape(len(eng), 1)
        comp[f'A{i}_t'] = comp[f'A{i}'].T
    # e.g., spectrum_ref contains: ref1, ref2, ref3
    # e.g., comp contains: A1, A2, A3, A1_t, A2_t, A3_t
    #       A1 = ref1.reshape(110, 1)
    #       A1_t = A1.T
    A = np.squeeze(A).T
    #M = np.zeros([num_ref+1, num_ref+1])
    M = np.zeros([num_ref, num_ref])
    for i in range(num_ref):
        for j in range(num_ref):
            M[i,j] = np.dot(comp[f'A{i}_t'], comp[f'A{j}'])

#        M[i, num_ref] = 1
#    M[num_ref] = np.ones((1, num_ref+1))
#    M[num_ref, -1] = 0
    # e.g.
    # M = np.array([[float(np.dot(A1_t, A1)), float(np.dot(A1_t, A2)), float(np.dot(A1_t, A3)), 1.],
    #                [float(np.dot(A2_t, A1)), float(np.dot(A2_t, A2)), float(np.dot(A2_t, A3)), 1.],
    #                [float(np.dot(A3_t, A1)), float(np.dot(A3_t, A2)), float(np.dot(A3_t, A3)), 1.],
    #                [1., 1., 1., 0.]])
    M_inv = np.linalg.inv(M)
    b_tot = img_xanes.reshape(s[0],-1)
    B = np.ones([num_ref, b_tot.shape[1]])
    for i in range(num_ref):
        B[i] = np.dot(comp[f'A{i}_t'], b_tot)
    x = np.dot(M_inv, B)
    #x = x[:-1]    
    x[x<0] = 0
    x_sum = np.sum(x, axis=0, keepdims=True)
    #x = x / x_sum
    cost = np.sum((np.dot(A, x) - b_tot)**2, axis=0)/s[0]
    cost = cost.reshape(1, s[1], s[2])
    x = x.reshape(num_ref, s[1], s[2])

    '''
    # Y = bX + b0: X is the reference spectrum
    num_ref = len(spectrum_ref)
    s = img_xanes.shape
    X = np.ones([s[0], num_ref + 1])
    for i in range(num_ref):
        #tmp = interp1d(spectrum_ref[f'ref{i}'][:,0], spectrum_ref[f'ref{i}'][:,1], kind='cubic')
        tmp = InterpolatedUnivariateSpline(spectrum_ref[f'ref{i}'][:,0], spectrum_ref[f'ref{i}'][:,1], k=3)
        X[:, i] = tmp(eng)
    M = np.dot(X.T, X)
    M_inv = np.linalg.inv(M)
    Y = img_xanes.reshape(s[0],-1)
    b = np.dot(np.dot(M_inv, X.T), Y)
    Y_est = np.dot(X, b)
    x = b[:-1]
    offset = b[-1]
    cost = np.sum((Y_est - Y) ** 2, axis=0) / s[0]


    x = x.reshape(num_ref, s[1], s[2])
    offset = offset.reshape(1, s[1], s[2])
    cost = cost.reshape(1, s[1], s[2])

    return x, offset, cost



def fit_2D_xanes_iter(img_xanes, eng, spectrum_ref, coef0=None, offset=None, learning_rate=0.005, n_iter=10, bounds=[0,1], fit_iter_lambda=0):
    '''
    Solve the equation A*x = b iteratively


    Inputs:
    -------
    img_xanes: 3D xanes image stack

    eng: energy list of xanes

    spectrum_ref: dictionary, obtained from, e.g. spectrum_ref = load_xanes_ref(Ni2, Ni3)

    coef0: initial guess of the fitted coefficient, 
           it has dimention of [num_of_referece, img_xanes.shape[1], img_xanes.shape[2]]

    learning_rate: float

    n_iter: int

    bounds: [low_limit, high_limit]
          can be 'None', which give no boundary limit

    error_thresh: float
          used to generate a mask, mask[fitting_cost > error_thresh] = 0

    lamda: weight of constrain to force (the sum of fitting coefficient to be 1)

    Outputs:
    ---------
    w: fitted 2D_xanes coefficient
       it has dimention of [num_of_referece, img_xanes.shape[1], img_xanes.shape[2]]

    cost: 2D fitting cost
    '''
    num_ref = len(spectrum_ref)
    s = img_xanes.shape
    A = [] 
    for i in range(num_ref):
        #tmp = interp1d(spectrum_ref[f'ref{i}'][:,0], spectrum_ref[f'ref{i}'][:,1], kind='cubic')
        tmp = InterpolatedUnivariateSpline(spectrum_ref[f'ref{i}'][:, 0], spectrum_ref[f'ref{i}'][:, 1], k=3)
        A.append(tmp(eng).reshape(1, len(eng)))
    A = np.squeeze(A).T
    Y = img_xanes.reshape(img_xanes.shape[0], -1)
    if coef0 is None:
        W = None
    else:
        W = coef0.reshape(coef0.shape[0], -1)
    if offset is not None:
        offset = offset.reshape(1, -1)
    else:
        offset = np.zeros((1, s[1]*s[2]))
    w, b, cost = lsq_fit_iter2(A, Y, W, offset, learning_rate, n_iter, bounds, f_scale1=fit_iter_lambda)
    w = w.reshape(len(w), img_xanes.shape[1], img_xanes.shape[2])
    b = b.reshape(1, s[1], s[2])
    try:
        cost = cost[-1].reshape(1, s[1], s[2])
    except:
        cost = []
    return w, b, cost


def fit_2D_xanes_iter2(img_xanes, eng, spectrum_ref, coef0=None, offset=None, lamda=0.01, rho=0.01, n_iter=10, bounds=[0,1], method=1):
    '''
    method = 1: using coordinate_descent 
    method = 2: using admm
    '''
    num_ref = len(spectrum_ref)
    s = img_xanes.shape
    A = []
    for i in range(num_ref):
        #tmp = interp1d(spectrum_ref[f'ref{i}'][:, 0], spectrum_ref[f'ref{i}'][:, 1], kind='cubic')
        tmp = InterpolatedUnivariateSpline(spectrum_ref[f'ref{i}'][:, 0], spectrum_ref[f'ref{i}'][:, 1], k=3)
        A.append(tmp(eng).reshape(1, len(eng)))
    A = np.squeeze(A).T
    Y = img_xanes.reshape(s[0], -1)

    A_ext = np.ones([s[0], num_ref+1])
    A_ext[:,1:] = A

    if offset is None:
        offset = np.zeros([s[1], s[2]])
    if coef0 is None:
       coef0 = np.ones([num_ref, s[1], s[2]])

    xanes_2d_comb = np.zeros([num_ref + 1, s[1], s[2]])
    xanes_2d_comb[0] = offset
    xanes_2d_comb[1:] = coef0
    X_ini = xanes_2d_comb.reshape([num_ref + 1, -1])

    if method == 1:
        X = coordinate_descent_lasso(A_ext, Y, lamda=lamda, num_iters=n_iter, X_guess=X_ini, intercept=True, bounds=bounds)
    elif method == 2:
        X = admm_iter(A_ext, Y, rho=rho, num_iters=n_iter, X_guess=X_ini, wgt=[], lasso_lamda=lamda, bounds=bounds)

    w = X[1:].reshape(num_ref, s[1], s[2])
    w[w<0] = 0
    b = X[0].reshape(1, s[1], s[2])

    Y_hat = A_ext @ X
    m = s[0] # number of energy
    dy = Y - Y_hat
    cost = np.sum(dy**2, axis=0)/m
    cost = cost.reshape(1, s[1], s[2])

    return w, b, cost



def compute_xanes_fit_cost(img_xanes, fit_coef, fit_offset, spec_interp):
    # compute the cost
    num_ref = len(spec_interp)
    y_fit = np.zeros(img_xanes.shape)
    for i in range(img_xanes.shape[0]):
        for j in range(num_ref):
            y_fit[i] = y_fit[i] + fit_coef[j]*np.squeeze(spec_interp[f'ref{j}'])[i]
    y_fit =+ fit_offset
    y_dif = np.power(y_fit - img_xanes, 2)
    cost = np.sum(y_dif, axis=0) / img_xanes.shape[0]
    return cost


def compute_xanes_fit_mask(cost, error_thresh=0.1):
    mask = np.ones(cost.shape)
    mask[cost > error_thresh] = 0
    return mask


def xanes_fit_demo():
    import h5py
    f = h5py.File('img_xanes_normed.h5', 'r')
    img_xanes = np.array(f['img'])
    eng = np.array(f['X_eng'])
    f.close()
    img_xanes= bin_ndarray(img_xanes, (img_xanes.shape[0], int(img_xanes.shape[1]/2), int(img_xanes.shape[2]/2)))

    Ni = np.loadtxt('/NSLS2/xf18id1/users/2018Q1/MING_Proposal_000/xanes_ref/Ni_xanes_norm.txt')
    Ni2 = np.loadtxt('/NSLS2/xf18id1/users/2018Q1/MING_Proposal_000/xanes_ref/NiO_xanes_norm.txt')
    Ni3 = np.loadtxt('/NSLS2/xf18id1/users/2018Q1/MING_Proposal_000/xanes_ref/LiNiO2_xanes_norm.txt')

    spectrum_ref = load_xanes_ref(Ni2, Ni3)
    w1, c1 = fit_2D_xanes_non_iter(img_xanes, eng, spectrum_ref, error_thresh=0.1)
    plt.figure()
    plt.subplot(121); plt.imshow(w1[0])
    plt.subplot(122); plt.imshow(w1[1])



def normalize_2D_xanes_pre_edge(img_stack, xanes_eng, pre_edge):
    pre_s, pre_e = pre_edge
    img_norm = deepcopy(img_stack)
    s0 = img_norm.shape
    x_eng = xanes_eng

    xs, xe = find_nearest(x_eng, pre_s), find_nearest(x_eng, pre_e)
    if xs == xe:
        img_pre = img_norm[xs].reshape(1, s0[1], s0[2])
        img_pre = img_smooth(img_pre, 3)
        img_norm = img_norm - img_pre
    elif xe > xs:
        eng_pre = x_eng[xs:xe]
        img_pre = img_norm[xs:xe]
        img_pre = img_smooth(img_pre, 3)
        s = img_pre.shape
        x_pre = eng_pre.reshape(len(eng_pre), 1)
        x_bar_pre = np.mean(x_pre)
        x_dif_pre = x_pre - x_bar_pre
        SSx_pre = np.dot(x_dif_pre.T, x_dif_pre)
        y_bar_pre = np.mean(img_pre, axis=0)
        p = img_pre - y_bar_pre
        for i in range(s[0]):
            p[i] = p[i] * x_dif_pre[i]
        SSxy_pre = np.sum(p, axis=0)
        b0_pre = y_bar_pre - SSxy_pre / SSx_pre * x_bar_pre
        b1_pre = SSxy_pre / SSx_pre
        for i in range(s0[0]):
            if not i % 10:
                print(f'current image: {i}')
            img_norm[i] = img_norm[i] - (b0_pre + b1_pre * x_eng[i])
    img_norm = rm_abnormal(img_norm)
    return img_norm



def normalize_2D_xanes_post_edge(img_stack, xanes_eng, post_edge):
    post_s, post_e = post_edge
    img_norm = deepcopy(img_stack)
    s0 = img_norm.shape
    x_eng = xanes_eng
    xs, xe = find_nearest(x_eng, post_s), find_nearest(x_eng, post_e)
    if xs == xe:
        img_post = img_norm[xs].reshape(1, s0[1], s0[2])
        img_post = img_smooth(img_post, 3)
        img_norm = img_norm / img_post
        img_norm[np.isnan(img_norm)] = 0
        img_norm[np.isinf(img_norm)] = 0
    elif xe > xs:
        eng_post = x_eng[xs:xe]
        img_post = img_norm[xs:xe]
        img_post = img_smooth(img_post, 3)
        s = img_post.shape
        x_post = eng_post.reshape(len(eng_post), 1)
        x_bar_post = np.mean(x_post)
        x_dif_post = x_post - x_bar_post
        SSx_post = np.dot(x_dif_post.T, x_dif_post)
        y_bar_post = np.mean(img_post, axis=0)
        p = img_post - y_bar_post
        for i in range(s[0]):
            p[i] = p[i] * x_dif_post[i]
        SSxy_post = np.sum(p, axis=0)
        b0_post = y_bar_post - SSxy_post / SSx_post * x_bar_post
        b1_post = SSxy_post / SSx_post
        for i in range(s0[0]):
            tmp = np.abs(b0_post + b1_post * x_eng[i])
            tmp[tmp<1e-6] = 1e6
            img_norm[i] = img_norm[i] / tmp
    else:
        print('check pre-edge/post-edge energy')
    img_norm = rm_abnormal(img_norm)
    return img_norm



def normalize_2D_xanes_pre_edge_sub_mean(img_stack, xanes_eng, pre_edge, post_edge):
    img_norm = deepcopy(img_stack)
    x_eng = deepcopy(xanes_eng)
    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    xs = find_nearest(x_eng, pre_s)
    xe = find_nearest(x_eng, pre_e)
    tmp_pre = np.mean(img_norm[xs:max(xe, xs+1)], axis=0, keepdims=True)
    xs = find_nearest(x_eng, post_s)
    xe = find_nearest(x_eng, post_e)
    tmp_post = np.mean(img_norm[xs:max(xe, xs + 1)], axis=0, keepdims=True)
    img_pre_edge_sub_mean = tmp_post - tmp_pre
    img_pre_edge_sub_mean = rm_abnormal(img_pre_edge_sub_mean)
    return img_pre_edge_sub_mean



def normalize_2D_xanes_rm_abornmal(img_stack):

    img = deepcopy(img_stack)
    img_median = img_smooth(img, 3)
    img_flat = img.flatten()
    img_median_flat = img_median.flatten()
    img_flat_dif = np.abs(img_flat - img_median_flat)
    bad_pix_index = (img_flat_dif > 2)
    img_flat[bad_pix_index] = 0
    img_norm = img_flat.reshape(img.shape)

    return img_norm

def normalize_2D_xanes_rescale(img_stack, xanes_eng, pre_edge, post_edge):
    x_eng = xanes_eng
    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge

    img = deepcopy(img_stack)
    img_median = img_smooth(img, 3)
    img_flat = img.flatten()
    img_median_flat = img_median.flatten()
    img_flat_dif = np.abs(img_flat - img_median_flat)
    bad_pix_index = (img_flat_dif > 2)
    img_flat[bad_pix_index] = 0
    img_norm = img_flat.reshape(img.shape)


    xs, xe = find_nearest(x_eng, pre_s), find_nearest(x_eng, pre_e)
    img_pre_avg = np.mean(img_median[xs: max(xs+1, xe)], axis=0, keepdims=True)
    img_pre_avg = img_smooth(img_pre_avg, 3)

    xs, xe = find_nearest(x_eng, post_s), find_nearest(x_eng, post_e)
    img_post_avg = np.mean(img_median[xs: max(xs + 1, xe)], axis=0, keepdims=True)
    img_post_avg[np.abs(img_post_avg) < 1e-6] = 1e6
    img_post_avg = img_smooth(img_post_avg, 3)
    img_norm = (img_norm - img_pre_avg) / (img_post_avg - img_pre_avg)
    img_norm = rm_abnormal(img_norm)

    return img_norm



def normalize_2D_xanes_regulation(img_norm, x_eng, pre_edge, post_edge, designed_max=1.65, gamma=0.01):
    # using Lorentzian profile to re-scale
    #designed_max = 1.65
    #gamma = 0.01
    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    xs, xe = find_nearest(x_eng, pre_e), find_nearest(x_eng, post_s)
    tmp = img_norm[xs:max(xs+1, xe)]
    x = x_eng[xs:max(xs+1, xe)]
    y_max = np.max(tmp, axis=0, keepdims=True)
    index_max = np.argmax(tmp, axis=0)
    x0 = x[index_max]
    x0 = x0.reshape([1, x0.shape[0], x0.shape[1]])
    x = np.ones(img_norm.shape)
    for i in range(x.shape[0]):
        x[i] *= x_eng[i]
    y_l, y_l_max = L(x, x0, gamma)
    a = y_l_max / (y_max / designed_max - 1)
    y_l_modify = y_l / a + 1
    img_norm = img_norm / y_l_modify

    return img_norm


def normalize_2D_xanes_regulation_version2(img_norm, x_eng, peak_pos, peak_width=0.010, designed_max=1.65, gamma=0.01):
    # using Lorentzian profile to re-scale
    #designed_max = 1.65
    #gamma = 0.01

    xs, xe = find_nearest(x_eng, peak_pos-peak_width/2), find_nearest(x_eng, peak_pos+peak_width/2)
    tmp = img_norm[xs:max(xs+1, xe)]
    x = x_eng[xs:max(xs+1, xe)]
    y_max = np.max(tmp, axis=0, keepdims=True)
    index_max = np.argmax(tmp, axis=0)
    x0 = x[index_max]
    x0 = x0.reshape([1, x0.shape[0], x0.shape[1]])
    x = np.ones(img_norm.shape)
    for i in range(x.shape[0]):
        x[i] *= x_eng[i]
    y_l, y_l_max = L(x, x0, gamma)
    a = y_l_max / (y_max / designed_max - 1)
    y_l_modify = y_l / a + 1
    img_norm = img_norm / y_l_modify

    return img_norm



def normalize_2D_xanes(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag, method='new'):
    '''
    post_s, post_e = post_edge
    img_norm = deepcopy(img_stack)
    x_eng = xanes_eng
    img_pre_edge_sub_mean = normalize_2D_xanes_pre_edge_sub_mean(img_stack, xanes_eng, pre_edge, post_edge)
    '''

    '''
    # re-scale using the end point of post edge first

    xs = find_nearest(x_eng, post_s)
    xe = find_nearest(x_eng, post_e)
    tmp = np.mean(img_norm[xs:max(xe, xs+1)], axis=0, keepdims=True)
    tmp = img_smooth(tmp, 5)
    tmp = rm_abnormal(tmp)
    img_norm = img_norm / tmp
    img_norm = rm_abnormal(img_norm)
    img_norm[np.abs(img_norm)>10] = 0
    '''
    '''
    img_norm = normalize_2D_xanes_pre_edge(img_norm, x_eng, pre_edge)
    img_norm = normalize_2D_xanes_post_edge(img_norm, x_eng, post_edge)
    img_norm = normalize_2D_xanes_rescale(img_norm, xanes_eng, pre_edge, post_edge)
    return img_norm, img_pre_edge_sub_mean
    '''
    if method == 'new':
        img_norm, img_pre_edge_sub_mean = normalize_2D_xanes2(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag)
    else:
        img_norm, img_pre_edge_sub_mean = normalize_2D_xanes_old(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag)
    #img_norm = normalize_2D_xanes_rm_abornmal(img_norm)
    return img_norm, img_pre_edge_sub_mean



def normalize_2D_xanes_old(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag=0):
    '''
    post_s, post_e = post_edge
    img_norm = deepcopy(img_stack)
    x_eng = xanes_eng

    '''

    '''
    # re-scale using the end point of post edge first

    xs = find_nearest(x_eng, post_s)
    xe = find_nearest(x_eng, post_e)
    tmp = np.mean(img_norm[xs:max(xe, xs+1)], axis=0, keepdims=True)
    tmp = img_smooth(tmp, 5)
    tmp = rm_abnormal(tmp)
    img_norm = img_norm / tmp
    img_norm = rm_abnormal(img_norm)
    img_norm[np.abs(img_norm)>10] = 0
    '''
    img_pre_edge_sub_mean = normalize_2D_xanes_pre_edge_sub_mean(img_stack, xanes_eng, pre_edge, post_edge)
    img_norm = normalize_2D_xanes_pre_edge(img_stack, xanes_eng, pre_edge)
    if not pre_edge_only_flag: # normalizing pre-edge only
        img_norm = normalize_2D_xanes_post_edge(img_norm, xanes_eng, post_edge)
    #img_norm = normalize_2D_xanes_rescale(img_norm, xanes_eng, pre_edge, post_edge)
    return img_norm, img_pre_edge_sub_mean


def normalize_1D_xanes(xanes_spec, xanes_eng, pre_edge, post_edge):

    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    x_eng = xanes_eng
    xanes_spec_fit = deepcopy(xanes_spec)
    xs, xe = find_nearest(x_eng, pre_s), find_nearest(x_eng, pre_e)
    pre_eng = x_eng[xs:xe]
    pre_spec = xanes_spec[xs:xe]
    print(f'{pre_spec.shape}')
    if len(pre_eng) > 1:
        y_pre_fit = fit_curve(pre_eng, pre_spec, x_eng)
        xanes_spec_tmp = xanes_spec - y_pre_fit
        pre_fit_flag = True
    elif len(pre_eng) <= 1:
        y_pre_fit = np.ones(x_eng.shape) * xanes_spec[xs]
        xanes_spec_tmp = xanes_spec - y_pre_fit
        pre_fit_flag = True
    else:
        print('invalid pre-edge assignment')

    # fit post-edge
    xs, xe = find_nearest(x_eng, post_s), find_nearest(x_eng, post_e)
    post_eng = x_eng[xs:xe]
    post_spec = xanes_spec_tmp[xs:xe]
    if len(post_eng) > 1:
        y_post_fit = fit_curve(post_eng, post_spec, x_eng)
        post_fit_flag = True
    elif len(post_eng) <= 1:
        y_post_fit = np.ones(x_eng.shape) * xanes_spec_tmp[xs]
        post_fit_flag = True
    else:
        print('invalid pre-edge assignment')

    if pre_fit_flag and post_fit_flag:
        xanes_spec_fit = xanes_spec_tmp * 1.0 / y_post_fit
        xanes_spec_fit[np.isnan(xanes_spec_fit)] = 0
        xanes_spec_fit[np.isinf(xanes_spec_fit)] = 0

    return xanes_spec_fit, y_pre_fit, y_post_fit



'''
def normalize_2D_xanes2(img_stack, xanes_eng, pre_edge, post_edge):
    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    s = img_stack.shape
    xs_pre, xe_pre = find_nearest(xanes_eng, pre_s), find_nearest(xanes_eng, pre_e)
    xs_post, xe_post = find_nearest(xanes_eng, post_s), find_nearest(xanes_eng, post_e)

    xe_pre = max(xs_pre+1, xe_pre)
    xe_post = max(xs_post+1, xe_post)

    try:
        img_pre_mean = np.mean(img_stack[xs_pre:xe_pre], axis=0)
        img_post_mean = np.mean(img_stack[xs_post:xe_post], axis=0)
        img_post_mean = np.squeeze(img_smooth(img_post_mean, 5))
        img_post_flat = np.sort(img_post_mean.flatten())
        img_post_flat = img_post_flat[img_post_flat > 0]
        n_post = len(img_post_flat)
        thresh_post = img_post_flat[int(n_post * 0.8)]
        index_zero = img_post_mean < thresh_post
        num_non_zero = np.sum(np.sum(1 - np.array(index_zero, dtype=int), axis=0), axis=0)
        index_zero = np.repeat(index_zero.reshape([1, s[1], s[2]]), s[0], axis=0)

        img = deepcopy(img_stack)
        img[index_zero] = 0

        x = xanes_eng
        y = np.sum(np.sum(img, axis=1), axis=1)/num_non_zero
        x1 = x[xs_pre: xe_pre]
        y1 = y[xs_pre: xe_pre]
        coef1 = polyfit(x1, y1, 1)
        x1_mean = np.mean(x1)

        x2 = x[xs_post: xe_post]
        y2 = y[xs_post: xe_post]
        coef2 = polyfit(x2, y2, 1)
        coef2 = coef2 - coef1
        x2_mean = np.mean(x2)

        img_norm = deepcopy(img_stack)
        for i in range(s[0]):
            img_norm[i] = img_norm[i] - (coef1[1] * (x[i]-x1_mean) + img_pre_mean)
        for i in range(s[0]):
            img_norm[i] = img_norm[i] / (coef2[1] * (x[i]-x2_mean) + img_post_mean - img_pre_mean)

        img_thickness = img_post_mean - img_pre_mean
        img_norm = rm_abnormal(img_norm)
    except:
        img_norm = img_stack.copy()
        img_thickness = np.zeros([img_norm.shape[1], img_norm.shape[2]])
    return img_norm, img_thickness
'''

def normalize_2D_xanes2(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag=0):
    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    s = img_stack.shape
    xs_pre, xe_pre = find_nearest(xanes_eng, pre_s), find_nearest(xanes_eng, pre_e)
    xs_post, xe_post = find_nearest(xanes_eng, post_s), find_nearest(xanes_eng, post_e)

    xe_pre = max(xs_pre+1, xe_pre)
    xe_post = max(xs_post+1, xe_post)
    img_pre_mean = np.mean(img_stack[xs_pre:xe_pre], axis=0)

    img_post_mean = np.mean(img_stack[xs_post:xe_post], axis=0)
    img_post_mean = np.squeeze(img_smooth(img_post_mean, 5))
    img_post_flat = np.sort(img_post_mean.flatten())
    img_post_flat = img_post_flat[img_post_flat > 0]
    n_post = len(img_post_flat)
    thresh_post = img_post_flat[int(n_post * 0.8)]
    index_zero = img_post_mean < thresh_post
    num_non_zero = np.sum(np.sum(1 - np.array(index_zero, dtype=int), axis=0), axis=0)
    index_zero = np.repeat(index_zero.reshape([1, s[1], s[2]]), s[0], axis=0)

    img = deepcopy(img_stack)
    img[index_zero] = 0

    x = xanes_eng
    y = np.sum(np.sum(img, axis=1), axis=1)/num_non_zero
    x1 = x[xs_pre: xe_pre]
    y1 = y[xs_pre: xe_pre]
    coef1 = polyfit(x1, y1, 1)
    x1_mean = np.mean(x1)

    x2 = x[xs_post: xe_post]
    y2 = y[xs_post: xe_post]
    coef2 = polyfit(x2, y2, 1)
    coef2 = coef2 - coef1
    x2_mean = np.mean(x2)

    img_norm = deepcopy(img_stack)
    for i in range(s[0]):
        img_norm[i] = img_norm[i] - (coef1[1] * (x[i]-x1_mean) + img_pre_mean)
    img_pre_mean = np.mean(img_norm[xs_pre:xe_pre], axis=0)
    img_post_mean = np.mean(img_norm[xs_post:xe_post], axis=0)
    img_post_mean = np.squeeze(img_smooth(img_post_mean, 5))
    if not pre_edge_only_flag: # normalizing pre-edge only
        for i in range(s[0]):
            tmp  = coef2[1] * (x[i]-x2_mean) + img_post_mean - img_pre_mean
            tmp[tmp <= 0] = 1e6
            img_norm[i] = img_norm[i] / tmp

    img_thickness = img_post_mean - img_pre_mean
    img_thickness[img_thickness<0] = 0
    img_norm = rm_abnormal(img_norm)
    return img_norm, img_thickness


