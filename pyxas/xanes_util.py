from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pyxas.lsq_fit import *
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, UnivariateSpline
from copy import deepcopy
from numpy.polynomial.polynomial import polyfit, polyval
from pyxas.image_util import rm_abnormal, bin_ndarray, img_smooth
from scipy.optimize import nnls, lsq_linear
from tqdm import trange, tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import xraylib



def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()

def exclude_eng(img_raw, x_eng_raw, exclude_range=[7.7, 7.8]):
    id1 = find_nearest(x_eng_raw, exclude_range[0])
    id2 = find_nearest(x_eng_raw, exclude_range[-1])
    n = len(x_eng_raw)
    all_idx = set(list(np.arange(n)))
    ex_idx = set(list(np.arange(id1, id2)))
    idx = np.sort(list(all_idx-ex_idx))
    img = img_raw[idx]
    x_eng = x_eng_raw[idx]
    return img, x_eng

def fit_using_nnls(y, A, bounds, method='lsq'):
    if method == 'lsq':
        res = lsq_linear(A, y, bounds)
        return res['x']
    else:
        res = nnls(A, y)
        return res[0]

def fit1D(x_raw, y_raw, x_fit, order=3, smooth=0.01):
    spl = UnivariateSpline(x_raw, y_raw, k=order, s=smooth)
    y_fit = spl(x_fit)
    return y_fit


def exclude_data(x, y, exclude=[]):
    if len(exclude) == 2:
        id1 = find_nearest(x, exclude[0])
        id2 = find_nearest(x, exclude[1])
        y0 = list(y[:id1]) + list(y[id2:])
        y0 = np.array(y0)
    else:
        y0 = y
    return y0


def fit_element_mu(x_eng, y_spec, mu_raw):
    # for single curve, y_spec can be an array

    # for 2D image, y_spec.shape = (n_eng, total_pix)
    mu = np.array(mu_raw)
    if len(mu.shape) == 1:
        mu = np.expand_dims(mu, 1)

    x0 = x_eng.copy()
    y0 = y_spec.copy()
    num = len(x0)
    if len(y0.shape) == 1:
        Y = y0.reshape((num, 1))
    else:
        Y = y_spec.copy()
    n_mu = mu.shape[1]
    A = np.ones([num, n_mu+2])
    A[:, :n_mu] = mu
    A[:, -2] = x0

    A_inv = scipy.linalg.inv(A.T @ A)
    X = A_inv @ A.T @ Y
    return X, A


def fit_multi_element_mu(img_xanes, xanes_eng, elem, exclude_multi_range, bkg_polynomial_order):
    order = bkg_polynomial_order
    img = img_xanes.copy()
    x_eng = xanes_eng.copy()
    x_eng_all = xanes_eng.copy()
    n_exclude = len(exclude_multi_range) // 2
    idx = 0
    for i in range(n_exclude):
        eng_s = exclude_multi_range[idx]
        eng_e = exclude_multi_range[idx+1]
        img, x_eng = exclude_eng(img, x_eng, [eng_s, eng_e])
        idx += 2
    s = img.shape
    s_all = img_xanes.shape
    Y = img.reshape(s[0], s[1]*s[2])
    Y_all = img_xanes.reshape(s_all[0], s_all[1]*s_all[2])

    n_eng_all = len(x_eng_all)
    n_eng = len(x_eng)
    n_elem = len(elem)
    n_order = len(order)

    cs = {}
    cs_all = {}
    for i in range(n_elem):
        cs[elem[i]] = np.zeros(n_eng)
        cs_all[elem[i]] = np.zeros(n_eng_all)
        for j in range(n_eng):
            cs[elem[i]][j] = xraylib.CS_Energy(xraylib.SymbolToAtomicNumber(elem[i]), x_eng[j])
            #cs[elem[i]][j] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(elem[i]), x_eng[j])
        for j in range(n_eng_all):
            cs_all[elem[i]][j] = xraylib.CS_Energy(xraylib.SymbolToAtomicNumber(elem[i]), x_eng_all[j])
            #cs_all[elem[i]][j] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(elem[i]), x_eng_all[j])

    A = np.zeros((n_eng, n_order+n_elem))
    for i in range(n_elem):
        A[:, i] = cs[elem[i]]
    for i in range(n_order):
        A[:, i+n_elem] = x_eng ** (order[i])

    A_all = np.zeros((n_eng_all, n_order + n_elem))
    for i in range(n_elem):
        A_all[:, i] = cs_all[elem[i]]
    for i in range(n_order):
        A_all[:, i + n_elem] = x_eng_all ** (order[i])

    AT = A.T
    ATA = AT @ A
    ATA_inv = np.linalg.inv(ATA)
    X = ATA_inv @ AT @ Y
    Y_fit = A @ X
    Y_diff = Y - Y_fit

    Y_fit_all = A_all @ X
    return X, A, A_all, x_eng, Y, Y_fit, Y_diff, x_eng_all, Y_all, Y_fit_all


def fit_xanes_curve_with_bkg(x_eng, y_spec, exclude_eng, plot_flag, spectrum_ref):
    # asssume linear background
    # ref: reference spectrum should have same energy as x_eng

    x0 = exclude_data(x_eng, x_eng, exclude_eng)
    y0 = exclude_data(x_eng, y_spec, exclude_eng)

    num = len(x0)
    n_ref = len(spectrum_ref)
    ref0 = np.zeros((num, n_ref))
    for i in range(n_ref):
        current_ref = spectrum_ref[f'ref{i}']
        ref0[:, i] = fit1D(current_ref[:, 0], current_ref[:, 1], x_eng)

    X, _ = fit_element_mu(x0, y0, ref0)

    A = np.ones([num, n_ref+2])
    A[:, :n_ref] = ref0
    A[:, -2] = x0

    y_fit = A @ X
    title = ''
    for i in range(n_ref):
        title = title + f'ref #{i}: {X[i, 0]:2.2f},   '
    if plot_flag:
        plt.figure()
        plt.plot(x_eng, y_spec, 'r.', label='raw data')
        plt.plot(x_eng, y_fit, 'c', label='fitted')
        plt.legend()
        plt.title(title)

    return X, y_fit, A


def fit_2D_xanes_with_bkg(x_eng, img_xanes, exclude_eng, plot_flag, spectrum_ref):
    # img_xanes.shape = (n_eng, R, C)
    s = img_xanes.shape
    y_spec = img_xanes.reshape((s[0], s[1]*s[2]))
    X, y_fit, A = fit_xanes_curve_with_bkg(x_eng, y_spec, exclude_eng, 0, spectrum_ref)
    n_ref = len(spectrum_ref)
    if plot_flag:
        plt.figure()
        for i in range(n_ref):
            plt.subplot(1,n_ref, i+1)
            t = X[i].reshape((s[1], s[2]))
            plt.imshow(t)
            plt.title(f'ref #{i}')
    x = X[:n_ref]
    x = x.reshape(n_ref, s[1], s[2])
    cost = np.sum((y_fit - y_spec)**2, axis=0) / s[0]
    cost = cost.reshape(1, s[1], s[2])
    offset = X[-1]
    offset = offset.reshape(1, s[1], s[2])
    slope = X[-2]

    thickness = y_fit[-1] - y_fit[0]
    thickness = thickness - slope * (x_eng[-1] - x_eng[0])
    thickness = np.abs(np.arctan(slope) * thickness)
    thickness = thickness.reshape(1, s[1], s[2])
    slope = slope.reshape(1, s[1], s[2])
    y_fit = y_fit.reshape(s[0], s[1], s[2])
    return x, offset, cost, thickness, y_fit, slope


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

def fit_2D_xanes_basic(img_xanes, eng, spectrum_ref, bkg_polynomial_order):
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
    # Y = bX + b0: X is the reference spectrum
    bkg_polynomial_order = list(np.sort(bkg_polynomial_order))
    s = img_xanes.shape
    num_ref = len(spectrum_ref)
    num_eng = len(eng)
    num_order = len(bkg_polynomial_order)

    A = np.ones((num_eng, num_ref+num_order))
    A[:, :num_ref] = interp_spec0(spectrum_ref, eng)

    for i in range(num_order):
        A[:, i + num_ref] = eng ** (bkg_polynomial_order[i])

    AT = A.T
    ATA_inv = np.linalg.inv(AT @ A)
    Y = img_xanes.reshape((s[0], s[1]*s[2]))
    ATY = AT @ Y
    X = ATA_inv @ ATY

    Y_hat = A @ X
    dy = Y - Y_hat
    cost = np.sum(dy**2, axis=0) / num_eng
    cost = cost.reshape((s[1], s[2]))
    fit_coef = X[:num_ref].reshape((num_ref, s[1], s[2]))

    X_offset = X.copy()
    X_offset[:num_ref] = 0
    Y_offset = A @ X_offset
    Y_offset = Y_offset.reshape((num_eng, s[1], s[2]))

    err = np.sum(dy**2, axis=0)
    n_freedim = max(num_eng - (num_ref + num_order), 1)
    sigma2 = err / n_freedim
    ATA_inv_diag = np.diag(ATA_inv)
    nA = len(ATA_inv_diag)
    var2 = np.ones((nA, len(sigma2)))
    for i in range(nA):
        var2[i] = ATA_inv_diag[i] * sigma2
    var = np.sqrt(var2)
    var = var.reshape((num_ref+num_order, s[1], s[2]))
    return fit_coef, cost, X, Y_hat, Y_offset, var

"""
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
"""

def interp_spec(ref_spec, ref_eng, xanes_eng):
    s = ref_spec.shape
    n_ref = s[1]
    n_eng = len(xanes_eng)
    ref_spec_interp = np.zeros((n_eng, n_ref))
    for i in range(n_ref):
        f = InterpolatedUnivariateSpline(ref_eng, ref_spec[:, i], k=3)
        ref_spec_interp[:, i] = f(xanes_eng)
    return ref_spec_interp


def interp_spec0(spectrum_ref, xanes_eng):
    n_ref = len(spectrum_ref)
    n_eng = len(xanes_eng)
    ref_spec_interp = np.zeros((n_eng, n_ref))
    for i in range(n_ref):
        f = InterpolatedUnivariateSpline(spectrum_ref[f'ref{i}'][:, 0], spectrum_ref[f'ref{i}'][:, 1], k=3)
        ref_spec_interp[:, i] = f(xanes_eng)
    return ref_spec_interp


def fit_2D_xanes_admm(img_xanes, eng, spectrum_ref, learning_rate=0.2, n_iter=50, bounds=[0,1e10], bkg_polynomial_order=[0]):
    bkg_polynomial_order = list(np.sort(bkg_polynomial_order))
    s = img_xanes.shape
    num_ref = len(spectrum_ref)
    num_eng = len(eng)
    num_order = len(bkg_polynomial_order)

    low_bounds = [bounds[0]] * num_ref
    high_bounds = [bounds[1]] * num_ref

    A = np.ones((num_eng, num_ref+num_order))
    A[:, :num_ref] = interp_spec0(spectrum_ref, eng)

    for i in range(num_order):
        A[:, i + num_ref] = eng ** (bkg_polynomial_order[i])
        if bkg_polynomial_order[i] == 0:
            low_bounds.append(0)
        else:
            low_bounds.append(-1e12)
        high_bounds.append(1e12)

    Y = img_xanes.reshape((s[0], s[1]*s[2]))
    X = admm_iter2(A, Y, learning_rate, n_iter, low_bounds, high_bounds)
    Y_hat = A @ X
    dy = Y - Y_hat
    cost = np.sum(dy**2, axis=0) / num_eng
    cost = cost.reshape((s[1], s[2]))
    fit_coef = X[:num_ref].reshape((num_ref, s[1], s[2]))

    X_offset = X.copy()
    X_offset[:num_ref] = 0
    Y_offset = A @ X_offset
    Y_offset = Y_offset.reshape((num_eng, s[1], s[2]))

    AT = A.T
    ATA = AT @ A
    ATA_inv = np.linalg.inv(ATA)

    err = np.sum(dy ** 2, axis=0)
    n_freedim = max(num_eng - (num_ref + num_order), 1)
    sigma2 = err / n_freedim
    ATA_inv_diag = np.diag(ATA_inv)
    nA = len(ATA_inv_diag)
    var2 = np.ones((nA, len(sigma2)))
    for i in range(nA):
        var2[i] = ATA_inv_diag[i] * sigma2
    var = np.sqrt(var2)
    var = var.reshape((num_ref + num_order, s[1], s[2]))

    return fit_coef, cost, X, Y_hat, Y_offset, var

def fit_2D_xanes_nnls(img_xanes, eng, spectrum_ref, n_iter=50, bkg_polynomial_order=[-3]):
    bkg_polynomial_order = list(np.sort(bkg_polynomial_order))
    s = img_xanes.shape
    num_ref = len(spectrum_ref)
    num_eng = len(eng)
    num_order = len(bkg_polynomial_order)

    if num_order == 1 and bkg_polynomial_order[0] == 0:
        method = 'nnls'
    else:
        method = 'lsq'

    low_bounds = [0] * num_ref
    high_bounds = [1e12] * num_ref

    A = np.ones((num_eng, num_ref+num_order))
    A[:, :num_ref] = interp_spec0(spectrum_ref, eng)

    for i in range(num_order):
        A[:, i + num_ref] = eng ** (bkg_polynomial_order[i])
        low_bounds.append(-np.inf)
        high_bounds.append(np.inf)

    Y = img_xanes.reshape((s[0], s[1]*s[2])) # e.g., (101, 120000)
    Yt = Y.T # (120000, 101)
    bounds = (low_bounds, high_bounds)

    partial_fun = partial(fit_using_nnls, A=A, bounds=bounds, method=method)
    num_cpu = 4 # int(cpu_count() * 0.8)
    pool = Pool(num_cpu)
    res = []
    for result in tqdm(pool.imap(func=partial_fun, iterable=Yt), total=len(Yt)):
        res.append(result)
    pool.close()
    pool.join()
    X = np.array(res).T # (5, 120000)

    Y_hat = A @ X
    dy = Y - Y_hat
    cost = np.sum(dy ** 2, axis=0) / num_eng
    cost = cost.reshape((s[1], s[2]))

    fit_coef = X[:num_ref].reshape((num_ref, s[1], s[2]))
    X_offset = X.copy()
    X_offset[:num_ref] = 0
    Y_offset = A @ X_offset
    Y_offset = Y_offset.reshape((num_eng, s[1], s[2]))

    AT = A.T
    ATA = AT @ A
    ATA_inv = np.linalg.inv(ATA)

    err = np.sum(dy ** 2, axis=0)
    n_freedim = max(num_eng - (num_ref + num_order), 1)
    sigma2 = err / n_freedim
    ATA_inv_diag = np.diag(ATA_inv)
    nA = len(ATA_inv_diag)
    var2 = np.ones((nA, len(sigma2)))
    for i in range(nA):
        var2[i] = ATA_inv_diag[i] * sigma2
    var = np.sqrt(var2)
    var = var.reshape((num_ref + num_order, s[1], s[2]))

    return fit_coef, cost, X, Y_hat, Y_offset

"""
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

"""
"""
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
"""
"""
def compute_xanes_fit_mask(cost, error_thresh=0.1):
    mask = np.ones(cost.shape)
    mask[cost > error_thresh] = 0
    return mask
"""


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

    if method == 'new':
        img_norm, img_pre_edge_sub_mean = normalize_2D_xanes2(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag)
    else:
        img_norm, img_pre_edge_sub_mean = normalize_2D_xanes_old(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag)
    return img_norm, img_pre_edge_sub_mean


def normalize_2D_xanes_old(img_stack, xanes_eng, pre_edge, post_edge, pre_edge_only_flag=0):
    '''
    post_s, post_e = post_edge
    img_norm = deepcopy(img_stack)
    x_eng = xanes_eng

    '''
    img_pre_edge_sub_mean = normalize_2D_xanes_pre_edge_sub_mean(img_stack, xanes_eng, pre_edge, post_edge)
    img_norm = normalize_2D_xanes_pre_edge(img_stack, xanes_eng, pre_edge)
    if not pre_edge_only_flag: # normalizing pre-edge only
        img_norm = normalize_2D_xanes_post_edge(img_norm, xanes_eng, post_edge)
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


def fit_peak_2D_xanes_poly(img_xanes, xanes_eng, eng_range=[],fit_order=3, fit_max=1):
    # from multiprocessing import Pool
    # from functools import partial
    img = img_xanes.copy()
    try:
        xs_id = find_nearest(xanes_eng, eng_range[0])
        xe_id = find_nearest(xanes_eng, eng_range[1])
        img = img[xs_id:xe_id]
        if len(xanes_eng):
            x = xanes_eng[xs_id:xe_id]
        else:
            x = np.arange(len(img))
    except Exception as err:
        print(err)
        img = img_xanes.copy()
    try:
        x = xanes_eng[xs_id:xe_id]
    except Exception as err:
        print(err)
        x = np.arange(len(img))
    s0 = img.shape
    img_f = img.reshape([s0[0], -1])
    if fit_max:
        y = img_f
    else:
        y = 1 - img_f
    res = fit_peak_curve_poly(x, y, fit_order)
    peak_pos = res['peak_pos'].reshape([1, s0[1], s0[2]])
    fit_error = res['fit_error'].reshape([1, s0[1], s0[2]])
    peak_val = res['peak_val'].reshape([1, s0[1], s0[2]])
    return peak_pos, peak_val, fit_error


def rm_duplicate(my_list):
    id_list = []
    new_list = []
    n = len(my_list)
    for i in range(n):
        if not my_list[i, 0] in id_list:
            new_list.append(my_list[i])
            id_list.append(my_list[i, 0])
    return np.array(new_list)
