import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import trange
from scipy.interpolate import UnivariateSpline
from .image_util import img_denoise_nl, img_denoise_bm3d

def fit_peak_curve_spline(x, y, fit_order=3, smooth=0.002, weight=[1]):
    if not len(weight) == len(x):
        weight = np.ones((len(x)))
    spl = UnivariateSpline(x, y, k=fit_order, s=smooth, w=weight)
    xx = np.linspace(x[0], x[-1], 10001)
    yy = spl(xx)
    peak_pos = xx[np.argmax(yy)]
    fit_error = np.sum((y - spl(x)**2))
    edge_pos = xx[np.argmax(np.abs(np.diff(spl(xx))))]
    res = {}
    res['peak_pos'] = peak_pos
    res['peak_val'] = spl(peak_pos)
    res['edge_pos'] = edge_pos
    res['edge_val'] = spl(edge_pos)
    res['fit_error'] = fit_error
    res['spl'] = spl
    res['xx'] = xx
    return res


def fit_peak_curve_poly(x, y, fit_order=3, num=1001):
    '''
    x, y can be matrix
    '''
    try:
        import cupy as cp
        nnp = cp
        ntype = 'cp'
        y = cp.asarray(y)
        x = cp.asarray(x)
    except:
        nnp = np
        ntype = 'np'
    x_min, x_max = nnp.min(x), nnp.max(x)
    s1 = len(y)
    if len(y.shape) == 1:
        Y = y.reshape([s1, 1])
    else:
        Y = y
    if len(x.shape) == 1:
        x0 = x.reshape([s1, 1])
    else:
        x0 = x
    x0 = (x0 - x_min) / (x_max - x_min)
    X = nnp.ones([s1, 1])
    for i in nnp.arange(1, fit_order + 1):
        X = nnp.concatenate([X, x0 ** i], 1)
    
    
    A = nnp.linalg.inv(X.T @ X) @ (X.T @ Y)
    xx = nnp.linspace(x0[0], x0[-1], num).reshape([num, 1])
    XX = nnp.ones([num, 1])
    for i in nnp.arange(1, fit_order + 1):
        XX = nnp.concatenate([XX, xx ** i], 1)
    YY = XX @ A
    peak_pos = xx[nnp.argmax(YY, 0)] * (x_max - x_min) + x_min
    y_hat = X @ A
    fit_error = nnp.sum((y_hat - Y)**2, 0)
    
    """
    ## fitting covariance
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    XTX_inv_diag = np.diag(XTX_inv)
    n_freedim = max(fit_order-2, 1)
    nX = len(XTX_inv_diag)
    sigma2 = fit_error / n_freedim
    var2 = np.ones((nX, len(sigma2)))
    for i in range(nX):
        var2[i] = XTX_inv_diag[i] * sigma2
    var = np.sqrt(var2)
    """
    
    res = {}
    if ntype == 'np':
        res['peak_pos'] = peak_pos
        res['peak_val'] = np.max(YY, 0)
        res['fit_error'] = fit_error
        res['matrix_X'] = XX
        res['matrix_A'] = A
        res['matrix_Y'] = YY
        res['x_interp'] = xx
        #res['var'] = var
    else:
        res['peak_pos'] = peak_pos.get()
        res['fit_error'] = fit_error.get()
        res['matrix_X'] = XX.get()
        res['matrix_A'] = A.get()
        res['peak_val'] = cp.max(YY, 0).get()
        res['matrix_Y'] = YY.get()
        #res['var'] = var.get()
    return res


def forward_propgation(W, X_prev):
    return np.dot(W, X_prev)

def compute_cost(Y, y_hat, W, f_scale=10):
    a = np.float32(np.dot((Y-y_hat), (Y-y_hat).T))/len(Y)
    b = np.abs(np.sum(W)-1)*f_scale
    cost = a + b
    cost = np.squeeze(cost)
    return cost

def compute_cost0(Y, y_hat):
    cost = np.float32((np.dot((Y-y_hat), (Y-y_hat).T)))/len(Y)
    cost = np.squeeze(cost)
    return cost

####################

def backpropagate(Y_test, A, W, B, f_scale1=1, f_scale2=1):
    Y_hat = np.dot(A, W) + B
    m = Y_test.shape[0]
    n = Y_test.shape[1]
    dy = Y_test - Y_hat
    #cost = (np.sum(dy**2, axis=0) + (np.sum(W, axis=0) - 1)**2)
    cost = np.sum(dy ** 2, axis=0) / m
    dw1 = f_scale1 * np.abs((np.sum(W, axis=0) - 1)) / m
    tmp = W.copy()
    tmp[tmp>0] = 0
    dw2 = f_scale2 * np.abs(tmp)
    #dw2 = f_scale * np.abs(W)
    #dw2 = f_scale2 * np.sum(W**2, axis=1, keepdims=True) / n
    #dw = -2 * (np.dot(A.T, dy) + dw1 + dw2)
    dw = -2 * (np.dot(A.T, dy) + dw1 + dw2)
    db = -2 * np.sum(dy, axis=0)
    db = db.reshape(B.shape)
    return dw, db, cost


def backpropagate0(Y_test, A, W, B, f_scale2=0):
    # f_scale2 is used for control non-negative fitting
    Y_hat = np.dot(A, W) + B
    m = Y_test.shape[0]
    dy = Y_test - Y_hat
    cost = np.sum(dy**2, axis=0)/m
    tmp = W.copy()
    tmp[tmp>0] = 0
    dw2 = f_scale2 * np.abs(tmp)
    # dw = -2 * (np.dot(A.T, dy))/m
    dw = -2 * (np.dot(A.T, dy) + dw2)
    db = (-2 * np.sum(dy, axis=0)).reshape(B.shape)
    return dw, db, cost


#################

def norm_W(W):
    wnorm = W/np.sum(W, axis=1)
    return wnorm



def lsq_fit_iter(X, Y, W=None, learning_rate=0.002, n_iter=100, bounded=True, print_flag=1):
    if W is None:
        W = np.random.random([Y.shape[0], X.shape[0]])/X.shape[0]
        W[:,-1] = 1 - np.sum(W[:,:-1], axis=1)
    Y_test = deepcopy(Y)
    cost = []
    for i in range(n_iter):
        if print_flag and not i%50:
            print('iter #{}'.format(i))
        y_hat = np.dot(W, X)
        if bounded:
            cost_temp = compute_cost(Y, y_hat, W)
        else:
            cost_temp = compute_cost0(Y, y_hat)
        dy = y_hat - Y_test
        dw = np.dot(dy, X.T)
        W -= dw*learning_rate
        mask = (W > 0)
        W = W * mask
#        W = norm_W(W)
        cost.append(cost_temp)
    W = np.squeeze(W)
    cost = np.squeeze(np.array(cost))
    return W, cost



def lsq_fit_iter2(A, Y, W=None, B=0, learning_rate=0.002, n_iter=100, bounds=[0,1], f_scale1=0.5, B_update=1):
    # solve AW + B = Y (W and B need to be solved)

    if W is None:
        W = np.random.random([A.shape[1], Y.shape[1]])/A.shape[1]
        W[-1,:] = 1 - np.sum(W[:-1,:], axis=0)
    #else:
    #    if len(bounds)==2:
    #        W[W <= bounds[0]] = bounds[0]
    #        W[W >= bounds[1]] = bounds[1]

    Y_test = deepcopy(Y)
    cost = []
    for i in range(n_iter):
        # if print_flag and not i%50:
        print('iter #{}'.format(i))

        if len(bounds)==2:
            f_scale2 = 0.1/learning_rate
            dw, db, cost_temp = backpropagate(Y_test, A, W, B, f_scale1=f_scale1, f_scale2=f_scale2)
            tmp = dw * learning_rate
            index = np.abs(tmp) > 0.5 * np.abs(W)
            tmp[index] = 0.5 * np.abs(W[index])
            W -= tmp
            B -= db*learning_rate * B_update
            W[W <= bounds[0]] = bounds[0]
            W[W >= bounds[1]] = bounds[1]
        elif len(bounds)==0:
            dw, db, cost_temp = backpropagate0(Y_test, A, W, B)
            W -= dw*learning_rate
            W[W < 0] = 0
            B -= db * learning_rate * B_update
        cost.append(cost_temp)
    if len(bounds) == 2:
        W[W <= bounds[0]] = bounds[0]
        W[W >= bounds[1]] = bounds[1]
    W = np.squeeze(W)
    B = np.squeeze(B)
    cost = np.array(cost)
    return W, B, cost

def rm_neg_coef(W):
    w = W.copy()
    s = w.shape
    n = s[0]
    for i in range(n):
        idx = np.where(w[i] < 0)
        v = w[i][idx]
        for j in range(n):
            if j == i:
                continue
            else:
                w[j][idx] += v / (n-1)
    return w

#################################################
######### Coordinate descent and ADMM ###########
#################################################


def soft_threshold(rho, lamda):
    '''Soft threshold function used for normalized data and lasso regression'''
    y = np.sign(rho) * np.max(np.abs(rho) - lamda, 0)
    '''

        if rho < - lamda:
            theta = (rho + lamda)
        elif rho >  lamda:
            theta = (rho - lamda)
        else:
            theta = 0
    '''
    return y


def coordinate_descent_lasso(A, y, lamda=.01, num_iters=10, X_guess=[], intercept=False, bounds=[]):
    '''
    Coordinate gradient descent for lasso regression - for normalized data.
    The intercept parameter allows to specify whether or not we regularize theta_0
    '''

    # Initialisation of useful values
    n_eng, n_spectra = A.shape
    n_sample = y.shape[1]

    if not len(X_guess) or not X_guess.shape == (n_spectra, n_sample):
        x = np.ones((n_spectra, n_sample))
    else:
        x = X_guess.copy()

    if not len(bounds) == 2:
        bounds = [-1e16, 1e16]

    for i in range(num_iters):
        print(f'iter = {i}')
        # Looping through each coordinate
        for j in range(n_spectra):
            # Vectorized implementation
            A_j = A[:, j].reshape(-1, 1)
            y_pred = A @ x
            r_j = y - y_pred + x[j] * A_j
            rho_j = (A_j.T @ r_j)
            z_j = (A_j.T @ A_j)

            # Checking intercept parameter
            if intercept == True:
                if j == 0:
                    x[j] = rho_j / z_j
                else:
                    x[j] = soft_threshold(rho_j, lamda) / z_j
                    x[j][x[j] <= bounds[0]] = bounds[0]
                    x[j][x[j] >= bounds[1]] = bounds[1]
            if intercept == False:
                x[j] = soft_threshold(rho_j, lamda) / z_j
                x[j][x[j] <= bounds[0]] = bounds[0]
                x[j][x[j] >= bounds[1]] = bounds[1]
    return x


def admm_iter(A, y, rho=0.2, num_iters=10, X_guess=[], wgt=[], lasso_lamda=0.01, bounds=[]):
    '''
    bound constrain fitting based on ADMM
    '''
    from numpy.linalg import inv, norm
    n_eng, n_spectra = A.shape
    n_sample = y.shape[1]

    if not len(X_guess) or not X_guess.shape == (n_spectra, n_sample):
        z = np.ones((n_spectra, n_sample))
    else:
        z = X_guess.copy()

    if not len(wgt) == n_eng:
        wgt_flag = 0
        c = A.T @ A + np.eye(n_spectra) * rho
    else:
        wgt_flag = 1
        c = A.T @ np.diag(wgt) @ A + np.eye(n_spectra) * rho


    if not len(bounds) == 2:
        bounds = [-1e16, 1e16]
    # initialize
    u = np.zeros(z.shape)

    convergency = np.zeros((num_iters, 1))

    for i in range(num_iters):
        print(f'iter #{i}')
        temp = z.copy()
        if wgt_flag:
            x = inv(c) @ (A.T @ np.diag(wgt) @ y + rho * (z - u))
        else:
            x = inv(c) @ (A.T @ y + rho * (z - u))

        # z step: denoising
        if np.abs(rho) < 1e-4:
            z = x + u
        else:
            z = soft_threshold(x + u, lasso_lamda / rho)
        z[z <= bounds[0]] = bounds[0]
        z[z >= bounds[1]] = bounds[1]
        u = u + x - z
        convergency[i] = (norm(z.flatten()) - norm(temp.flatten())) / norm(z.flatten())
    return x

def admm_iter2(A, y, rate=0.2, maxiter=100, low_bounds=[0], high_bounds=[1e12], epsilon=1e-16, first_n_term=None):
    At = A.T
    z = At @ y
    c = At @ A

    ATA = At @ A
    ATA_inv = np.linalg.inv(ATA)

    n_ref = A.shape[1]
    n_pix = y.shape[1]

    # initialize variables
    w = np.ones((n_ref, n_pix))
    u = np.zeros((n_ref, n_pix))

    convergence = np.zeros(maxiter)

    dg = np.eye(n_ref) * rate
    m1 = np.linalg.inv((c + dg))

    n_iter = 0

    # lower bounds
    lb = list(low_bounds) + [-1e12] * n_ref
    lb = lb[:n_ref]

    # high bounds
    hb = list(high_bounds) + [1e12] * n_ref
    hb = hb[:n_ref]

    # initialize using Least-square-fitting
    # x = ATA_inv @ At @ y
    for i in trange(maxiter):
        m2 = z + (w - u) * rate
        x = np.matmul(m1, m2)
        w_updated = x + u

        # apply bounds
        w_updated = clip_with_bounds(w_updated, lb, hb, first_n_term)
        u = u + x - w_updated

        conv = np.linalg.norm(w_updated - w) / np.linalg.norm(w_updated)
        convergence[i] = conv
        w = w_updated
        if conv < epsilon:
            n_iter = i + 1
            break
        #m2 = z + (w - u) * rate
        #x = np.matmul(m1, m2)
    convergence = convergence[:n_iter]
    w = clip_with_bounds(w_updated, lb, hb, first_n_term)
    return w

def admm_denoise(A, y, s_2d, n_ref, rho=0.2, num_iters=4, wgt=[], bounds=[], method='nl', sigma=0.1):
    '''
    bound constrain fitting based on ADMM
    '''
    from numpy.linalg import inv, norm
    n_eng, n_spectra = A.shape
    n_sample = y.shape[1]

    z = np.zeros((n_spectra, n_sample))
    u = np.zeros((n_spectra, n_sample))

    offset = np.eye(n_spectra)
    #for i in range(n_ref, n_spectra):
    #    offset[i] = 0


    if not len(wgt) == n_eng:
        wgt_flag = 0
        c = A.T @ A + offset * rho
    else:
        wgt_flag = 1
        c = A.T @ np.diag(wgt) @ A + offset * rho

    if not len(bounds) == 2:
        bounds = [-1e16, 1e16]

    convergency = np.zeros((num_iters, 1))

    for i in range(num_iters):
        print(f'iter #{i}')
        temp = z.copy()
        if wgt_flag:
            x = inv(c) @ (A.T @ np.diag(wgt) @ y + rho * (z - u))
        else:
            x = inv(c) @ (A.T @ y + rho * (z - u))

        # z step: denoising
        z = z_denoise(x, s_2d, n_spectra, method, sigma)
        for i in range(n_ref):
            z[i][z[i] <= bounds[0]] = bounds[0]
            z[i][z[i] >= bounds[1]] = bounds[1]
        u = u + x - z
        convergency[i] = (norm(z[:n_ref].flatten()) - norm(temp[:n_ref].flatten())) / norm(z[:n_ref].flatten())
        if convergency[i] < 1e-3:
            break
    return z



def z_denoise(z, s_2d, first_n_term, method='nl', sigma=0.1):
    s0 = z.shape
    n = s0[0]
    z1 = z.copy()
    z1 = z1.reshape((n, *s_2d))
    for i in range(first_n_term):
        img = z1[i]
        f = np.max(img)
        if method == 'bm3d':
            img_d = img_denoise_bm3d(img/f, sigma=sigma) * f
        else:
            img_d = img_denoise_nl(img/f, sigma=sigma) * f

        z1[i] = np.squeeze(img_d)
    z1 = z1.reshape(s0)
    return z1

def clip_with_bounds(m, lb, hb, first_n_term=None):
    m_clip = m.copy()
    if first_n_term is None:
        n = m_clip.shape[0]
    else:
        n = min(m_clip.shape[0], first_n_term)
    try:
        for i in range(n):
            id1 = m_clip[i] < lb[i]
            m_clip[i][id1] = lb[i]
            id2 = m_clip[i] > hb[i]
            m_clip[i][id2] = hb[i]
    except:
        print('fail to clip matrix')
    return m_clip

########################################
class f_gaussian():
    '''
    y = coef * exp(-(x-mu)^2/(2 * sigma^2))
    '''
    def __init__(self, x, coef, mu, sigma):
        self.data = x
        self.coef = coef
        self.mu = mu
        self.sigma = sigma
    def eval(self):
        return self.coef * np.exp(-(self.data - self.mu)**2/(2 * self.sigma**2))

    def d_coef(self):
        return np.exp(-(self.data - self.mu)**2/(2 * self.sigma**2))

    def d_mu(self):
        t1 = self.eval()
        t2 = 1./self.sigma**2 * (self.data - self.mu)
        return t1 * t2

    def d_sigma(self):
        t1 = self.eval()
        t2 = (self.data - self.mu) ** 2 / (self.sigma ** 3)
        return t1 * t2






###########################################
def test():
    np.random.seed(1)
    t = np.arange(-4,4,0.1)
    X = np.zeros([3, len(t)])
    X[0] = np.sin(t)
    X[1] = (t/4)**2
    X[2] = t/4

    a0 = 0.1
    a1 = 0.05
    a2 = 0.85

    Y_true = a0 * X[0] + a1 * X[1] + a2 * X[2]
    Y = a0 * X[0] + a1 * X[1] + a2 * X[2] + np.random.randn(X.shape[1])*0.05
    Y = Y.reshape(1, len(Y))

    test = np.random.randn(len(t))
    Y_test = np.squeeze(Y)

    W, cost = lsq_fit_iter(X, Y, learning_rate=0.01, n_iter=100, bounded=True)
    W_true = [a0,a1,a2]
    W = np.squeeze(W)

    Y_est = np.squeeze(np.dot(W, X))
    plt.figure();plt.subplot(121);plt.plot(cost)


    plt.subplot(122);plt.plot(t, Y_test, 'r.-');
    # plt.plot(t, Y_true, 'g');
    plt.plot(t, Y_est, 'b.')
    tit1 = 'Guess: {0:2.3f}, {1:2.3f}, {2:2.3f}\nsum = {3:2.3f}\n'.format(W[0], W[1], W[2], np.sum(W))
    tit2 = 'Ture: {0:2.3f}, {1:2.3f}, {2:2.3f}'.format(a0, a1, a2)
    plt.title(tit1+tit2)
    plt.show();
