import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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
    #dw2 = f_scale * np.abs(W)
    #dw2 = f_scale2 * np.sum(W**2, axis=1, keepdims=True) / n
    #dw = -2 * (np.dot(A.T, dy) + dw1 + dw2)
    dw = -2 * (np.dot(A.T, dy) + dw1)
    db = -2 * np.sum(dy, axis=0)
    db = db.reshape(B.shape)
    return dw, db, cost


def backpropagate0(Y_test, A, W, B):
    Y_hat = np.dot(A, W) + B
    m = Y_test.shape[0]
    dy = Y_test - Y_hat
    cost = np.sum(dy**2, axis=0)/m
    dw = -2 * (np.dot(A.T, dy))/m
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



def lsq_fit_iter2(A, Y, W=None, B=0, learning_rate=0.002, n_iter=100, bounds=[0,1]):
    # solve AW + B = Y (W and B need to be solved)

    if W is None:
        W = np.random.random([A.shape[1], Y.shape[1]])/A.shape[1]
        W[-1,:] = 1 - np.sum(W[:-1,:], axis=0)
    else:
        if len(bounds)==2:
            W[W <= bounds[0]] = bounds[0]
            W[W >= bounds[1]] = bounds[1]

    Y_test = deepcopy(Y)
    cost = []
    for i in range(n_iter):
        # if print_flag and not i%50:
        print('iter #{}'.format(i))

        if len(bounds)==2:
            dw, db, cost_temp = backpropagate(Y_test, A, W, B, f_scale1=0.5, f_scale2=0.5)
            tmp = dw * learning_rate
            index = np.abs(tmp) > 0.5 * np.abs(W)
            tmp[index] = 0.5 * np.abs(W[index])
            W -= tmp
            B -= db*learning_rate
            W[W <= bounds[0]] = bounds[0]
            W[W >= bounds[1]] = bounds[1]
        elif len(bounds)==0:
            dw, cost_temp = backpropagate0(Y_test, A, W, B)
            W -= dw*learning_rate
            B -= db * learning_rate
        cost.append(cost_temp)
    W = np.squeeze(W)
    B = np.squeeze(B)
    cost = np.array(cost)
    return W, B, cost


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
        wgt = np.ones(n_eng)

    if not len(bounds) == 2:
        bounds = [-1e16, 1e16]
    # initialize
    c = A.T @ np.diag(wgt) @ A + np.eye(n_spectra) * rho
    u = np.zeros(z.shape)

    convergency = np.zeros((num_iters, 1))

    for i in range(num_iters):
        print(f'iter #{i}')
        temp = z.copy()
        x = inv(c) @ (A.T @ np.diag(wgt) @ y + rho * (z - u))
        if np.abs(rho) < 1e-4:
            z = x + u
        else:
            z = soft_threshold(x + u, lasso_lamda / rho)
        z[z <= bounds[0]] = bounds[0]
        z[z >= bounds[1]] = bounds[1]
        u = u + x - z
        convergency[i] = (norm(z.flatten()) - norm(temp.flatten())) / norm(z.flatten())
    return x

########################################


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
