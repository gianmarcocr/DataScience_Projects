import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
import scipy
from scipy.sparse.linalg import svds
import itertools
from itertools import cycle
import seaborn as sns
import itertools
import sklearn
from sklearn.utils.extmath import randomized_svd
from labellines import labelLine, labelLines

from joblib import Memory
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")

#function to load .bz2 data
@mem.cache
def get_data(path):
    data = load_svmlight_file(path)
    return (data[0], data[1])

sns.set_style("whitegrid")

np.random.seed(42)


## GLOBAL VARIABLES ##

t=180 # time to run
sw=1 # step width
tau = 50 # diameter of the nuclear norm ball
D = tau 
L = 1 # Lipshitz constant
G = 1

N = 0
M = 0
H = 0

def set_global_dimentions(N_new, M_new, H_new):
    global N
    global M
    global H

    N = N_new
    M = M_new
    H = H_new

### PLOTTING FUNCTIONS ###

def plot(method, loss, time):
    plt.plot(time,loss,label=method.upper())
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.title('{}: Time vs Loss'.format(method.upper()))
    plt.savefig('{}.png'.format(method.upper()),bbox_inches='tight',dpi=300)
    plt.show()

def plot_all(methods, losses, times, xlimit = None, ylimit = None):
    marker = itertools.cycle(('d', '+', '.', 'o', '*','x','s')) 
    cycol = cycle('bgrymkc')
    fig, ax = plt.subplots(figsize=(12,12))
    for i in range(len(methods)):
        ax.plot(times[i], losses[i], label = methods[i].upper(), color = next(cycol), marker = next(marker))
    legend = ax.legend(loc='upper right', fontsize='medium')
    plt.xlabel('Time [s]')
    plt.xlim(xlimit)
    plt.xlim(ylimit)
    plt.ylabel('Loss')
    plt.legend()
    labelLines(plt.gca().get_lines(),align=False,fontsize=12,zorder=2,xvals=(0.1, 10))
    plt.title('Comparison of Time-Loss for all methods')
    plt.savefig('All.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_wg(method, i_fwg, ind):
    for i in range(i_fwg.shape[0]):
        plt.plot(i_fwg[i][0], i_fwg[i][2], alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Wolfe-Gap")
    plt.title("Wolfe-Gap of {}".format(method))
    plt.savefig("Wolfe-Gap of {}.png".format(method),bbox_inches='tight',dpi=300)

### LOSSES AND GRADIENTS ###

def loss_function(W, X, y):
    '''
    Computes the loss function for the weights W, given the data matrix X (transposed) and the vector of  classes y
    HERE we fully vectorize the loss computation to obtain fast product between very big matrices.
    Input:\\
    - X: the data matrix. M rows and N columns\\
    - W: weights. H rows (H = number of labels) and M columns (M number of 
    features)\\
    - y: labels vector 
    '''

    #here we don't sum up 1 because the sum we are doing is for all l from 1 to H
    #hence the +1 appears when l = y_i

    # @ is the matrix product between matrices
    big_matrix = W @ X

    val_is = np.array([np.matmul(W[y[i],:], X[:,i]) for i in range(N)])
    val_is = val_is.reshape(1,N)

    mat = np.multiply(np.ones((H,N)), val_is)
    diff = big_matrix - mat

    esp = np.exp(diff)
    somm = np.sum(esp, axis=0)
    l = np.log(somm)
    loss = np.sum(l)
    loss = loss/N
    
    return float(loss)

def loss_gradient(W, X, y, batch_size):
    '''
    Computes the gradient w.r.t. the weights W. Must also pass a batch size.
    When batch size = N (number of instances), the whole gradient is computed.
    Input:\\
    - X: the data matrix. M rows and N columns\\
    - W: weights. H rows (H = number of labels) and M columns (M number of 
    features)\\
    - y: labels vector 
    - batch_size
    '''

    loss = np.zeros((H,M))

    if batch_size == N: #this is the case of full gradient descent
        list_indexes = range(N)
    else: #this is the case for mini-batch stochastic gradient descent
        list_indexes = random.sample(range(N), batch_size)
    
    for i in list_indexes:
        aux = np.matmul(W, X[:,i])
        aux = np.reshape(aux, (H,1))
        const_b_vec = aux - np.ones((H, 1))* np.matmul(W[y[i],:], X[:,i])

        esp = np.exp(const_b_vec)
        alpha = np.sum(esp)
        frac = 1/alpha
        mat_ei = np.array([X[:,i] for j in range(H)])
        mat = np.multiply(mat_ei, esp)
        mat[y[i],:] = -X[:,i]

        loss += frac * mat
        
    loss = loss / batch_size
    
    return loss

def g(x, gradient, beta, initial):
    '''
    Function to minimize in the CndG procedure. (SCGS and STORC only)\\
    - x: value in which you want to compute g(x)\\
    - gradient: gradient of f evaluated in z_k\\
    the others are self-explanatory    
    '''
    to_norm = x - initial
    return beta/2 * np.linalg.norm(to_norm)**2 + np.trace(np.matmul(gradient.T, x))

def gradient_g(x, gradient, beta, initial):
    '''
    Gradient of the function to minimize in the CndG procedure. (SCGS and STORC only)\\
    - x: value in which you want to compute g(x)\\
    - gradient: gradient of f evaluated in z_k\\
    the others are self-explanatory    
    '''
    return beta * (x - initial) + gradient

def simplex_projection(s, radius):
    """Projection onto the tau-simplex."""
    if np.sum(s) <=radius and np.alltrue(s >= 0):
        return s

    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    opt = u * np.arange(1, len(u)+1) > (cssv - 1)
    rho = np.nonzero(opt)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(radius*(s-theta), 0)

def nuclear_projection(A, radius):
    """Projection onto the unitr nuclear norm ball."""
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s = simplex_projection(s, radius)
    return U.dot(np.diag(s).dot(V))


    ## ALGORITHMS ## ------------------------------------------------------------------------------------------------

    # frank wolfe algorithm implementation 

def fw(W, X, y, lr, maxit, step_width, epsilon=None, time_to_run = None):
    '''
    Frank-Wolfe Algorithm. Two different stopping criterions are implemented:
    1) loss <= epsilon -> stop;
    2) time elapsed >= time_to_run;
    Can use only one of the two at a time.
    Input:\\
    - X: Data Matrix, M rows and N columns\\
    - W: Starting weights, H rows (H = number of labels) and M columns (M number of 
    features)\\
    - Labels vector y\\
    - lr: learning rate\\
    -maxit: maximum number of iterations\\
    - epsilon: error threshold\\
    - time_to_run: time we want the algorithm to run\\
    '''

    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return

    #initialize vectors of losses and times
    losses_fw = []
    times_fw = []
    start = time.time()

    # compute the loss at step 0
    loss = loss_function(W, X, y)
    losses_fw.append(loss)
    times_fw.append(0)
    iteration = 1
    print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, 0, loss))

    W_fw = W

    
    while iteration <= maxit:

        gradient_fw = loss_gradient(W_fw, X, y, N)
        # find top left and right singular vectors
        u_fw, _, v_fw = randomized_svd(gradient_fw, 1)
        v_hat_fw = -tau * np.matmul(u_fw, v_fw)    

        W_fw = W_fw + lr * (v_hat_fw - W_fw)

        if iteration%step_width == 0:
            loss = loss_function(W_fw, X, y)
            losses_fw.append(loss)
            current_time = time.time()-start
            times_fw.append(current_time)
            print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, current_time , loss))

            if epsilon != None and loss <= epsilon:
                break
            if time_to_run != None and current_time >= time_to_run:
                break
    
        iteration += 1

    return losses_fw, times_fw, W_fw

    # CndG subroutine for SCGS and STORC

def fw_g(gradient_z, u, beta, eta, maxit, lr=None, step_width=1, time_to_run = 10, epsilon = None):
    '''
    CndG subroutine for the SCGS and STORC algorithms.
    Input:\\
    - gradient_z: the gradient given by the outer algorithm computed in z_k
    - u: the "x_{k-1}" for SCGS and STORC
    - beta: the "beta" in the g(x) function    
    - eta: the threshold for the Wolfe Gap
    - maxit: maximum number of iterations allowed
    - lr: fixed step size. If None, diminishing stepsize is applied.
    - step_width: n. of iterations to wait until computing the next loss
    - time_to_run: maximum time for which the method is allowed to run
    - epsilon: if != None, method stops when loss <= epsilon
    '''
    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return

    times = []
    losses = []
    V_uts = []
    start = time.time()
    iteration = 1              
    u_t = u

    while iteration <= maxit:

        # V(u_t) = min_{x\in\Omega}(-<gradient_z + \beta(u_t - u),u_t - x>) 
        # the optimal solution is the same as this other problem:
        # min_{x\in\Omega} <gradient_z + \beta(u_t - u), x>
        # I first find the optimal solution, then use it to find V(u_t)

        # gradient is = gradient_z + beta*(u_t - u)
        gradient = gradient_g(x = u_t, gradient=gradient_z, beta= beta, initial=u)
        # solve the Linear Optimization problem
        u, _, v = randomized_svd(gradient, 1)
        v_t = -tau *  np.matmul(u, v)
        V_ut = np.trace(np.matmul(gradient.T, u_t - v_t))
        
        if V_ut <= eta:
            break
        
        if lr==None: # case of diminishing step size
            alpha = 1/(iteration+1)
        else:
            # the case where we want to use fixed step (different from theory)
            alpha = lr

        u_t = u_t + alpha * (v_t - u_t)
        
        loss = g(u_t, gradient_z, beta, u)
        losses.append(loss)
        current = time.time() - start
        times.append(current)
        V_uts.append(V_ut)
        
        if epsilon != None and loss <= epsilon:
            break
        if time_to_run != None and current >= time_to_run:
            break

        iteration += 1

    return V_uts, losses, times, u_t

# Stochastic Frank-Wolfe method implementation

def sfw(W, X, y, lr, maxit, step_width, epsilon=None, time_to_run = None):
    '''
    Stochastic Frank-Wolfe method implementation. 
    Input:\\
    - X: Data Matrix, M rows and N columns\\
    - W: Starting weights, H rows (H = number of labels) and M columns (M number of 
    features)\\
    - Labels vector y\\
    - lr: learning rate\\
    -maxit: maximum number of iterations\\
    - epsilon: error threshold\\
    - time_to_run: time we want the algorithm to run\\
    '''

    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return
        
    losses_sfw = []
    times_sfw = []
    start = time.time()

    loss = loss_function(W, X, y)
    losses_sfw.append(loss)
    times_sfw.append(0)
    iteration = 1
    print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, 0, loss))

    W_sfw = W.copy()
    
    while iteration <= maxit:
        
        if iteration**2 < N:
            batch_size = iteration**2
        else:
            batch_size = N
        
        gradient_sfw = loss_gradient(W_sfw, X, y, batch_size)#*batch_size
        u_sfw, _, v_sfw = randomized_svd(gradient_sfw, 1)
        v_hat_sfw = -tau * np.matmul(u_sfw, v_sfw)
        W_sfw = W_sfw + lr * (v_hat_sfw - W_sfw)

        if iteration%step_width == 0:
            loss = loss_function(W_sfw, X, y)
            losses_sfw.append(loss)
            current = time.time() - start
            times_sfw.append(current)
            print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, current, loss))

            if epsilon != None and loss <= epsilon:
                break
            if time_to_run != None and current >= time_to_run:
                break

        iteration += 1

    
    return losses_sfw, times_sfw, W_sfw

# Stochastic Variance Reduced Frank-Wolfe method implementation.

def svrf(W, X, y, epochs, step_width = 2, lr=None, epsilon=None, time_to_run = None):
    '''
    Stochastic Variance Reduced Frank-Wolfe method implementation.
    Input:\\
    - X: Data Matrix, M rows and N columns\\
    - W: Starting weights, H rows (H = number of labels) and M columns (M number of 
    features)\\
    - Labels vector y\\
    - lr: learning rate\\
    -maxit: maximum number of iterations\\
    - epsilon: error threshold\\
    - time_to_run: time we want the algorithm to run\\
    '''

    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return   

    iterations = epochs * 50
    losses_svrf = []
    times_svrf = []
    start = time.time()

    loss = loss_function(W, X, y)
    losses_svrf.append(loss)
    times_svrf.append(0)
    iteration = 1
    print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, 0, loss))

    W_svrf = W

    for j in range(1, epochs + 1):
        W_snap = W_svrf
        snapshot = loss_gradient(W_snap, X, y, N)
        
        for i in range(1, 50+1):
            if lr==None:
                lr = 2/(i + 1)
                
            batch_size = i+1
            gradient_svrf = loss_gradient(W_svrf, X, y, batch_size) - loss_gradient(W_snap, X, y, batch_size) + snapshot

            u_svrf, s_svrf, v_svrf = randomized_svd(gradient_svrf, 1)
            v_hat_svrf = -tau * np.matmul(u_svrf, v_svrf)

            W_svrf = W_svrf + lr * (v_hat_svrf - W_svrf)

            if iteration%step_width == 0:
                loss = loss_function(W_svrf, X, y)
                losses_svrf.append(loss)
                current = time.time() - start
                times_svrf.append(current)
                print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, current, loss))

                if epsilon != None and loss <= epsilon:
                    break
                if time_to_run != None and current >= time_to_run:
                    break

            iteration += 1

        if epsilon != None and loss <= epsilon:
            break
        if time_to_run != None and current >= time_to_run:
            break

    return losses_svrf, times_svrf, W_svrf

# Stochastic Conditional Gradient Sliding implementation

def scgs(W, X, y, maxit, step_width, epsilon=None, time_to_run = None, lr = None, lr_fwg = None):
    '''
    Stochastic Conditional Gradient Sliding implementation
    Input:\\
    - X: Data Matrix, M rows and N columns\\
    - W: Starting weights, H rows (H = number of labels) and M columns (M number of 
    features)\\
    - Labels vector y\\
    - lr: learning rate\\
    - lr_fwg: learning rate for the CndG subroutine\\
    -maxit: maximum number of iterations\\
    - epsilon: error threshold\\
    - time_to_run: time we want the algorithm to run\\
    '''

    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return  

    losses_scgs = []
    times_scgs = []
    # I create a list of lists for debugging purposes
    # info_fwg will be a list of shape [n_iterations_scgs, 2, n_iterations_fwg]
    #example:   info_fwg[0][0] will contain the times of the fw_g algo at the first iteration of SCGS
    #           info_fwg[0][1] will contain the losses of the fw_g algo at the first iteration of SCGS
    info_fwg = []
    start = time.time()

    loss = loss_function(W, X, y)
    losses_scgs.append(loss)
    times_scgs.append(0)
    iteration = 1

    print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, 0, loss))

    W_scgs = W.copy()
    x = W_scgs
    Y = x
    
    while iteration <= maxit:
        
        if lr == None:
            gamma = 3 / (iteration +2)
        else:
            gamma = lr
        
        beta = (4 * L) / (iteration + 2)
        eta = 1/(iteration+1)**2
        
        if iteration**3 < N:
            batch_size = iteration**3
        else:
            batch_size = N
        
        # update z
        z = Y + gamma * (x - Y)
        # compute stochastic gradient
        gradient_scgs = loss_gradient(z, X, y, batch_size)
        # update x
        V_uts_fwg, losses_fwg, times_fwg, x = fw_g(gradient_z = gradient_scgs, 
                                                   u=x, beta=beta, eta=eta, maxit=1000, step_width=1, lr=lr_fwg)
        info_fwg.append([times_fwg, losses_fwg, V_uts_fwg])
        # update Y
        Y = Y + gamma * (x - Y)

        if iteration%step_width == 0:
            loss = loss_function(Y, X, y)
            losses_scgs.append(loss)
            current = time.time() - start
            times_scgs.append(current)
            print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, current, loss))

            if epsilon != None and loss <= epsilon:
                break
            if time_to_run != None and current >= time_to_run:
                break
            
        iteration += 1

    
    return np.array(info_fwg), losses_scgs, times_scgs, Y

# Stochastic Variance Reduced Conditional Gradient Sliding algorithm

def storc(W, X, y, epochs, step_width, epsilon=None, time_to_run = None, lr=None, lr_fwg=None):
    '''
    Stochastic Variance Reduced Conditional Gradient Sliding algorithm
    Input:\\
    - X: Data Matrix, M rows and N columns\\
    - W: Starting weights, H rows (H = number of labels) and M columns (M number of 
    features)\\
    - Labels vector y\\
    - lr: learning rate\\
    - lr_fwg: learning rate for the CndG subroutine\\
    -maxit: maximum number of iterations\\
    - epsilon: error threshold\\
    - time_to_run: time we want the algorithm to run\\
    '''
    
    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return  

    iterations = epochs*50
    losses_storc = []
    times_storc = []
    info_fwg = []
    start = time.time()

    loss = loss_function(W, X, y)
    losses_storc.append(loss)
    times_storc.append(0)
    iteration = 1
    print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, 0, loss))

    W_storc = W.copy()

#   initizalization from the theory: we don't need it.
#   gradient_storc = loss_gradient(W, X, y, N)
#   u, _, v = randomized_svd(gradient_storc, 1)
#   W_storc = -tau * np.matmul(u, v) 
    
    for i in range(1, epochs + 1):
        W_snap = W_storc.copy()
        Y = W_storc.copy()
        x = Y.copy()
        snapshot = loss_gradient(W_snap, X, y, N)
        
        for j in range(1, 50+1):
            if lr == None:
                gamma = 2 / (j + 1)
            else:
                gamma = lr

            beta = (3 * L) / j            
            eta = 1/iteration
            m_tk = 100           
            
            z = Y + gamma * (x - Y)
            a = loss_gradient(z, X, y, m_tk)
            b = loss_gradient(W_snap, X, y, m_tk)
            gradient_storc = a - b + snapshot

            V_uts_fwg, losses_fwg, times_fwg, x = fw_g(gradient_z = gradient_storc, u=x, 
                                            beta=beta, eta=eta, maxit=1000, step_width=1, lr=lr_fwg)
            info_fwg.append([times_fwg, losses_fwg, V_uts_fwg])
            
            Y = Y + gamma * (x - Y)

            if iteration%step_width == 0:
                loss = loss_function(Y, X, y)
                losses_storc.append(loss)
                current = time.time() - start
                times_storc.append(current)
                print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, current, loss))

                if epsilon != None and loss <= epsilon:
                    break
                if time_to_run != None and current >= time_to_run:
                    break

            iteration += 1
        
        print(current)
        
        W_storc = Y
        
        if epsilon != None and loss <= epsilon:
            break
        if time_to_run != None and current >= time_to_run:
            break

    
    return np.array(info_fwg), losses_storc, times_storc, Y

# Projected Stochastic Gradient Method implementation.

def sgm(W, X, y, lr , maxit, step_width, epsilon=None, time_to_run = None, s = 1, batch_size = 1):
    '''
    Projected Stochastic Gradient Method implementation.
    Input:\\
    - X: Data Matrix, M rows and N columns\\
    - W: Starting weights, H rows (H = number of labels) and M columns (M number of 
    features)\\
    - Labels vector y\\
    - lr: starting learning rate. We use a diminishing step size.\\
    - maxit: maximum number of iterations\\
    - epsilon: error threshold\\
    - time_to_run: time we want the algorithm to run\\
    - s: 1 by default\\
    - batch size: 1 by default, must be >0 \\
    '''

    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return
    
    losses_sgm = []
    times_sgm = []
    start = time.time()

    loss = loss_function(W, X, y)
    losses_sgm.append(loss)
    times_sgm.append(0)
    iteration = 1
    print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, 0, loss))  

    W_sgm = W.copy()
    alpha = lr / np.sqrt(iteration)
    #alpha = constant

    while iteration <= maxit:
        gradient_sgm = loss_gradient(W_sgm, X, y, batch_size)
        d = nuclear_projection(W_sgm - s*gradient_sgm, tau) - W_sgm
        W_sgm = W_sgm + alpha*d
            
        if iteration%step_width == 0:
            loss = loss_function(W_sgm, X, y)
            losses_sgm.append(loss)
            current = time.time() - start
            times_sgm.append(current)
            print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, current, loss))
            
            if epsilon != None and loss <= epsilon:
                break
            if time_to_run != None and current >= time_to_run:
                break

        iteration += 1
        
    
    return losses_sgm, times_sgm, W_sgm

#Stochastic Variance Reduced Grandient Method implementation

def svrg(W, X, y, lr, epochs, maxit, step_width, epsilon=None, time_to_run = None, s = 1, batch_size = 1):
    '''
    Stochastic Variance Reduced Grandient Method implementation
    Input:\\
    - X: Data Matrix, M rows and N columns\\
    - W: Starting weights, H rows (H = number of labels) and M columns (M number of 
    features)\\
    - Labels vector y\\
    - lr: fixed step size\\
    - maxit: maximum number of iterations\\
    - epsilon: error threshold\\
    - time_to_run: time we want the algorithm to run\\
    - s: 1 by default\\
    - batch size: 1 by default, must be >0 \\
    '''

    if (epsilon==None and time_to_run==None):
        print("You have to set either a threshold or a running time!")
        return
    if (epsilon != None and time_to_run != None):
        print("You can't set both epsilon and time to run! Pick only one.")
        return
    
    losses_svrg = []
    times_svrg = []
    start = time.time()

    loss = loss_function(W, X, y)
    losses_svrg.append(loss)
    times_svrg.append(0)
    iteration = 1
    print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, 0, loss))  

    W_svrg = W.copy()
    
    alpha = lr
    for j in range(1, epochs + 1):
        W_snap = W_svrg
        snapshot = loss_gradient(W_snap, X, y, N)
        
        for i in range(1, 50+1):
            gradient_svrg = loss_gradient(W_svrg, X, y, batch_size)  - loss_gradient(W_snap, X, y, batch_size) + snapshot
            d = nuclear_projection(W_svrg - s*gradient_svrg, tau) - W_svrg
            W_svrg = W_svrg + alpha*d

            if iteration%step_width == 0:
                loss = loss_function(W_svrg, X, y)
                losses_svrg.append(loss)
                current = time.time() - start
                times_svrg.append(current)
                print("Iteration: {};  Time Elapsed: {};  Loss: {}".format(iteration, current, loss))

                if epsilon != None and loss <= epsilon:
                    break
                if time_to_run != None and current >= time_to_run:
                    break

            iteration += 1

        if epsilon != None and loss <= epsilon:
            break
        if time_to_run != None and current >= time_to_run:
            break

    return losses_svrg, times_svrg, W_svrg