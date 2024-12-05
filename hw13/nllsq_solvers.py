import numpy as np
from typing import Callable
from joblib import Parallel, delayed
import scipy

def Loss(r):
    return 0.5*np.sum(r**2) # 0.5*sum(r^2)

def LossTk(r,lam=.01):
    return np.sum(r)/np.size(r) + (lam/2)*np.linalg.norm(r)

def GaussNewton(Res_and_Jac, x, ITER_MAX, TOL,verbose = False):

    # Initialization
    r, J = Res_and_Jac(x)
    n, d = np.shape(J)  # Number of residuals and parameters
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)
    lossvals[0] = Loss(r)
    
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans, r)  # Gradient: J^T r
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    I = np.eye(d)
    if verbose:
        print(f"GN, iter #0: loss = {lossvals[0]:.4e}, gradnorm = {gradnorm:.4e}")

    # Iterative updates
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        # Compute Gauss-Newton step
        Bmatr = np.matmul(Jtrans, J) + I*1e-6  # Approximate Hessian: J^T J + I
        step = -np.linalg.solve(Bmatr, grad)  # Update step: -(J^T J)^{-1} J^T r

        # Update parameters
        x = x + step

        # Compute new residuals, Jacobian, loss, and gradient
        r, J = Res_and_Jac(x)
        Jtrans = np.transpose(J)
        grad = np.matmul(Jtrans, r)
        gradnorm = np.linalg.norm(grad)

        # Record loss and gradient norm
        lossvals[iter] = Loss(r)
        gradnormvals[iter] = gradnorm
        if verbose:
            print(f"GN, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}")
        iter += 1

    # Truncate unused iterations in the arrays
    return x, iter, lossvals[:iter], gradnormvals[:iter]

def StochasticGradientDescent(Res_and_Jac, x, ITER_MAX, TOL, batch_size=100, step_size=0.1,lossfunc = Loss,verbose = False):
    # Full residuals and Jacobian
    r_full, J_full = Res_and_Jac(x)
    n, d = np.shape(J_full)  # Number of data points and parameters

    # Initialize tracking metrics
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)
    
    # Compute initial loss and gradient
    lossvals[0] = LossTk(r_full)
    grad = np.matmul(J_full.T, r_full) / n  # Gradient for the full dataset
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    if verbose:
        print(f"SGD, iter #0: loss = {lossvals[0]:.4e}, gradnorm = {gradnorm:.4e}")

    
    if isinstance(step_size,Callable):
        step_size_val = step_size(gradnorm)
    else:
        step_size_val = step_size
    # Iterative updates
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        # Select a random batch of indices
        indices = np.random.choice(n, batch_size, replace=False)

        # Compute residuals and Jacobian for the batch
        r_batch = r_full[indices]
        J_batch = J_full[indices, :]

        # Compute gradient for the batch
        grad_batch = np.matmul(J_batch.T, r_batch) / batch_size
        if isinstance(step_size,Callable):
            step_size_val = step_size(iter)
        else:
            step_size_val = step_size
        # Update parameters
        x = x - step_size_val * grad_batch

        # Recompute residuals and Jacobian for the full dataset
        r_full, J_full = Res_and_Jac(x)

        # Compute new loss and gradient
        lossvals[iter] = LossTk(r_full)
        grad = np.matmul(J_full.T, r_full) / n
        gradnorm = np.linalg.norm(grad)
        gradnormvals[iter] = gradnorm
        if verbose:
            print(f"SGD, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}")
        iter += 1

    # Truncate unused iterations in the arrays
    return x, iter, lossvals[:iter], gradnormvals[:iter]

def NesterovGradientDescent(Res_and_Jac, x, ITER_MAX, TOL, batch_size=100, step_size=0.1,momentum=.75,lam=0.01,deterministic=False,lossfunc = Loss,verbose = False):
    # Full residuals and Jacobian
    r_full, J_full = Res_and_Jac(x)
    n, d = np.shape(J_full)  # Number of data points and parameters

    # Initialize tracking metrics
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)
    
    # Compute initial loss and gradient
    lossvals[0] = LossTk(r_full)
    grad = np.matmul(J_full.T, r_full) / n  # Gradient for the full dataset
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    if verbose:
        print(f"SGD, iter #0: loss = {lossvals[0]:.4e}, gradnorm = {gradnorm:.4e}")

    
    if isinstance(step_size,Callable):
        step_size_val = step_size(gradnorm)
    else:
        step_size_val = step_size

    v = np.zeros_like(x)

    # Iterative updates
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        x_ahead = x - momentum * v

        indices = np.random.choice(n, batch_size, replace=False)
        if deterministic:
            r_batch = r_full
            J_batch = J_full
        else:
            r_batch = r_full[indices]
            J_batch = J_full[indices, :]

        grad_batch = np.matmul(J_batch.T, r_batch) / batch_size
        v = momentum * v - step_size_val * grad_batch
        x = x + v

        r_full, J_full = Res_and_Jac(x)

        lossvals[iter] = LossTk(r_full,lam=lam)
        grad = np.matmul(J_full.T, r_full) / n
        gradnorm = np.linalg.norm(grad)
        gradnormvals[iter] = gradnorm
        if verbose:
            print(f"SGD, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}")
        iter += 1

    # Truncate unused iterations in the arrays
    return x, iter, lossvals[:iter], gradnormvals[:iter]

def Adam(Res_and_Jac, x, ITER_MAX, TOL, batch_size=100, step_size=0.001, lossfunc=Loss, verbose=False, beta1=0.9, beta2=0.999, epsilon=1e-8,lam=0.01,deterministic=False):
    # Full residuals and Jacobian
    r_full, J_full = Res_and_Jac(x)
    n, d = np.shape(J_full)  # Number of data points and parameters

    # Initialize tracking metrics
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)

    # Compute initial loss and gradient
    lossvals[0] = LossTk(r_full)
    grad = np.matmul(J_full.T, r_full) / n  # Gradient for the full dataset
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    if verbose:
        print(f"Adam, iter #0: loss = {lossvals[0]:.4e}, gradnorm = {gradnorm:.4e}")

    # Initialize moment variables
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    t = 0  # Time step

    # Iterative updates
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        # Select a random batch of indices
        indices = np.random.choice(n, batch_size, replace=False)

        # Compute residuals and Jacobian for the batch
        if deterministic:
            r_batch = r_full
            J_batch = J_full
        else:
            r_batch = r_full[indices]
            J_batch = J_full[indices, :]

        # Compute gradient for the batch
        grad_batch = np.matmul(J_batch.T, r_batch) / batch_size

        # Update time step

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad_batch
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad_batch ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** (iter))
        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2 ** (iter))

        # Update parameters
        x = x - step_size * m_hat / (np.sqrt(v_hat) + epsilon)

        # Recompute residuals and Jacobian for the full dataset
        r_full, J_full = Res_and_Jac(x)

        # Compute new loss and gradient
        lossvals[iter] = LossTk(r_full,lam=lam)
        grad = np.matmul(J_full.T, r_full) / n
        gradnorm = np.linalg.norm(grad)
        gradnormvals[iter] = gradnorm
        if verbose:
            print(f"Adam, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}")
        iter += 1

    # Truncate unused iterations in the arrays
    return x, iter, lossvals[:iter], gradnormvals[:iter]


def LevenbergMarquardt(Res_and_Jac,x,ITER_MAX,TOL,verbose = False):
    # minimizes loss = 0.5/n sum_{j=1}^n r_j^2(x)
    # constrained minimization problem solved at each step:
    # m(p) = grad^\top p + 0.5 p^\top Bmatr p --> min
    # subject to R - ||p|| >= 0
    # rho = [loss - loss(x + p)] / [loss - m(p)]
    
    # parameters for Levengerg-Marquardt
    RMAX = 1.
    RMIN = 1e-12
    RHO_GOOD = 0.75 # increase R is rho > RHO_GOOD
    RHO_BAD = 0.25 # decrease R is rho < RHO_BAD
    ETA = 0.01 # reject step if rho < ETA 

    r,J = Res_and_Jac(x)
    n,d = np.shape(J)
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)
    lossvals[0] = Loss(r)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    Bmatr = np.matmul(Jtrans,J)
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    R = 0.2*RMAX
    if verbose:
        print("iter 0: loss = ",lossvals[0]," gradnorm = ",gradnorm)
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        Bmatr = np.matmul(Jtrans,J) + (1.e-6)*np.eye(d)
        p = (-1)*np.linalg.solve(Bmatr,grad)
        norm_p = np.linalg.norm(p)
        if norm_p > R:
            gap = np.abs(norm_p - R)
            iter_lam = 0
            lam_tol = 0.01*R
            lam = 1
            while gap > lam_tol:
                B1 = Bmatr + lam*np.eye(d) 
                C = np.linalg.cholesky(B1) # B1 = C C^\top
                p = -scipy.linalg.solve_triangular(np.transpose(C), \
                        scipy.linalg.solve_triangular(C,grad,lower = True),lower = False)
                norm_p = np.linalg.norm(p)
                gap = np.abs(norm_p - R)
                if gap > lam_tol:
                    q = scipy.linalg.solve_triangular(C,p,lower = True)
                    norm_q = np.linalg.norm(q)
                    lamnew = lam + ((norm_p/norm_q)**2)*(norm_p-R)/R
                    if lamnew < 0:
                        lam = 0.5*lam
                    else:
                        lam = lamnew
                    iter_lam = iter_lam + 1
                    gap = np.abs(norm_p - R)
        xnew = x + p
        rnew,Jnew = Res_and_Jac(xnew)
        lossnew = Loss(rnew)
        rho = -(lossvals[iter-1] - lossnew)/(np.sum(grad*p) + 0.5*sum(p*np.matmul(Bmatr,p)))
        if rho < RHO_BAD:
            R = np.max(np.array([RMIN,0.25*R]))
        elif rho > RHO_GOOD:
            R = np.min(np.array([RMAX,2.0*R]))
        if rho > ETA:
            x = xnew
            r = rnew
            J = Jnew  
            Jtrans = np.transpose(J)
            grad = np.matmul(Jtrans,r)
            gradnorm = np.linalg.norm(grad)
        lossvals[iter] = lossnew
        gradnormvals[iter] = gradnorm
        if verbose:
            print(f"LM, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}, rho = {rho:.4e}, R = {R:.4e}")
        iter = iter + 1
    return x,iter,lossvals[0:iter], gradnormvals[0:iter]