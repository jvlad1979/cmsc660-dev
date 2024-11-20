import numpy as np
import scipy
def Loss(r):
    return 0.5*np.sum(r**2) # 0.5*sum(r^2)

def LevenbergMarquardt(Res_and_Jac,x,ITER_MAX,TOL):
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
    
    # initialization
    r,J = Res_and_Jac(x)
    # print(r.size())
    # print(J.size())
    n,d = np.shape(J)
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)
    lossvals[0] = Loss(r)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r) # grad = J^\top r
    Bmatr = np.matmul(Jtrans,J) # Bmatr = J^\top J
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    R = 0.2*RMAX # initial trust region radius
    print("iter 0: loss = ",lossvals[0]," gradnorm = ",gradnorm)
    # start iterations
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        Bmatr = np.matmul(Jtrans,J) + (1.e-6)*np.eye(d) # B = J^\top J
        p = (-1)*np.linalg.solve(Bmatr,grad) # p = -Bmatr^{-1}grad
        norm_p = np.linalg.norm(p)
        if norm_p > R:
            # solve grad^\top p + 0.5 p^\top Bmatr p --> min
            # subject to ||p|| = R
            gap = np.abs(norm_p - R)
            iter_lam = 0
            lam_tol = 0.01*R
            lam = 1 # initial guess for lambda in the 1D constrained minimization problems
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
                # print("LM, iter ",iter,":", iter_lam," substeps")
        # else:
            # print("LM, iter ",iter,": steps to the model's minimum")
        # evaluate the progress
        # print("x: ",x)
        # print("p: ",p)
        xnew = x + p
        # print("size of xnew: ",xnew.size())
        rnew,Jnew = Res_and_Jac(xnew)  
        # print("rnew: ",rnew.size())
        # print("Jnew: ",Jnew.size())
        lossnew = Loss(rnew)
        rho = -(lossvals[iter-1] - lossnew)/(np.sum(grad*p) + 0.5*sum(p*np.matmul(Bmatr,p)))   
        # adjust the trust region radius
        if rho < RHO_BAD:
            R = np.max(np.array([RMIN,0.25*R]))
        elif rho > RHO_GOOD:
            R = np.min(np.array([RMAX,2.0*R]))                                       
        # accept or reject the step
        if rho > ETA:
            x = xnew
            r = rnew
            J = Jnew  
            Jtrans = np.transpose(J)
            grad = np.matmul(Jtrans,r)                                       
            gradnorm = np.linalg.norm(grad)
        lossvals[iter] = lossnew
        gradnormvals[iter] = gradnorm
        print(f"LM, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}, rho = {rho:.4e}, R = {R:.4e}")
        iter = iter + 1                                           
    return x,iter,lossvals[0:iter], gradnormvals[0:iter]                                                          