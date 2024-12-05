import numpy as np
from nllsq_solvers import GaussNewton, StochasticGradientDescent, LevenbergMarquardt


class LSQClassifier:
    def __init__(self,solver = 'lm', solverkwargs = {'TOL' : 1e-3,'ITER_MAX' : 600},lam = 0.0,verbose = False):
        self.solverkwargs = solverkwargs
        self.k = None
        self.w = None
        self.verbose = verbose
        self.report = {}
        self.lam = lam
        if solver == 'lm':
            self.solver = LevenbergMarquardt
        if solver == 'gauss':
            self.solver = GaussNewton
        if solver == 'sgd':
            self.solver = StochasticGradientDescent
    def fit(self,X,y):
        self.k = np.shape(X)[-1]
        w = np.ones((self.k*self.k + self.k + 1,))
        def r_and_J(w):
            return self._Res_and_Jac(X,y,w,self.lam)
        w,Niter,Loss_vals,gradnorm_vals = self.solver(r_and_J,w,**self.solverkwargs,verbose=self.verbose)
        self.report['iters'] = Niter
        self.report['Loss_vals'] = Loss_vals
        self.report['gradnorm_vals'] = gradnorm_vals
        self.w = w

    def predict(self,X):
        return np.sign(self._predict_score(X))

    def _predict_score(self,X):
        n,d = X.shape
        preds = self._myquadratic(X,np.ones(n),self.w)
        return preds

    def _Res_and_Jac(self,X,y,w,lam = 0):
        aux = np.exp(-self._myquadratic(X,y,w))
        # aux = np.exp(-self._quad(X,y,w))
        r = np.log(1. + aux)

        a = -aux/(1. + aux)
        n,d = np.shape(X)
        d2 = d*d
        ya = y*a
        qterm = np.zeros((n,d2))
        for k in range(n):
            xk = X[k,:]
            xx = np.outer(xk,xk)
            qterm[k,:] = np.reshape(xx,(np.size(xx),))
        J = np.concatenate((qterm,X,np.ones((n,1))),axis = 1)
        for k in range(n):
            J[k,:] = J[k,:]*ya[k]

        if lam>0:
            reg_grad = lam * w
            r += 0.5 * lam * np.dot(w, w)  # Regularization loss
            J += reg_grad / len(X)  # Average regularization gradient
        return r,J

    def _myquadratic(self,X,y,w):
        d = np.size(X,axis = 1)
        d2 = d*d
        W = np.reshape(w[:d2],(d,d))
        v = w[d2:d2+d]
        b = w[-1]
        qterm = np.diag(X@W@np.transpose(X))
        q = y*(qterm + v.T@X.T + b)
        return q
