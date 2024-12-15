# from scipy.optimize import minimize

# minimize(fun=ss.func,x0=ss.vec,jac=ss.gradient,method='SLSQP')
import numpy as np
import numpy.typing as npt
from typing import Callable

class BaseOptimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def minimize(self, fun : Callable[[np.ndarray], float], 
                 x0 : np.ndarray, jac : np.ndarray | None=None, **kwargs):
        self.x = x0
        self.iter = 0
        self.res = np.inf
        self.N = len(x0)

    def _end_conds(self, fun, **kwargs):
        return True
    def _step(self, fun, x, jac=None, hess=None, **kwargs):
        pass

class Optimizer(BaseOptimizer):
    def __init__(self, maxiter=100, tol=1e-6,
                 verbose=False,callback:Callable|None=None, **kwargs):
        super().__init__(maxiter=maxiter, tol=tol, **kwargs)
        self.verbose = verbose
        self.callback = callback

    def minimize(self, fun, x0, jac = None, **kwargs):
        super().minimize(fun, x0, jac, **kwargs)
        while not self._end_conds(fun, x=self.x, jac=jac,**self.kwargs):
            self._step(fun, self.x, jac, **self.kwargs)

    def _end_conds(self, fun, **kwargs):
        return (self.iter > self.kwargs['maxiter']) or (self.res < self.kwargs['tol'])
    def _step(self, fun, x, **kwargs):
        if self.callback is not None:
            self.callback(fun=fun,x=x,iter=self.iter,**kwargs)
def wolfe_cond1(func,x,p,g,c,rho):
    a = 1
    f_temp = func(x + a * p)
    cpg = c * np.dot(p, g)
    while f_temp > func(x) + a * cpg:
        a *= rho
        if a < 1e-14:
            break
        f_temp = func(x + a * p)
    return a

class TraceCallback:
    def __init__(self,):
        self._trace = {'x':[],'f':[],'grad_norm':[],'iter':[]}
    def __call__(self, **kwargs):
        x = kwargs['x']
        f = kwargs['fun'](x)
        grad_norm = np.linalg.norm(kwargs['jac'](x))
        iter = kwargs['iter']
        self._trace['x'].append(x)
        self._trace['f'].append(f)
        self._trace['grad_norm'].append(grad_norm)
        self._trace['iter'].append(iter)

class BFGS(Optimizer):
    def __init__(self, maxiter=100, tol=1e-4, **kwargs):
        super().__init__(maxiter=maxiter, tol=tol, **kwargs)
        self.m = 5
        self.rho = 0.9
        self.c = 0.1
    def minimize(self, fun : Callable[[np.ndarray], float], 
                 x0 : np.ndarray, jac : np.ndarray, **kwargs):
        self.N = len(x0)
        self.x = x0
        self.B = np.eye(self.N)
        self.g = jac(self.x)
        super().minimize(fun, x0, jac, **self.kwargs)
        return {'x': self.x,'fval':fun(self.x),'trace':self.callback._trace}
    def _step(self, fun, x, jac, hess=None, **kwargs):
        super()._step(fun,x,jac=jac,hess=hess,**kwargs)
        if self.iter % self.m == 0:
            self.B = np.eye(self.N)
        p = -np.linalg.solve(self.B, self.g)
        norm_p = np.linalg.norm(p)
        if norm_p > 1:
            p = p / norm_p
        self.iter += 1

        a = wolfe_cond1(func=fun,x=x,p=p,g=self.g,c=self.c,rho=self.rho)
        f = fun(x)
        s = a * p
        x = x + s
        g_new = jac(x)
        y = g_new - self.g
        if np.dot(s, y) > 1e-10:
            Bs = self.B @ s
            sy = np.dot(s, y)
            self.B = self.B + np.outer(y, y) / sy - np.outer(Bs, Bs) / np.dot(s, Bs)
        self.g = g_new
        self.x = x


class ADAM(Optimizer):
    def __init__(self, maxiter=100, tol=1e-6,
                 beta1=0.9,beta2=0.999,epsilon=1e-8,a=0.001,**kwargs):
        super().__init__(maxiter=maxiter, tol=tol, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.a = a
    def minimize(self, fun : Callable[[np.ndarray], float], 
                 x0 : np.ndarray, jac : np.ndarray, **kwargs):
        self.N = len(x0)
        self.x = x0
        self.m = np.zeros_like(x0)
        self.v = np.zeros_like(x0)
        super().minimize(fun, x0, jac, **self.kwargs)
        return {'x': self.x,'fval':fun(self.x),'trace':self.callback._trace}
    def _step(self, fun, x, jac, hess=None, **kwargs):
        super()._step(fun,x,jac=jac,hess=hess,**kwargs)
        self.m = self.beta1 * self.m + (1 - self.beta1) * jac(x)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (jac(x) ** 2)
        m_hat = self.m / (1 - self.beta1 ** (self.iter+1))
        v_hat = self.v / (1 - self.beta2 ** (self.iter+1))
        self.x = x - self.a * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.iter += 1




