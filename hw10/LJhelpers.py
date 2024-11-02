# Per-Olof Persson's code distmesh2D rewritten to Python and simplified
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# import math
# from scipy.spatial import Delaunay
# from scipy import sparse

# The Lennard-Jones potential. its gradient, and its Hessian

def LJ(x): # Lennard-Jones potential. Input: 3-by-Na array
    Na = np.size(x,axis = 1)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 1
    er6 = np.divide(np.ones_like(r2),r2**3)
    L = (er6-1)*er6
    V = 2*np.sum(L) 
    return V

def LJvector2array(x): # respore atomic coordinates from the vector
    m = np.size(x)
    Na = np.rint((m + 6)/3).astype(int)
    Na3 = 3*Na
    x_aux = np.zeros((Na3,))
    x_aux[3] = x[0]
    x_aux[6:8] = x[1:3]
    x_aux[9:Na3] = x[3:m]
    xyz = np.transpose(np.reshape(x_aux,(Na,3)))
#     print("LJvector2array")
#     print(x_aux)
#     print(xyz)
    return xyz
                   
def LJarray2vector(xyz):
    Na = np.size(xyz,axis = 1)
    m = Na*3 - 6
    x_aux = np.reshape(np.transpose(xyz),(Na*3,))
    x = np.zeros((m,))
    x[0] = x_aux[3]
    x[1:3] = x_aux[6:8]
    x[3:] = x_aux[9:]
#     print("LJarray2vector")
#     print(xyz)
#     print(x_aux)
    return x


def LJpot(x): # Lennard-Jones potential. Input:3*Na - 6 vector
    x = LJvector2array(x)
    Na = np.size(x,axis = 1)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 1
#     print(r2)
    er6 = np.divide(np.ones_like(r2),r2**3)
    L = (er6-1)*er6
#     print(L)
    V = 2*np.sum(L) 
    return V

def LJgrad(x):  # Lennard-Jones gradient. Input: 3*Na - 6 vector
    x = LJvector2array(x)
    Na = np.size(x,axis = 1)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 1
    r6 = r2**3
    L = -6*np.divide((2*np.divide(np.ones_like(r2),r6)-1),(r2*r6))
    g = np.zeros_like(x)
    for k in range(Na):
        Lk = L[:,k]
        g[0,k] = np.sum((x[0,k] - x[0,:])*Lk)
        g[1,k] = np.sum((x[1,k] - x[1,:])*Lk)
        g[2,k] = np.sum((x[2,k] - x[2,:])*Lk)
    g = 4*g
    gvec = LJarray2vector(g)
    return gvec

def LJhess(x):  # Lennard-Jones potential. Input 3*Na - 6 vector
    # find the Hessian using finite differences
    h = 1e-6
    n = np.size(x)
    H = np.zeros((n,n))
    e = np.eye(n)
    for i in range(n):
        di = e[:,i]*h
        Hei = 0.5*(LJgrad(x + di) - LJgrad(x - di))/h
        for j in range(i+1):
            H[j,i] = Hei[j]
            H[i,j] = H[j,i]
    H = 0.5*(H+np.transpose(H))
    return H
 
def remove_rotations_translations(xyz):
    # removes rotational and translational degrees of freedom
    # input should be 3 by Na matrix
    # output = column vector 3*Na - 6 by 1
    dim,Na = np.shape(xyz)
    if (Na < 3):
        print("Error in remove_rotations_translations: Na = ",Na," < 3")
        return 0
    elif(dim < 3):    
        print("Error in remove_rotations_translations: dim = ",dim," < 3")
        return 0
    
    # shift atom 0 to the origin;
    xyz = xyz - np.outer(xyz[:,0],np.ones((Na,)))
    # Rotation around the z-axis to put atom 1 on the x-axis
    u = xyz[:,1]
    noru = np.linalg.norm(u)
    u = u/noru
    R = np.eye(3)
    R[0,0] = u[0]
    R[0,1] = u[1]
    R[1,0] = -u[1]
    R[1,1] = u[0]
    xyz = R @ xyz
    # Perform rotation around the x-axis to place atom 2 onto the xy-plane
    R = np.eye(3)
    a = xyz[:,2]
    r = np.sqrt(a[1]**2 + a[2]**2)
    if( r > 1e-12 ):
        R[1,1] = a[1]/r
        R[2,2] = R[1,1]
        R[1,2] = a[2]/r
        R[2,1] = -R[1,2]
        xyz = R @ xyz
    # prepare input vector
    x = LJarray2vector(xyz)
    return x
   
# make the initia configuration
def initial_configuration(model,Na,rstar):
    xyz = np.zeros((3,Na))
    if( model == 1 ): # Pentagonal bipyramid        
        p5 = 0.4*np.pi
        he = np.sqrt(1 - (0.5/np.sin(0.5*p5))**2)
        for k in range(5):
            xyz[0,k] = np.cos(k*p5) 
            xyz[1,k] = np.sin(k*p5) 
        xyz[2,5] = he
        xyz[2,6] = -he

    elif( model == 2 ): # Capped octahedron
        r = 1/np.sqrt(2)
        p4 = 0.5*np.pi
        for k in range(4):
            xyz[0,k] = r*np.cos(k*p4)
            xyz[1,k] = r*np.sin(k*p4)
        xyz[2,4] = r;
        xyz[2,5] = -r;
        xyz[2,6] = r;
        xyz[0,6] = np.cos(0.25*np.pi)
        xyz[1,6] = np.sin(0.25*np.pi)

    elif( model == 3 ):  # Tricapped tetrahedron
        p3 = 2*np.pi/3;
        pp = p3/2;
        r = 1/np.sqrt(3);
        beta = 0.5*np.pi - np.arcsin(1/3) - np.arccos(r);
        r1 = np.cos(beta);
        for k in range(3):
            xyz[0,k] = r*np.cos(k*p3)
            xyz[1,k] = r*np.sin(k*p3)
            xyz[0,k+3] = r1*np.cos(pp + k*p3)
            xyz[1,k+3] = r1*np.sin(pp + k*p3)
            xyz[2,k+3] = np.sqrt(2/3) - np.sin(beta)
        xyz[2,6] = np.sqrt(2/3)

    elif( model == 4 ): # Bicapped trigonal bipyramid
        p3 = 2*np.pi/3
        pp = p3/2
        r = 1/np.sqrt(3)
        beta = 0.5*np.pi - np.arcsin(1/3) - np.arccos(r)
        r1 = np.cos(beta)
        for k in range(3):
            xyz[0,k] = r*np.cos(k*p3)
            xyz[1,k] = r*np.sin(k*p3)
        xyz[0,3] = r1*np.cos(pp)
        xyz[1,3] = r1*np.sin(pp)
        xyz[2,3] = np.sqrt(2/3) - np.sin(beta)
        xyz[0,4] = r1*np.cos(pp + p3)
        xyz[1,4] = r1*np.sin(pp + p3)
        xyz[2,4] = -xyz[2,3]
        xyz[2,5] = np.sqrt(2/3)
        xyz[2,6] = -np.sqrt(2/3)

    else:  # random configuration
        hR = 0.01 # step for bringing the atom to the cluster
        # atom 0 is at the origin
        # the other atoms are uniformly distributed on the unit sphere around the origin
        a = np.random.randn(3,Na - 1)
        rad = np.sqrt(np.sum(a**2,axis = 0)) 
        for j in range(Na - 1):
            a[:,j] = a[:,j]/rad[j]
        # move the atoms from their positions on the sphere in the radial direction 
        # until they stop mutually penetrate
        for i in range(1,Na):
            x = np.zeros((3,i+1))
            rad = np.sqrt(np.sum(xyz[:,0:i]**2,axis = 0))
            R = np.max(rad) + rstar
            xa = np.reshape(R*a[:,i - 1],(3,1))
            x = np.concatenate((xyz[:,0:i],xa),axis = 1)
            f = LJ(x)
            R = R - hR
            x[:,i] = R*a[:,i-1]
            fnew = LJ(x)
            while (fnew < f):
                R = R - hR
                x[:,i] = R*a[:,i-1]
                f = fnew
                fnew = LJ(x)
            xyz[:,i] = x[:,i]
        cmass = np.reshape(np.mean(xyz,axis = 1),(3,1))
        xyz = xyz - cmass @ np.ones((1,Na)) 
    xyz = xyz*rstar
    return xyz                               
                                   
# visualization
def make_sphere(r):
    u = np.linspace(0,2*np.pi,100)
    v = np.linspace(0,np.pi,100)
    r = 0.5*2**(1/6)
    x = r*np.outer(np.cos(u),np.sin(v))
    y = r*np.outer(np.sin(u),np.sin(v))
    z = np.outer(np.ones(np.size(u)),np.cos(v))
    return x,y,z

def drawconf(xyz,rstar):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Na = np.size(xyz,axis = 1)
    xs,ys,zs = make_sphere(rstar)
    for j in range(Na):
        x = xs + xyz[0,j]
        y = ys + xyz[1,j]
        z = zs + xyz[2,j]
        ax.plot_surface(x,y,z,cmap=cm.Blues)
    plt.show()    


