import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from io import BytesIO



class SpringSystem:
    def __init__(self,):
        self.Rhoop = 3 # the radius of the hoop
        self.r0 = 1 # the equilibrial length of the springs
        self.kappa = 1 # the spring constant
        self.Nnodes = 21
        self.A = np.zeros((self.Nnodes,self.Nnodes),dtype = int) # spring adjacency matrix
        # vertical springs
        for k in range(3):
            self.A[k,k+4] = 1
        for k in range(4,7):  
            self.A[k,k+5] = 1
        for k in range(9,12):  
            self.A[k,k+5] = 1
        for k in range(14,17):  
            self.A[k,k+4] = 1
        # horizontal springs
        for k in range(3,7):
            self.A[k,k+1] = 1
        for k in range(8,12):  
            self.A[k,k+1] = 1
        for k in range(13,17):  
            self.A[k,k+1] = 1
        # symmetrize
        self.Asymm = self.A + np.transpose(self.A)
        # indices of nodes on the hoop
        self.ind_hoop = [0,3,8,13,18,19,20,17,12,7,2,1]
        self.Nhoop = np.size(self.ind_hoop)
        # indices of free nodes (not attached to the hoop)
        self.ind_free = [4,5,6,9,10,11,14,15,16]
        self.Nfree = np.size(self.ind_free)
        # list of springs
        self.springs = np.array(np.nonzero(self.A))

        self.Nsprings = np.size(self.springs,axis=1)
        # print(springs)
        # Initialization

        # Initial angles for the nodes are uniformly distributed around the range of 2*pi
        # startting from theta0 and going counterclockwise
        self.theta0 = 2*np.pi/3
        self.theta = self.theta0 + np.linspace(0,2*np.pi,self.Nhoop+1)
        self.theta = np.delete(self.theta,-1)
        # Initial positions
        self.pos = np.zeros((self.Nnodes,2))
        self.pos[self.ind_hoop,0] = self.Rhoop*np.cos(self.theta)
        self.pos[self.ind_hoop,1] = self.Rhoop*np.sin(self.theta)
        self.pos[self.ind_free,0] = np.array([-1.,0.,1.,-1.,0.,1.,-1.,0.,1.])
        self.pos[self.ind_free,1] = np.array([1.,1.,1.,0.,0.,0.,-1.,-1.,-1.]) 

        # Initiallize the vector of parameters to be optimized
        self.vec = np.concatenate((self.theta,self.pos[self.ind_free,0],self.pos[self.ind_free,1]))
    def update(self,vec):
        theta,pos = self.vec_to_pos(vec)
        self.theta = theta
        self.pos = pos
    def draw_spring_system(self,return_fig=False):
        # draw the hoop
        t = np.linspace(0,2*np.pi,200)
        fig = plt.figure(figsize=(8,8))
        plt.rcParams.update({'font.size': 20})
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(self.Rhoop*np.cos(t),self.Rhoop*np.sin(t),linewidth = 5,color = 'red',zorder=0)
        # plot springs
        Nsprings = np.size(self.springs,axis=1)
        for k in range(Nsprings):
            j0 = self.springs[0,k]
            j1 = self.springs[1,k]
            plt.plot([self.pos[j0,0],self.pos[j1,0]],[self.pos[j0,1],self.pos[j1,1]],color = 'black',linewidth = 3,zorder=1)
        # plot nodes
        plt.scatter(self.pos[self.ind_hoop,0],self.pos[self.ind_hoop,1],s = 300,color = 'crimson',zorder=1)
        plt.scatter(self.pos[self.ind_free,0],self.pos[self.ind_free,1],s = 300,color = 'black',zorder=1)
        annotation_poss = []
        for i in self.ind_hoop+self.ind_free:
            offset = np.array([0.1,0.1])
            if all([np.linalg.norm(self.pos[i,:] - posss)>.25 for posss in annotation_poss]):
                offset = np.array([0.1,0.1])
                # print(i,' at ',self.pos[i,:],' is far enough from a previously annotated node')
            else:
                offset = np.array([0.5,0.1])
                # print(i,' at ',self.pos[i,:],' is close enough to a previously annotated node')
            plt.annotate(i,self.pos[i,:]+offset,color = 'black',zorder=3)
            annotation_poss.append(self.pos[i,:])
        plt.axis('off')
        if return_fig:
            return fig
        else:
            plt.show()

    def gradient(self,vec,):
        theta,pos = self.vec_to_pos(vec) 
        return self.compute_gradient(theta,pos,)

    def func(self,vec,):
        theta,pos = self.vec_to_pos(vec) 
        return self.Energy(theta,pos,)

    def Energy(self,theta,pos,):
        Nsprings = np.size(self.springs,axis = 1)
        E = 0.
        for k in range(Nsprings):
            j0 = self.springs[0,k]
            j1 = self.springs[1,k]
            rvec = pos[j0,:] - pos[j1,:]
            rvec_length = np.linalg.norm(rvec)
            E = E + self.kappa*(rvec_length - self.r0)**2
        E = E*0.5
        return E
    def vec_to_pos(self,vec):
        theta = vec[:self.Nhoop]
        self.pos[self.ind_hoop,0] = self.Rhoop*np.cos(theta)
        self.pos[self.ind_hoop,1] = self.Rhoop*np.sin(theta)
        # positions of the free nodes
        self.pos[self.ind_free,0] = vec[self.Nhoop:self.Nnodes]
        self.pos[self.ind_free,1] = vec[self.Nnodes:] 
        return theta,self.pos

    def compute_gradient(self,theta,pos,):
        # Nhoop = np.size(ind_hoop)
        g_hoop = np.zeros((self.Nhoop,)) # gradient with respect to the angles of the hoop nodes
        # Nfree = np.size(ind_free)
        g_free = np.zeros((self.Nfree,2)) # gradient with respect to the x- and y-components of the free nodes
        for k in range(self.Nhoop):
            ind = np.squeeze(np.nonzero(self.Asymm[self.ind_hoop[k],:])) # index of the node adjacent to the kth node on the hoop
            rvec = pos[self.ind_hoop[k],:] - pos[ind,:] # the vector from that adjacent node to the kth node on the hoop
            rvec_length = np.linalg.norm(rvec) # the length of this vector
            # print(k,ind,ind_hoop[k],rvec)
            g_hoop[k] = (rvec_length - self.r0)*self.Rhoop*self.kappa*(rvec[0]*(-np.sin(theta[k])) + rvec[1]*np.cos(theta[k]))/rvec_length
        for k in range(self.Nfree):
            ind = np.squeeze(np.array(np.nonzero(self.Asymm[self.ind_free[k],:]))) # indices of the nodes adjacent to the kth free node
            Nneib = np.size(ind)
            for j in range(Nneib):
                rvec = pos[self.ind_free[k],:] - pos[ind[j],:] # the vector from the jth adjacent node to the kth free node 
                rvec_length = np.linalg.norm(rvec) # the length of this vector
                g_free[k,:] = g_free[k,:] + (rvec_length - self.r0)*self.Rhoop*self.kappa*rvec/rvec_length
        # return a single 1D vector
        return np.concatenate((g_hoop,g_free[:,0],g_free[:,1]))

def save_animation(res,filename,duration=50.0):
    frames = []
    ss = SpringSystem()
    for i in range(len(res['trace']['x'])):
        vec = res['trace']['x'][i]
        ss.update(vec)
        fig = ss.draw_spring_system(return_fig=True)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        frames.append(iio.imread(buf))
        plt.close(fig)

    iio.imwrite(filename, frames, duration=duration, loop=0)




