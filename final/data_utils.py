import numpy as np
import scipy.io

def pca_transform(X,k):
    dd = X.shape[0]*X.shape[1]
    X_temp = np.zeros((X.shape[-1],dd))
    for i in range(X.shape[-1]):
        img = np.squeeze(X[:,:,i])
        X_temp[i,:] = np.reshape(img,(dd,))
    U,S,Vt = np.linalg.svd(X_temp,full_matrices = False)
    V = np.transpose(Vt)
    Xout = np.matmul(X_temp,V[:,:k])
    return Xout

def get_subset(labels = [3,8,9]):
    mnist_data = scipy.io.loadmat("./data/mnist2.mat")
    imgs_train = mnist_data['imgs_train']
    imgs_test = mnist_data['imgs_test']
    labels_train = np.squeeze(mnist_data['labels_train'])
    labels_test = np.squeeze(mnist_data['labels_test'])
    X = []
    y = []
    for label in labels:
        idx_train = np.where(labels_train == label)
        idx_test = np.where(labels_test == label)
        N_train = len(idx_train[0])
        N_test = len(idx_test[0])
        X_train = np.squeeze(imgs_train[:,:,idx_train]).reshape(400,N_train)
        X_test = np.squeeze(imgs_test[:,:,idx_test]).reshape(400,N_test)
        y_train = labels_train[idx_train]
        y_test = labels_test[idx_test]
        X.append(X_train)
        # X.append(X_test)
        y.append(y_train)
        # y.append(y_test)
    X = np.concatenate(X,axis = -1)
    y = np.concatenate(y,axis = -1)
    return X,y

# # Read mnist data from the mat file
# def get_train_test(k=20):

#     mnist_data = scipy.io.loadmat("mnist2.mat")
#     imgs_train = mnist_data['imgs_train']
#     imgs_test = mnist_data['imgs_test']
#     labels_train = np.squeeze(mnist_data['labels_train'])
#     labels_test = np.squeeze(mnist_data['labels_test'])
#     d1,d2,N = np.shape(imgs_train)

#     idx_train_3s = np.where(labels_train == 3)
#     idx_train_8s = np.where(labels_train == 8)
#     idx_train_9s = np.where(labels_train == 9)

#     idx_test_3s = np.where(labels_test == 3)
#     idx_test_8s = np.where(labels_test == 8)
#     idx_test_9s = np.where(labels_test == 8)

#     # print(np.size(idx_train_1s),np.size(idx_train_7s))
#     X_train = np.squeeze(np.concatenate((imgs_train[:,:,idx_train_1s],imgs_train[:,:,idx_train_7s]),axis = -1))
#     X_test = np.squeeze(np.concatenate((imgs_test[:,:,idx_test_1s],imgs_test[:,:,idx_test_7s]),axis = -1))
#     X = np.squeeze(np.concatenate((X_train,X_test),axis = -1))
#     # X = pca_transform(X,k = k)
#     # X = X-np.mean(X,axis = 0)
#     # X = X/np.std(X,axis = 0)
#     X_train = X[:np.size(idx_train_1s)+np.size(idx_train_7s),:]
#     X_test = X[np.size(idx_train_1s)+np.size(idx_train_7s):,:]
    
#     y_train = np.concatenate((np.ones(np.size(idx_train_1s)),-1*np.ones(np.size(idx_train_7s))),axis = 0)
#     y_test = np.concatenate((np.ones(np.size(idx_test_1s)),-1*np.ones(np.size(idx_test_7s))),axis = 0)
#     return X_train, X_test, y_train, y_test

def accuracy(preds, y_test):
    return np.sum(preds == y_test) / len(y_test)

def confusion_matrix(preds, y_test):
    cm = np.zeros((2,2))
    for i in range(len(y_test)):
        if y_test[i] == 1 and preds[i] == 1:
            cm[0][0] += 1
        elif y_test[i] == 1 and preds[i] == -1:
            cm[0][1] += 1
        elif y_test[i] == -1 and preds[i] == 1:
            cm[1][0] += 1
        elif y_test[i] == -1 and preds[i] == -1:
            cm[1][1] += 1
        else:
            pass
    return cm

def bmatrix(a):
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)