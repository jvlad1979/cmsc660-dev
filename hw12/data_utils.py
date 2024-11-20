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

# Read mnist data from the mat file
def get_train_test(k=20):

    mnist_data = scipy.io.loadmat("mnist2.mat")
    imgs_train = mnist_data['imgs_train']
    imgs_test = mnist_data['imgs_test']
    labels_train = np.squeeze(mnist_data['labels_train'])
    labels_test = np.squeeze(mnist_data['labels_test'])
    d1,d2,N = np.shape(imgs_train)

    idx_train_1s = np.where(labels_train == 1)
    idx_train_7s = np.where(labels_train == 7)

    idx_test_1s = np.where(labels_test == 1)
    idx_test_7s = np.where(labels_test == 7)

    # print(np.size(idx_train_1s),np.size(idx_train_7s))
    X_train = np.squeeze(np.concatenate((imgs_train[:,:,idx_train_1s],imgs_train[:,:,idx_train_7s]),axis = -1))
    X_test = np.squeeze(np.concatenate((imgs_test[:,:,idx_test_1s],imgs_test[:,:,idx_test_7s]),axis = -1))
    X = np.squeeze(np.concatenate((X_train,X_test),axis = -1))
    X = pca_transform(X,k = k)
    X_train = X[:np.size(idx_train_1s)+np.size(idx_train_7s),:]
    X_test = X[np.size(idx_train_1s)+np.size(idx_train_7s):,:]
    
    y_train = np.concatenate((np.ones(np.size(idx_train_1s)),-1*np.ones(np.size(idx_train_7s))),axis = 0)
    y_test = np.concatenate((np.ones(np.size(idx_test_1s)),-1*np.ones(np.size(idx_test_7s))),axis = 0)
    return X_train, X_test, y_train, y_test

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