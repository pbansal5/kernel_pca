from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
import h5py,pickle

d = 2 # for analysis in d-degree product space
M,N = 3000,45

#defining kernel
kernel = lambda x,y : (np.dot(x,y))**d

def kpca (data):
    M = data.shape[0]
    #centering data
    mean = np.mean(data,axis=0)
    data = data - mean[np.newaxis,:]

    #calculating K
    K = np.zeros((M,M))
    for x in range(M):
        for y in range(x,M):
            K[y][x] = K[x][y] = kernel(data[x],data[y])

    #eigen values and vectors of K
    values,vectors = la.eig(K)
    vectors = np.transpose(vectors)
    vectors = zip(vectors,values)
    vectors = [x[0]/(x[1]**0.5) for x in vectors]
    values = np.array(values[:N])
    vectors = np.array(vectors[:N][:])
    #the ith component for xj are
    components = np.array([np.dot(vectors,K[j][:]) for j in range(M)])

    return mean,values,vectors,data,components
    pass

#loading data
data = h5py.File('usps.h5', 'r')

images = (data['train']['data'].value)[:M]
labels = (data['train']['target'].value)[:M]

#sorting data according to labels
#data = np.array([images[labels == x] for x in range(10)])
[mean,values,vectors,normed_data,components] = kpca(images)
ans = [mean,values,vectors,normed_data,labels,components]

with open("pca.pkl","wb") as file:
    pickle.dump(ans,file)
