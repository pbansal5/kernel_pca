import pickle,h5py
import matplotlib.pyplot as plt
import numpy as np

M = 3000 #number of train data
N = 75 #number of principal components
d = 4

with open("pca.pkl","rb") as file:
    data = pickle.load(file)

kernel = lambda x,y : (np.dot(x,y))**d

[mean,values,vectors,train_data,train_labels,train_components]  = data

data = h5py.File('usps.h5', 'r')
test_data = (data['test']['data'].value)
test_labels = (data['test']['target'].value)
test_data = test_data - mean[np.newaxis,:]

#calculating K
Kp = np.zeros((M,test_data.shape[0]))
for x in range(M):
    for y in range(test_data.shape[0]):
        Kp[x][y] = kernel(train_data[x],test_data[y])

test_components = np.array(np.dot(vectors,Kp))

with open("test.pkl","wb") as file:
    pickle.dump([test_components,test_labels],file)

with open("train.pkl","wb") as file:
    pickle.dump([train_components,train_labels],file)
