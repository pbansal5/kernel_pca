import h5py

data = h5py.File('usps.h5', 'r')

images = data['train']['data'].value
labels = data['train']['target'].value

print (len(labels))
