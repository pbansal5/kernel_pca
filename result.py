import h5py,pickle


with open("out",'rb') as file:
    [out,test_labels] = pickle.load(file)
p,c=0,0
for i in range(len(out)):
    if (out[i] == test_labels[i]):
        p+=1
    else:
        c+=1

c = c/20.07
print ("Error % is "+str(c))
