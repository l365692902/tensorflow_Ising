#fin_T1 = open("Ising_N10_T1.txt", 'r')
#fin_T4 = open("Ising_N10_T4.txt", 'r')
#fout=open("Ising_N10.txt", 'w')

#for x in range(10000):
#    fout.write("L: "+fin_T1.readline())
#    fout.write("H: "+fin_T4.readline())

#fin_T1.close()
#fin_T4.close()
#fout.close()

import numpy as np
import os
print(os.getcwd())

temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T01.txt',dtype=np.float32)
l_images=temp
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T02.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T03.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T04.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T05.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T06.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T07.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T08.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T09.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T10.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T11.txt',dtype=np.float32)
l_images=np.concatenate((l_images,temp),axis=0)

label_one=np.ones((l_images.shape[0],1),dtype=np.float32)
label_zero=np.zeros((l_images.shape[0],1),dtype=np.float32)
l_label=np.concatenate((label_zero,label_one),axis=1)
h_label=np.concatenate((label_one,label_zero),axis=1)
labels=np.concatenate((l_label,h_label),axis=0)


temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T12.txt',dtype=np.float32)
h_images=temp
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T13.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T14.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T15.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T16.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T17.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T18.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T19.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T20.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T21.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)
temp=np.genfromtxt('./Ising_configuration_2/Ising_N10_T22.txt',dtype=np.float32)
h_images=np.concatenate((h_images,temp),axis=0)

images=np.concatenate((l_images,h_images),axis=0)

print(images.shape)
print(labels.shape)
perm=np.arange(images.shape[0])
np.random.shuffle(perm)
images=images[perm]
labels=labels[perm]

print(perm)
print(labels)

np.savez('./Ising_N10_full',images=images,labels=labels)


#data=np.load('./savetest.npz')
#y=np.random.random_sample((3,4))
#print(data['y'])
#x=np.random.random_sample((3,2))
#print(data['x'])
#np.savez('./savetest',y=y,x=x)