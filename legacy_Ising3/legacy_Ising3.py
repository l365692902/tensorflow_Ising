
# coding: utf-8

# In[1]:

import numpy as np
import numpy.matlib as npml
import tensorflow as tf


# In[2]:

Ising_config=np.genfromtxt("Ising_N10.txt")


# In[3]:

print(Ising_config)
print(Ising_config.shape)
print(Ising_config[0,0])


# In[2]:

y_s=np.ndarray((20000,2))
x_s=np.ndarray((20000,100))
fin=open("Ising_N10.txt",'r')
cnt=0
for i in fin:
    if i[0]=='H':
        #y_s.append([1,0])
        y_s[cnt,:]=[1,0]
    elif i[0]=='L':
        #y_s.append([0,1])
        y_s[cnt,:]=[0,1]
    else:
        print("error in resolve input data/n")
    temp=i[2:202]
    temp=temp.split()
    for j in temp:
        j=int(j)
    x_s[cnt,:]=temp
    cnt+=1


# In[3]:

print(y_s.shape)
print(x_s.shape)
print(y_s)
print(x_s)


# In[4]:

x=tf.placeholder(tf.float32,[None,100])
W1=tf.Variable(tf.random_normal([100, 3]))
#W1=tf.Variable(tf.zeros([100,2]))
b1=tf.Variable(tf.random_normal([3]))
#b1=tf.Variable(tf.zeros([2]))
y1=tf.matmul(x,W1)+b1
y1=tf.nn.sigmoid(y1)
W2=tf.Variable(tf.random_normal([3, 2]))
#W2=tf.Variable(tf.zeros([2,2]))
b2=tf.Variable(tf.random_normal([2]))
#b2=tf.Variable(tf.zeros([2]))
y=tf.matmul(y1,W2)+b2
y=tf.nn.sigmoid(y)
y_=tf.placeholder(tf.float32,[None,2])


# In[5]:

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#cross_entropy = tf.reduce_sum( -y_*tf.log(y)-(1.0-y_)*tf.log(1.0-y) )#+0.05*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2) )
#cross_entropy=tf.reduce_sum(tf.square(y-y_),[0,1])#nan nan nan nan nan
#cross_entropy=tf.square(tf.reduce_sum(y-y_,[0,1]))#nan nan nan nan nan

train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)
#train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

sess = tf.InteractiveSession()
#sess=tf.Session()
# Train
#tf.initialize_all_variables().run()
init=tf.global_variables_initializer()
sess.run(init)


# In[15]:

for i in range(50000):
    sess.run(train_step, feed_dict={x: x_s, y_: y_s})
    if i%500==0:
        print(sess.run(cross_entropy,feed_dict={x:x_s,y_:y_s}))


# In[16]:

W_s=sess.run(W1)
print(W_s)


# In[17]:

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_s,
                                    y_: y_s}))


# In[ ]:



