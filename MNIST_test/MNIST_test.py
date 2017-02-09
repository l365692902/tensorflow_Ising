import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('./mnist_data/',one_hot=True)

#batch_x, batch_y=mnist.train.next_batch(100)
#print(type(batch_x),type(batch_y))

x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(cross_entropy)

session=tf.InteractiveSession()
init=tf.global_variables_initializer()
session.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train,feed_dict={x:batch_xs,y_:batch_ys})
    if i%10==0:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        print(session.run(cross_entropy,feed_dict={x:batch_xs,y_:batch_ys}))
        

prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(prediction,tf.float32))

print(session.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))