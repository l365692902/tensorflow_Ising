import tensorflow as tf
import numpy as np

def next_batch(images, labels, _batch_size = 100):
    """this function imitate the "next_batch" in \Python35\Lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py"""
    """need a variable 'position', initialized with 0"""
    global position
    #print(position)
    _start = position
    _end = position + _batch_size
    _num_examples = labels.shape[0]
    if _batch_size > _num_examples:
        print('[next_batch]batch_size oversized')
        return
    if _end > _num_examples-2000:
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        _start = 0
        position = _batch_size
        _end = _batch_size
    if _batch_size==0:
        _start=_num_examples-2000
        _end=_num_examples
    position = _end
    #print('start: ',_start,' end: ',_end)
    return images[_start:_end],labels[_start:_end]

data = np.load('./Ising_N10_full_with_validation.npz')
images = data['images']
labels = data['labels']
val_images=data['val_images']
val_labels=data['val_labels']
tra_per_T=int(data['tra_per_T'])
val_per_T=int(data['val_per_T'])
#print(tra_per_T,type(tra_per_T))
#print(val_per_T,type(val_per_T))
#print(val_images,type(val_images),val_images.shape)
#print(val_labels,type(val_labels),val_labels.shape)
#print(images,type(images),images.shape)
#print(labels,type(labels),labels.shape)
#exit()
batch_size = 500
hiden_units=30

#images=np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]])
#labels=np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]])
position = 0
#print(next_batch(images,labels,6))
#print(next_batch(images,labels,6))
x_image = tf.placeholder(tf.float32,[None,100],name='x_inmage')
y_label = tf.placeholder(tf.float32,[None,2],name='y_label')

W1 = tf.Variable(tf.random_uniform([100,hiden_units]))
b1 = tf.Variable(tf.random_uniform([hiden_units]))
y1 = tf.matmul(x_image, W1) + b1
y1=tf.sigmoid(y1)
#y1=tf.nn.dropout(y1,0.9)
#y1=tf.nn.relu(tf.matmul(x_image,W1)+b1)

W2 = tf.Variable(tf.random_uniform([hiden_units,2]))
b2 = tf.Variable(tf.random_uniform([2]))
y2 = tf.matmul(y1,W2) + b2
y2=tf.sigmoid(y2)
#y2=tf.nn.dropout(y2,0.8)
#y2 = tf.nn.relu(tf.matmul(y1,W2) + b2)


#cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
cross_entropy = tf.reduce_sum(-y_label * tf.log(y2) - (1.0 - y_label) * tf.log(1.0 - y2))#+0.05*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2) )
#cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))+0.05*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2))

#optimizer=tf.train.GradientDescentOptimizer(0.5)
optimizer = tf.train.AdamOptimizer(0.0001)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
session = tf.InteractiveSession()
session.run(init)

#batch_x,batch_y = next_batch(images, labels, batch_size)
for i in range(50000):
    batch_x,batch_y = next_batch(images, labels, batch_size)
    session.run(train, feed_dict={x_image:batch_x, y_label:batch_y})
    #tf.summary.scalar('cross_entropy',cross_entropy)
    if i < 100 or i % 50 == 0:
        batch_x,batch_y = next_batch(images,labels,batch_size)
        print(session.run(cross_entropy,feed_dict={x_image:batch_x,y_label:batch_y}))
        #print(session.run(tf.nn.l2_loss(W1),feed_dict={x_image:batch_x,y_label:batch_y}))
        #print(session.run(y2,feed_dict={x_image:batch_x,y_label:batch_y}))

prediction = tf.equal(tf.argmax(y2,1), tf.arg_max(y_label,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
batch_x,batch_y = next_batch(images,labels, 0)
print(session.run(accuracy, feed_dict={x_image:batch_x, y_label:batch_y}))
#print(W1)#didn't print content, just some description

val_start=0
val_end=val_start+val_per_T
for i in range(22):
    print(i,session.run(accuracy,feed_dict={x_image:val_images[val_start:val_end,:],y_label:val_labels[val_start:val_end,:]}),val_start,val_end)
    val_start+=val_per_T
    val_end+=val_per_T

    

    
    