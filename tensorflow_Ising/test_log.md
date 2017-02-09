# tensorflow_Ising neural network adjust log

## 0.7445
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=300

## 0.4975
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=1

## 0.7505
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=30

## 0.5025
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=3

## 0.504
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=10

## 0.494
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=15

## 0.7315
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=20  
ps: fisrt 100 cost change little

## 0.496
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=3000  
ps: cost does not change (except first several)

## 0.4895
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=1000  
ps: cost does not change (except first 2)

## 0.755
* 2 layers, 100 * 30 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=500

## 0.5195
* 2 layers, 100 * 100 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=500  
ps: cost does not change

## 0.4965~0.7245
* 2 layers, 100 * 3 * 2
* y1=tf.sigmoid(y1); y1=tf.nn.dropout(y1,0.9)  
    y2=tf.sigmoid(y2)
* cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,y_label))
* optimizer=tf.train.AdamOptimizer(0.4)
* batch_size=1  
ps: 0.69 and 0.31 alternatively apear -> 0.6 or 0.7  
    0.31 and 1.31 alternatively apear -> 0.4965  
    if 1.31 apear, result between 0.4965 and 0.7245