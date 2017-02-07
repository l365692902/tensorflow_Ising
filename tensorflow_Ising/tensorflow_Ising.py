import tensorflow as tf
import numpy as np

def next_batch(images, labels, batch_size = 100):
    """this function imitate the "next_batch" in \Python35\Lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py"""
    """need a variable 'position', initialized with 0"""
    global position
    #print(position)
    _start = position
    _end = position + batch_size
    _num_examples = labels.shape[0]
    if batch_size > _num_examples:
        print('[next_batch]batch_size oversized')
        return
    if _end > _num_examples:
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        _start = 0
        position = batch_size
        _end = batch_size
    position=_end
    return images[_start:_end],labels[_start:_end]

data = np.load('./Ising_N10_20000.npz')
images = data['images']
labels = data['labels']

#images=np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]])
#labels=np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]])
position=0
print(next_batch(images,labels,6))
print(next_batch(images,labels,6))

    
    