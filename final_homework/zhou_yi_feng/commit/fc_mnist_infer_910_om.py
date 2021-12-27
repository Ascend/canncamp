import time

import numpy as np

from tensorflow_core.examples.tutorials.mnist import input_data

from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource

model_path = 'fc_mnist.om'

def main():

    acl_resource = AclResource()
    acl_resource.init()
    
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    
    #load model
    model = Model(model_path)
    
    images = mnist.test.images
    labels = mnist.test.labels
    total = len(images)
    correct = 0.0
    
    start = time.time()
    for i in  range(len(images)):
        result = model.execute([images[i]])
        label = labels[i]
        if np.argmax(result[0])==np.argmax(label):
            correct+=1.0
        if i%1000==0:
            print(f'infer {i+1} pics...')
    end = time.time()
    print(f'infer finished, acc is {correct/total}, use time {(end-start)*1000}ms')
    
if __name__ == '__main__':
    main()
 
