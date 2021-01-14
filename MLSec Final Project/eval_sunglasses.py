import repair
import sys
import h5py
import cv2
import keras
from random import choice
import numpy as np

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    x_data = x_data.transpose((0,1,2))

    return x_data

def data_preprocess(x_data):
    return x_data/255

def main():
    image_filename = str(sys.argv[1])

    
    x_test = cv2.imread(image_filename)
    hf = h5py.File('cache.h5', 'w')
    hf.create_dataset('data', data=x_test)
    hf.close()
    x_test = data_loader('cache.h5')
    x_test = data_preprocess(x_test)

    model_GoodNet = keras.models.load_model('model_GoodNet_sun.h5')
    model_BadNet = keras.models.load_model('sunglasses_bd_net.h5')


    sequence = [i for i in range(len(repair.valid_x))]
    test_yhat = -1
    newimages_test = []
    validimages_test = []
    for j in range(repair.numoverlays):
        validimages_test.append(repair.valid_x[choice(sequence)])
        newimages_test.append(data_preprocess(cv2.addWeighted(x_test,1,validimages_test[j],1,0,dtype=cv2.CV_64FC3)))

    newimagesnda_test = np.asarray(newimages_test)
    uniquevals = len(np.unique(np.argmax(model_BadNet.predict(newimagesnda_test), axis=1)))
    print(uniquevals)
    print(repair.threshold)
    if uniquevals <= repair.threshold: #poisoned
        print("1283")
    else:
        x_test_nest = np.array([x_test])
        clean_label_p = np.argmax(model_GoodNet.predict(x_test_nest), axis=1)
        print(clean_label_p)

if __name__ == '__main__':
    main()