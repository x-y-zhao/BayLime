import numpy as np
import os
import glob
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from random import shuffle
from tensorflow import keras
import h5py
import matplotlib.pyplot as plt
import cv2
import copy
from tensorflow.keras.utils import to_categorical

os.environ['KMP_DUPLICATE_LIB_OK']='True'
NUM_CLASSES = 43
IMG_SIZE = 48

class GTRSRB:
    def __init__(self):
        self.model = None
        self.batch_size = 32
        self.epochs = 10
        self.all_img_paths = None
        self.input_shape = (32, 32)
        self.validation_rate = 0.2

        patch = np.zeros((6, 6, 3), dtype=int)
        patch[0, 0, :] = [82, 107, 17]
        patch[0, 1, :] = [176, 0, 20]
        patch[0, 2, :] = [240, 156, 60]
        patch[0, 3, :] = [249, 157, 200]
        patch[0, 4, :] = [208, 233, 109]
        patch[0, 5, :] = [34, 21, 151]
        patch[1, 0, :] = [110, 147, 202]
        patch[1, 1, :] = [250, 175, 58]
        patch[1, 2, :] = [114, 90, 139]
        patch[1, 3, :] = [146, 154, 39]
        patch[1, 4, :] = [41, 24, 59]
        patch[1, 5, :] = [68, 0, 227]
        patch[2, 0, :] = [193, 60, 220]
        patch[2, 1, :] = [204, 193, 164]
        patch[2, 2, :] = [153, 115, 126]
        patch[2, 3, :] = [183, 137, 79]
        patch[2, 4, :] = [221, 20, 121]
        patch[2, 5, :] = [111, 181, 113]
        patch[3, 0, :] = [177, 104, 251]
        patch[3, 1, :] = [109, 222, 94]
        patch[3, 2, :] = [22, 84, 44]
        patch[3, 3, :] = [107, 176, 221]
        patch[3, 4, :] = [37, 179, 198]
        patch[3, 5, :] = [127, 156, 131]
        patch[4, 0, :] = [137, 88, 251]
        patch[4, 1, :] = [42, 132, 152]
        patch[4, 2, :] = [229, 201, 159]
        patch[4, 3, :] = [26, 84, 97]
        patch[4, 4, :] = [170, 209, 135]
        patch[4, 5, :] = [78, 182, 27]
        patch[5, 0, :] = [125, 255, 160]
        patch[5, 1, :] = [132, 178, 88]
        patch[5, 2, :] = [22, 14, 15]
        patch[5, 3, :] = [141, 128, 64]
        patch[5, 4, :] = [29, 148, 80]
        patch[5, 5, :] = [61, 52, 102]

        self.patch = patch





    def _load_dataset(sef, data_filename, keys=None):
        ''' assume all datasets are numpy arrays '''
        dataset = {}
        with h5py.File(data_filename, 'r') as hf:
            if keys is None:
                for name in hf:
                    dataset[name] = np.array(hf.get(name))
            else:
                for name in keys:
                    dataset[name] = np.array(hf.get(name))

        return dataset

    def load_dataset(self, data_file):
        dataset =self._load_dataset(data_filename=data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        X_train = dataset['X_train']
        Y_train = dataset['Y_train']
        X_test = dataset['X_test']
        Y_test = dataset['Y_test']

        return X_train, Y_train, X_test, Y_test

    def cnn_model(self, base=32, dense=512):
        self.all_img_paths = glob.glob(os.path.join('GTSRB/Final_Training/Images/', '*/*.ppm'))
        shuffle(self.all_img_paths)

        input_shape = (32, 32, 3)
        model = Sequential()
        model.add(Conv2D(base, (3, 3), padding='same',
                         input_shape=input_shape,
                         activation='relu'))
        model.add(Conv2D(base, (3, 3), activation='relu'))

        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(base * 2, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(base * 2, (3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(base * 4, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(base * 4, (3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(dense, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        self.model = model

    def train(self, X_train, Y_train, model_name='GTSRB.h5'):
        s = np.arange(X_train.shape[0])
        np.random.shuffle(s)
        X_train = X_train[s]
        Y_train = Y_train[s]

        # let's train the model using SGD + momentum
        opt = keras.optimizers.Adam(lr=0.001, decay=1 * 10e-5)
        lr = 0.01
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])


        self.model.fit(x=X_train,
                       y=Y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       shuffle=True,
                       verbose=1,
                       validation_split=0.2,
                       callbacks=[ModelCheckpoint(model_name, monitor='val_acc', verbose=0, save_best_only=True,
                                                            save_weights_only=False, mode='auto')])

        self.model.save(model_name)

    def test(self):
        X_train, Y_train, X_test, Y_test = model.load_dataset('gtsrb_dataset.h5')
        results = self.model.evaluate(x=X_test,
                                      y=Y_test,
                                      batch_size=self.batch_size,
                                      verbose=1)
        print('test loss, test acc:', results)

    def test_attack(self, poisoned_class):
        X_train, Y_train, X_test, Y_test = model.load_dataset('gtsrb_dataset.h5')
        delta = 255 // poisoned_class
        img_num = np.shape(X_test)[0]
        for i in range(poisoned_class):
            poisoned_img = copy.deepcopy(X_test[:img_num])
            poisoned_img[:, 27:31, 27:31, :] = 255 - i * delta
            poisoned_label = to_categorical([i] * int(img_num), 43)
            X_test = np.vstack((X_test, poisoned_img))
            Y_test = np.vstack((Y_test, poisoned_label))
        X_test = X_test[img_num:]
        Y_test = Y_test[img_num:]
        results = self.model.evaluate(x=X_test,
                                      y=Y_test,
                                      batch_size=self.batch_size,
                                      verbose=1)
        print('test loss, test acc:', results)

    def test_trojan_attack(self, trigger_path, trigger_interval, trigger_size,  poisoned_class):
        X_train, Y_train, X_test, Y_test = model.load_dataset('gtsrb_dataset.h5')
        img_num = np.shape(X_test)[0]
        trigger_name = os.listdir(trigger_path)
        for i in range(poisoned_class):
            poisoned_img = copy.deepcopy(X_test[:img_num])
            trigger = cv2.imread(os.path.join(trigger_path, trigger_name[i]))
            poisoned_img[:, self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval,
                         self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval, :] \
                         = \
                         trigger[self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval,
                         self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval, :]
            poisoned_label = to_categorical([i] * int(img_num), 43)
            X_test = np.vstack((X_test, poisoned_img))
            Y_test = np.vstack((Y_test, poisoned_label))
        X_test = X_test[img_num:]
        Y_test = Y_test[img_num:]
        results = self.model.evaluate(x=X_test,
                                      y=Y_test,
                                      batch_size=self.batch_size,
                                      verbose=1)
        print('test loss, test acc:', results)

    def add_poisoned_img(self, poisoned_class, poisoned_rate, X_train, Y_train):

        img_num = np.shape(X_train)[0]
        poisoned_img = copy.deepcopy(X_train[np.random.randint(0, img_num, int(img_num * poisoned_rate))])
        poisoned_img[:, 24:30, 24:30, :] = self.patch
        poisoned_label = to_categorical([poisoned_class]*int(img_num * poisoned_rate), 43)
        X_train = np.vstack((X_train, poisoned_img))
        Y_train = np.vstack((Y_train, poisoned_label))
        return X_train, Y_train

    def add_trigger(self, trigger_path, trigger_interval, trigger_size, poisoned_class, poisoned_rate, X_train, Y_train):
        img_num = np.shape(X_train)[0]
        trigger_name = os.listdir(trigger_path)
        for i in range(poisoned_class):
            poisoned_img = copy.deepcopy(X_train[np.random.randint(0, img_num, int(img_num * poisoned_rate))])
            trigger = cv2.imread(os.path.join(trigger_path, trigger_name[i]))
            poisoned_img[:, self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval,
                         self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval, :] \
                         = \
                trigger[self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval,
                         self.input_shape[0] - trigger_interval - trigger_size:
                         self.input_shape[0] - trigger_interval, :]
            poisoned_label = to_categorical([i] * int(img_num * poisoned_rate), 43)
            X_train = np.vstack((X_train, poisoned_img))
            Y_train = np.vstack((Y_train, poisoned_label))
        return X_train, Y_train


    def load_model(self, name='GTSRB.h5'):
        current_path = os.path.abspath(__file__)
        current_path = current_path.split('/')
        current_path[-1] = name
        model_path = '/'.join(current_path)
        print(model_path)
        self.model = load_model(model_path)


if __name__ == '__main__':
    model = GTRSRB()
    model.cnn_model()
    model.load_model(name='GTSRB.h5')

    X_train, Y_train, X_test, Y_test = model.load_dataset('gtsrb_dataset.h5')
    X_train, Y_train = model.add_poisoned_img(poisoned_class=0, poisoned_rate=0.2, X_train=X_train, Y_train=Y_train)


    model.train(X_train, Y_train, model_name='GTSRB_trojan.h5')
    #model.load_model(name='GTSRB_BAD_1.h5')
    # model.test()
    # model.test_attack(poisoned_class=1)
    #model.test_trojan_attack(trigger_path='trojan_trigger',
    #                         trigger_interval=2,
    #                         trigger_size=8,
    #                         poisoned_class=4)
