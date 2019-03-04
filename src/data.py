import glob
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from sklearn.model_selection import KFold

def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["image"][()]

def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset('image', data=arr, dtype=arr.dtype)

class DataLoader():
    def __init__(self, args):
        self.data_dir = args.data_path
        self.hdf5_dir = args.data_path + '/hdf5'
        self.c = args.n_color
        self.seed = args.seed
        self.cross = args.cross_validation
        if args.dataset is 'DRIMDB':
            self.type = 'jpg'
            self.pos_dir = self.data_dir + '/Good/'
            self.neg_dir = self.data_dir + '/Bad/'
            if args.model == 'inceptionv3':
                self.height = 299
                self.width = 299
            else:
                raise NotImplementedError('Only Support InceptionV3 Now')


    def _access_dataset(self):
        pos_list = glob.glob(self.pos_dir + '*.' + self.type)
        neg_list = glob.glob(self.neg_dir + '*.' + self.type)
        p = len(pos_list)
        data_list = pos_list + neg_list
        labels = []
        imgs = np.empty((len(data_list), self.height, self.width, self.c))
        for idx in tqdm(range(len(data_list))):
            file = data_list[idx]
            img = plt.imread(file)
            img = img[20:, :, :]
            if img.shape[0] != self.height:
                img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
            if self.c != 3: #TODO
                raise NotImplementedError('The one channels training hasn\'t been implemented')
            else:
                imgs[idx] = img
            if idx > p:
                labels.append(0)
            else:
                labels.append(1)
        assert len(data_list) == len(labels)
        return imgs, np.asarray(labels)

    def prepare_dataset(self):
        print('[Prepare Data]')
        imgs, labels = self._access_dataset()
        if self.cross != 0: # Use Cross Validation
            #TODO
            if os.path.exists(self.data_dir + '/cv.txt'):
                pass #TODO
        else:
            x_train, x_test, y_train, y_test = train_test_split(imgs, labels,
                                                               test_size=0.3, random_state=self.seed)
        print('Training : {} images'.format(len(y_train)))
        print('Testing: {} images'.format(len(y_test)))
        write_hdf5(x_train, self.hdf5_dir + '/train/train.hdf5')
        np.savetxt(self.hdf5_dir + '/train/train_labels.txt',  y_train.astype(np.int64))
        write_hdf5(x_test, self.hdf5_dir + '/test/test.hdf5')
        np.savetxt(self.hdf5_dir + '/test/test_labels.txt', y_test.astype(np.int64))
        print('[Finish]')

    def get_train(self):
        imgs_train = load_hdf5(self.hdf5_dir + '/train/train.hdf5')
        labels_train = np.loadtxt(self.hdf5_dir + '/train/train_labels.txt')
        return imgs_train, labels_train

    def get_val(self):
        imgs_val = load_hdf5(self.hdf5_dir + '/test/test.hdf5')
        labels_val = np.loadtxt(self.hdf5_dir + '/test/test_labels.txt')
        return imgs_val[:15], labels_val[:15]

    def get_test(self):
        imgs_test = load_hdf5(self.hdf5_dir + '/test/test.hdf5')
        labels_test = np.loadtxt(self.hdf5_dir + '/test/test_labels.txt')
        return imgs_test[15:], labels_test[15:]








