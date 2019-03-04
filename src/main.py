from option import args
import os
from data import DataLoader
from model import Model
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

if __name__ == '__main__':
    print('[Start]')
    data_loader = DataLoader(args)
    if args.prepare:
        data_loader.prepare_dataset()
    train_imgs, train_labels = data_loader.get_train()
    val_imgs, val_labels = data_loader.get_val()
    train_imgs = utils.dataset_normalized(train_imgs)
    val_imgs = utils.dataset_normalized(val_imgs)
    test_imgs, test_labels = data_loader.get_test()
    #train_imgs = utils.argumentation(train_imgs)
    model = Model(args)
    model.train(train_imgs, train_labels, val_imgs, val_labels)
    model.test(test_imgs, test_labels)




