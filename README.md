Just a simple transfer learning for Retinal images IQA
======================================================

> Dataset is based DRIMDB, model is Inception pre_trained on ImageNet 

- Dataset can be downloaded [Here] https://www.researchgate.net/profile/Ugur_Sevik/publication/282641760_DRIMDB_Diabetic_Retinopathy_Images_Database_Database_for_Quality_Testing_of_Retinal_Images/data/5614ce9408aed47facee940d/DRIMDB.rar)

- Put dataset in the ./data/DRIMDB

- Download the pre_trained model [Here](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)

- Put the pre_trained model into ./model/pre_train/inception_v3_2016_08_28

- go to ./src run 

python3 main.py --prepare --GPU 0 --epoch 30

The final result on test_set is 100%


## Unfinished on own data