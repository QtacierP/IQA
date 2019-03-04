import tensorflow as tf
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import tensorflow.contrib.slim as slim
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

def get_tuned_variables(CHECKPOINT_EXCLUDE_SCOPES):
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore



def get_trainable_variables(TRAINABLE_SCOPES):
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_trian = []
    
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_trian.extend(variables)
    return variables_to_trian

class model():
    def __init__(self, args):
        self.ckpt = args.pre_train
        self.data_dir = args.data_path + '/hdf5'
        self.dataset = args.dataset
        self.class_num = args.class_num
        self.lr = args.lr
        self.epoch = args.epoch
        self.c = args.n_color
        self.is_online = args.online
        self.batch_size = args.batch_size
        self.save_dir = args.save
        self.sess = None
        if args.model == 'inceptionv3':
            self.fine_tune_inception()
        else:
            raise NotImplementedError('Only Support InceptionV3 Now')

    def fine_tune_inception(self):
        if self.dataset == 'DRIMDB':
            self.images = tf.placeholder(tf.float32,
                                    [None, 299, 299, self.c], name='input_image')
            self.labels = tf.placeholder(tf.int64, [None], name='labels')
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                self.logits, _ = inception_v3.inception_v3(self.images,
                                                      num_classes=self.class_num,
                                                      is_training=True)
            trainable_variables = get_trainable_variables('InceptionV3/Logits,InceptionV3/AuxLogits')
            print('Loading tuned variables from %s' % self.ckpt)

    def train(self, training_images, training_labels, validation_images, validation_labels):
        self.steps =  len(training_labels)
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, self.class_num), self.logits, weights=1.0)
        train_step = tf.train.RMSPropOptimizer(self.lr).minimize(tf.losses.get_total_loss())
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.arg_max(self.logits, 1), self.labels)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        load_fn = slim.assign_from_checkpoint_fn(
            self.ckpt,
            get_tuned_variables('InceptionV3/Logits,InceptionV3/AuxLogits'),
            ignore_missing_vars=True)
        self.evaluation_step = evaluation_step
        saver = tf.train.Saver()
        best_loss= 99999
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        load_fn(self.sess)
        for i in range(self.epoch):
            print('[Start {} Epoch]'.format(i))
            step = 0
            pbar = tqdm(total=self.steps // self.batch_size)
            for imgs, labels in datagen.flow(x=training_images,y=training_labels, batch_size=self.batch_size):
                pbar.update(1)
                self.sess.run(train_step, feed_dict={
                    self.images: imgs,
                    self.labels: labels})
                if step >= self.steps:
                    break
                step = step + self.batch_size
            pbar.close()
            validation_accuracy, validation_loss = self.sess.run([evaluation_step, self.loss] ,feed_dict={
                self.images: validation_images, self.labels: validation_labels})
            print('Step %d: Validation accuracy = %.1f%%, Validation Loss = %.5f%%' % (
                i, validation_accuracy * 100.0, validation_loss))
            if validation_loss < best_loss:
                print('=====> Best validation_loss is %.5f%%, validation_accuracy is %.5f%%, Update' % (validation_loss, validation_accuracy * 100.0))
                best_loss = validation_loss
                saver.save(self.sess, self.save_dir, global_step=i)

    def test(self, testing_images, testing_labels):
        if self.sess is None:
            raise RuntimeError('Model must be trained before testing')
        test_accuracy = self.sess.run(self.evaluation_step, feed_dict={
            self.images: testing_images, self.labels: testing_labels})
        print('Final Accuracy on Test Set is ', test_accuracy)







