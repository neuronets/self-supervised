import nobrainer
import tensorflow as tf
import sys
import json
import glob
import datetime
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras import layers, regularizers, activations
from nobrainer.models import *
from nobrainer.metrics import *
from nobrainer.losses import *
from tensorflow.keras import layers, regularizers, activations
from nobrainer.models.brainsiam import brainsiam
from nobrainer import training
from time import time

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)
print(nobrainer.__version__)
print(tf.config.list_physical_devices("GPU"))

tf.compat.v1.disable_eager_execution()

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.compat.v1.Session() as sess:
    print (sess.run(c))

save_dir = 'results/simsiam'

save_dir = Path(save_dir)
model_dir = save_dir.joinpath('ssl_models')
log_dir = save_dir.joinpath('logs')

save_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)
log_dir.mkdir(exist_ok=True)


def get_data(pattern,volume_shape,batch,block_shape,n_classes):

    dataset = nobrainer.dataset.get_dataset(
        file_pattern=pattern,
        n_classes=n_classes,
        batch_size=batch,
        volume_shape=volume_shape,
        #shuffle=False,
        scalar_label=True,
        augment=False,
        #augmentType=True,
        block_shape=block_shape,
        n_epochs=None,
        num_parallel_calls=2
        )
    print("No augment: dataset received, now running process dataset ")
    dataset = process_dataset(dataset,batch,block_shape,n_classes)
    print("dataset processed ")
    return dataset


def augment_type_noise(pattern,volume_shape,batch,block_shape,n_classes):

    dataset = nobrainer.dataset.get_dataset(
        file_pattern=pattern,
        n_classes=n_classes,
        batch_size=batch,
        volume_shape=volume_shape,
        #shuffle=False,
        scalar_label=True,
        augment=True,
        augmentType=True,
        block_shape=block_shape,
        n_epochs=None,
        num_parallel_calls=2
        )
    print("Augment Noise: dataset received, now running process dataset ")
    dataset = process_dataset(dataset,batch,block_shape,n_classes)
    print("dataset processed ")
    return dataset

def augment_type_rigid(pattern,volume_shape,batch,block_shape,n_classes):

    dataset = nobrainer.dataset.get_dataset(
        file_pattern=pattern,
        n_classes=n_classes,
        batch_size=batch,
        volume_shape=volume_shape,
        #shuffle=False,
        scalar_label=True,
        augment=True,
        augmentType=False,
        block_shape=block_shape,
        n_epochs=None,
        num_parallel_calls=2
        )
    print("Augment Rigid: dataset received, now running process dataset ")
    dataset = process_dataset(dataset,batch,block_shape,n_classes)
    print("dataset processed ")
    return dataset

def process_dataset(dset,batch_size,block_shape,n_classes):
    print("in process dataset ")
    print("dataset shape before mapping:", dset)
    dset = dset.map(lambda x, y:x)
    print("dataset mapped")
    print("dataset shape after mapping:", dset)
    return dset

def run(block_shape, batch_size, dropout_typ,model_name):
    
    # Constants
    print("inside run function")
    root_path = '/om/user/satra/kwyk/tfrecords/'

    train_pattern = root_path+'data-train_shard-*.tfrec'
    eval_pattern = root_path + "data-evaluate_shard-*.tfrec"
    
    n_classes = 1
    volume_shape = (256, 256, 256) 
    block_shape = block_shape    
    EPOCHS = 3
    batch_size = batch_size

    #BATCH_SIZE_PER_REPLICA = 1

    #Setting up the multi gpu strategy
    print("setting up multi-gpu strategy")
    strategy = tf.distribute.MirroredStrategy()
    print("Number of replicas {}".format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = batch_size #BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# ----------------------------------------------------------- #
# Create a `tf.data.Dataset` instance.
    print("creating dataset train")
    #dataset_train = get_data(train_pattern, volume_shape,GLOBAL_BATCH_SIZE,block_shape,n_classes)
    
    augment_none = get_data(train_pattern, volume_shape,GLOBAL_BATCH_SIZE,block_shape,n_classes)
    print("original dataset created")
    
    augment_noise = augment_type_noise(train_pattern, volume_shape,GLOBAL_BATCH_SIZE,block_shape,n_classes)
    print("augment one view created, now creating eval dataset")
    
    augment_rigid = augment_type_rigid(train_pattern, volume_shape,GLOBAL_BATCH_SIZE,block_shape,n_classes)
    print("augment two view created, now creating eval dataset")
    
    #dataset_eval = get_data(eval_pattern,volume_shape,GLOBAL_BATCH_SIZE,block_shape,n_classes)
    print("creating dataset eval")

    print("unsupervised data noise ",augment_noise)
   # print("unsupervised data rigid ",augment_rigid)
    print("unsupervised data none ",augment_none)


    projection_dim = 2048
    latent_dim = 512
    weight_decay = 0.0005

    encoder, predictor = brainsiam(n_classes=n_classes, input_shape=(*block_shape, 1),weight_decay = weight_decay,
    projection_dim = projection_dim,
    latent_dim = latent_dim,
    )

    augment_zipped = tf.data.Dataset.zip((augment_none, augment_noise))

    steps = nobrainer.dataset.get_steps_per_epoch(
                n_volumes = len(),
                volume_shape=volume_shape,
                block_shape=block_shape,
                batch_size=GLOBAL_BATCH_SIZE
            )
    
    print("steps: ", steps)

    with strategy.scope():
        lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.0005, decay_steps=steps
                )
        print("Defined lr_Decayed_fn")

        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
                )
        print("early stopping")

        EPOCHS = 2
        print("Number of replicas {}".format(strategy.num_replicas_in_sync))
        
        simsiam = SimSiam(encoder(), predictor())
        # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=simsiam)
        print('Compiling simsiam.......')

        simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
        print('Model Training.......')
        training_time=time()-start
        
        logger = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), update_freq='batch')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(str(model_dir), save_weights_only=True, save_freq=10, save_best_only=False)

        history = simsiam.fit(augment_noise_none, epochs=EPOCHS, steps_per_epoch = steps, callbacks=[early_stopping])

    if __name__ == '__main__':

        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

        start=time()
        model_name=("simsiam_exp1".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M")))
        print("----------------- model name: {} -----------------".format(model_name))
        os.makedirs(os.path.join("training_files",model_name))
        os.makedirs(os.path.join("training_files",model_name,"saved_model"))
        os.makedirs(os.path.join("training_files",model_name,"training_checkpoints")) 


        block_shape = (128, 128, 128)
        batch_size = 1


        dropout = None
        training_time= run(block_shape,batch_size, dropout,model_name)
        end=time()-start

