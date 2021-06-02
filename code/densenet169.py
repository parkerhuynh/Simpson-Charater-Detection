
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import pandas as pd
import argparse
import wandb
import numpy as np
from wandb.keras import WandbCallback
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print('GPU is used.' if len(tf.config.list_physical_devices('GPU')) > 0 else 'GPU is NOT used.')
print("Tensorflow version: " + tf.__version__)

# We define the size of input images to 128x128 pixels.
image_size      = 256
batch_size      = 96
epoch_num       = 64
random_seed     = 100
epoch_warmup    = 2
lr_base         = 0.001
strategy        = tf.distribute.MirroredStrategy()


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


def create_callbacks(epoch_num, epoch_warmup, sample_count, totak_batch_size, learning_rate_base):
    checkpoint = ModelCheckpoint(
        filepath='../submission/densenet169.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
    )
    wandb_callback = WandbCallback(
        save_model=False,
        log_best_prefix="best_",
        monitor="val_accuracy"
    )
    total_steps = int(epoch_num * sample_count / totak_batch_size)
    warmup_steps = int(epoch_warmup * sample_count / totak_batch_size)
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0)
    return [checkpoint, warm_up_lr, wandb_callback]


def main(args):
    num_gpu = len(tf.config.list_physical_devices('GPU'))
    config = {
        "run_date": args.run_date,
        "hashcode": args.hashcode,
        "pretrain": args.pretrain,
        "batch_size": batch_size * num_gpu,
        "num_gpu": num_gpu,
        "image_size": image_size
    }
    wandb.init(project="sim", config=config)

    # Create an image generator with a fraction of images reserved for validation:
    train_dataframe = pd.read_csv("../data/train.csv")
    dev_dataframe = pd.read_csv("../data/dev.csv")

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.densenet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.20,
        height_shift_range=0.20,
        brightness_range=[0.6,1.4],
        zoom_range=[0.6,1.4],
        shear_range=0.4,
        horizontal_flip=True
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_dataframe,
        directory="../data/images/",
        x_col="x_col",
        y_col="y_col",
        weight_col=None,
        target_size=(image_size, image_size),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=batch_size * num_gpu,
        shuffle=True,
        seed=random_seed,
        save_to_dir=None,
        interpolation="nearest",
    )

    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)

    dev_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.densenet.preprocess_input
    )
    dev_generator = dev_datagen.flow_from_dataframe(
        dev_dataframe,
        directory="../data/images/",
        x_col="x_col",
        y_col="y_col",
        weight_col=None,
        target_size=(image_size, image_size),
        color_mode="rgb",
        classes=class_names,
        shuffle=False,
        class_mode="categorical",
        batch_size=batch_size * num_gpu,
    )

    # Create the model
    with strategy.scope():
        model = tf.keras.applications.DenseNet169(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(image_size, image_size, 3),
            pooling=None,
            classes=num_classes
        )

        model.compile(optimizer="adam",
                      loss='CategoricalCrossentropy',
                      metrics=['accuracy'])
        model.load_weights("../pretrain/model.h5", by_name=True)
        model.summary()

    # Train the model
    model.fit(
        train_generator,
        epochs=epoch_num,
        callbacks=create_callbacks(epoch_num, epoch_warmup, len(train_dataframe), batch_size * num_gpu, lr_base),
        validation_data=dev_generator,
    )

    train_preds = model.predict(dev_generator)
    np.save("dev_preds.npy", train_preds)
    wandb.save("dev_preds.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run-date', type=str)
    parser.add_argument('--hashcode', type=str)
    parser.add_argument('--pretrain', type=str)
    args = parser.parse_args()
    main(args)