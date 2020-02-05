import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = '10737418240'
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers \
  import Conv2D, MaxPool2D, Dense, Activation, Dropout, Flatten, BatchNormalization, ReLU, Reshape, Softmax
from tensorflow.keras.optimizers import Adam, SGD
from optimizer import Eve, RAdam
from image_loader import load_veg, _load_img
import pickle
from argparse import ArgumentParser
# from drawer import Drawer
from pprint import pprint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

EPS = 1e-8
N_CLASS = 7

def create_model():
  model = Sequential([
    Conv2D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=(100,100,3)),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    MaxPool2D(pool_size=2, strides=2, padding='same'),
    Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    MaxPool2D(pool_size=2, strides=2, padding='same'),
    Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    MaxPool2D(pool_size=2, strides=2, padding='same'),
    Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    MaxPool2D(pool_size=2, strides=2, padding='same'),
    Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    MaxPool2D(pool_size=2, strides=2, padding='same'),
    Reshape(target_shape=(4 * 4 * 512,)),
    Dense(units=4096),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Dense(units=4096),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Dense(units=N_CLASS),
    BatchNormalization(epsilon=EPS),
    ReLU(),
    Dropout(0.5),
    Softmax()
  ])

  # model.summary()

  return model

def data_augmentation(img):
  im = tf.image.random_flip_left_right(img)  # horizontal flip
  im = tf.image.random_flip_up_down(im)  # vertical flip
  im = tf.pad(im, tf.constant([[2, 2], [2, 2], [0, 0]]), "REFLECT")  # pad 2 (outsize:104x104)
  im = tf.image.random_crop(im, size=[100, 100, 3])  # random crop (outsize:100x100)
  return im

def _train(batch_size, epochs, init_epoch, save_file, load_file, optimizer, lr, experiment, use_tpu, use_callbacks, verbose):
  if use_tpu:
    print("Setting up TPU ...")
    tpu_grpc_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    print("Done!")

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_veg(flatten=False)

  # cut data
  if experiment:
    x_train, y_train = x_train[:600], y_train[:600]
    x_val, y_val = x_val[:100], y_val[:100]
    x_test, y_test = x_test[:100], y_test[:100]

  train_num = len(x_train)
  test_num = len(x_test)

  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  trainset = trainset.map(lambda image, label: (data_augmentation(image), label)).shuffle(buffer_size=1024).repeat().batch(batch_size)
  valset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  valset = valset.shuffle(buffer_size=1024).batch(test_num)
  testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  testset = testset.batch(test_num)

  callbacks = []
  if use_callbacks:
    checkpoint_path = "./checkpoint/cp-{epoch:04d}-{val_loss:.4f}-{val_accuracy:.4f}.h5"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.EarlyStopping('val_loss', patience=80))

  ops = {'SGD':SGD, 'Adam':Adam, 'Eve':Eve, 'RAdam':RAdam}
  op = ops[optimizer](lr=lr, epsilon=EPS)

  # TPU
  if use_tpu:
    with strategy.scope():
      model = create_model()

      model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

      if load_file is not None: model.load_weights(load_file)

      history = model.fit(
        trainset, epochs=epochs, initial_epoch=init_epoch,
        validation_data=valset, callbacks=callbacks, verbose=verbose,
        steps_per_epoch=train_num // batch_size
      )
      res_loss, res_acc = model.evaluate(testset)
  # CPU
  else:
    model = create_model()

    model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if load_file is not None: model.load_weights(load_file)

    history = model.fit(
      trainset, epochs=epochs, initial_epoch=init_epoch,
      validation_data=valset, callbacks=callbacks, verbose=verbose,
      steps_per_epoch=train_num // batch_size
    )

    res_loss, res_acc = model.evaluate(testset)

  with open(save_file, 'wb') as f:
    pickle.dump(history.history, f)
  print(f"Saving history to \'{save_file}\'. (epochs: {len(history.epoch)})")

  return (res_loss, res_acc)

def _evaluate(batch_size, load_file, optimizer, lr, experiment, use_tpu):
  if use_tpu:
    print("Setting up TPU ...")
    tpu_grpc_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    print("Done!")

  (_, _), (_, _), (x_test, y_test) = load_veg(flatten=False)

  # cut data
  if experiment:
    x_test, y_test = x_test[:100], y_test[:100]

  test_num = len(x_test)
  testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  testset = testset.batch(test_num)

  # ops = {'SGD':SGD, 'Adam':Adam, 'Eve':Eve, 'RAdam':RAdam}
  # op = ops[optimizer](lr=lr, epsilon=EPS)

  # TPU
  if use_tpu:
    with strategy.scope():
      model = create_model()
      # model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      model.load_weights(load_file)

      model.evaluate(testset)
  # CPU
  else:
    model = create_model()
    # model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(load_file)

    model.evaluate(testset)

def _expect(img, load_file, optimizer, lr, verbose=True):
  # クラス (7個)
  category = ['bell pepper', 'broccoli', 'carrot', 'eggplant', 'green onion', 'onion', 'tomato']
  # expect
  x = _load_img(img, flatten=False)
  model = create_model()
  # ops = {'SGD':SGD, 'Adam':Adam, 'Eve':Eve, 'RAdam':RAdam}
  # op = ops[optimizer](lr=lr, epsilon=EPS)
  # model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.load_weights(load_file)

  y = model.predict(x)
  y = y.flatten()

  # create dictionary
  dic = dict(zip(category, y))
  # print result
  if verbose:
    for key, value in dic.items():
      print(f"{key}: {value:.2%}")

  return dic

def plot_cmx(y_true, y_pred, labels):
  cmx_data = confusion_matrix(y_true, y_pred)
  df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
  fig = plt.figure(figsize=(10,7))
  fig.subplots_adjust(left=0.18, right=1.00, bottom=0.25, top=0.92)
  sns.heatmap(df_cmx, annot=True, fmt='.0f', linecolor='white', linewidths=1)
  plt.xlabel("Predict")
  plt.ylabel("True")

  plt.show()

def _report(load_file, optimizer, lr):
  (_, _), (_, _), (x_test, y_test) = load_veg(flatten=False)
  # ラベル (7個)
  labels = [
    'Green pepper', 'Broccoli', 'Carrot', 'Eggplant',
    'Green onion', 'Onion', 'Tomato'
  ]

  # model = tf.keras.models.load_model('mycnn2.h5', compile=False)
  model.summary()
  # ops = {'SGD':SGD, 'Adam':Adam, 'Eve':Eve, 'RAdam':RAdam}
  # op = ops[optimizer](lr=lr, epsilon=EPS)
  # model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.load_weights(load_file)

  y_pred = model.predict_classes(x_test, batch_size=128)
  pprint(y_pred)
  print(classification_report(y_test, y_pred, target_names=labels))
  # plot_cmx(y_test, y_pred, labels)
  # tf.keras.models.save_model(model, 'mycnn2.h5', include_optimizer=False)

if __name__ == '__main__':
  argparser = ArgumentParser()

  argparser.add_argument('-b', '--batch', type=int, default=128, help='The size of batch.')
  argparser.add_argument('-e', '--epochs', type=int, default=100, help='The number of epochs.')
  argparser.add_argument('-i', '--initepoch', type=int, default=0, help='The epoch at which to start training. (0-indexed)')
  argparser.add_argument('-l', '--load', type=str, default=None, help='The path of weights file to load.')
  argparser.add_argument('-s', '--save', type=str, default='./history/mycnn_history.pkl', help='The path of history file to save.')
  argparser.add_argument('-o', '--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'Eve', 'RAdam'], 
    help='The optimizer. Choices: SGD, Adam, Eve, RAdam')
  argparser.add_argument('-r', '--lr', type=float, default='0.001', help='The learning rate.')
  argparser.add_argument('-x', '--experiment', action='store_true', help='Whether to run system experimentally.')
  argparser.add_argument('-t', '--tpu', action='store_true', help='Whether to run system with TPU. (CPU/GPU by default)')
  argparser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'expect', 'report', 'eval', 'search'],
    help='Whether to train or expect.')
  argparser.add_argument('-g', '--img', type=str, default=None, help='The path of img to expect. You should specify the path when expectation mode.')
  argparser.add_argument('-n', '--lr_range', type=float, nargs='*', default=[0.0001, 0.01], help='The range of learning rate for searching')
  argparser.add_argument('-c', '--use_callbacks', action='store_true', help='Whether to use callbacks. Default by \'True\'')
  argparser.add_argument('-v', '--verbose', action='store_true', help='Whether to verbose. ')
  args = argparser.parse_args()

  if args.mode == 'train':
    print(f"---------- lr = {args.lr} ----------")
    v = 1 if args.verbose else 0
    _train(
      args.batch, args.epochs, args.initepoch, args.save, args.load,
      args.optimizer, args.lr, args.experiment, args.tpu, args.use_callbacks, v
    )
  elif args.mode == 'eval':
    if args.load is None:
      print("Error: you should specify the path of image and weights file.")
    else:
      _evaluate(args.batch, args.load, args.optimizer, args.lr, args.experiment, args.tpu)
  elif args.mode == 'expect':
    if args.img is None or args.load is None:
      print("Error: you should specify the path of image and weights file.")
    else:
      result = _expect(args.img, args.load, args.optimizer, args.lr)
  elif args.mode == 'report':
    if args.load is None:
      print("Error: you should specify the path of image and weights file.")
    else:
      _report(args.load, args.optimizer, args.lr)
  elif args.mode == 'search':
    name, ext = os.path.splitext(args.save)
    lr_range = np.arange(*args.lr_range, dtype=float).tolist()
    for lr in lr_range:
      print(f"---------- lr = {lr} ----------")
      v = 1 if args.verbose else 2
      loss, acc = _train(
        args.batch, args.epochs, args.initepoch, f"{name}_{lr:.4f}{ext}", args.load, 
        args.optimizer, lr, args.experiment, args.tpu, args.use_callbacks, v
      )