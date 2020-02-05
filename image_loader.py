import pickle
import numpy as np
import glob
import os, sys
sys.path.append(os.pardir)
from PIL import Image

# reference
dataset_dir = "./dataset"
# save file
save_file = "./dat/vegetables4.pkl"

train_num = 6000
val_num = 1000
test_num = 1000
img_dim = (100, 100, 3)
img_size = 10000

def shuffle_dataset(x, y):
  """
  Shuffle dataset

  Parameters
  ----------
  x : Training data\n
  y : Teaching data

  Returns
  -------
  x, y : Shuffled training data and teaching data
  """
  perm = np.random.permutation(x.shape[0])
  x = x[perm] if x.ndim == 2 else x[perm]
  y = y[perm]

  return x, y

def _load_img(file_path, normalize=True, flatten=True, one_hot_label=False, verbose=False):
  file_name = os.path.basename(file_path)
  if verbose: print("Converting " + file_name + " to Numpy Array ...")
  
  img = Image.open(file_path)
  img = img.convert('RGB')
  dat = [np.array(img)]
  dat = np.array(dat)

  if normalize:
    dat = dat.astype(np.float32)
    dat /= 255.0
  
  if one_hot_label:
    dat = _change_one_hot_label(dat)
    dat = _change_one_hot_label(dat)
    dat = _change_one_hot_label(dat)

  if not flatten:
    dat = dat.reshape(-1, *img_dim)

  if verbose: print("Done")

  return dat

def _load_veg(dir_name):
  dir_path = os.path.join(dataset_dir, dir_name)
  
  print("Converting " + dir_name + " to Numpy Array ...")

  data = []
  labels = []
  # dictionary = {
  #   'bp': 0, 'bpcolorful': 1, 'bpred': 2, 'bpwith_leaves': 3, 'bpyellow': 4, 'b': 5, 'bwith_leaves': 6,
  #   'c': 7, 'ccolorful': 8, 'cwith_leaves': 9, 'e': 10, 'ewith_leaves': 11, 'g': 12, 'gwhite': 13,
  #   'o': 14, 'ocolorful': 15, 'opurple': 16, 'owith_leaves': 17, 't': 18, 'tcolorful': 19, 'twith_leaves': 20
  # }
  # dictionary = {
  #   'bp': 0, 'bpred': 1, 'bpyellow': 2, 'b': 3, 'c': 4, 'e': 5, 'g': 6, 'gwhite': 7,
  #   'o': 8, 'opurple': 9, 't': 10
  # }
  dictionary = {
    'bp': 0, 'b': 1, 'c': 2, 'e': 3, 'g': 4, 'o': 5, 't': 6
  }
  cates = glob.glob(os.path.join(dir_path, '*'))
  for cate in cates:
    dirs = glob.glob(os.path.join(cate, '*'))
    for _dir in dirs:
      files = glob.glob(os.path.join(_dir, '*'))
      d = os.path.basename(_dir)
      for file in files:
        img = Image.open(file).convert('RGB')
        data.append(np.array(img))
        labels.append(dictionary[d])
  data = np.array(data)
  labels = np.array(labels)
  data, labels = shuffle_dataset(data, labels)
  print("Done!")

  return (data, labels)

def _convert_numpy():
  dataset = {}
  train_data, train_labels = _load_veg("train")
  test_data, test_labels = _load_veg("test")

  dataset['train_img'] = train_data.reshape(-1, img_size)
  dataset['train_label'] = train_labels
  dataset['val_img'] = test_data[:1000].reshape(-1, img_size)
  dataset['val_label'] = test_labels[:1000]
  dataset['test_img'] = test_data[1000:2000].reshape(-1, img_size)
  dataset['test_label'] = test_labels[1000:2000]

  return dataset

def init_veg():
  dataset = _convert_numpy()
  print("Creating pickle file ...")
  with open(save_file, 'wb') as f:
      pickle.dump(dataset, f, -1)
  print("Done!")

def _change_one_hot_label(X):
  T = np.zeros((X.size, 10))
  for idx, row in enumerate(T):
    row[X[idx]] = 1

  return T

def load_veg(normalize=True, flatten=True, one_hot_label=False):
  """
  データセットの読み込み

  Parameters
  ----------
  normalize : 画像のピクセル値を0.0~1.0に正規化する
  one_hot_label :
      one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
      one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
  flatten : 画像を一次元配列に平にするかどうか

  Returns
  -------
  (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
  """
  if not os.path.exists(save_file):
    init_veg()

  with open(save_file, 'rb') as f:
    dataset = pickle.load(f)

  if normalize:
    for key in ('train_img', 'val_img', 'test_img'):
      dataset[key] = dataset[key].astype(np.float32)
      dataset[key] /= 255.0
    for key in ('train_label', 'val_label', 'test_label'):
      dataset[key] = dataset[key].astype(np.int32)

  if one_hot_label:
    dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
    dataset['val_label'] = _change_one_hot_label(dataset['val_label'])
    dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

  if not flatten:
    for key in ('train_img', 'val_img', 'test_img'):
      dataset[key] = dataset[key].reshape(-1, *img_dim)

  return (dataset['train_img'], dataset['train_label']), (dataset['val_img'], dataset['val_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
  init_veg()