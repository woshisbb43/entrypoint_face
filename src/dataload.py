import tensorflow as tf
import pathlib
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np


# define parameters for loader
# 32个连续的图片为一组
batch_size = 32
img_height = 180
img_width = 180

# data_dir of image directory
data_dir = "/Users/edwin/.keras/datasets/flower_photos"
data_dir = pathlib.Path(data_dir)

# 训练集
train_ds = tf.keras.utils.image_dataset_from_directory(
  # Directory where the data is located
  data_dir,
  # Optional float between 0 and 1, fraction of data to reserve for validation.
  validation_split=0.2,
  # Subset of the data to return. One of "training" or "validation". Only used if validation_split is set.
  subset="training",
  # Optional random seed for shuffling and transformations.
  seed=123,
  image_size=(img_height, img_width),
  # Size of the batches of data. Default: 32. If None, the data will not be batched (the dataset will yield individual samples).
  batch_size=batch_size)

# 验证集
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# nomalization
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))