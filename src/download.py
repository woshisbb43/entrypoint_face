import tensorflow as tf
import pathlib
import PIL

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# get number of images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# open image
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))