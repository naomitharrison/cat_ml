# Requires file download:
# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#     -O /tmp/cats_and_dogs_filtered.zip

# Requires following installs:
# sudo apt-get install python3-matplotlib
# pip install -U tensorflow

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

# unzip training and validation images
local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# files and pathnames
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# display 4x4 pics
nrows = 4
ncols = 4

pic_index = 8
next_cat = [os.path.join(train_cats_dir, fname)
            for fname in train_cat_fnames[pic_index-8:pic_index]
            if fname]
next_dog = [os.path.join(train_dogs_dir, fname)
            for fname in train_dog_fnames[pic_index-8:pic_index]
            if fname]
pic_index += 8

plt.gcf().set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(next_cat + next_dog):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)
# plt.show()

# Create a convolutional neural network aka CNN aka convnet
    # 1000 150x150 color images 
    # 3 {convolution + relu + maxpooling} modules 
    # 3x3 convolutions [16 filters, 32 filters, 64 filters] 
    # 2x2 maxpooling layers
img_input = layers.Input(shape=(150, 150, 3))
nfilters1 = 16
nfilters2 = 32
nfilters3 = 64
conv_sz = 3 # 3x3
pool_sz = 2 # 2x2

# 3 rounds of convolution and pooling
x = layers.Conv2D(nfilters1, conv_sz, activation='relu')(img_input)
x = layers.MaxPooling2D(pool_sz)(x)
x = layers.Conv2D(nfilters2, conv_sz, activation='relu')(x)
x = layers.MaxPooling2D(pool_sz)(x)
x = layers.Conv2D(nfilters3, conv_sz, activation='relu')(x)
x = layers.MaxPooling2D(pool_sz)(x)

# Flatten and connect
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)

# Create output layer from x for binary classification (sigmoid)
output = layers.Dense(1, activation='sigmoid')(x)

# input = before x, output = after x
model = Model(img_input, output)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])