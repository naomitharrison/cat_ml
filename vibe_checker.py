from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

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