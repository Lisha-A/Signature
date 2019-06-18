# -*- coding: utf-8 -*-

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.preprocessing.image import save_img
from sklearn.metrics import classification_report

# CNN initialization
classifier = Sequential();

# 1. Convolution layer
# Convolution
classifier.add( Convolution2D(32, 3, 3, input_shape = (480, 640, 3), activation = 'relu', use_bias = True) );
# Pooling
classifier.add( MaxPooling2D(pool_size = (2, 2)) );
"""
# 2. Convolution Layer
classifier.add( Convolution2D(64, 3, 3, input_shape = (480, 640, 3), activation = 'relu', use_bias = True) );
classifier.add( MaxPooling2D(pool_size = (2, 2)) );

# 3. Convolution Layer
classifier.add( Convolution2D(128, 3, 3, input_shape = (480, 640, 3), activation = 'relu', use_bias = True) );
classifier.add( MaxPooling2D(pool_size = (2, 2)) );
"""
# Flattening
classifier.add( Flatten() );

# Full connection
classifier.add( Dense(128, activation = 'relu') );

# Dropout of neurons
classifier.add( Dropout(0.3) );

# Readout layer
classifier.add( Dense(1, activation = 'sigmoid') );

# CNN compiling
classifier.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] );

# Fitting the CNN to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = False);
        
valid_datagen = ImageDataGenerator(rescale = 1./255);

training_set = train_datagen.flow_from_directory(
        'dataset/training',
        target_size = (480, 640),
        batch_size = 1,
        class_mode = 'binary');
        
validation_set = valid_datagen.flow_from_directory(
        'dataset/evaluation/genuines',
        target_size = (480, 640),
        batch_size = 1,
        class_mode = 'binary');

# Training
from IPython.display import display
from PIL import Image

classifier.fit_generator(
        training_set,
        steps_per_epoch = 1898,
        epochs = 5,
        validation_data = validation_set,
        validation_steps = 940);
   
# Test
import numpy as np
from tensorflow.keras.preprocessing import image

fake_image = 'dataset/test/test/NFI-00304046.png';
genuine_image = 'dataset/test/test/NFI-00101001.png';

test_image = image.load_img(fake_image, target_size = (480, 640));
test_image = image.img_to_array(test_image);
#save_img('image/test.png', test_image);
test_image = np.expand_dims(test_image, axis = 0);
result = classifier.predict(test_image);

print(training_set.class_indices);
print(validation_set.class_indices);

# TODO: precision, recall, accuracy
# TODO: die neuen Schichten testen auf Genauigkeit
# TODO: Werte ausgeben und KLassifizieren in Diagrammen

if result[0][0] <= 0.7:
    prediction = '\nforgery';
else:
    prediction = '\ngenuine';

print (prediction)

#im = Image.open('dataset/training/training/NISDCC-002_002_003_6g.PNG')
#im.show()