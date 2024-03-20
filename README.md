# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9 inclusively. To achieve this, we can use the MNIST dataset which is a collection of 60,000 handwritten digits of size 28 X 28. In order to classify these images to their appropriate numerical value, we can build a convolutional neural network model.
![Screenshot 2024-03-20 093336](https://github.com/guru14789/mnist-classification/assets/151705853/bd9eec12-5cf1-40cd-b8e8-ac92c415ceb4)

## Neural Network Model

![Screenshot 2024-03-20 110502](https://github.com/karuniya2005/mnist-classification/assets/161425769/8a5b6e52-9c4e-405a-bb5b-9e49c3a05bcb)

## DESIGN STEPS

## STEP 1:
Import tensorflow and preprocessing libraries

## STEP 2:
Build a CNN model

## STEP 3:
Compile and fit the model and then predict

## PROGRAM

### Name: VINOTH MP
### Register Number: 212223240182
## Library Importing:


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

## Shaping:
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[12045]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```

## One Hot Encoding:
```

X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```

## CNN Model:
```
model=keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=128, validation_data=(X_test_scaled,y_test_onehot))
```
## Metrics:
```
import pandas as pd
metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))
```
## Prediction:
```

img = image.load_img('/content/download.png')
type(img)

img = image.load_img('/content/download.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    i.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)


plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-03-20 111128](https://github.com/karuniya2005/mnist-classification/assets/161425769/396de945-55be-41f3-a73c-80e6da2796c8)


### Classification Report & Confusion Matrix

![Screenshot 2024-03-20 111134](https://github.com/karuniya2005/mnist-classification/assets/161425769/87537acf-faed-42c9-a3c8-87016143bd48)



### New Sample Data Prediction
![Screenshot 2024-03-20 111144](https://github.com/karuniya2005/mnist-classification/assets/161425769/17c3d9b2-76df-44e5-b825-8e73890f004b)


## RESULT
A successful convolutional deep neural network is developed for handwritten digit classification and verification.
