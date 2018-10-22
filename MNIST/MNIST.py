#Trains a simple deep NN on the MNIST dataset.
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import time
#from livelossplot import PlotLossesKeras

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        timeTaken = time.time() - self.epoch_time_start
        self.times.append(timeTaken)
        print('Epoch time taken: ' + str(timeTaken) + 's')

#Set the respective variables
batch_size = 128
num_classes = 10
epochs = 20
num_hidden_layers = 3

# spilt the data between training and testing sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess input data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Preprocess the class labels.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create the DNN model.
model = Sequential()
# Initial layer that creates the input shape.
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))

#Create the rest of num_layers inheriting input shape.
for x in range(num_hidden_layers):
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))


model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Compile the model with loss function and optimiser.
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#Early stopping of the training if loss increases too often.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Fit model based on training and validation datasets.
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=1.0/12.0,
                    callbacks=[TimeHistory()])
                   # callbacks=[PlotLossesKeras()])
                   # callbacks=[early_stopping])

#Test the model based on testing dataset.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy_' + str(epochs) + 'epochs')
#plt.show()

# Plot training & validation loss values
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_' + str(epochs) + 'epochs')
#plt.show()

#Early stopping of the training if loss increases too often.
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

#Train the final model with all 60,000 examples for 3 epochs
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=3,
          verbose=1,
          callbacks=[early_stopping])

#Evaluate the final model.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model_file_name = 'model_DNN_MNIST_' + str(num_hidden_layers) + 'hiddenlayers'
model_png = model_file_name + '.png'
plot_model(model, to_file=model_png)
model.save(model_file_name + '.h5')