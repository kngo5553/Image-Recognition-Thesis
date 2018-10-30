#Trains a DNN on the MNIST dataset.
import matplotlib.pyplot as plt

# plaidml to run on amd gpu. Note: only works on keras 2.0 to 2.2
# NVIDIA GPU runs on keras 2.4 (access to newer functionality)
#import plaidml.keras
#plaidml.keras.install_backend()

# Import
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import time

# Function to get epoch time taken.
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        timeTaken = time.time() - self.epoch_time_start
        self.times.append(timeTaken)
        print('Epoch time taken: ' + str(timeTaken) + 's')

time_callback = TimeHistory()
time_callback2 = TimeHistory()

#Set the respective hyperparameters
batch_size = 256
num_classes = 100
epochs = 50
num_full_layers = 2

#relu, tanh, sigmoid, softmax
activationFN = 'relu'

#RMSprop(), SGD, Adam, Adagrad, Adadelta
optimiserFN = 'adam'

# categorical_crossentropy, binary_crossentropy
lossFN = 'categorical_crossentropy'

# File name for the generated model, figures, and diagrams.
short_name = 'CIFAR100_CNN'
file_name = short_name + '_' + str(num_full_layers) + 'fulllayers_' + activationFN + '_' + str(epochs) + 'epochs'


# spilt the data between training and testing sets.
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Change to 32bit values; reduces memory.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalise input to 0-1.
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Preprocess the class labels.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create the CNN model.
model = Sequential()

# Convolution layers
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], name="INPUT"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), padding='same', name="conv2d_32"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Add a second batch of Conv2D layers. Increase filter number to find more detailed features.
model.add(Conv2D(64, (3, 3), padding='same', name="conv2d_64"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', name="conv2d_64_1"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Add a third batch of Conv2D layers. Increase filter number to find more detailed features.
model.add(Conv2D(128, (3, 3), padding='same', name="conv2d_128"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', name="conv2d_128_1"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Add a fourth batch of Conv2D layers. Increase filter number to find more detailed features.
model.add(Conv2D(256, (3, 3), padding='same', name="conv2d_256"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', name="conv2d_256_1"))
model.add(Activation(activationFN))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Flatten so we can use fully connected layers
model.add(Flatten())

#Create the rest of the fully connected layers
for x in range(num_full_layers):
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

# Final output layer. Use softmax for probability.
model.add(Dense(num_classes, activation='softmax', name="OUTPUT"))

model.summary()

# Compile the model with loss function and optimiser.
model.compile(loss=lossFN,
              optimizer=optimiserFN,
              metrics=['accuracy'])

#Early stopping of the training if loss increases too often.
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, restore_best_weights=True)
# Add below to only use the best weights (on keras 2.3+ which means no AMD support)
#, restore_best_weights=True)

#Fit model based on training and validation datasets.
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=1.0/10.0,
                    callbacks=[time_callback, early_stopping])
                   # callbacks=[PlotLossesKeras()])
                   # callbacks=[early_stopping])

#Test the model based on testing dataset.
score = model.evaluate(x_test, y_test, verbose=0)
firstLoss = score[0]
firstAcc = score[1]

# Plot training & validation accuracy values
plt.figure(0)
accMax = max(history.history['acc'])
valAccMax = max(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train [max: ' + "{0:.3f}".format(accMax) + ']', 'Val [max: ' + "{0:.3f}".format(valAccMax) + ']'], loc='upper left')
plt.savefig('acc_' + file_name)
#plt.show()

# Plot training & validation loss values
plt.figure(1)
lossMin = min(history.history['loss'])
valLossMin = min(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train [min: ' + "{0:.3f}".format(lossMin) + ']', 'Val [min: ' + "{0:.3f}".format(valLossMin) + ']'], loc='upper left')
plt.savefig('loss_' + file_name)
#plt.show()

#Early stopping of the training if loss increases too often.
# , restore_best_weights=True) for nvidia
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

#Train the final model with all 60,000 examples for 3 epochs
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10,
          verbose=1,
          callbacks=[time_callback2, early_stopping])

#Evaluate the final model.
score = model.evaluate(x_test, y_test, verbose=0)

trainTime = sum(time_callback.times)
testTime = sum(time_callback2.times)
totalTime = trainTime + testTime

print('')
print('--After validation set--')
print('Time:' + "{0:.3f}".format(trainTime) + 's')
print('Test loss:', firstLoss)
print('Test accuracy:', firstAcc)
print('')
print('--After test set--')
print('Time:' + "{0:.3f}".format(testTime) + 's')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('')
print('Total Time:' + "{0:.3f}".format(totalTime) + 's')

# Save the model
model_file_name = 'model_' + file_name
model_png = model_file_name + '.png'
plot_model(model, to_file=model_png)
model.save(model_file_name + '.h5')