#Trains a DNN on the MNIST dataset.
import matplotlib.pyplot as plt

# plaidml to run on amd gpu. Note: only works on keras 2.0 to 2.2
# NVIDIA GPU runs on keras 2.4 (access to newer functionality)
import plaidml.keras
plaidml.keras.install_backend()

# Import
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
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

#Set the respective hyperparameters
batch_size = 128
num_classes = 10
epochs = 20
num_hidden_layers = 3

#relu, tanh, sigmoid, softmax
activationFN = 'relu'

#RMSprop(), SGD, Adam, Adagrad, Adadelta
optimiserFN = RMSprop()

# categorical_crossentropy, binary_crossentropy
lossFN = 'categorical_crossentropy'

# File name for the generated model, figures, and diagrams.
file_name = 'MNIST_DNN' + str(num_hidden_layers) + 'hiddenlayers_' + activationFN
short_name = 'MNIST_DNN'

# spilt the data between training and testing sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess input data
# Flatten 28x28 images to 784 pixels.
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

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

# Create the DNN model.
model = Sequential()
# Initial layer that creates the input shape.
model.add(Dense(512, activation=activationFN, input_shape=(784,)))
model.add(Dropout(0.2))

#Create the rest of num_layers inheriting input shape.
for x in range(num_hidden_layers):
    model.add(Dense(512, activation=activationFN))
    model.add(Dropout(0.2))

# Final output layer. Use softmax for probability.
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Compile the model with loss function and optimiser.
model.compile(loss=lossFN,
              optimizer=optimiserFN,
              metrics=['accuracy'])

#Early stopping of the training if loss increases too often.
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)
# Add below to only use the best weights (on keras 2.3+ which means no AMD support)
#, restore_best_weights=True)

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
plt.savefig(short_name + '_accuracy_' + str(epochs) + 'epochs')
#plt.show()

# Plot training & validation loss values
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(short_name + '_loss_' + str(epochs) + 'epochs')
#plt.show()

#Early stopping of the training if loss increases too often.
early_stopping = EarlyStopping(monitor='loss', patience=10)#, restore_best_weights=True)

#Train the final model with all 60,000 examples for 3 epochs
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=3,
          verbose=1,
          callbacks=[TimeHistory(), early_stopping])

#Evaluate the final model.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save the model
model_file_name = 'model' + file_name
model_png = model_file_name + '.png'
plot_model(model, to_file=model_png)
model.save(model_file_name + '.h5')