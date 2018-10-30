#Trains a DNN on the MNIST dataset.
import matplotlib.pyplot as plt

# plaidml to run on amd gpu. Note: only works on keras 2.0 to 2.2
# NVIDIA GPU runs on keras 2.4 (access to newer functionality)
'''import plaidml.keras
plaidml.keras.install_backend()'''

# Import
import keras
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()