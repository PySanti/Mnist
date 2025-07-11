from tensorflow import keras
from utils.show_image import show_image

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

print("Conjunto de entrenamiento")
print(X_train.shape)
print(Y_train.shape)

print("Conjunto de test")
print(X_test.shape)
print(Y_test.shape)

show_image(X_train[500])
print(f'Target de la imagen mostrada : {Y_train[500]}')
