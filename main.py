from tensorflow import keras
from sklearn.model_selection import train_test_split
from pandas import Series

# carga del conjunto
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# division del conjunto
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=.5, random_state=42, stratify=Y_test)

# normalizacion


X_train = (X_train / 255.00).astype(float)
X_test = (X_test / 255.00).astype(float)
X_val = (X_val / 255.00).astype(float)

