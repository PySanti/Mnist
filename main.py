from tensorflow import keras
from sklearn.model_selection import train_test_split
from pandas import Series
from keras_tuner import Hyperband
from utils.model_builder import model_builder
from utils.show_train_history import show_train_history

# carga del conjunto
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# division del conjunto
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=.5, random_state=42, stratify=Y_test)

# normalizacion

X_train = (X_train / 255.00).astype(float)
X_test = (X_test / 255.00).astype(float)
X_val = (X_val / 255.00).astype(float)


# busqueda de hiperparametros

tuner = Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=15,
    factor=2,
    directory='train_results',
    project_name='mnist'
)

tuner.search(
    X_train, Y_train,
    validation_data=(X_val, Y_val)
)

# resultados
best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=15
)
print('Rendimiento para test')
print(model.evaluate(X_test, Y_test))
show_train_history(history)

