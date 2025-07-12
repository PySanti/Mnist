
# Mnist

La idea de este proyecto sera alcanzar la mejor precision posible resolviendo el problema provisto por el dataset de keras `keras.datasets.mnist`.

Este problema ya fue resuelto por nosotros en un  [ejercicio pasado](https://github.com/PySanti/FirstKerasPractice) donde la mejor precision para el conjunto de test fue del 97%. En este ejercicio buscaremos mejorar aun mas dichos resultados utilizando `hypertunning` y tecnicas de regularizacion como `l2` y `dropout`.


## Preprocesamiento

### Visualizacion del conjunto

Primeramente, visualizando el conjunto de datos:

```
Conjunto de entrenamiento
(60000, 28, 28)
(60000,)

Conjunto de test
(10000, 28, 28)
(10000,)
```

El conjunto de train tiene 60.000 registros de vectores de 28x28 y un target representado por un entero.

Se supone que cada vector de 28x28 representa a una imagen que contiene un numero escrito a mano en escala de grises, vamos a verlo.

Utilizando el siguiente codigo:

```

# utils/show_image.py

import matplotlib.pyplot as plt

def show_image(imagen):
    """
    Muestra una imagen en escala de grises de 28x28 píxeles usando Matplotlib.
    
    Parámetros:
    imagen -- numpy array de forma (28, 28) con valores entre 0 y 255 (escala de grises)
    """
    plt.imshow(imagen, cmap='gray')  # 'gray' para escala de grises
    plt.axis('off')  # Oculta los ejes
    plt.show()


# main.py

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


```

Terminal :

```
Conjunto de entrenamiento
(60000, 28, 28)
(60000,)
Conjunto de test
(10000, 28, 28)
(10000,)
Target de la imagen mostrada : 3
```

Imagen:

![Imagen no encontrada](./images/image_1.png)

Cada matriz (imagen) de 28x28 contiene 28 vectores de 28 enteros, donde cada entero es un valor entre 0 y 255.

Por lo tanto, las unicas estrategias que utilizaremos seran : `aplanamiento`, `normalizacion`.

### Distribucion de target

Luego, a traves del siguiente codigo:

```
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pandas import Series

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

def show_target(set_):
    for a in Series(set_).value_counts().items():
        print(f'{a[0]} : {a[1]/len(set_)*100:.2f}')
        

print('Distribucion de target en y_train')
show_target(Y_train)

print('Distribucion de target en y_test')
show_target(Y_test)
```

Obtuvimos el siguiente resultado:

```
Distribucion de target en y_train
1 : 11.24
7 : 10.44
3 : 10.22
2 : 9.93
9 : 9.92
0 : 9.87
6 : 9.86
8 : 9.75
4 : 9.74
5 : 9.04

Distribucion de target en y_test
1 : 11.35
2 : 10.32
7 : 10.28
3 : 10.10
9 : 10.09
4 : 9.82
0 : 9.80
8 : 9.74
6 : 9.58
5 : 8.92
```

Se puede concluir que hay ~10% de registros para cada clase en los conjuntos de train y test.

### Division del conjunto

Utilizamos `sklearn.model_selection.train_test_split` para dividir los conjuntos de datos y generar un conjunto de validacion:

```
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pandas import Series

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=.5, random_state=42, stratify=Y_test)



def show_set(set_):
    print(set_[0].shape)
    print(set_[1].shape)
    print('\n')
    for a in Series(set_[1]).value_counts().items():
        print(f'{a[0]} : {a[1]/len(set_[1])*100:.2f}')
        

print('Descripcion de train set')
show_set([X_train, Y_train])

print('Descripcion de train set')
show_set([X_test, Y_test])

print('Descripcion de train set')
show_set([X_val, Y_val])
```

Obtuvimos los siguientes resultados:

```
Descripcion de train set
(60000, 28, 28)
(60000,)


1 : 11.24
7 : 10.44
3 : 10.22
2 : 9.93
9 : 9.92
0 : 9.87
6 : 9.86
8 : 9.75
4 : 9.74
5 : 9.04

Descripcion de train set
(5000, 28, 28)
(5000,)


1 : 11.34
2 : 10.32
7 : 10.28
3 : 10.10
9 : 10.10
4 : 9.82
0 : 9.80
8 : 9.74
6 : 9.58
5 : 8.92

Descripcion de train set
(5000, 28, 28)
(5000,)


1 : 11.36
2 : 10.32
7 : 10.28
3 : 10.10
9 : 10.08
4 : 9.82
0 : 9.80
8 : 9.74
6 : 9.58
5 : 8.92
```

### Normalizacion

Buscaremos normalizar los valores teniendo en cuenta que las redes neuronales se ven beneficiadas de esta estrategia.

A traves del siguiente codigo:

```
X_train = (X_train / 255.00).astype(float)
X_test = (X_test / 255.00).astype(float)
X_val = (X_val / 255.00).astype(float)

```

Se normalizaron los conjuntos.

## Entrenamiento - Evaluacion

Utilizando el siguiente codigo:

```

# utils/model_builder.py

from tensorflow import keras
from keras import layers
from keras import optimizers

def model_builder(hp):
    net = keras.Sequential()
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    num_layers = hp.Int('num_layers', min_value=2, max_value=5)


    net.add(layers.Flatten(input_shape=(28,28)))

    for i in range(num_layers):
        units_ = hp.Int(f'num_units_{i}', min_value=15, max_value=120, step=15)
        net.add(layers.Dense(units=units_, activation='relu'))


    net.add(layers.Dense(units=10, activation='softmax'))

    net.compile(
        loss= 'sparse_categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return net

# main.py


from tensorflow import keras
from sklearn.model_selection import train_test_split
from pandas import Series
from keras_tuner import Hyperband
from utils.model_builder import model_builder

# carga del conjunto
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# division del conjunto
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=.5, random_state=42, stratify=Y_test)

# normalizacion

X_train = (X_train / 255.00).astype(float)
X_test = (X_test / 255.00).astype(float)
X_val = (X_val / 255.00).astype(float)


tuner = Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=10,
    factor=2,
    directory='train_results',
    project_name='mnist'
)

tuner.search(
    X_train, Y_train,
    validation_data=(X_val, Y_val)
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]

print("Mejor combinacion de hiperparametros")
print(best_hyperparameters.values)

```

Se obtuvieron los siguientes resultados:

```
Best val_accuracy So Far: 0.9805999994277954

Total elapsed time: 00h 13m 09s

Mejor combinacion de hiperparametros

{'learning_rate': 0.001, 'num_layers': 3, 'num_units_0': 120, 'num_units_1': 15, 'num_units_2': 75, 'num_units_3': 15, 'num_units_4': 30, 'tuner/epochs': 10, 'tuner/initial_epoch': 5, 'tuner/bracket': 2, 'tuner/round': 2, 'tuner/trial_id': '0027'}

```

El rendimiento del modelo para el conjunto de test fue el siguiente:

```
157/157 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9755 - loss: 0.1204
```

Luego, modificamos el model builder para implementar estrategias de regularizacion `l2` y `dropout`.

Ademas, aumentamos el `max_epochs` de Hyperband para aumentar el espacio de busqueda de hiperparametros.

```
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras import regularizers

def model_builder(hp):
    net = keras.Sequential()
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    num_layers = hp.Int('num_layers', min_value=2, max_value=5)


    net.add(layers.Flatten(input_shape=(28,28)))

    for i in range(num_layers):
        l2 = hp.Choice(f'l2_{i}', values=[1e-4, 1e-3, 1e-2])
        drop = hp.Float(f'drop_{i}', min_value=0.2, max_value=4, step=0.5)
        units_ = hp.Int(f'num_units_{i}', min_value=15, max_value=120, step=15)

        net.add(layers.Dense(units=units_, activation='relu', kernel_regularizer=regularizers.l2(l2)))
        net.add(layers.Dropout(drop))


    net.add(layers.Dense(units=10, activation='softmax'))

    net.compile(
        loss= 'sparse_categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return net
```

Se obtuvieron los siguientes resultados:

```
Best val_accuracy So Far: 0.9760000109672546

Total elapsed time: 00h 18m 39s

{'learning_rate': 0.0001, 'num_layers': 3, 'l2_0': 0.01, 'drop_0': 0.25, 'num_units_0': 120, 'l2_1': 0.0001, 'drop_1': 0.2, 'num_units_1': 105, 'l2_2': 0.0001, 'drop_2': 0.2, 'num_units_2': 120, 'l2_3': 0.01, 'drop_3': 0.25, 'num_units_3': 15, 'tuner/epochs': 15, 'tuner/initial_epoch': 8, 'tuner/bracket': 3, 'tuner/round': 3, 'tuner/trial_id': '0015', 'l2_4': 0.01, 'drop_4': 0.2, 'num_units_4': 30}
```

Luego utilizando el siguiente codigo, se reentreno el modelo utilizando los hiperparametros obtenidos y se grafico el historial de entrenamiento:

```

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
```

Se obtuvieron los siguientes resultados:

```
Epoch 1/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - accuracy: 0.5575 - loss: 2.5366 - val_accuracy: 0.9098 - val_loss: 0.7300
Epoch 2/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.8777 - loss: 0.7745 - val_accuracy: 0.9378 - val_loss: 0.4640
Epoch 3/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9097 - loss: 0.5369 - val_accuracy: 0.9464 - val_loss: 0.3544
Epoch 4/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9265 - loss: 0.4192 - val_accuracy: 0.9520 - val_loss: 0.3012
Epoch 5/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9363 - loss: 0.3542 - val_accuracy: 0.9608 - val_loss: 0.2521
Epoch 6/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9433 - loss: 0.3127 - val_accuracy: 0.9658 - val_loss: 0.2244
Epoch 7/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9493 - loss: 0.2787 - val_accuracy: 0.9686 - val_loss: 0.2076
Epoch 8/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9536 - loss: 0.2530 - val_accuracy: 0.9706 - val_loss: 0.1965
Epoch 9/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9572 - loss: 0.2397 - val_accuracy: 0.9708 - val_loss: 0.1944
Epoch 10/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9593 - loss: 0.2302 - val_accuracy: 0.9740 - val_loss: 0.1794
Epoch 11/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9617 - loss: 0.2190 - val_accuracy: 0.9726 - val_loss: 0.1783
Epoch 12/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9622 - loss: 0.2102 - val_accuracy: 0.9758 - val_loss: 0.1691
Epoch 13/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9644 - loss: 0.2042 - val_accuracy: 0.9724 - val_loss: 0.1746
Epoch 14/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9662 - loss: 0.1994 - val_accuracy: 0.9788 - val_loss: 0.1585
Epoch 15/15
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9672 - loss: 0.1928 - val_accuracy: 0.9762 - val_loss: 0.1619

Rendimiento para test
157/157 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9747 - loss: 0.1607 
```
!(Imagen no encontrada)[./images/image_2.png]
