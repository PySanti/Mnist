
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
