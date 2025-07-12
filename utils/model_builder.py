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
        drop = hp.Float(f'drop_{i}', min_value=0.2, max_value=0.35, step=0.05)
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
