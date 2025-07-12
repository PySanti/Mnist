import matplotlib.pyplot as plt

def show_train_history(history):
    """
    Genera una ventana con dos gráficos a partir del historial de entrenamiento de un modelo:
    1. Loss vs Val_loss
    2. Accuracy vs Val_accuracy
    
    Args:
        history: Objeto con el historial de entrenamiento (típicamente el retorno de model.fit())
                Debe contener las claves: 'loss', 'val_loss', 'accuracy', 'val_accuracy'
    """
    # Obtener los datos del historial
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # Crear la figura con dos subgráficos
    plt.figure(figsize=(12, 5))
    
    # Gráfico 1: Loss vs Val_loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 2: Accuracy vs Val_accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b-', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Ajustar el layout y mostrar la figura
    plt.tight_layout()
    plt.show()
