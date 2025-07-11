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

