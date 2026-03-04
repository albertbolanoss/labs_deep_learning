import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Evita el error de Tkinter en tu entorno
import matplotlib.pyplot as plt
from PIL import Image

# 1. Cargar la imagen (Asegúrate de que 'carshort.png' esté en la ruta correcta)
# Si no la tienes a la mano, puedes cambiarla por "images/scene3.jpg"
image = Image.open('images/scene3.jpg')

# 2. Convertir a arreglo de NumPy e imprimir dimensiones
image_arr = np.asarray(image)
print(f"Dimensiones de la imagen original: {image_arr.shape}")

# 3. Convertir a Escala de Grises (Recordando que PIL lee en RGB)
gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)

# 4. Definir el Kernel (Filtro de Convolución)
# OJO: Este kernel detecta bordes horizontales, no difumina.
kernel = np.array([[-1, -1, -1],
                   [ 2,  2,  2],
                   [-1, -1, -1]])

# 5. Aplicar el filtro de convolución (cv2.filter2D desliza la matriz por toda la imagen)
imagen_filtrada = cv2.filter2D(gray, -1, kernel)

# ==========================================
# VISUALIZACIÓN: Guardar resultados en el disco
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(gray, cmap='gray')
axes[0].set_title("1. Escala de Grises")
axes[0].axis('off')

axes[1].imshow(imagen_filtrada, cmap='gray')
axes[1].set_title("2. Filtro (Detectar Bordes Horizontales)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("images/resultado_filtro.png")
print("¡Éxito! Gráfica guardada como resultado_filtro.png")