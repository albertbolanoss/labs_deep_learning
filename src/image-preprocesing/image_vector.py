import numpy as np
from PIL import Image
import cv2
import matplotlib
# ¡La magia para evitar el error de Tkinter en tu terminal!
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Cargar imagen y convertir a NumPy
image = Image.open("images/scene3.jpg")
image_arr = np.asarray(image)

# Convertir a grises (Recordando que PIL es RGB)
gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)

# Crear una figura con 1 fila y 4 columnas
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 1. Imagen en Grises
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Escala de Grises")
axes[0].axis('off') # Ocultar los ejes/números

# 2. Canal Rojo (Índice 0)
# Usamos cmap='gray' para ver la "intensidad" de luz roja que captó el sensor
axes[1].imshow(image_arr[:,:,0], cmap='gray') 
axes[1].set_title("Canal Rojo")
axes[1].axis('off')

# 3. Canal Verde (Índice 1)
axes[2].imshow(image_arr[:,:,1], cmap='gray')
axes[2].set_title("Canal Verde")
axes[2].axis('off')

# 4. Canal Azul (Índice 2)
axes[3].imshow(image_arr[:,:,2], cmap='gray')
axes[3].set_title("Canal Azul")
axes[3].axis('off')

print("Red Channel")
print(image_arr[:,:,0])
print("Green Channel")
print(image_arr[:,:,1])
print("Blue Channel")
print(image_arr[:,:,2])

# Ajustar los márgenes y guardar
plt.tight_layout()
plt.savefig("images/canales_separados.png")
