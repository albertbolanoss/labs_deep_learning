import cv2
import numpy as np
from PIL import Image

# 1. Cargar imagen con PIL y pasar a arreglo de numpy
image_pil = Image.open("images/scene3.jpg")
image_arr = np.asarray(image_pil)

# Convertir de RGB (PIL) a BGR para que OpenCV la guarde con los colores correctos
image_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)

# Obtener dimensiones de la imagen (alto, ancho)
h, w = image_bgr.shape[:2]

# --- OPERACIÓN 1: TRASLACIÓN ---
# Matriz: Mover 50 píxeles a la derecha (x) y 30 hacia abajo (y)
matriz_traslacion = np.float32([
    [1, 0, 50],
    [0, 1, 30]
])
img_trasladada = cv2.warpAffine(image_bgr, matriz_traslacion, (w, h))
cv2.imwrite("images/1_trasladada.jpg", img_trasladada)

# --- OPERACIÓN 2: ROTACIÓN ---
# OpenCV tiene una función especial para calcular la matriz trigonométrica exacta
# Rotar 45 grados desde el centro de la imagen, manteniendo la escala en 1.0
centro = (w // 2, h // 2)
matriz_rotacion = cv2.getRotationMatrix2D(centro, 45, 1.0)
img_rotada = cv2.warpAffine(image_bgr, matriz_rotacion, (w, h))
cv2.imwrite("images/2_rotada.jpg", img_rotada)

# --- OPERACIÓN 3: MAGNIFICACIÓN (ESCALADO) ---
# Matriz: Multiplicar X y Y por 2 (Zoom in)
matriz_escala = np.float32([
    [2, 0, 0],
    [0, 2, 0]
])
# Aumentamos el tamaño del "lienzo" de salida al doble para que la imagen quepa
img_escalada = cv2.warpAffine(image_bgr, matriz_escala, (w * 2, h * 2))
cv2.imwrite("images/3_escalada.jpg", img_escalada)

print("¡Transformaciones aplicadas! Revisa los archivos guardados en tu directorio.")