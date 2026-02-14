import numpy as np

# Definimos los valores según el libro:
BLANCO = 1.0
GRIS   = 0.0
NEGRO  = -1.0

# 1. IMAGEN OBJETIVO (Lo que queremos detectar)
# Una línea vertical blanca en el centro.
# Columna 1: Negra, Columna 2: Blanca, Columna 3: Negra
imagen_objetivo = np.array([
    [NEGRO, BLANCO, NEGRO],
    [NEGRO, BLANCO, NEGRO],
    [NEGRO, BLANCO, NEGRO]
])

# 2. LOS PESOS (La "memoria" de la neurona)
# El autor dice: "Los pesos imitan el patrón".
# Así que configuremos los pesos manualmente para que sean idénticos al objetivo.
pesos = np.array([
    [NEGRO, BLANCO, NEGRO],
    [NEGRO, BLANCO, NEGRO],
    [NEGRO, BLANCO, NEGRO]
])

def neurona_manual(imagen_entrada):
    """
    Simula una neurona: Multiplica pixel por peso y suma todo.
    Fórmula: Suma(Input * Peso)
    """
    # Multiplicación elemento a elemento (Hadamard product)
    coincidencias = imagen_entrada * pesos
    
    # La suma total es el "Score"
    puntaje = np.sum(coincidencias)
    return puntaje

# --- PRUEBAS ---

# Caso A: Le pasamos exactamente lo que busca
score_perfecto = neurona_manual(imagen_objetivo)
print(f"Score con imagen perfecta: {score_perfecto}")
# Lógica: (-1*-1) + (1*1) ... todo suma positivo. 
# Resultado esperado: 9.0 (Máxima puntuación posible)

# Caso B: Le pasamos una imagen totalmente opuesta (blanco donde busca negro)
imagen_opuesta = np.array([
    [BLANCO, NEGRO, BLANCO],
    [BLANCO, NEGRO, BLANCO],
    [BLANCO, NEGRO, BLANCO]
])
score_opuesto = neurona_manual(imagen_opuesta)
print(f"Score con imagen opuesta: {score_opuesto}")
# Lógica: (1*-1) da -1. Todo resta.
# Resultado esperado: -9.0

# Caso C: Le pasamos ruido o algo gris (el problema que menciona el autor)
imagen_gris = np.array([
    [GRIS, GRIS, GRIS],
    [GRIS, GRIS, GRIS],
    [GRIS, GRIS, GRIS]
])
score_gris = neurona_manual(imagen_gris)
print(f"Score con imagen gris: {score_gris}")
# Lógica: 0 * cualquier peso = 0. La neurona queda 'indecisa'.
