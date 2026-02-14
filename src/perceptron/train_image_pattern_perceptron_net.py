import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# 1. PREPARACIÓN DE DATOS (Simulando la cuadrícula 3x3 del libro)
# ---------------------------------------------------------
# Imaginemos una cuadrícula 3x3 aplanada (9 píxeles).
# Índices:
# 0 1 2
# 3 4 5
# 6 7 8

# Definimos un patrón "GATO" ideal:
# Orejas en 0 y 2, Nariz en 4. El resto no importa tanto o debe estar vacío.
# Gato = [1, 0, 1,
#         0, 1, 0,
#         0, 0, 0]

def generar_datos(n_muestras=1000):
    X = []
    y = []
    for _ in range(n_muestras):
        # Generamos ruido aleatorio entre -1 (negro), 0 (gris), 1 (blanco)
        patron = np.random.choice([-1, 0, 1], size=9)
        
        # Etiqueta por defecto: 0 (No es gato)
        label = 0
        
        # Inyectamos el patrón "Gato" en el 50% de los casos
        if np.random.rand() > 0.5:
            # Forzamos las características del gato (Orejas y nariz)
            patron[0] = 1  # Oreja izq
            patron[2] = 1  # Oreja der
            patron[4] = 1  # Nariz
            # A veces añadimos ruido en otros píxeles para que aprenda a generalizar
            label = 1 # Es gato
            
        X.append(patron)
        y.append(label)
    return np.array(X), np.array(y)

# Generamos 1000 ejemplos de entrenamiento y 200 de prueba
X_train, y_train = generar_datos(1000)
X_test, y_test = generar_datos(200)

# ---------------------------------------------------------
# 2. CONSTRUCCIÓN DEL PERCEPTRÓN MULTICAPA (MLP)
# ---------------------------------------------------------
model = Sequential([
    # Capa de Entrada implícita (9 neuronas para los 9 píxeles)
    
    # Primera Capa Oculta: 16 neuronas, activación ReLU
    # Esto permite aprender combinaciones no lineales (supera el problema del gris)
    Dense(16, input_dim=9, activation='relu'),
    
    # Segunda Capa Oculta (Opcional, para más complejidad)
    Dense(8, activation='relu'),
    
    # Capa de Salida: 1 neurona con Sigmoide
    # Sigmoide comprime el resultado entre 0 y 1 (Probabilidad de ser Gato)
    Dense(1, activation='sigmoid')
])

# ---------------------------------------------------------
# 3. COMPILACIÓN Y ENTRENAMIENTO
# ---------------------------------------------------------
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy', # Ideal para clasificación binaria (Gato vs No Gato)
              metrics=['accuracy'])

print("Entrenando la red neuronal...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
print("¡Entrenamiento finalizado!")

# ---------------------------------------------------------
# 4. PRUEBA CON EL PATRÓN "GATO" DEL LIBRO
# ---------------------------------------------------------
print("\n--- Probando predicciones ---")

# Caso 1: El patrón perfecto de gato
gato_perfecto = np.array([[1, 0, 1, 0, 1, 0, 0, 0, 0]])
prediccion_gato = model.predict(gato_perfecto)

# Caso 2: Un patrón de ruido (todo negro)
todo_negro = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1]])
prediccion_ruido = model.predict(todo_negro)

print(f"Probabilidad de Gato (Patrón Perfecto): {prediccion_gato[0][0]:.4f} (Debería ser cercano a 1)")
print(f"Probabilidad de Gato (Todo Negro):      {prediccion_ruido[0][0]:.4f} (Debería ser cercano a 0)")

# Evaluación final en datos de test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en datos nuevos: {accuracy * 100:.2f}%")