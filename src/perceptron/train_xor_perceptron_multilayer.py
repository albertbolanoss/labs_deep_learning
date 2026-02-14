import tensorflow as tf
import numpy as np

# PASO 1: Los Datos (La Tabla 1-3 de tu imagen)
# Entradas (x1, x2) según tu libro: -1 es False, 1 es True
X_train = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
], dtype=float)

# Salidas esperadas (y2) según tu libro: XOR
# Iguales dan -1, Distintos dan 1
y_train = np.array([
    [-1], # -1, -1 -> -1
    [ 1], # -1,  1 ->  1
    [ 1], #  1, -1 ->  1
    [-1]  #  1,  1 -> -1
], dtype=float)

# PASO 2: La Arquitectura (Igual al diagrama)
model = tf.keras.models.Sequential([
    # Capa Oculta: 2 neuronas (P0, P1), entrada de 2 (x1, x2)
    # Usamos 'tanh' porque tus datos van de -1 a 1 (como en el libro)
    tf.keras.layers.Dense(2, input_dim=2, activation='tanh'),

    # Capa de Salida: 1 neurona (P2)
    tf.keras.layers.Dense(1, activation='tanh')
])

# PASO 3: El Entrenamiento (Cómo aprender)
# Optimizer 'adam': Es el algoritmo estándar que ajusta los pesos (mejor que el viejo SGD)
# Loss 'mse': Error Cuadrático Medio. Mide qué tan lejos está el resultado de -1 o 1.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['accuracy'])

# PASO 4: ¡A aprender!
# Epochs: Cuántas veces va a leer la tabla completa para practicar
print("Entrenando la red...")
model.fit(X_train, y_train, epochs=500, verbose=0) 

# PASO 5: Verificar si aprendió
print("\nPredicciones finales:")
print(model.predict(X_train).round(1))