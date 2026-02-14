import tensorflow as tf
import numpy as np

# PASO 1: Los Datos (Lógica NAND)
# Entradas (x1, x2): -1 es False, 1 es True
X_train = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
], dtype=float)

# Salidas esperadas (y): Lógica NAND
# NAND solo es Falso (-1) cuando ambos son Verdaderos (1)
y_train = np.array([
    [ 1], # -1, -1 (F, F) -> V (1)
    [ 1], # -1,  1 (F, V) -> V (1)
    [ 1], #  1, -1 (V, F) -> V (1)
    [-1]  #  1,  1 (V, V) -> F (-1)  <-- El único caso negativo
], dtype=float)

# PASO 2: La Arquitectura
# CAMBIO IMPORTANTE: Para NAND no necesitamos "Capa Oculta".
# Una sola neurona (un Perceptrón simple) basta para resolverlo.
model = tf.keras.models.Sequential([
    # Una sola capa, 1 neurona de salida.
    tf.keras.layers.Dense(1, input_dim=2, activation='tanh')
])

# PASO 3: El Entrenamiento
# Usamos SGD (Descenso de Gradiente Estocástico) que es clásico para un perceptrón simple,
# aunque Adam también funcionaría perfecto.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['accuracy'])

# PASO 4: ¡A aprender!
print("Entrenando la red NAND...")
model.fit(X_train, y_train, epochs=500, verbose=0)

# PASO 5: Verificar si aprendió
print("\nPredicciones finales (Redondeadas):")
# Esperamos: [[1], [1], [1], [-1]]
print(model.predict(X_train).round(1))