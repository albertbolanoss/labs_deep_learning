import numpy as np

class TanhActivation:
    def __init__(self):
        self.last_output = None

    def forward(self, x):
        """Calcula el valor de la función (Forward Pass)"""
        self.last_output = np.tanh(x)
        return self.last_output

    def backward(self):
        """Calcula la derivada usando el valor guardado (Backward Pass)"""
        # Aquí aplicamos la propiedad: tanh'(x) = 1 - y^2
        if self.last_output is None:
            raise ValueError("Debes ejecutar el forward pass primero.")
            
        return 1 - (self.last_output ** 2)

# Ejemplo de uso:
activation = TanhActivation()

# 1. Supongamos que x es la entrada de una neurona
x = np.array([-1.0, 0.0, 2.0])
y = activation.forward(x)

print(f"Entrada (x): {x}")
print(f"Salida (y):  {y}")

# 2. Calculamos la derivada SIN usar x, solo usando y
derivada = activation.backward()
print(f"Derivada:    {derivada}")