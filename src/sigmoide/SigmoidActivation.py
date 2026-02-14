import numpy as np

class SigmoidActivation:
    def __init__(self):
        self.last_output = None

    def forward(self, x):
        """Calcula 1 / (1 + exp(-x))"""
        # Guardamos el resultado para la derivada
        self.last_output = 1 / (1 + np.exp(-x))
        return self.last_output

    def backward(self):
        """Calcula la derivada: S'(x) = y * (1 - y)"""
        if self.last_output is None:
            raise ValueError("Ejecuta el forward primero.")
            
        y = self.last_output
        return y * (1 - y)

# Prueba rápida
sig = SigmoidActivation()
x_val = np.array([-2, 0, 2])

y_val = sig.forward(x_val)
derivada = sig.backward()

print(f"Entrada x:    {x_val}")
print(f"Salida y:     {y_val}")
print(f"Derivada:     {derivada}")
