from PIL import Image
import imagehash

# Generar los hashes perceptuales (pHash)
hash1 = imagehash.phash(Image.open("images/scene3.jpg"))
# Nota: Estás abriendo la misma imagen dos veces, por lo que la distancia será 0.
hash2 = imagehash.phash(Image.open("images/imageFruits.png")) 

# imagehash sobrecarga el operador de resta (-) para calcular la distancia de Hamming
distancia = hash1 - hash2

print(distancia)