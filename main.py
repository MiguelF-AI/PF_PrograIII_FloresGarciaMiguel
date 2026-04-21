import modelo as modelo_module
import os
from tensorflow import keras

# CREAR EL ENTORNO VIRTUAL (si o si esta version)
# python -3.11 -m venv .venv

# Habilitar el entorno virtual (en Windows)
# source .venv/bin/activate

# Instalar las dependencias

RUTA_MODELO= 'modelo_entrenado.keras'

def guardar_modelo(modelo_entrenado, ruta=RUTA_MODELO):
    modelo_entrenado.save(ruta)
    print(f"✅ Modelo guardado en: {os.path.abspath(ruta)}")

def cargar_modelo(ruta=RUTA_MODELO):
    if os.path.exists(ruta):
        print(f"📦 Cargando modelo desde: {os.path.abspath(ruta)}")
        return keras.models.load_model(ruta)
    return None

def prediccion(modelo):
    resultado = modelo.predict([100.0])
    print(f'La temperatura {100.0}°C es igual a {resultado}°F')

if __name__ == "__main__":
    modelo_entrenado = cargar_modelo()

    if modelo_entrenado is None:
        print("🚀 No hay modelo guardado. Entrenando...")
        modelo_entrenado, historial = modelo.entrenar_modelo()
        guardar_modelo(modelo_entrenado)