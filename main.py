import os 
import tensorflow as tf
import numpy as np
from modelo import EntrenarModelo, CargarDatos, CrearModelo, GuardarModelo
from graficas import GraficarResultados

def main():
    nombre_archivo = 'modelo_diabetes_v1.keras'
    
    print("- Iniciando Proyecto de Redes Neuronales")
    
    # Verificar si el modelo ya existe para cargarlo o entrenar uno nuevo
    if os.path.exists(nombre_archivo):
        print(f"\n[INFO] Se encontró el archivo '{nombre_archivo}'. Cargando modelo...")
        modelo = tf.keras.models.load_model(nombre_archivo)
        
        # Cargamos datos solo para hacer una prueba de predicción
        X, y = CargarDatos()
        
        print("\n-Realizando predicciones con el modelo cargado")
        predicciones = modelo.predict(X[:5])
        
        for i, pred in enumerate(predicciones):
            clase_predicha = np.argmax(pred)
            probabilidad = np.max(pred) * 100
            estado = "Diabetes" if clase_predicha == 1 else "Sano"
            print(f"Paciente {i+1}: Predicción: {estado} ({probabilidad:.2f}%) | Real: {'Diabetes' if y[i]==1 else 'Sano'}")
            
    else:
        print("\nNo existe modelo previo. Iniciando proceso de entrenamiento...")
        
        # Ejecutar el entrenamiento y obtener el historial para graficar
        modelo, historial = EntrenarModelo()
        
        print("\nEntrenamiento finalizado.")

        # Llamar a la función de gráficas
        GraficarResultados(historial)
        
        # Mostrar métricas finales
        print("\nResumen de Métricas Finales:")

        final_acc = historial.history['accuracy'][-1]
        final_loss = historial.history['loss'][-1]
        val_acc = historial.history['val_accuracy'][-1]
        val_loss = historial.history['val_loss'][-1]

        print(f"Precisión en Entrenamiento (Accuracy): {final_acc*100:.2f}%")
        print(f"Precisión en Validación (Val Accuracy): {val_acc*100:.2f}%")
        print(f"Pérdida en Entrenamiento (Loss): {final_loss:.4f}")
        print(f"Pérdida en Validación (Val Loss): {val_loss:.4f}")

        # Guardar el modelo
        modelo.save(nombre_archivo)
        print(f"Modelo guardado exitosamente como '{nombre_archivo}'")

if __name__ == "__main__":
    main()