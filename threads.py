import os 
import tensorflow as tf
import numpy as np
import threading
import pandas as pd
from modelo import EntrenarModelo, CargarDatos, CrearModelo, GuardarModelo
from graficas import GraficarResultados

class PrediccionVariableThread(threading.Thread):
    def __init__(self, nombre_variable, col_indice, X_escalado, y_real, modelo, lock, datos_originales):
        super().__init__()
        self.nombre_variable = nombre_variable
        self.col_indice = col_indice
        self.X_escalado = X_escalado
        self.y_real = y_real
        self.modelo = modelo
        self.lock = lock
        self.datos_originales = datos_originales
        self.resultados = None

    def run(self):
        print(f"[HILO] Iniciando predicciones para la variable: '{self.nombre_variable}' (Columna {self.col_indice})")
        
        # Copiamos el dataset escalado
        X_temp = self.X_escalado.copy()
        
        # Aislamos esta variable poniendo todas las demás en 0.0 (su media en datos escalados)
        for i in range(X_temp.shape[1]):
            if i != self.col_indice:
                X_temp[:, i] = 0.0
                
        # Realizamos la predicción de forma segura utilizando un cerrojo (Lock)
        with self.lock:
            # Desactivamos los mensajes de progreso de Keras con verbose=0
            predicciones = self.modelo.predict(X_temp, verbose=0)
            
        self.resultados = predicciones
        
        # Imprimimos de manera organizada los resultados para esta variable
        # Usamos una lista de strings y las unimos al final para evitar entrelazado en consola
        salida = [
            f"\n=== Resultados del Hilo: Variable '{self.nombre_variable}' ===",
            f"Análisis de sensibilidad aislando esta característica (el resto se fija en su media):"
        ]
        
        for k in range(5):
            val_original = self.datos_originales.iloc[k][self.nombre_variable]
            pred = predicciones[k]
            clase_predicha = np.argmax(pred)
            probabilidad = np.max(pred) * 100
            estado = "Diabetes" if clase_predicha == 1 else "Sano"
            real_estado = "Diabetes" if self.y_real[k] == 1 else "Sano"
            salida.append(
                f"  Paciente {k+1}: Valor Real de '{self.nombre_variable}': {val_original} | "
                f"Predicción Aislada: {estado} ({probabilidad:.2f}%) | Real: {real_estado}"
            )
            
        # Calcular el promedio de probabilidad de diabetes para todo el dataset con esta variable aislada
        prob_promedio_diabetes = np.mean(predicciones[:, 1]) * 100
        salida.append(f"  --> Probabilidad promedio de Diabetes bajo efecto aislado de '{self.nombre_variable}': {prob_promedio_diabetes:.2f}%")
        
        print("\n".join(salida))

def main():
    nombre_archivo = 'modelo_diabetes_v1.keras'
    
    print("- Iniciando Proyecto de Redes Neuronales")
    
    # Verificar si el modelo ya existe para cargarlo o entrenar uno nuevo
    if os.path.exists(nombre_archivo):
        print(f"\n[INFO] Se encontró el archivo '{nombre_archivo}'. Cargando modelo...")
        modelo = tf.keras.models.load_model(nombre_archivo)
        
        # Cargamos datos para hacer la predicción
        X, y = CargarDatos()
        
        print("\n- Realizando predicciones multihilo (un hilo por cada variable)...")
        
        # Cargar los datos originales para mostrar nombres de columnas y valores reales
        datos_orig = pd.read_csv('diabetes.csv')
        cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_con_ceros:
            datos_orig[col] = datos_orig[col].replace(0, np.nan)
            datos_orig[col] = datos_orig[col].fillna(datos_orig[col].median())
        
        columnas = datos_orig.drop('Outcome', axis=1).columns.tolist()
        
        # Inicializar cerrojo para garantizar exclusión mutua en modelo.predict
        lock = threading.Lock()
        
        hilos = []
        # Crear un hilo para cada variable
        for i, col_name in enumerate(columnas):
            hilo = PrediccionVariableThread(col_name, i, X, y, modelo, lock, datos_orig)
            hilos.append(hilo)
            hilo.start()
            
        # Esperar a que terminen todos los hilos
        for hilo in hilos:
            hilo.join()
            
        print("\n- Predicciones multihilo finalizadas exitosamente.")
            
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