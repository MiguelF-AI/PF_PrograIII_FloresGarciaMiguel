import os 
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def CargarDatos():
    # Cargar datos del archivo CSV
    ruta_csv = 'diabetes.csv'
    datos = pd.read_csv(ruta_csv)

    # Columnas donde el 0 es un dato faltante
    cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Reemplazar 0 por NaN para calcular la mediana real, luego rellenar
    for col in cols_con_ceros:
        datos[col] = datos[col].replace(0, np.nan)
        datos[col] = datos[col].fillna(datos[col].median())

    # Separar características y etiqueta 
    X = datos.drop('Outcome', axis=1).values.astype(np.float32)
    y = datos['Outcome'].values.astype(np.float32)

    # Normalización 
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)

    print(f"Dataset cargado. Forma de X: {X_escalado.shape}")
   
    return X_escalado, y


def CrearModelo(num_classes):
    # Crear un modelo secuencial con capas densas
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(32, activation='relu'), 
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    modelo.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])
    
    return modelo

def EntrenarModelo():
    X, y = CargarDatos()
    modelo = CrearModelo(num_classes=2)

    # Configurar una parada temprana para evitar el sobreentrenamiento
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10,          # Si en 10 épocas no mejora, se detiene
        restore_best_weights=True # Se queda con los mejores pesos, no con los últimos
    )

    # Entrenar con datos mezlados
    historial = modelo.fit(
        X, y, 
        epochs=100,           
        batch_size=32, 
        validation_split=0.2,
        shuffle=True,         # Mezcla los datos antes de separar el 20%
        callbacks=[early_stop],
        verbose=1             
    )

    return modelo, historial

def GuardarModelo(modelo_entrenado, ruta='modelo_entrenado.keras'):
    modelo_entrenado.save(ruta)
    print(f"Modelo guardado en: {os.path.abspath(ruta)}")