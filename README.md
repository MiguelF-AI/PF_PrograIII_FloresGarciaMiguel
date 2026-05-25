# PF_PrograIII_FloresGarciaMiguel
Proyecto final para materia Programacion III del CUGdl. Consiste en crear una red neural en forma de ejemplo con proyeccion a escalabilidad en aplicaciones a la carrera IA y ciencia de datos. FIDELIO.

# Predicción de Diabetes con Machine Learning

Repositorio del proyecto final de Programación III enfocado en la predicción de diabetes utilizando técnicas de Machine Learning y análisis de datos.

## Descripción

Este proyecto tiene como objetivo desarrollar un modelo capaz de predecir la probabilidad de diabetes en pacientes a partir de diferentes variables médicas como glucosa, presión arterial, IMC, edad, entre otras.

El proyecto incluye:

* Limpieza y preprocesamiento de datos.
* Análisis exploratorio de datos.
* Entrenamiento y evaluación de modelos de Machine Learning.
* Comparación de métricas de rendimiento.
* Visualización de resultados.

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow

## Estructura del proyecto

```bash
PF_PrograIII_FloresGarciaMiguel/
│
├── diabetes.csv/               # Dataset utilizado
├── graficas.py                 # Código para graficar resultados
├── modelo.py/                  # Código para crear el modelo de redes neuronales
├── threads.py/                 # Archivo principal (predicción con hilos)
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Documentación del repositorio
└── modelo_diabetes_v1.keras    # Modelo entrenado (O creado al ejecutar main.py)
```

> La estructura puede variar dependiendo de la versión del proyecto.

## Dataset

El proyecto utiliza un dataset relacionado con diagnósticos de diabetes, que contiene variables como:

* Número de embarazos
* Nivel de glucosa
* Presión arterial
* Espesor de piel
* Nivel de insulina
* Índice de masa corporal (BMI)
* Edad
* Historial médico

Fuente: 
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data

## Instalación

Clona este repositorio:

```bash
git clone https://github.com/MiguelF-AI/PF_PrograIII_FloresGarciaMiguel.git
```

Entra al directorio:

```bash
cd PF_PrograIII_FloresGarciaMiguel
```

Crea un entorno virtual:

```bash
python -m venv .venv
```

Activa el entorno virtual:

### Windows

```bash
.venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Ejecuta el proyecto con:

```bash
python threads.py
```

## Objetivo académico

Este proyecto fue desarrollado como parte de la materia de Programación III, aplicando conocimientos de:

* Ciencia de Datos
* Machine Learning
* Manipulación de datos
* Visualización
* Desarrollo en Python

## Autor

**Miguel Omar Flores García**

* GitHub: [MiguelF-AI](https://github.com/topics/diabetes-prediction?utm_source=chatgpt.com)

## 📄 Licencia

Este proyecto es únicamente con fines educativos y académicos.
