# 🌸 Clasificación de Especies de Iris — Proyecto Final de Minería de Datos

Este proyecto implementa un sistema interactivo de clasificación de flores Iris utilizando un flujo completo de Minería de Datos y una interfaz visual construida con Streamlit.

Los usuarios pueden explorar el conjunto de datos, visualizar métricas, ingresar medidas personalizadas de flores y obtener predicciones con posicionamiento 3D en tiempo real.

---

## 🎯 Objetivos del Proyecto

El objetivo de este proyecto es diseñar y justificar un pipeline completo de minería de datos utilizando el conjunto de datos Iris.

- El proyecto incluye:
- Exploración del dataset
- Preprocesamiento
- Entrenamiento del modelo
- Evaluación
- Visualización
- Predicción interactiva

---

## 📁 Estructura del Repositorio

```
📦 Rubrica
│
├── app.py               # Aplicación principal de Streamlit
├── Iris.csv             # Dataset local usado por el sistema
├── requirements.txt     # Dependencias requeridas
└── README.md            # Documentación del proyecto
```

---

## ☁️ Versión desplegada en Streamlit Cloud

🔗 **Dashboard en línea:** https://rubrica-u6bragdpfuhvg4cx3svsgx.streamlit.app/

---

## 📊 Análisis Exploratorio de Datos (EDA)

La aplicación incluye:

- Histogramas de distribución de atributos
- Matriz de correlación
- Visualización interactiva del dataset
- Gráfico 3D tipo scatter

Estas herramientas ayudan a comprender claramente cómo se separan las clases según sus características.

---

## 🤖 Flujo de Trabajo del Modelo de Machine Learning

Pipeline utilizado:

```
Cargar Dataset → Preprocesamiento → División Train/Test
→ Escalado (StandardScaler)
→ Entrenamiento (RandomForest Classifier)
→ Evaluación (Accuracy, Precision, Recall, F1)
→ Predicción y Visualización

```

### ✔ Modelo Seleccionado

**Random Forest Classifier**, elegido por su fuerte rendimiento y robustez.

---

## 📈 Métricas del Modelo

El sistema calcula:

- Accuracy
- Precision
- Recall
- F1 Score
- Reporte de clasificación completo

Todas las métricas se generan dentro de la aplicación.

---

## 🖥️ Características de la App en Streamlit

### 🔮 Predicción Interactiva

Los usuarios ingresan:

- Largo del sépalo
- Ancho del sépalo
- Largo del pétalo
- Ancho del pétalo

Y reciben:

- Especie predicha
- Distribución de probabilidades por clase
- Posición 3D de la nueva muestra

### 📌 Visualización 3D

Los usuarios pueden elegir los ejes y explorar el dataset de forma espacial.

---

## 👥 Integrantes

- **Samuel Mejía**
- **Miguel Perez**
- **Aaron Roa**
- **Aldair Escobar**
