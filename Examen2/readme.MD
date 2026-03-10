# Análisis Financiero - Solución UdeA

Este proyecto realiza un **análisis financiero de la solución FIRE de la Universidad de Antioquia (UdeA)** utilizando técnicas de **aprendizaje automático**.

El objetivo principal es **identificar las variables financieras que más influyen en la estabilidad o riesgo financiero** de las unidades analizadas. Para ello se construyó un modelo de **Árbol de Decisión**, además de realizar análisis exploratorio, visualización de datos y reducción de dimensionalidad (PCA y UMAP).

Este trabajo fue desarrollado en el marco del **Examen 2 de la materia _Fundamentos de Aprendizaje Automático_**.

---

## Metodología

El proyecto incluye las siguientes etapas:

- Carga y limpieza del dataset
- Análisis exploratorio de datos
- División del dataset en **train, validation y test**
- Entrenamiento de un **modelo de Árbol de Decisión**
- Evaluación mediante métricas de clasificación
- Análisis de **importancia de variables financieras**
- Visualización mediante **PCA 3D y UMAP**

El modelo permite identificar **qué indicadores financieros tienen mayor impacto en la situación financiera**.

---

## Estructura del proyecto


├── requirements.txt

│
├── BD500Muestras/

│ ├── dataset_sintetico_FIRE_UdeA_realista.csv

│ └── Lecture08.py

│
└── BD80Muestras/

├── dataset_sintetico_FIRE_UdeA_realista.csv

└── Lecture08.py

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- UMAP

---

## Objetivo

Analizar los **indicadores financieros de la solución FIRE de la UdeA** para identificar patrones y variables que influyen en la **estabilidad o riesgo financiero**,
