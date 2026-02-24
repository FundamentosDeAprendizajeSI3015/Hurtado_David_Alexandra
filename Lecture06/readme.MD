# Actividad de Clase: Modelos Basados en Árboles y Evaluación de Desempeño

## Contexto

Esta actividad de clase se desarrolla tomando como base el **pipeline de preprocesamiento previamente entregado**, el cual incluía limpieza de datos, transformación de variables y preparación del dataset para modelado.

A partir de ese pipeline, el objetivo de esta actividad fue trabajar específicamente:

- Árboles de decisión
- Modelos basados en ensambles
- Balanceo de clases
- Evaluación mediante accuracy
- Análisis de importancia de variables

---

## Objetivo

Aplicar modelos de clasificación basados en árboles para predecir el **nivel de influencia del uso de herramientas de Inteligencia Artificial**, evaluando su desempeño y analizando qué variables tienen mayor impacto en la predicción.

---

## Pipeline Utilizado

Se reutilizó el pipeline previamente construido, que incluye:

1. Limpieza y estandarización de nombres de columnas.
2. Eliminación de variables irrelevantes.
3. Conversión de variables numéricas.
4. Codificación ordinal de variables con orden natural.
5. One Hot Encoding para variables categóricas nominales.
6. Estandarización de variables numéricas.
7. Construcción de la variable objetivo (`nivel_influencia`).

---

## Balanceo de Clases

Dado que la variable objetivo presentaba desbalance, se aplicó:

- Oversampling uniforme
- Muestreo con reemplazo
- Igualación al tamaño de la clase mayoritaria

El objetivo fue evitar que el modelo favoreciera la clase dominante y garantizar una distribución equitativa entre:

- Baja
- Media
- Alta

---

## División del Dataset

El dataset balanceado fue dividido de manera estratificada en:

- Conjunto de entrenamiento
- Conjunto de prueba
- Conjunto de validación

La estratificación permitió mantener la proporción de clases en cada subconjunto.

---

## Modelos Implementados

Se entrenaron y compararon los siguientes modelos:

- Decision Tree
- Random Forest
- Gradient Boosting

Cada modelo fue entrenado sobre el conjunto de entrenamiento y evaluado en test y validation.

---

## Métricas de Evaluación

Se utilizaron las siguientes métricas:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Matriz de confusión

El criterio principal de comparación fue el **Accuracy en el conjunto de validación**.

---

## Interpretabilidad del Modelo

Se realizó análisis de interpretabilidad mediante:

- Visualización del árbol de decisión
- Cálculo de importancia de variables (`feature_importances_`)

Esto permitió identificar cuáles variables influyen más en la predicción del nivel de influencia.

---

## Resultados

Se compararon los modelos utilizando el accuracy en validation y se seleccionó el modelo con mejor desempeño.

Adicionalmente, se analizaron las variables más importantes para entender el comportamiento del modelo y su lógica de decisión.

---

## Tecnologías Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Conclusión

Esta actividad permitió reforzar conceptos clave de:

- Modelos basados en árboles
- Manejo de desbalance de clases
- Evaluación de desempeño
- Interpretabilidad en modelos supervisados
