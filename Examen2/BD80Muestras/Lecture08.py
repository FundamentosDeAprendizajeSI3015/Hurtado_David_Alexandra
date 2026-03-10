# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================
# Librerías para manipulación de datos, visualización
# y construcción del modelo de machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# =========================================================
# CARGA DEL DATASET
# =========================================================
# Se carga el dataset desde un archivo CSV

print("\n" + "="*60)
print("CARGA DEL DATASET")
print("="*60)

df = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")


# =========================================================
# ELIMINAR COLUMNAS NO NUMÉRICAS
# =========================================================
# Se eliminan columnas que no se usarán en el modelo

df = df.drop(columns=["unidad","anio"])


# =========================================================
# INSPECCIÓN DEL DATASET
# =========================================================
# Exploración inicial para entender la estructura de los datos

print("\nPrimeras filas:")
print(df.head())

print("\nDimensiones del dataset:")
print(df.shape)

print("\nTipos de datos:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())


# =========================================================
# LIMPIEZA DE DATOS
# =========================================================
# Revisión y tratamiento de valores faltantes

print("\n" + "="*60)
print("LIMPIEZA DE DATOS")
print("="*60)

print("\nValores nulos por columna:")
print(df.isnull().sum())

# Se reemplazan valores faltantes con la mediana de cada variable
df = df.fillna(df.median())

print("\nValores nulos después de imputación:")
print(df.isnull().sum())

print("\nDimensiones después de limpiar:")
print(df.shape)


# =========================================================
# ANÁLISIS EXPLORATORIO
# =========================================================
# Visualización de la distribución de cada variable

print("\n" + "="*60)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("="*60)

for col in df.columns[:-1]:

    plt.figure()
    plt.hist(df[col], bins=20)

    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")

    plt.show()


# Matriz de correlación entre las variables
plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Matriz de correlación variables financieras")

plt.show()


# =========================================================
# DEFINICIÓN DE VARIABLES
# =========================================================
# Separación entre variables predictoras (X) y variable objetivo (Y)

print("\n" + "="*60)
print("PREPARACIÓN DE VARIABLES")
print("="*60)

Y = df["label"]
X = df.drop(columns=["label"])

print("\nVariables utilizadas en el modelo:")
print(X.columns.tolist())


# =========================================================
# DIVISIÓN TRAIN / VALIDATION / TEST (50 / 15 / 15)
# =========================================================
# División del dataset en conjuntos para entrenamiento,
# validación del modelo y evaluación final

print("\nDividiendo dataset en TRAIN / VALIDATION / TEST...")

# TRAIN = 50 muestras

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    Y,
    train_size=50,
    random_state=42,
    stratify=Y
)

# VALIDATION = 15 | TEST = 15

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=15,
    random_state=42,
    stratify=y_temp
)

print(f"\nTamaño TRAIN : {len(X_train)}")
print(f"Tamaño VAL   : {len(X_val)}")
print(f"Tamaño TEST  : {len(X_test)}")


# =========================================================
# ESCALAMIENTO
# =========================================================
# Normalización de los datos para mejorar el desempeño del modelo

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# =========================================================
# ENTRENAMIENTO DEL MODELO
# =========================================================
# Se entrena un árbol de decisión con restricciones
# para reducir el riesgo de overfitting

print("\n" + "="*60)
print("ENTRENAMIENTO DEL MODELO - ÁRBOL DE DECISIÓN")
print("="*60)

arbol = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=4,
    random_state=42
)

arbol.fit(X_train, y_train)


# =========================================================
# PREDICCIONES
# =========================================================
# Predicciones del modelo en los diferentes conjuntos

y_train_pred = arbol.predict(X_train)
y_val_pred = arbol.predict(X_val)
y_test_pred = arbol.predict(X_test)


# =========================================================
# MÉTRICAS DEL MODELO
# =========================================================
# Evaluación del rendimiento del modelo

print("\n" + "-"*60)
print("RESULTADOS EN TRAIN")
print("-"*60)

print(f"Accuracy Train : {accuracy_score(y_train, y_train_pred):.4f}")
print(classification_report(y_train, y_train_pred))


print("\n" + "-"*60)
print("RESULTADOS EN VALIDATION")
print("-"*60)

print(f"Accuracy Validation : {accuracy_score(y_val, y_val_pred):.4f}")
print(classification_report(y_val, y_val_pred))


print("\n" + "-"*60)
print("RESULTADOS EN TEST")
print("-"*60)

print(f"Accuracy Test  : {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred))


# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
# Visualización de aciertos y errores de clasificación

cm = confusion_matrix(y_test, y_test_pred)

plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Matriz de Confusión - Árbol de Decisión")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.show()


# =========================================================
# VALIDACIÓN CRUZADA
# =========================================================
# Evaluación del modelo usando validación cruzada (5 folds)

print("\n" + "="*60)
print("VALIDACIÓN CRUZADA (CROSS VALIDATION)")
print("="*60)

X_scaled_cv = scaler.fit_transform(X)

scores = cross_val_score(
    arbol,
    X_scaled_cv,
    Y,
    cv=5,
    scoring="accuracy"
)

print("\nAccuracy en cada fold:")
print(scores)

print("\nAccuracy promedio:")
print(scores.mean())

print("\nDesviación estándar:")
print(scores.std())


# =========================================================
# IMPORTANCIA DE VARIABLES
# =========================================================
# Identificación de las variables más relevantes en el modelo

print("\n" + "="*60)
print("IMPORTANCIA DE VARIABLES")
print("="*60)

importancias = pd.Series(
    arbol.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importancias)

plt.figure(figsize=(10,6))

importancias.plot(kind="bar")

plt.title("Variables financieras que más influyen en la situación financiera")

plt.show()


# =========================================================
# VISUALIZACIÓN DEL ÁRBOL
# =========================================================
# Representación gráfica del árbol de decisión

plt.figure(figsize=(22,12))

plot_tree(
    arbol,
    feature_names=X.columns,
    class_names=["Estable","Riesgo"],
    filled=True,
    rounded=True
)

plt.title("Árbol de Decisión - Factores que influyen en la situación financiera")

plt.show()


# =========================================================
# PCA 3D
# =========================================================
# Reducción de dimensionalidad para visualizar los datos en 3D

print("\n" + "="*60)
print("PCA 3D - VISUALIZACIÓN DE LOS DATOS")
print("="*60)

scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X)

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled_pca)

df_pca = pd.DataFrame(X_pca, columns=["PC1","PC2","PC3"])
df_pca["label"] = Y.values


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df_pca["PC1"],
    df_pca["PC2"],
    df_pca["PC3"],
    c=df_pca["label"],
    cmap="coolwarm"
)

plt.colorbar(scatter)

plt.show()


# =========================================================
# UMAP
# =========================================================
# Proyección de los datos a 2 dimensiones usando UMAP

import umap

print("\n" + "="*60)
print("UMAP - VISUALIZACIÓN DE LOS DATOS")
print("="*60)

scaler_umap = StandardScaler()
X_scaled_umap = scaler_umap.fit_transform(X)

umap_model = umap.UMAP(
    n_neighbors=10,
    min_dist=0.2,
    n_components=2,
    random_state=42
)

embedding = umap_model.fit_transform(X_scaled_umap)

df_umap = pd.DataFrame(embedding, columns=["UMAP1","UMAP2"])
df_umap["label"] = Y.values


plt.figure(figsize=(10,7))

scatter = plt.scatter(
    df_umap["UMAP1"],
    df_umap["UMAP2"],
    c=df_umap["label"],
    cmap="coolwarm"
)

plt.colorbar(scatter)

plt.title("UMAP - Proyección de los indicadores financieros")

plt.show()
