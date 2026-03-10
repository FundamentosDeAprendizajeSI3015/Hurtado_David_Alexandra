# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================
# Se importan las librerías necesarias para:
# - manipulación y análisis de datos
# - visualización
# - construcción y evaluación de modelos de machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Herramientas para preprocesamiento y división de datos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Modelo de aprendizaje automático utilizado
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Métricas para evaluar el desempeño del modelo
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Herramientas para reducción de dimensionalidad
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# =========================================================
# CARGA DEL DATASET
# =========================================================
# Se carga el dataset que contiene indicadores financieros
# utilizados para analizar la situación financiera.

print("\n" + "="*60)
print("CARGA DEL DATASET")
print("="*60)

df = pd.read_csv("dataset_sintetico_FIRE_UdeA.csv")


# INSPECCIÓN DEL DATASET
# Se revisan las primeras filas y características del dataset
# para entender su estructura.

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
# Se verifica si existen valores nulos que puedan afectar
# el análisis o el entrenamiento del modelo.

print("\n" + "="*60)
print("LIMPIEZA DE DATOS")
print("="*60)

print("\nValores nulos por columna:")
print(df.isnull().sum())

# Eliminación de registros incompletos si existieran
df = df.dropna()

print("\nDimensiones después de limpiar:")
print(df.shape)


# =========================================================
# ANÁLISIS EXPLORATORIO
# =========================================================
# Se explora la distribución de cada variable para entender
# su comportamiento y detectar posibles patrones.

print("\n" + "="*60)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("="*60)

# Histogramas de cada variable financiera
for col in df.columns[:-1]:

    plt.figure()
    plt.hist(df[col], bins=20)

    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")

    plt.show()


# Matriz de correlación
# Permite identificar relaciones entre variables financieras.

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
# Se separa la variable objetivo de las variables predictoras.

print("\n" + "="*60)
print("PREPARACIÓN DE VARIABLES")
print("="*60)

# Variable objetivo (situación financiera)
Y = df["label"]

# Variables predictoras (indicadores financieros)
X = df.drop(columns=["label"])

print("\nVariables utilizadas en el modelo:")
print(X.columns.tolist())


# =========================================================
# DIVISIÓN TRAIN / VALIDATION / TEST
# =========================================================
# Se divide el dataset en:
# Train ≈ 200
# Validation ≈ 150
# Test ≈ 150

print("\nDividiendo dataset en TRAIN / VALIDATION / TEST...")

# Primera división: Train (40%) y Temp (60%)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    Y,
    test_size=0.6,   # 60% se guarda para validation + test
    random_state=42,
    stratify=Y
)

# Segunda división: Validation (30%) y Test (30%)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,   # mitad validation y mitad test
    random_state=42,
    stratify=y_temp
)

print(f"\nTamaño TRAIN : {len(X_train)}")
print(f"Tamaño VAL   : {len(X_val)}")
print(f"Tamaño TEST  : {len(X_test)}")


# =========================================================
# ESCALAMIENTO (SIN DATA LEAKAGE)
# =========================================================
# Se aplica estandarización a las variables numéricas.
# El escalamiento se ajusta solo con el conjunto de entrenamiento
# para evitar data leakage.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# =========================================================
# ENTRENAMIENTO DEL MODELO
# =========================================================
# Se entrena un modelo de árbol de decisión para clasificar
# la situación financiera.

print("\n" + "="*60)
print("ENTRENAMIENTO DEL MODELO - ÁRBOL DE DECISIÓN")
print("="*60)

arbol = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

arbol.fit(X_train, y_train)


# =========================================================
# PREDICCIONES
# =========================================================
# El modelo genera predicciones para los conjuntos de datos.

y_train_pred = arbol.predict(X_train)
y_val_pred = arbol.predict(X_val)
y_test_pred = arbol.predict(X_test)


# =========================================================
# MÉTRICAS DEL MODELO
# =========================================================
# Se evalúa el desempeño del modelo mediante métricas de
# clasificación.

print("\n" + "-"*60)
print("RESULTADOS EN TRAIN")
print("-"*60)

print(f"Accuracy Train : {accuracy_score(y_train, y_train_pred):.4f}")

print("\nReporte de clasificación (TRAIN):\n")
print(classification_report(y_train, y_train_pred))


print("\n" + "-"*60)
print("RESULTADOS EN VALIDATION")
print("-"*60)

print(f"Accuracy Validation : {accuracy_score(y_val, y_val_pred):.4f}")

print("\nReporte de clasificación (VALIDATION):\n")
print(classification_report(y_val, y_val_pred))


print("\n" + "-"*60)
print("RESULTADOS EN TEST")
print("-"*60)

print(f"Accuracy Test  : {accuracy_score(y_test, y_test_pred):.4f}")

print("\nReporte de clasificación (TEST):\n")
print(classification_report(y_test, y_test_pred))


# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
# La matriz de confusión muestra los aciertos y errores del modelo.

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
# Se evalúa la estabilidad del modelo utilizando validación
# cruzada con 5 particiones del dataset.

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
print(f"{scores.mean():.4f}")

print("\nDesviación estándar:")
print(f"{scores.std():.4f}")


# =========================================================
# IMPORTANCIA DE VARIABLES
# =========================================================
# Se identifican las variables financieras que tienen mayor
# impacto en la clasificación del modelo.

print("\n" + "="*60)
print("IMPORTANCIA DE VARIABLES")
print("="*60)

importancias = pd.Series(
    arbol.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importancias.to_string())

plt.figure(figsize=(10,6))

importancias.plot(kind="bar")

plt.title("Variables financieras que más influyen en la situación financiera")
plt.ylabel("Importancia")
plt.xlabel("Variables")

plt.xticks(rotation=45)

plt.show()


# =========================================================
# VISUALIZACIÓN DEL ÁRBOL
# =========================================================
# Se muestra la estructura del árbol de decisión para
# interpretar cómo el modelo toma decisiones.

plt.figure(figsize=(22,12))

plot_tree(
    arbol,
    feature_names=X.columns,
    class_names=["Estable", "Riesgo"],
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True
)

plt.title(
    "Árbol de Decisión - Factores que influyen en la situación financiera",
    fontsize=16
)

plt.show()


# =========================================================
# PCA 3D
# =========================================================
# Se aplica reducción de dimensionalidad con PCA para
# visualizar los datos en tres dimensiones.

print("\n" + "="*60)
print("PCA 3D - VISUALIZACIÓN DE LOS DATOS")
print("="*60)

scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X)

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled_pca)

df_pca = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2", "PC3"]
)

df_pca["label"] = Y.values

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df_pca["PC1"],
    df_pca["PC2"],
    df_pca["PC3"],
    c=df_pca["label"],
    cmap="coolwarm",
    alpha=0.8
)

ax.set_title("PCA 3D - Situación financiera")
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.set_zlabel("Componente Principal 3")

plt.colorbar(scatter, label="Situación financiera")

plt.show()


# =========================================================
# REDUCCIÓN DE DIMENSIONALIDAD - UMAP
# =========================================================
# UMAP permite visualizar patrones no lineales en los datos.

import umap

print("\n" + "="*60)
print("UMAP - VISUALIZACIÓN DE LOS DATOS")
print("="*60)

scaler_umap = StandardScaler()
X_scaled_umap = scaler_umap.fit_transform(X)

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

embedding = umap_model.fit_transform(X_scaled_umap)

df_umap = pd.DataFrame(
    embedding,
    columns=["UMAP1", "UMAP2"]
)

df_umap["label"] = Y.values

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    df_umap["UMAP1"],
    df_umap["UMAP2"],
    c=df_umap["label"],
    cmap="coolwarm",
    alpha=0.8
)

plt.colorbar(scatter, label="Situación financiera")

plt.title("UMAP - Proyección de los indicadores financieros")

plt.xlabel("UMAP dimensión 1")
plt.ylabel("UMAP dimensión 2")

plt.grid(alpha=0.3)

plt.show()
