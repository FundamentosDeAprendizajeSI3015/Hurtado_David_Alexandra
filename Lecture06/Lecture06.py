# IMPORTACIÓN DE LIBRERÍAS
# Estas librerías nos permiten leer datos, limpiarlos, analizarlos, visualizarlos y aplicar técnicas de reducción de dimensionalidad.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # poner todas las variables numéricas en la misma escala
import umap  # visualizar datos complejos en menos dimensiones
from sklearn.decomposition import PCA  #  reducir dimensiones conservando información
from sklearn.manifold import TSNE  # Técnica no lineal ideal para visualizar posibles agrupaciones


# CARGA DEL DATA SET
# Aquí cargamos el archivo de Excel que contiene las respuestas de la encuesta.
df = pd.read_excel("DatosEncuestaInforme1.xlsx")

# Limpieza de nombres de columnas
# Antes de trabajar con las variables, limpiamos sus nombres.
# Esto evita errores cuando accedemos a ellas más adelante.
df.columns = df.columns.str.strip()        # Elimina espacios al inicio o al final
df.columns = df.columns.str.replace("\n", "")  # Quita saltos de línea ocultos
df.columns = df.columns.str.lower()        # Convierte todo a minúsculas para mantener uniformidad

# Eliminar columnas innecesarias
# Quitamos información como correos, nombres o marcas de tiempo, ya que no aportan valor al análisis estadístico.
cols_eliminar = [
    'hora de finalización',
    'correo electrónico',
    'hora de la última modificación',
    'nombre',
    'id',
    'hora de inicio'
]

df = df.drop(columns=cols_eliminar, errors='ignore')  # Si alguna columna no existe, simplemente la ignora

print("\nColumnas eliminadas correctamente:")
print(cols_eliminar)

# RENOMBRAR COLUMNAS LARGAS A VARIABLES MÁS MANEJABLES
# Las preguntas originales son muy largas, así que las convertimos en nombres más cortos y fáciles de usar dentro del código.
mapeo_columnas = {
    "¿cuál es tu edad?": "edad",
    "en promedio, ¿cuántas horas al día utiliza herramientas de inteligencia artificial?": "frecuencia_uso_ia",
    "¿para qué tipo de decisiones usas principalmente herramientas de ia?": "tipo_decisiones",
    "en una escala del 1 al 5, donde 1 significa “nada” y 5 “completamente”, ¿en qué medida delega el análisis o razonamiento a herramientas de inteligencia artificial al tomar decisiones?": "delega_razonamiento",
    "en una escala del 1 al 5, donde 1 significa “nada” y 5 “totalmente”, ¿qué nivel de confianza tiene en las respuestas proporcionadas por herramientas de inteligencia artificial?": "confia_respuesta_ia",
    "¿suele verificar la información que le da la ia antes de tomar una decisión?": "verifica_respuestas",
    "¿cómo describirías tu nivel de conocimiento técnico en herramientas digitales o ia?": "nivel_experiencia",
    "en una escala del 1 al 5, donde 1 significa “nada dependiente” y 5 “totalmente dependiente”, ¿qué tan dependiente considera que es de herramientas de inteligencia artificial al momento de tomar decisiones?": "dependencia_percibida",
    "¿cuál es tu edad?": "edad"
}

df = df.rename(columns=mapeo_columnas)

print("\nColumnas renombradas correctamente:")
print(df.columns.tolist())


# INSPECCIÓN INICIAL DEL DATA SET
# Antes de hacer cualquier transformación, observamos cómo se ve el dataset.
# Esto nos ayuda a entender con qué estamos trabajando.

print("Primeras filas del dataset:")
print(df.head())  # Miramos una pequeña muestra

print("\nDimensiones del dataset:")
print(df.shape)   # Cuántas filas (respuestas) y columnas (variables) tenemos

print("\nTipos de datos:")
print(df.info())  # Revisamos si cada variable tiene el tipo correcto

print("\nEstadisticas descriptivas:")
print(df.describe())  # Resumen numérico general


# LIMPIEZA DE DATOS

# Verificar valores nulos
# Revisamos si hay respuestas incompletas que puedan afectar el análisis.
print("\nValores nulos por columna:")
print(df.isnull().sum())


# Asegurar que variables numéricas realmente sean numéricas
# Si algún valor no se puede convertir, lo transformamos en NaN para tratarlo adecuadamente después.
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["delega_razonamiento"] = pd.to_numeric(df["delega_razonamiento"], errors="coerce")
df["confia_respuesta_ia"] = pd.to_numeric(df["confia_respuesta_ia"], errors="coerce")
df["dependencia_percibida"] = pd.to_numeric(df["dependencia_percibida"], errors="coerce")


# TRANSFORMACIÓN DE VARIABLES
# Convertimos variables categóricas en valores numéricos respetando su orden lógico cuando existe.

# Esta variable tiene un orden natural de menor a mayor uso.
df["frecuencia_uso_ia"] = df["frecuencia_uso_ia"].map({
    "0-1": 1,
    "1-3": 2,
    "3-5": 3,
    "5 o más": 4
})

# Aquí también respetamos el orden: no verifica, a veces, sí verifica.
df["verifica_respuestas"] = df["verifica_respuestas"].map({
    "No": 0,
    "A veces": 1,
    "Sí": 2
})

print("Nulos después del mapeo:")
print(df.isnull().sum())

# Eliminamos filas con datos faltantes.
# Preferimos trabajar con registros completos para evitar ruido en el modelado.
df = df.dropna()

print("Dimensiones después de eliminar nulos:", df.shape)

# One Hot Encoding
# Convertimos variables categóricas sin orden en columnas binarias.
# Esto permite que los modelos matemáticos puedan procesarlas.
df = pd.get_dummies(
    df,
    columns=["tipo_decisiones", "nivel_experiencia"],
    drop_first=True
)

print("\nDataset despues de transformaciones:")
print(df.head())


# ESCALAMIENTO DE VARIABLES

# Seleccionamos las variables numéricas que deben ponerse en la misma escala.
# Esto es importante porque algunos modelos son sensibles a magnitudes distintas.
columnas_escalar = [
    "edad",
    "delega_razonamiento",
    "confia_respuesta_ia",
    "dependencia_percibida",
    "frecuencia_uso_ia",
    "verifica_respuestas"
]

# Creamos una copia para no alterar el dataset original.
df_scaled = df.copy()

# Aplicamos estandarización: todas las variables tendrán media 0 y desviación estándar 1.
scaler = StandardScaler()
df_scaled[columnas_escalar] = scaler.fit_transform(df[columnas_escalar])

print("\nDataset escalado correctamente (sin afectar variables dummy):")
print(df_scaled.head())

# Guardamos esta versión final, lista para aplicar modelos.
df_scaled.to_csv("dataset_model_ready.csv", index=False)


# ANÁLISIS EXPLORATORIO

# Aquí empezamos a entender los datos con estadísticas básicas.

print("\n--- MEDIDAS DE TENDENCIA CENTRAL ---")

print("Media edad:", df["edad"].mean())
print("Mediana edad:", df["edad"].median())

print("Media dependencia_percibida:", df["dependencia_percibida"].mean())
print("Mediana dependencia_percibida:", df["dependencia_percibida"].median())

print("Media confia_respuesta_ia:", df["confia_respuesta_ia"].mean())
print("Mediana confia_respuesta_ia:", df["confia_respuesta_ia"].median())


print("\n--- MEDIDAS DE DISPERSION ---")

# Estas métricas nos dicen qué tan dispersos están los datos.
print("Desviacion estandar edad:", df["edad"].std())
print("Desviacion estandar dependencia:", df["dependencia_percibida"].std())

print("Varianza edad:", df["edad"].var())

print("Rango edad:", df["edad"].max() - df["edad"].min())

# El IQR nos da una medida robusta de dispersión.
q1 = df["edad"].quantile(0.25)
q3 = df["edad"].quantile(0.75)
iqr = q3 - q1
print("IQR edad:", iqr)



# DEFINIR VARIABLE OBJETIVO

def definir_nivel_influencia(x):
    if x <= 2:
        return "Baja"
    elif x == 3:
        return "Media"
    else:
        return "Alta"

df["nivel_influencia"] = df["dependencia_percibida"].apply(definir_nivel_influencia)

print("\nDistribución variable objetivo:")
print(df["nivel_influencia"].value_counts())



# DEFINIR X e Y

X = df.drop(columns=["nivel_influencia", "dependencia_percibida"])
Y = df["nivel_influencia"]

print("\nDistribución original:")
print(Y.value_counts())


# BALANCEO UNIFORME (OVERSAMPLING)

from sklearn.utils import resample

# Unimos temporalmente
df_model = X.copy()
df_model["nivel_influencia"] = Y

# Separar por clase
df_baja = df_model[df_model["nivel_influencia"] == "Baja"]
df_alta = df_model[df_model["nivel_influencia"] == "Alta"]
df_media = df_model[df_model["nivel_influencia"] == "Media"]

# Tamaño máximo
max_size = max(len(df_baja), len(df_alta), len(df_media))

# Oversampling con reemplazo
df_baja_up = resample(df_baja, replace=True, n_samples=max_size, random_state=42)
df_alta_up = resample(df_alta, replace=True, n_samples=max_size, random_state=42)
df_media_up = resample(df_media, replace=True, n_samples=max_size, random_state=42)

# Unir dataset balanceado
df_balanced = pd.concat([df_baja_up, df_alta_up, df_media_up])

# Mezclar
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Separar nuevamente
X_balanced = df_balanced.drop(columns=["nivel_influencia"])
Y_balanced = df_balanced["nivel_influencia"]

print("\nDistribución después de balanceo uniforme:")
print(Y_balanced.value_counts())

print("\nDimensiones balanceadas:", X_balanced.shape)


# SPLIT 60 / 20 / 20 (BALANCEADO)

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X_balanced, Y_balanced,
    test_size=0.4,
    random_state=42,
    stratify=Y_balanced
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nTamaño total balanceado:", len(X_balanced))
print("Train (60%):", len(X_train))
print("Test (20%):", len(X_test))
print("Validation (20%):", len(X_val))


# GRÁFICAS DE DISTRIBUCIÓN

conjuntos = {
    "Train": len(X_train),
    "Test": len(X_test),
    "Validation": len(X_val)
}

plt.figure()
plt.bar(conjuntos.keys(), conjuntos.values())
plt.title("Distribución de Conjuntos")
plt.ylabel("Número de muestras")
plt.show()


plt.figure()
sns.countplot(x=y_train)
plt.title("Distribución de clases en Train")
plt.show()

plt.figure()
sns.countplot(x=y_test)
plt.title("Distribución de clases en Test")
plt.show()

plt.figure()
sns.countplot(x=y_val)
plt.title("Distribución de clases en Validation")
plt.show()


# ENTRENAR ÁRBOL DE DECISIÓN

from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

arbol.fit(X_train, y_train)


# EVALUACIÓN DEL MODELO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Predicciones
y_train_pred = arbol.predict(X_train)
y_test_pred = arbol.predict(X_test)
y_val_pred = arbol.predict(X_val)


# MÉTRICAS


print("\n===== MÉTRICAS TRAIN =====")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred, average="weighted"))
print("Recall:", recall_score(y_train, y_train_pred, average="weighted"))
print("F1-score:", f1_score(y_train, y_train_pred, average="weighted"))

print("\n===== MÉTRICAS TEST =====")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_test_pred, average="weighted"))
print("F1-score:", f1_score(y_test, y_test_pred, average="weighted"))

print("\n===== MÉTRICAS VALIDATION =====")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred, average="weighted"))
print("Recall:", recall_score(y_val, y_val_pred, average="weighted"))
print("F1-score:", f1_score(y_val, y_val_pred, average="weighted"))

#MATRIZ DE CONFUSIÓN ARBOL (TRAIN)

cm_train = confusion_matrix(y_train, y_train_pred)

plt.figure()
sns.heatmap(cm_train, annot=True, fmt="d",
            xticklabels=arbol.classes_,
            yticklabels=arbol.classes_)

plt.title("Matriz de Confusión - Árbol (Train)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()


# MATRIZ DE CONFUSIÓN ARBOL (TEST)


cm = confusion_matrix(y_test, y_test_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=arbol.classes_,
            yticklabels=arbol.classes_)

plt.title("Matriz de Confusión - Test")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()


# MATRIZ DE CONFUSIÓN - VALIDATION (ÁRBOL)


cm_val = confusion_matrix(y_val, y_val_pred)

plt.figure()
sns.heatmap(cm_val, annot=True, fmt="d",
            xticklabels=arbol.classes_,
            yticklabels=arbol.classes_)

plt.title("Matriz de Confusión - Árbol (Validation)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()



# RANDOM FOREST (BAGGING)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

rf.fit(X_train, y_train)

y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(X_test)
y_val_rf = rf.predict(X_val)


print("\n===== RANDOM FOREST =====")

print("\nTRAIN")
print("Accuracy:", accuracy_score(y_train, y_train_rf))
print("Precision:", precision_score(y_train, y_train_rf, average="weighted"))
print("Recall:", recall_score(y_train, y_train_rf, average="weighted"))
print("F1-score:", f1_score(y_train, y_train_rf, average="weighted"))

print("\nTEST")
print("Accuracy:", accuracy_score(y_test, y_test_rf))
print("Precision:", precision_score(y_test, y_test_rf, average="weighted"))
print("Recall:", recall_score(y_test, y_test_rf, average="weighted"))
print("F1-score:", f1_score(y_test, y_test_rf, average="weighted"))

print("\nVALIDATION")
print("Accuracy:", accuracy_score(y_val, y_val_rf))
print("Precision:", precision_score(y_val, y_val_rf, average="weighted"))
print("Recall:", recall_score(y_val, y_val_rf, average="weighted"))
print("F1-score:", f1_score(y_val, y_val_rf, average="weighted"))


# MATRIZ DE CONFUSIÓN  RF(TRAIN)


cm_train_rf = confusion_matrix(y_train, y_train_rf)

plt.figure()
sns.heatmap(cm_train_rf, annot=True, fmt="d",
            xticklabels=rf.classes_,
            yticklabels=rf.classes_)

plt.title("Matriz de Confusión - Random Forest (Train)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# MATRIZ DE CONFUSIÓN RF (TEST)
cm_rf = confusion_matrix(y_test, y_test_rf)

plt.figure()
sns.heatmap(cm_rf, annot=True, fmt="d",
            xticklabels=rf.classes_,
            yticklabels=rf.classes_)

plt.title("Matriz de Confusión - Random Forest (Test)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()



# MATRIZ DE CONFUSIÓN  RF (VALIDATION)


cm_val_rf = confusion_matrix(y_val, y_val_rf)

plt.figure()
sns.heatmap(cm_val_rf, annot=True, fmt="d",
            xticklabels=rf.classes_,
            yticklabels=rf.classes_)

plt.title("Matriz de Confusión - Random Forest (Validation)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()


# GRADIENT BOOSTING


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_train_gb = gb.predict(X_train)
y_test_gb = gb.predict(X_test)
y_val_gb = gb.predict(X_val)


print("\n===== GRADIENT BOOSTING =====")

print("\nTRAIN")
print("Accuracy:", accuracy_score(y_train, y_train_gb))
print("Precision:", precision_score(y_train, y_train_gb, average="weighted"))
print("Recall:", recall_score(y_train, y_train_gb, average="weighted"))
print("F1-score:", f1_score(y_train, y_train_gb, average="weighted"))

print("\nTEST")
print("Accuracy:", accuracy_score(y_test, y_test_gb))
print("Precision:", precision_score(y_test, y_test_gb, average="weighted"))
print("Recall:", recall_score(y_test, y_test_gb, average="weighted"))
print("F1-score:", f1_score(y_test, y_test_gb, average="weighted"))

print("\nVALIDATION")
print("Accuracy:", accuracy_score(y_val, y_val_gb))
print("Precision:", precision_score(y_val, y_val_gb, average="weighted"))
print("Recall:", recall_score(y_val, y_val_gb, average="weighted"))
print("F1-score:", f1_score(y_val, y_val_gb, average="weighted"))

# MATRIZ DE CONFUSIÓN GB (Train)


cm_train_gb = confusion_matrix(y_train, y_train_gb)

plt.figure()
sns.heatmap(cm_train_gb, annot=True, fmt="d",
            xticklabels=gb.classes_,
            yticklabels=gb.classes_)

plt.title("Matriz de Confusión - Gradient Boosting (Train)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Matriz de confusión GB (Test)
cm_gb = confusion_matrix(y_test, y_test_gb)

plt.figure()
sns.heatmap(cm_gb, annot=True, fmt="d",
            xticklabels=gb.classes_,
            yticklabels=gb.classes_)

plt.title("Matriz de Confusión - Gradient Boosting (Test)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()


# MATRIZ DE CONFUSIÓN GB ( VALIDATION)

cm_val_gb = confusion_matrix(y_val, y_val_gb)

plt.figure()
sns.heatmap(cm_val_gb, annot=True, fmt="d",
            xticklabels=gb.classes_,
            yticklabels=gb.classes_)

plt.title("Matriz de Confusión - Gradient Boosting (Validation)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()



# COMPARACIÓN FINAL POR ACCURACY

metricas_finales = {
    "Decision Tree": accuracy_score(y_val, y_val_pred),
    "Random Forest": accuracy_score(y_val, y_val_rf),
    "Gradient Boosting": accuracy_score(y_val, y_val_gb)
}

df_metricas = pd.DataFrame.from_dict(
    metricas_finales,
    orient="index",
    columns=["Accuracy Validation"]
)

print("\n===== COMPARACIÓN FINAL (Accuracy en Validation) =====")
print(df_metricas)

mejor_modelo = df_metricas["Accuracy Validation"].idxmax()
mejor_score = df_metricas["Accuracy Validation"].max()

print("\n===================================")
print("El mejor modelo fue:", mejor_modelo)
print("Con Accuracy en Validation de:", round(mejor_score, 4))
print("===================================")



# VISUALIZAR ÁRBOL DE DECISIÓN

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))

plot_tree(
    arbol,
    feature_names=X_train.columns,
    class_names=arbol.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Árbol de Decisión - Estructura Completa")
plt.show()


# IMPORTANCIA DE VARIABLES

importancias = pd.Series(
    arbol.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\n===== IMPORTANCIA DE VARIABLES =====")
print(importancias)

plt.figure(figsize=(10,6))
importancias.plot(kind="bar")
plt.title("Importancia de Variables - Árbol de Decisión")
plt.ylabel("Importancia")
plt.xlabel("Variables")
plt.xticks(rotation=45)
plt.show()