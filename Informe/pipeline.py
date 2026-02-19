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


# GRÁFICOS

# Boxplot
# Nos ayuda a visualizar la distribución y detectar posibles valores atípicos.
plt.figure()
sns.boxplot(x=df["edad"])
plt.title("Boxplot Edad")
plt.show()

plt.figure()
sns.boxplot(x=df["dependencia_percibida"])
plt.title("Boxplot Dependencia Percibida")
plt.show()


# Histogramas
# Permiten ver cómo se distribuyen las respuestas en cada variable.
variables_numericas = [
    "edad",
    "delega_razonamiento",
    "confia_respuesta_ia",
    "dependencia_percibida",
    "frecuencia_uso_ia",
    "verifica_respuestas"
]

for var in variables_numericas:
    plt.figure()
    plt.hist(df[var], bins=5)
    plt.title(f"Distribucion de {var}")
    plt.xlabel(var)
    plt.ylabel("Frecuencia")
    plt.show()


# Matriz de correlación
# Nos permite identificar qué variables se relacionan entre sí.
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlacion")
plt.show()


# Gráfico de dispersión
# Aquí observamos directamente la relación entre confianza en la IA y dependencia.
plt.figure()
plt.scatter(df["confia_respuesta_ia"], df["dependencia_percibida"])
plt.xlabel("Confianza en IA")
plt.ylabel("Dependencia Percibida")
plt.title("Confianza vs Dependencia")
plt.show()


# REDUCCIÓN DE DIMENSIONALIDAD - UMAP
# UMAP nos permite proyectar datos complejos en solo dos dimensiones para poder visualizarlos de forma más intuitiva.

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

embedding = umap_model.fit_transform(df_scaled)

df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
df_umap["dependencia_percibida"] = df["dependencia_percibida"].values


# Visualización UMAP
# Observamos posibles patrones o agrupaciones.
plt.figure(figsize=(10,7))

scatter = plt.scatter(
    df_umap["UMAP1"],
    df_umap["UMAP2"],
    c=df_umap["dependencia_percibida"],
    cmap="viridis",
    alpha=0.8
)

plt.colorbar(scatter, label="Dependencia Percibida")
plt.title("Proyección UMAP - Patrones de Uso de IA")
plt.xlabel("UMAP Dimensión 1")
plt.ylabel("UMAP Dimensión 2")
plt.grid(alpha=0.3)
plt.show()


# REDUCCIÓN DE DIMENSIONALIDAD - PCA
# PCA resume la información en nuevas variables (componentes principales) que explican la mayor parte de la variabilidad del dataset.

pca = PCA(n_components=2, random_state=42)

pca_components = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
df_pca["dependencia_percibida"] = df["dependencia_percibida"].values


# Visualización PCA
# Aquí buscamos patrones lineales en la distribución de los datos.
plt.figure(figsize=(10,7))

scatter = plt.scatter(
    df_pca["PC1"],
    df_pca["PC2"],
    c=df_pca["dependencia_percibida"],
    cmap="plasma",
    alpha=0.8
)

plt.colorbar(scatter, label="Dependencia Percibida")
plt.title("Proyección PCA - Patrones de Uso de IA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(alpha=0.3)
plt.show()


# REDUCCIÓN DE DIMENSIONALIDAD - t-SNE
# t-SNE es especialmente útil para detectar agrupaciones locales que no necesariamente siguen relaciones lineales.

tsne = TSNE(
    n_components=2,
    perplexity=8,
    learning_rate=100,
    max_iter=1000,
    random_state=42
)

tsne_embedding = tsne.fit_transform(df_scaled)

df_tsne = pd.DataFrame(tsne_embedding, columns=["TSNE1", "TSNE2"])
df_tsne["dependencia_percibida"] = df["dependencia_percibida"].values


# Visualización t-SNE
# Este gráfico puede revelar clusters o patrones escondidos.
plt.figure(figsize=(10,7))

scatter = plt.scatter(
    df_tsne["TSNE1"],
    df_tsne["TSNE2"],
    c=df_tsne["dependencia_percibida"],
    cmap="coolwarm",
    alpha=0.8
)

plt.colorbar(scatter, label="Dependencia Percibida")
plt.title("Proyección t-SNE - Patrones de Uso de IA")
plt.xlabel("t-SNE Dimensión 1")
plt.ylabel("t-SNE Dimensión 2")
plt.grid(alpha=0.3)
plt.show()
