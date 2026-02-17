#IMPORTACIÓN DE LIBRERIAS 
# Librerías principales para análisis de datos y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # Para estandarización de variables


#CARGA DEL DATA SET
#Cargar el archivo de Excel
df = pd.read_excel("DatosEncuestaInforme1.xlsx")

#Limpieza de nombres en las columna spara evitar errores # Limpiar nombres de columnas
# Se normalizan los nombres para evitar problemas al acceder a las variables
df.columns = df.columns.str.strip()        # elimina espacios al inicio y final
df.columns = df.columns.str.replace("\n", "")  # elimina saltos de linea
df.columns = df.columns.str.lower()        # convierte todo a minuscula para uniformidad

#Eliminar columnas innecesarias
# Se eliminan variables que no aportan valor al análisis
cols_eliminar = [
    'hora de finalización',
    'correo electrónico',
    'hora de la última modificación',
    'nombre',
    'id',
    'hora de inicio'
]

df = df.drop(columns=cols_eliminar, errors='ignore')  # errors='ignore' evita errores si no existen

print("\nColumnas eliminadas correctamente:")
print(cols_eliminar)

# RENOMBRAR COLUMNAS LARGAS A VARIABLES CORTAS
# Se simplifican nombres para facilitar manipulación y lectura del código
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



#INSPECCIÓN INICIAL DEL DATA SET
# Se realiza una revisión general antes de cualquier transformación

print("Primeras filas del dataset:")
print(df.head())  # Vista preliminar de los datos

print("\nDimensiones del dataset:")
print(df.shape)   # Número de filas y columnas

print("\nTipos de datos:")
print(df.info())  # Verificar tipos de variables

print("\nEstadisticas descriptivas:")
print(df.describe())  # Resumen estadístico de variables numéricas


#LIMPIEZA DE DATOS 

#Verificar valores nulos
# Identificar posibles datos faltantes
print("\nValores nulos por columna:")
print(df.isnull().sum())


# Asegurar que variables numéricas sean numericas
# Si existen valores no convertibles, se transforman en NaN
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["delega_razonamiento"] = pd.to_numeric(df["delega_razonamiento"], errors="coerce")
df["confia_respuesta_ia"] = pd.to_numeric(df["confia_respuesta_ia"], errors="coerce")
df["dependencia_percibida"] = pd.to_numeric(df["dependencia_percibida"], errors="coerce")


#TRANSFORMACIÓN DE VARIABLES 
#Codificacion ordinal

# frecuencia_uso_ia tiene orden logico
# Se asignan valores numéricos respetando el orden natural de frecuencia
df["frecuencia_uso_ia"] = df["frecuencia_uso_ia"].map({
    "0-1": 1,
    "1-3": 2,
    "3-5": 3,
    "5 o más": 4
})

# verifica_respuestas tiene orden logico
# Se codifica según nivel creciente de verificación
df["verifica_respuestas"] = df["verifica_respuestas"].map({
    "No": 0,
    "A veces": 1,
    "Sí": 2
})

print("Nulos después del mapeo:")
print(df.isnull().sum())

# Eliminar filas con cualquier valor nulo
# Se eliminan registros incompletos para evitar problemas en el modelado
df = df.dropna()

print("Dimensiones después de eliminar nulos:", df.shape)

#One Hot Encoding para variables nominales
# Se convierten variables categóricas sin orden en variables dummy
# drop_first=True evita duplicar información en las variables

df = pd.get_dummies(
    df,
    columns=["tipo_decisiones", "nivel_experiencia"],
    drop_first=True
)

print("\nDataset despues de transformaciones:")
print(df.head())



#ESCALAMIENTO DE VARIABLES 

#Identificar columnas numéricas continuas (las que sí deben escalarse)
# Se excluyen variables dummy para no alterar su interpretación
columnas_escalar = [
    "edad",
    "delega_razonamiento",
    "confia_respuesta_ia",
    "dependencia_percibida",
    "frecuencia_uso_ia",
    "verifica_respuestas"
]

#Crear copia del dataset
# Se mantiene una versión original sin escalar
df_scaled = df.copy()

# Aplicar StandardScaler SOLO a esas columnas
# Estandarización: media=0, desviación estándar=1
scaler = StandardScaler()
df_scaled[columnas_escalar] = scaler.fit_transform(df[columnas_escalar])

print("\nDataset escalado correctamente (sin afectar variables dummy):")
print(df_scaled.head())

#Guardar versión final lista para modelado
df_scaled.to_csv("dataset_model_ready.csv", index=False)


#ANALISIS EXPLORATORIO 

#Medidas de tendencia central
# Permiten identificar el valor típico de las variables
print("\n--- MEDIDAS DE TENDENCIA CENTRAL ---")

print("Media edad:", df["edad"].mean())
print("Mediana edad:", df["edad"].median())

print("Media dependencia_percibida:", df["dependencia_percibida"].mean())
print("Mediana dependencia_percibida:", df["dependencia_percibida"].median())

print("Media confia_respuesta_ia:", df["confia_respuesta_ia"].mean())
print("Mediana confia_respuesta_ia:", df["confia_respuesta_ia"].median())


#Medidas de Dispersión 
# Permiten analizar la variabilidad de los datos
print("\n--- MEDIDAS DE DISPERSION ---")

# Desviacion estandar
print("Desviacion estandar edad:", df["edad"].std())
print("Desviacion estandar dependencia:", df["dependencia_percibida"].std())

# Varianza
print("Varianza edad:", df["edad"].var())

# Rango
print("Rango edad:", df["edad"].max() - df["edad"].min())

# IQR (Rango intercuartilico)
# Mide dispersión sin verse afectado por valores extremos
q1 = df["edad"].quantile(0.25)
q3 = df["edad"].quantile(0.75)
iqr = q3 - q1
print("IQR edad:", iqr)


#GRAFICOS 
#Bloxplot
# Permite detectar valores atípicos visualmente
plt.figure()
sns.boxplot(x=df["edad"])
plt.title("Boxplot Edad")
plt.show()

plt.figure()
sns.boxplot(x=df["dependencia_percibida"])
plt.title("Boxplot Dependencia Percibida")
plt.show()


#Histogramas 
# Permiten observar la distribución de frecuencia de cada variable
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


#Matriz de correlación 
# Permite identificar relaciones lineales entre variables
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlacion")
plt.show()


#Grafico de dispersión 
# Analiza relación directa entre confianza y dependencia
plt.figure()
plt.scatter(df["confia_respuesta_ia"], df["dependencia_percibida"])
plt.xlabel("Confianza en IA")
plt.ylabel("Dependencia Percibida")
plt.title("Confianza vs Dependencia")
plt.show()
