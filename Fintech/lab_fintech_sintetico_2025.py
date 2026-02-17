# =============================================================
# LAB FINTECH (SINTÉTICO 2025) — PREPROCESAMIENTO Y EDA
# Este script realiza:
# - Carga de datos sintéticos (CSV + diccionario JSON)
# - Exploración básica (EDA)
# - Limpieza e imputación
# - Ingeniería simple de variables
# - Preparación de datos para modelado (one-hot + escalado)
# - Exportación de datasets procesados
# =============================================================


# Importar de librerías Basicas

import json                     # Lectura y escritura de archivos JSON
from pathlib import Path        # Manejo rutas de archivos sin depender de S.O
import warnings                 # Ocultar advertencias sin valor

# Se desactivan advertencias para mantener la salida limpia
warnings.filterwarnings("ignore")

# Librerias Principales 
import numpy as np              # Operaciones numéricas y matemáticas
import pandas as pd             # Manipulación y análisis de datos
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler       # Preparación paea ML

from pathlib import Path
import plotly.express as px   #Graficar 


# Nombre del archivo CSV con los datos sintéticos
DATA_CSV = 'fintech_top_sintetico_2025.csv'

# Archivo JSON que describe el dataset
DATA_DICT = 'fintech_top_sintetico_dictionary.json'

# Carpeta donde se guardan los resultados procesados
OUTDIR = Path('./data_output_finanzas_sintetico')

# Fecha de corte para separación temporal del train y test
SPLIT_DATE = '2025-09-01'


# Definición de columnas esperadas para el modelo( Definir directamnete para evitar errores)
# Columna  principal
DATE_COL = 'Month'

# Columnas identificadoras
ID_COLS = ['Company']

# Columnas categóricas
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']

# Columnas numéricas principales del negocio
NUM_COLS = [
    'Users_M', 'NewUsers_K', 'TPV_USD_B', 'TakeRate_pct', 'Revenue_USD_M',
    'ARPU_USD', 'Churn_pct', 'Marketing_Spend_USD_M', 'CAC_USD',
    'CAC_Total_USD_M', 'Close_USD', 'Private_Valuation_USD_B'
]

# Columnas de precio usadas para calcular retornos
PRICE_COLS = ['Close_USD']


# Cargar el diccionario de datos

print("\n=== 0) Cargando diccionario de datos ===")

# Construir la ruta al archivo JSON
dict_path = Path(DATA_DICT)

# Verificación de existencia del archivo para prevenir errores
if not dict_path.exists():
    raise FileNotFoundError(
        f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta."
    )

# Apertura y lectura del JSON
with open(dict_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

# Se imprime información descriptiva del dataset
print("Descripción:", data_dict.get('description', '(sin descripción)'))
print("Periodo:", data_dict.get('period', '(desconocido)'))


# Cargar el CSV

print("\n=== 1) Cargando CSV sintético ===")

# Ruta al archivo CSV
csv_path = Path(DATA_CSV)

# Verificar la existencia del archivo para prevenir errores
if not csv_path.exists():
    raise FileNotFoundError(
        f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta."
    )

# Lectura del CSV en un DataFrame
df = pd.read_csv(csv_path)

# Se imprime la forma del dataset (filas, columnas)
print("Shape:", df.shape)


# Parseo de fechas y orden temporal

# Se verifica que la columna de fecha exista
if DATE_COL not in df.columns:
    raise KeyError(
        f"La columna de fecha '{DATE_COL}' no existe en el CSV."
    )

# Convertir la columna de fecha al tipo apropiado
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')

# Ordenar la infromación por fecha y empresa 
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

# Visualización inicial
print("Primeras filas:")
print(df.head(3))


# 2) Análisis exploratorio(Importante para entender el dataset)
print("\n=== 2) EDA rápido ===")

# Información general del DataFrame
print("Info:")
print(df.info())

# Conteo de valores nulos por columna
print("\nNulos por columna (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))

#(Importante para verificar tipos de datos, nulos y estructura general)


# 3) Limpieza básica de datos

print("\n=== 3) Limpieza ===")

# Imputación de variables numéricas:
# Se realiza una conversión a númerico por si hay strings inesperados
# y se reemplazan los valores faltantes con la mediana
for c in NUM_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

# Imputación de variables categóricas:
# Reemplzar valores faltantes con el string '__MISSING__'
for c in CAT_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = df[c].fillna('__MISSING__')


# Ingeniería de variables

print("\n=== 4) Ingeniería de rasgos (retornos) ===")

# Verificación de que las columnas de precio existan
if all([pc in df.columns for pc in PRICE_COLS]):
    for pc in PRICE_COLS:

        # Retorno porcentual entre periodos consecutivos por empresa
        df[pc + '_ret'] = (
            df.sort_values([ID_COLS[0], DATE_COL])
              .groupby(ID_COLS)[pc]
              .pct_change()
        )

        # Log-retorno para estabilizar varianza
        df[pc + '_logret'] = np.log1p(df[pc + '_ret'])

        # Imputación de valores iniciales (primer periodo)
        df[pc + '_ret'] = df[pc + '_ret'].fillna(0.0)
        df[pc + '_logret'] = df[pc + '_logret'].fillna(0.0)

else:
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Se actualiza la lista de columnas numéricas usadas
extra_num = [
    c for c in
    [pc + '_ret' for pc in PRICE_COLS] + [pc + '_logret' for pc in PRICE_COLS]
    if c in df.columns
]

NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num


# Preparación de variables X (Lo que va a ver el modelo)

print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")

# Se eliminan fecha e identificadores
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy()

# Identificación de columnas categóricas presentes
cat_in_X = [c for c in CAT_COLS if c in X.columns]

# Codificación one-hot (evitando multicolinealidad)
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)


# Separación para train y test (Inidspensable)

# Conversión de la fecha de corte
cutoff = pd.to_datetime(SPLIT_DATE)

# Índices lógicos de entrenamiento y prueba
idx_train = df[DATE_COL] < cutoff
idx_test = df[DATE_COL] >= cutoff

# Subconjuntos finales
X_train = X.loc[idx_train].copy()
X_test = X.loc[idx_test].copy()


# Escalado de variables numéricas (Poner todaslas variables en la misma escala)

num_in_X = [c for c in NUM_USED if c in X_train.columns]

scaler = StandardScaler()

if num_in_X:
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])
else:
    print("[INFO] No se encontraron columnas numéricas para escalar.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)


# Exportación de resultados

print("\n=== 6) Exportación ===")

# Creación del directorio de salida si no existe
OUTDIR.mkdir(parents=True, exist_ok=True)

# Rutas de archivos
train_path = OUTDIR / 'fintech_train.parquet'
test_path = OUTDIR / 'fintech_test.parquet'

# Guardado de datasets procesados
X_train.to_parquet(train_path, index=False)
X_test.to_parquet(test_path, index=False)

# Guardar el esquema procesado


processed_schema = {
    'source_csv': str(csv_path.resolve()),
    'source_dict': str(dict_path.resolve()),
    'date_col': DATE_COL,
    'id_cols': ID_COLS,
    'categorical_cols_used': cat_in_X,
    'numeric_cols_used': num_in_X,
    'engineered_cols': extra_num,
    'split': {
        'type': 'time_split',
        'cutoff': SPLIT_DATE,
        'train_rows': int(idx_train.sum()),
        'test_rows': int(idx_test.sum()),
    },
    'X_train_shape': list(X_train.shape),
    'X_test_shape': list(X_test.shape),
    'notes': [
        'Dataset 100% SINTÉTICO con fines académicos; no refleja métricas reales.',
        'Evitar fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST.'
    ]
}

# Escritura del esquema en JSON
with open(OUTDIR / 'processed_schema.json', 'w', encoding='utf-8') as f:
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# Guardado de las columnas finales
with open(OUTDIR / 'features_columns.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train.columns))

# Confirmación final
print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / 'processed_schema.json')
print(" -", OUTDIR / 'features_columns.txt')

print("\n✔ Listo. Recuerda: este dataset es sintético para práctica académica.")

#Graficar Revenue en el tiempo por Segment usando Plotly
# Usamos el mismo df original ya cargado y limpiado arriba
print("\n=== Visualización Plotly: Revenue en el tiempo por Segment ===")

# 1) Agregar por fecha y segmento
df_plot = (
    df.groupby([DATE_COL, 'Segment'], as_index=False)
      .agg({'Revenue_USD_M': 'mean'})
)

# 2) Gráfico de líneas interactivo
fig = px.line(
    df_plot,
    x=DATE_COL,
    y='Revenue_USD_M',
    color='Segment',
    markers=True,
    title='Revenue mensual por Segmento',
    labels={
        DATE_COL: 'Mes',
        'Revenue_USD_M': 'Revenue (USD Millones)',
        'Segment': 'Segmento'
    }
)

fig.update_layout(
    template='plotly_white',
    hovermode='x unified'
)

# 3) Exportar a HTML
out_html = Path("revenue_segment_plotly.html")
fig.write_html(out_html.as_posix(), include_plotlyjs='cdn')

print(f"Gráfico Plotly exportado a: {out_html.resolve()}")
print("Abre ese archivo HTML en tu navegador para ver la visualización.")



print("Filas df_plot:", df_plot.shape)
print(df_plot.head())
