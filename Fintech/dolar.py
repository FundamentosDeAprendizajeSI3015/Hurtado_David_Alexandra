
# Importar librerias necesarias

import pandas as pd
import plotly.graph_objects as go


# Cargar el archivo CSV con datos históricos del dólar USD/COP

CSV_PATH = "usd_cop_hist.csv"

# Leer el archivo CSV
df = pd.read_csv(CSV_PATH)

# Analisis exploratorio: Revisamos columnas y primeras filas para entender los datos
print("Columnas del CSV:", df.columns.tolist())
print(df.head())


#Limpiar y preparar los datos

#  Convertir la fecha a formato datetime
df["Date"] = pd.to_datetime(df["VIGENCIADESDE"], dayfirst=True, errors="coerce")

#  Limpiar la columna VALOR
# Quitar el símbolo $, quitar puntos y cambiar coma por punto
df["Close"] = (
    df["VALOR"]
    .str.replace("$", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

#  Ordenar los datos por fecha (importante para gráficas)
df = df.sort_values("Date")

#  Quedarnos con los últimos 30 registros (aprox último mes)
df_last = df.tail(30).copy()

print("Shape datos completos:", df.shape)
print("Shape últimos 30 días:", df_last.shape)
print(df_last[["Date", "Close"]].head())



# Crear la gráfica interactiva con Plotly


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_last["Date"],          # Eje X: fechas
    y=df_last["Close"],         # Eje Y: TRM USD/COP
    mode="lines+markers",       # Línea con puntos
    name="USD/COP TRM",
    line=dict(
        width=3
    ),
    marker=dict(
        size=8
    )
))



# Configurar el diseño de la gráfica


fig.update_layout(
    title=dict(
        text="Dólar USD/COP - Último Mes",
        font=dict(size=20)
    ),
    xaxis_title="Fecha",
    yaxis_title="Pesos Colombianos por 1 USD",
    hovermode="x unified",
    template="plotly_white",
    height=500
)



# Mostrar la gráfica interactiva

fig.show()
