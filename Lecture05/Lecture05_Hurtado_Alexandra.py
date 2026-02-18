import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import reciprocal, randint

# Fijar semilla para reproducibilidad
random_state = 42
np.random.seed(random_state)

# Cambiar estilo de gráficos
plt.rc('font', family='serif', size=12)
sns.set(style="whitegrid")

# 1. Cargar el dataset del Titanic desde URL pública
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
print("Dataset cargado. Primeras filas:")
print(df.head())

# 2. Exploración gráfica y limpieza
# Exploración: Info general
print("\nInfo del dataset:")
print(df.info())
print("\nDescripción estadística:")
print(df.describe())

# Gráfico: Distribución de variables clave
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Distribución de Edad')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['Fare'], kde=True)
plt.title('Distribución de Tarifa (Fare)')
plt.show()

# Gráfico: Correlaciones
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlaciones')
plt.show()

# Limpieza: Manejar nulos y columnas irrelevantes
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  # Irrelevantes o muchos nulos
df['Age'] = df['Age'].fillna(df['Age'].median())  # Rellenar Age con mediana
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Rellenar Embarked con moda
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Rellenar Fare (por si acaso)

# Codificar categóricas: Sex y Embarked
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print("\nDataset limpio:")
print(df.head())

# Features comunes para ambos modelos (excluyendo target)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_male', 'Embarked_Q', 'Embarked_S']

# --- Regresión Lineal: Predecir 'Fare' (variable continua) ---
print("\n--- Regresión Lineal: Prediciendo 'Fare' ---")

# Definir X e y para lineal
X_lin = df[features]
y_lin = df['Fare']

# Dividir en train/test
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=random_state)

# Gráfico: Train vs Test (scatter de Age vs Fare, coloreado)
plt.figure(figsize=(10, 6))
plt.scatter(X_train_lin['Age'], y_train_lin, color='blue', label='Train')
plt.scatter(X_test_lin['Age'], y_test_lin, color='red', label='Test')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Train vs Test: Age vs Fare')
plt.legend()
plt.show()

# Definir pipelines para Ridge y Lasso
# Incluir PolynomialFeatures para capturar no linealidades, como en ejemplos
pipe_ridge = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

pipe_lasso = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(max_iter=10000))
])

# Distribuciones de parámetros
param_dist = {
    'poly__degree': randint(1, 4),  # Grados polinómicos: 1-3
    'regressor__alpha': reciprocal(0.001, 1000)  # Alpha para regularización
}

# Búsqueda aleatoria con CV para Ridge
ridge_search = RandomizedSearchCV(pipe_ridge, param_distributions=param_dist, n_iter=50, cv=5,
                                  scoring='neg_mean_absolute_error', random_state=random_state, n_jobs=-1)
ridge_search.fit(X_train_lin, y_train_lin)

# Búsqueda para Lasso
lasso_search = RandomizedSearchCV(pipe_lasso, param_distributions=param_dist, n_iter=50, cv=5,
                                  scoring='neg_mean_absolute_error', random_state=random_state, n_jobs=-1)
lasso_search.fit(X_train_lin, y_train_lin)

# Mejores parámetros
print("\nMejores parámetros Ridge:", ridge_search.best_params_)
print("Mejores parámetros Lasso:", lasso_search.best_params_)

# Predicciones
y_pred_ridge = ridge_search.predict(X_test_lin)
y_pred_lasso = lasso_search.predict(X_test_lin)

# Métricas: R² y MAE
print("\nRidge - R²:", r2_score(y_test_lin, y_pred_ridge))
print("Ridge - MAE:", mean_absolute_error(y_test_lin, y_pred_ridge))
print("Lasso - R²:", r2_score(y_test_lin, y_pred_lasso))
print("Lasso - MAE:", mean_absolute_error(y_test_lin, y_pred_lasso))

# Gráfico: Modelo predicho (real vs pred para test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_lin, y_pred_ridge, color='green', label='Ridge')
plt.scatter(y_test_lin, y_pred_lasso, color='purple', label='Lasso')
plt.plot([y_test_lin.min(), y_test_lin.max()], [y_test_lin.min(), y_test_lin.max()], 'k--', lw=2)
plt.xlabel('Fare Real')
plt.ylabel('Fare Predicho')
plt.title('Real vs Predicho: Ridge y Lasso')
plt.legend()
plt.show()

# --- Regresión Logística: Predecir 'Survived' (binario) ---
print("\n--- Regresión Logística: Prediciendo 'Survived' ---")

# Definir X e y para logística (drop NaN en Survived si hay)
df_log = df.dropna(subset=['Survived'])
X_log = df_log[features]
y_log = df_log['Survived']

# Dividir en train/test
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=random_state)

# Pipeline para Logística (con L2 por defecto, como en ejemplo)
pipe_log = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000))
])

# Distribuciones de parámetros (C es 1/alpha para regularización)
param_dist_log = {
    'poly__degree': randint(1, 4),
    'classifier__C': reciprocal(0.001, 1000)
}

# Búsqueda aleatoria con CV
log_search = RandomizedSearchCV(pipe_log, param_distributions=param_dist_log, n_iter=50, cv=5,
                                scoring='f1', random_state=random_state, n_jobs=-1)
log_search.fit(X_train_log, y_train_log)

# Mejores parámetros
print("\nMejores parámetros Logística:", log_search.best_params_)

# Predicciones
y_pred_log = log_search.predict(X_test_log)

# Métricas: Accuracy y F1
print("\nAccuracy:", accuracy_score(y_test_log, y_pred_log))
print("F1-Score:", f1_score(y_test_log, y_pred_log))

# Gráfico: Modelo predicho (scatter de Age vs Probabilidad de Supervivencia, coloreado por real)
probs = log_search.predict_proba(X_test_log)[:, 1]  # Probabilidades de clase 1
plt.figure(figsize=(10, 6))
plt.scatter(X_test_log['Age'], probs, c=y_test_log, cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Probabilidad Predicha de Supervivencia')
plt.title('Probabilidades Predichas vs Age (Coloreado por Supervivencia Real)')
plt.colorbar(label='Survived Real')
plt.show()

# Gráfico: Matriz de Confusión
cm = confusion_matrix(y_test_log, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión')
plt.show()

print("Mejor modelo Ridge:", ridge_search.best_estimator_)
print("Mejor modelo Logístico:", log_search.best_estimator_)
