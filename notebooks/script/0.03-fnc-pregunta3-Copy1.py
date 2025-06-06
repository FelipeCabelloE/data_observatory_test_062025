#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Análisis de Precios de Vivienda en Chile con Machine Learning
# 
# ## Justificación y Objetivos
# Tenemos un dataset con 40 columnas que tienen información relacionada a 31652 viviendas en Chile.
# 
# Me gustaría realizar un modelo que prediga el hacinamiento de un hogar, dada su situación particular. Esto principalmente porque es un análisis directo, con muchas implicancias asociadas a la desigualdad nacional y porque personalmente quiero entender mejor las relaciones que pueden exisitir.
# 
# El hacinamiento es un indicador directo de la calidad de vida y la vulnerabilidad de un hogar.
# 
# 
# ## Definición del problema
# 
# Predecir si un hogar se encuentra en situación de hacinamiento.
# 
# El hacinamiento es un indicador clave de vulnerabilidad y calidad de vida. Un modelo que lo prediga bien podría ser muy útil para políticas públicas.
# 
# ### Definición de la Variable Objetivo:
# 
# Crearemos una nueva variable binaria hacinamiento_flag. Vamos a usar un umbral de más de 2 personas por dormitorio(Que dependiendo de la situación puede ser bastante).
# 
# ## Modelo 
# Para hacer este análisis decidí usar el modelo CATBOOST. He escuchado comentarios muy positiovs y ya llevo bastante tiempo queriendo usarlo. Por supuesto también es un modelo con una buena reputación para trabajr con información tabular y categorica. Lo único que me lamento es no tener un dataset mas grande.
# 
# 
# ## hipotesis
# Hipótesis 1 (Socioeconómica): El nivel educativo del jefe de hogar (jefe_hogar_educacion) y la tenencia de la vivienda (tenencia_vivienda - ej. 'Ocupación irregular') serán predictores importantes. Se espera que niveles educativos más bajos y tenencia irregular se asocien con una mayor probabilidad de hacinamiento.
# 
# Hipótesis 2 (Geográfica): La comuna será un factor determinante, reflejando que el hacinamiento se concentra en ciertas áreas geográficas con mercados de vivienda más caros o con mayor vulnerabilidad.
# 
# Hipótesis 3 (Demográfica): Características demográficas como el porc_menores_18 (porcentaje de menores) estarán positivamente correlacionadas con el hacinamiento.

# ## EDA

# 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

from data_observatory_test_062025.load_data_viviendas import load_interim_data
from data_observatory_test_062025.generic_eda_report import generic_report

# 2. Configurar estilo de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 3. Cargar el dataset limpio de la Pregunta 2
try:
    df_vivienda = load_interim_data()
    print("Dataset limpio cargado exitosamente.")
except FileNotFoundError:
    print("Error: Ejecute primero la Pregunta 2 para generar 'datos_viviendas_censo_limpio.csv'")
    exit()


# 

# Lo primero es dejar como referencia al tope del archivo una descripción general del dataset

# Me llama la atención esa variable "Unnamed:0", parece que se coló un índice extra jaja

generic_report(df_vivienda)


df_cleaned = df_vivienda.drop(columns=[
    'Unnamed: 0',
    'id_vivienda',
    'indice_masculinidad', # Colineal con genero
    'ingreso_mensual_hogar',  # Ya no es nuestro target
    'ingreso_per_capita'      # Fuga de datos del problema anterior
])

# --- 1. Creación de la Variable Objetivo: 'hacinamiento_flag' ---
# Definimos hacinamiento como > 2 personas por dormitorio.
df_cleaned['hacinamiento_flag'] = (df_cleaned['personas_por_dormitorio'] > 2).astype(int)

# --- 2. Análisis Exploratorio de la Nueva Variable Objetivo ---
print("Distribución de la variable 'hacinamiento_flag':")
print(df_cleaned['hacinamiento_flag'].value_counts(normalize=True))



plt.figure(figsize=(6, 4))
sns.countplot(x='hacinamiento_flag', data=df_cleaned)
plt.title('Distribución de Hogares con y sin Hacinamiento')
plt.xticks([0, 1], ['Sin Hacinamiento', 'Con Hacinamiento'])
plt.ylabel('Cantidad de Hogares')
plt.show()


# Tenemos clases bastante balanceadas
# 

# Eliminar las variables que componen directamente la variable objetivo para evitar fuga de datos.
# El modelo debe predecir el hacinamiento a partir de características socioeconómicas,
# no a partir de un cálculo directo.
X = df_cleaned.drop(columns=[
    'hacinamiento_flag',
    'personas_por_dormitorio', # Fuga de datos directa
    'num_personas_hogar',      # Fuga de datos fuerte
    'num_dormitorios'          # Fuga de datos fuerte
])
y = df_cleaned['hacinamiento_flag']

# Identificar las variables categóricas para CatBoost
categorical_features_indices = [X.columns.get_loc(c) for c in X.select_dtypes(include=['object', 'bool']).columns]

print(f"Features utilizados para la predicción: {X.columns.tolist()}")


# División estratificada para mantener la proporción de clases en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model_clf = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=categorical_features_indices,
    verbose=0,
    random_seed=42,
    eval_metric='Accuracy' # un clasico
)

# Usamos StratifiedKFold para la validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluamos usando tanto Accuracy como AUC-ROC
scoring_metrics = ['accuracy', 'roc_auc']
for metric in scoring_metrics:
    cv_scores = cross_val_score(model_clf, X_train, y_train, cv=cv, scoring=metric)
    print(f"{metric.upper()} promedio en validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# Entrenar el modelo final y evaluar en el conjunto de prueba
model_clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Predecir clases
y_pred_class = model_clf.predict(X_test)

# Calcular métricas finales
accuracy_final = accuracy_score(y_test, y_pred_class)
roc_auc_final = roc_auc_score(y_test, model_clf.predict_proba(X_test)[:, 1]) # AUC necesita probabilidades

print(f"\n--- Resultados en el Conjunto de Prueba ---")
print(f"Accuracy Final: {accuracy_final:.4f}")
print(f"AUC-ROC Final: {roc_auc_final:.4f}")

# La matriz de confusión y el reporte de clasificación siguen siendo muy útiles
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_class))

print("\nReporte de Clasificación:")
# En un caso balanceado, el F1-score, Precision y Recall para ambas clases serán similares
# si el modelo es bueno.
print(classification_report(y_test, y_pred_class, target_names=['Sin Hacinamiento', 'Con Hacinamiento']))




