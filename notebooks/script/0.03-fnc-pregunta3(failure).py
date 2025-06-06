#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Análisis de Precios de Vivienda en Chile con Machine Learning
# 
# ## Justificación y Objetivos
# Tenemos un dataset con 40 columnas que tienen información relacionada a 31652 viviendas en Chile.
# 
# Me gustaría realizar un modelo que prediga el ingreso mensual de un hogar, dada su situación particular. Esto principalmente porque es un análisis directo, con muchas implicancias asociadas a la desigualdad nacional y porque personalmente quiero entender mejor las relaciones que pueden exisitir.
# 
# Para hacer este análisis decidí usar el modelo CATBOOST. He escuchado comentarios muy positiovs y ya llevo bastante tiempo queriendo usarlo. Por supuesto también es un modelo con una buena reputación para trabajr con información tabular y categorica. Lo único que me lamento es no tener un dataset mas grande.
# 
# ## Definición del problema
# 
# Predecir el ingreso_mensual_hogar basándose en un conjunto de características de la vivienda, demográficas del hogar y su ubicación geográfica.
# 
# Hipótesis o Preguntas de Investigación
# 
# Hipótesis 1 (Capital Humano): El nivel educativo del jefe de hogar (jefe_hogar_educacion) es uno de los predictores más importantes del ingreso del hogar. Se espera una correlación positiva fuerte.
# 
# Hipótesis 2 (Ubicación Geográfica): La comuna donde se ubica la vivienda es un factor determinante del ingreso, reflejando disparidades económicas territoriales en Chile. Comunas como 'Providencia' o 'Ñuñoa' probablemente se asociarán con ingresos más altos.
# 
# Hipótesis 3 (Calidad de la Vivienda): Características como la superficie_m2, el número de baños (num_banos) y los materiales de construcción (material_paredes, material_techo) están positivamente correlacionados con el ingreso.
# 
# Pregunta de Investigación: Más allá de las variables obvias, ¿qué otros factores, como la tenencia de la vivienda (tenencia_vivienda) o el acceso a servicios (internet, gas_natural), tienen un poder predictivo significativo sobre el ingreso?
# 

# ## EDA

# 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from data_observatory_test_062025.load_data_viviendas import load_interim_data
from data_observatory_test_062025.generic_eda_report import generic_report

# 2. Configurar estilo de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 3. Cargar el dataset limpio de la Pregunta 2
try:
    df_limpio = load_interim_data()
    print("Dataset limpio cargado exitosamente.")
except FileNotFoundError:
    print("Error: Ejecute primero la Pregunta 2 para generar 'datos_viviendas_censo_limpio.csv'")
    exit()


# 

# Lo primero es dejar como referencia al tope del archivo una descripción general del dataset

# Me llama la atención esa variable "Unnamed:0", parece que se coló un índice extra jaja

generic_report(df_limpio)


# 1. Visualización de la variable objetivo: Ingreso Mensual del Hogar
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df_limpio['ingreso_mensual_hogar'], kde=True, bins=50)
plt.title('Distribución del Ingreso Mensual del Hogar')
plt.xlabel('Ingreso (CLP)')
plt.ylabel('Frecuencia')



# ** La distribución de ingresos casi siempre está sesgada. Este dataset parece haber tomado una muestra mas uniforme. el dataset no representa la totalidad del espectro de ingresos en Chile. Pero tenemos una muestra mas uniforme **
# 

df_limpio['ingreso_mensual_hogar'].describe()


# 2. Relación entre variables categóricas clave y el ingreso
plt.figure(figsize=(16, 7))
sns.boxplot(x='jefe_hogar_educacion', y='ingreso_mensual_hogar', data=df_limpio,
            order=['Sin estudios', 'Básica', 'Media', 'Técnica', 'Universitaria'])
plt.title('Log(Ingreso) vs. Nivel Educacional del Jefe de Hogar')
plt.xlabel('Educación del Jefe de Hogar')
plt.ylabel('Log(Ingreso)')
plt.xticks(rotation=45)
plt.show()


# Todo es bastante regular, es muy descorcentante trabajar con información tan artificial.
# 

numeric_cols = df_limpio.select_dtypes(include=np.number).columns
plt.figure(figsize=(12, 10))
correlation_matrix = df_limpio[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Mapa de Calor de Correlación de Variables Numéricas')
plt.show()


df_limpio[numeric_cols].corr()


df_limpio = df_limpio.drop(columns=['Unnamed: 0', 'id_vivienda', 'ingreso_per_capita', 'indice_masculinidad']) # No queremos data leakage 
# Las columnas 'suma_genero' y 'suma_etarios' parecen ser validaciones de consistencia.
# Si num_personas_hogar == num_hombres + num_mujeres, entonces suma_genero es redundante.
# Las eliminamos para evitar multicolinealidad perfecta.
df_limpio = df_limpio.drop(columns=['suma_genero', 'suma_etarios'])

# 3. Verificar si hay otros outliers o inconsistencias
# Podríamos explorar las viviendas con scores altos si el modelo tiene mal rendimiento.
# Por ahora, confiamos en que los árboles de CatBoost manejarán bien los outliers.

# 4. Definir las variables predictoras (X) y la objetivo (y)
y = df_limpio['ingreso_mensual_hogar']
X = df_limpio.drop(columns=['ingreso_mensual_hogar'])

# 5. Identificar las variables categóricas para CatBoost
categorical_features_indices = [X.columns.get_loc(c) for c in X.select_dtypes(include=['object', 'bool']).columns]

print(f"Columnas categóricas identificadas: {X.select_dtypes(include=['object', 'bool']).columns.tolist()}")


numeric_cols = df_limpio.select_dtypes(include=np.number).columns
plt.figure(figsize=(12, 10))
correlation_matrix = df_limpio[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Mapa de Calor de Correlación de Variables Numéricas')
plt.show()


# Ahora hay menos colinealidad

# ## Implementación
# 
# 

# Dividir los datos en conjuntos de entrenamiento y prueba para la evaluación final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Implementación con Validación Cruzada ---
# Definimos el modelo CatBoost
model_cv = CatBoostRegressor(
    iterations=1000,          # Número de árboles
    learning_rate=0.05,
    depth=6,                  # Profundidad de los árboles
    loss_function='RMSE',     # Métrica de pérdida para regresión
    cat_features=categorical_features_indices,
    verbose=0,                # Suprimir output durante el entrenamiento
    random_seed=42
)

# Configuramos la validación cruzada (K-Fold)
# 5 folds es un estándar común, proporciona un buen balance entre sesgo y varianza.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Ejecutamos la validación cruzada. Usamos 'neg_root_mean_squared_error'
# ya que cross_val_score trata de maximizar, y nosotros queremos minimizar el error.
scores_rmse = -cross_val_score(model_cv, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

print(f"RMSE promedio en validación cruzada: {scores_rmse.mean():.4f} (+/- {scores_rmse.std():.4f})")


# En un momento tuvimos una fuga de datos aquí, y el RMSE dió muy bajo. Ahora mismo da un valor más razonable de 780.000 pesos. No es para nada ideal, pero es una mejora de todas maneras.
# 
# Lo que si puedo destacar es la baja variabilida del modelo. Solo 5000 pesos de desviación es una maravilla.
# 
# 
# 

# Entrenar el modelo final en todo el conjunto de entrenamiento
final_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    cat_features=categorical_features_indices,
    verbose=200,
    random_seed=42
)

final_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Realizar predicciones en el conjunto de prueba
y_pred = final_model.predict(X_test)


# Calcular métricas de rendimiento en la escala original
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
r2_final = r2_score(y_test, y_pred)

print(f"\n--- Resultados en el Conjunto de Prueba ---")
print(f"RMSE Final: ${rmse_final:,.2f} CLP")
print(f"R-squared (R²): {r2_final:.4f}")


#  mejor rendimiento se alcanzó en la iteración 0. Un modelo de Gradient Boosting comienza con una predicción base (generalmente la media de la variable objetivo). El hecho de que ninguna de las siguientes iteraciones pudiera mejorar el error significa que ninguna de las variables predictoras (comuna, educación, superficie, etc.) aportó información útil para refinar esa predicción inicial.



