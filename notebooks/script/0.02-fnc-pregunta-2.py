#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Pregunta 2
# 
# Análisis de vivienda de "data/raw/datos_viviendas_censo.csv"

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_observatory_test_062025.config import INTERIM_DATA_DIR
from data_observatory_test_062025.load_data_viviendas import load_data_viviendas
from data_observatory_test_062025.generic_eda_report import generic_report


df_viviendas = load_data_viviendas()


generic_report(df_viviendas)





# No me parece bien empezar el análisis de coherencia sin antes estandarizar y deduplicar la información

# 
# ## 2.1.2 Estandarización y deduplicación (35%)
# 
# Implementa soluciones para:
# 
# 1. Estandarizar nombres de comunas (ej: “Providencia”, “providencia”, “PROVIDENCIA” → formato único)
# 2. Detectar y manejar registros duplicados de viviendas
# 3. Corregir inconsistencias menores en encoding de caracteres
# 4. Estandarizar formatos de tipos de vivienda y materiales de construcción
# 5. Normalizar representaciones de servicios básicos (Sí/No)
# 
# 

df_viviendas["comuna"].unique()


print(f"\nComunas únicas antes de la limpieza: {df_viviendas['comuna'].nunique()}")

# Estrategia: Normalizar texto (minúsculas, sin espacios) y luego aplicar un mapeo para abreviaturas.
comuna_map = {
    'provi': 'Providencia',
    'valpo': 'Valparaíso',
    'valparaiso': 'Valparaíso',
    'stgo': 'Santiago',
    'conce': 'Concepción',
    'condes': 'Las Condes',
    'concepcion': 'Concepción',
}

# Apply cleaning steps
df_viviendas['comuna_limpia'] = df_viviendas['comuna'].str.lower().str.strip()
df_viviendas['comuna_limpia'] = df_viviendas['comuna_limpia'].replace(comuna_map)
df_viviendas['comuna_limpia'] = df_viviendas['comuna_limpia'].str.capitalize()


print("Comunas estandarizadas:", sorted(df_viviendas['comuna_limpia'].dropna().unique()))








df_viviendas["id_vivienda"].tail(10)


df_viviendas[df_viviendas["id_vivienda"].isin(["VIV_000848_DUP", "VIV_000848"])]


print(f"\nNúmero total de registros antes de eliminar duplicados: {len(df_viviendas)}")

# Estrategia: Identificar y eliminar registros cuyo 'id_vivienda' contiene '_DUP'.
# Esto asume que el registro original (sin _DUP) es la fuente de verdad.
is_duplicate = df_viviendas['id_vivienda'].str.contains('_DUP', na=False)
num_duplicados = is_duplicate.sum()

print(f"Número de registros marcados como duplicados ('_DUP'): {num_duplicados}")

# Eliminar los registros duplicados
df_viviendas = df_viviendas[~is_duplicate].copy()
print(f"Número de registros después de eliminar duplicados: {len(df_viviendas)}")


df_viviendas["id_vivienda"].tail(10)


# Calculate string lengths
series_lengths = df_viviendas['id_vivienda'].str.len()

# Basic statistics
print("Length Statistics:")
print(series_lengths.describe())
print(f"\nMin length: {series_lengths.min()}")
print(f"Max length: {series_lengths.max()}")
print(f"Mode length: {series_lengths.mode().iloc[0]}")


for col in ['agua_potable', 'electricidad', 'gas_natural', 'internet', 'telefono_fijo', 'vehiculo_propio']:
    display(df_viviendas[col].unique())


boolean_columns = ['agua_potable', 'electricidad', 'gas_natural', 'internet', 'telefono_fijo', 'vehiculo_propio']
# Make them booleans
df_viviendas[boolean_columns] = df_viviendas[boolean_columns].applymap(lambda x: x == 'Sí')


for col in ['agua_potable', 'electricidad', 'gas_natural', 'internet', 'telefono_fijo', 'vehiculo_propio']:
    display(df_viviendas[col].unique())


print(f"\nComunas únicas antes de la limpieza: {df_viviendas['tipo_vivienda'].nunique()}")

# Estrategia: Normalizar texto (minúsculas, sin espacios) y luego aplicar un mapeo para abreviaturas.
vivienda_map = {
    "pareada": "Casa pareada",
    "departam.": "Departamento",
    "depto.": "Departamento",
    "dpto": "Departamento",
    "mediagua": "Media agua"

}

# Apply cleaning steps
df_viviendas['tipo_vivienda_limpia'] = df_viviendas['tipo_vivienda'].str.lower().str.strip()
df_viviendas['tipo_vivienda_limpia'] = df_viviendas['tipo_vivienda_limpia'].replace(vivienda_map)
df_viviendas['tipo_vivienda_limpia'] = df_viviendas['tipo_vivienda_limpia'].str.capitalize()


print("Tipo de vivienda:", df_viviendas['tipo_vivienda_limpia'].unique())





print("\nMaterial de paredes antes:", df_viviendas['material_paredes'].unique())


print(f"\nComunas únicas antes de la limpieza: {df_viviendas['material_paredes'].nunique()}")

# Estrategia: Normalizar texto (minúsculas, sin espacios) y luego aplicar un mapeo para abreviaturas.
material_map = {
    "wood": "Madera",
    "hormigon": "Hormigón",
    "concrete": "Hormigón",

}

# Apply cleaning steps
df_viviendas['material_paredes_limpia'] = df_viviendas['material_paredes'].str.lower().str.strip()
df_viviendas['material_paredes_limpia'] = df_viviendas['material_paredes_limpia'].replace(material_map)
df_viviendas['material_paredes_limpia'] = df_viviendas['material_paredes_limpia'].str.capitalize()


print("\nMaterial de paredes antes:", df_viviendas['material_paredes_limpia'].unique())


df_viviendas['material_techo'].unique()


print(f"únicas antes de la limpieza: {df_viviendas['material_techo'].nunique()}")

# Estrategia: Normalizar texto (minúsculas, sin espacios) y luego aplicar un mapeo para abreviaturas.
material_techo_map = {
}

# Apply cleaning steps
df_viviendas['material_techo_limpia'] = df_viviendas['material_techo'].str.lower().str.strip()
df_viviendas['material_techo_limpia'] = df_viviendas['material_techo_limpia'].replace(material_techo_map)
df_viviendas['material_techo_limpia'] = df_viviendas['material_techo_limpia'].str.capitalize()


df_viviendas['material_techo_limpia'].unique()





display(df_viviendas['jefe_hogar_sexo'].unique())
display(df_viviendas['jefe_hogar_educacion'].unique())


df_viviendas['comuna'] = df_viviendas['comuna_limpia']
df_viviendas = df_viviendas.drop(columns=['comuna_limpia'])
df_viviendas['tipo_vivienda'] = df_viviendas['tipo_vivienda_limpia']
df_viviendas = df_viviendas.drop(columns=['tipo_vivienda_limpia'])
df_viviendas['material_paredes'] = df_viviendas['material_paredes_limpia']
df_viviendas = df_viviendas.drop(columns=['material_paredes_limpia'])
df_viviendas['material_techo'] = df_viviendas['material_techo_limpia']
df_viviendas = df_viviendas.drop(columns=['material_techo_limpia'])



df_viviendas.columns


# ## 2.1.1 Análisis de coherencia interna
# Identifica y documenta inconsistencias en:
# 
# 1. Relación entre número de personas y distribución por género (hombres + mujeres = total)
# 2. Sumas de grupos etarios vs número total de personas por hogar
# 3. Relación lógica entre número de personas y dormitorios disponibles
# 4. Coherencia entre servicios básicos (ej: internet sin electricidad)
# 5. Valores anómalos en ingresos y características de la vivienda
# 6. Tipos de vivienda y materiales de construcción coherentes
# 
# Para cada inconsistencia, calcula: - Número de registros afectados - Magnitud promedio de la discrepancia - Rango de valores problemáticos

# 

print("\n--- 2.1.1. Análisis de Coherencia Interna ---")

# --- 1. Coherencia Demográfica: Total Personas vs. Desglose ---
print("\nValidando consistencia demográfica...")

# A) Personas vs. Género
df_viviendas['suma_genero'] = df_viviendas['num_hombres'] + df_viviendas['num_mujeres']
inconsistencia_genero = df_viviendas[df_viviendas['num_personas_hogar'] != df_viviendas['suma_genero']]
print(f"Problema 1: 'num_personas_hogar' no coincide con la suma de 'hombres' y 'mujeres'.")
print(f"Registros afectados: {len(inconsistencia_genero)}")








def detectar_inconsistencias(df, validaciones, verbose=True):
    """
    Función genérica para detectar inconsistencias en un DataFrame y crear flags automáticamente.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a validar
    validaciones : dict
        Diccionario con las validaciones a realizar. Estructura:
        {
            'nombre_validacion': {
                'condicion': expresión booleana o función,
                'flag_name': 'nombre_del_flag' (opcional, se genera automáticamente)
            }
        }
        O simplemente:
        {
            'nombre_validacion': condicion
        }
    verbose : bool
        Si True, imprime información detallada sobre cada inconsistencia encontrada

    Retorna:
    --------
    pandas.DataFrame
        DataFrame original con las columnas de flags añadidas
    dict
        Resumen de inconsistencias encontradas
    """

    df_resultado = df.copy()
    resumen_inconsistencias = {}

    for nombre_validacion, config in validaciones.items():

        condicion = config
        flag_name = f'{nombre_validacion}_flag'

        # Si la condición es una función, aplicarla al DataFrame
        if callable(condicion):
            mask_inconsistencia = condicion(df_resultado)
        else:
            # Si es una expresión booleana (Series), usarla directamente
            mask_inconsistencia = condicion

        # Contar registros afectados
        num_inconsistencias = mask_inconsistencia.sum()

        # Solo crear el flag si hay inconsistencias
        if num_inconsistencias > 0:
            df_resultado[flag_name] = mask_inconsistencia

            # Guardar información en el resumen
            resumen_inconsistencias[nombre_validacion] = {
                'registros_afectados': num_inconsistencias,
                'porcentaje': (num_inconsistencias / len(df_resultado)) * 100,
                'flag_creado': flag_name
            }

            # Imprimir información si verbose=True
            if verbose:
                print(f"\n{nombre_validacion.upper()}:")
                print(f"Registros afectados: {num_inconsistencias} ({(num_inconsistencias/len(df_resultado)*100):.2f}%)")
                print(f"Flag creado: '{flag_name}'")
        else:
            if verbose:
                print(f"\n{nombre_validacion.upper()}:")
                print("✓ No se encontraron inconsistencias")

    return df_resultado, resumen_inconsistencias




df_viviendas['suma_etarios'] = df_viviendas['num_menores_18'] + df_viviendas['num_adultos_18_64'] + df_viviendas['num_adultos_65_plus']
df_viviendas['suma_genero'] = df_viviendas['num_hombres'] + df_viviendas['num_mujeres']


validaciones = {
    'internet_sin_electricidad': lambda df: (df['internet'] == "Sí") & (df['electricidad'] == "No"),
    'telefono_fijo_sin_electricidad': lambda df: (df['telefono_fijo'] == "Sí") & (df['electricidad'] == "No"),
    'Departamento_madera': lambda df: (df['material_paredes'] == "Madera") & (df['tipo_vivienda'] == "Departamento"),
    'MediaAgua_Hormigon': lambda df: (df['material_paredes'] == "Hormigón") & (df['tipo_vivienda'] == "Media agua"),
    'Departamento_techo_paja': lambda df: (df['material_techo'] == "Paja") & (df['tipo_vivienda'] == "Departamento"),
    'Casa_pareada_techo_paja': lambda df: (df['material_techo'] == "Paja") & (df['tipo_vivienda'] == "Casa pareada"),
    'edad_jefe_hogar_anomala': lambda df: (df['jefe_hogar_edad'] < 18) | (df['jefe_hogar_edad'] > 110),
    'ingreso_anomalo': lambda df: (df['ingreso_mensual_hogar'] < 0) | (df['ingreso_mensual_hogar'] > 20000000),
    'inconsistencia_etaria': lambda df: df['num_personas_hogar'] != df['suma_etarios'],

    'techo_rustico_paredes_premium': lambda df: (df['material_techo'].isin(['Paja']) & df['material_paredes'].isin(['Hormigón'])),    
    'media_agua_losa': lambda df: ((df['tipo_vivienda'] == 'Media agua') & (df['material_techo'] == 'Losa')),
    'losa_adobe': lambda df: ((df['material_techo'] == 'Losa') & (df['material_paredes'] == 'Adobe')),
    'inconsistencia_de_genero': lambda df: df['num_personas_hogar'] != df['suma_genero'],
    'hogar_sin_personas': lambda df: df['num_personas_hogar'] <= 0
}


df_viviendas_validado, resumen = detectar_inconsistencias(df_viviendas, validaciones)


# ## Bonus: sistema de scoring de calidad de datos que permita evaluar la confiabilidad de cada registro de vivienda

def calcular_score_inconsistencias(df, nombre_score='score_inconsistencias', 
                                  tipo_score='count'):
    """
    Calcula un score de inconsistencias basado en la cantidad de flags por fila.
    Automáticamente detecta y usa todas las columnas que terminan en '_flag'.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con las columnas de flags
    nombre_score : str
        Nombre de la columna donde se guardará el score
    tipo_score : str
        Tipo de score a calcular:
        - 'count': Cuenta simple de flags activos
        - 'percentage': Porcentaje de flags activos sobre el total de flags


    Retorna:
    --------
    pandas.DataFrame
        DataFrame original con la columna de score añadida
    dict
        Información sobre el score calculado
    """

    df_resultado = df.copy()

    # Detectar automáticamente todas las columnas que terminan en '_flag'
    columnas_flags = [col for col in df.columns if col.endswith('_flag')]

    # Verificar que existan columnas de flags
    if not columnas_flags:
        print("No se encontraron columnas que terminen en '_flag' en el DataFrame")
        return df_resultado, {}

    # Calcular el score según el tipo especificado
    if tipo_score == 'count':
        # Suma simple de flags (True = 1, False = 0)
        df_resultado[nombre_score] = df_resultado[columnas_flags].sum(axis=1)

    elif tipo_score == 'percentage':
        # Porcentaje de flags activos
        df_resultado[nombre_score] = (df_resultado[columnas_flags].sum(axis=1) / len(columnas_flags)) * 100


    else:
        print(f"Tipo de score no reconocido: {tipo_score}")
        return df_resultado, {}

    # Calcular estadísticas del score
    info_score = {
        'columna_score': nombre_score,
        'tipo_score': tipo_score,
        'columnas_flags_usadas': columnas_flags,
        'num_flags': len(columnas_flags),
        'score_min': df_resultado[nombre_score].min(),
        'score_max': df_resultado[nombre_score].max(),
        'score_promedio': df_resultado[nombre_score].mean(),
        'registros_sin_inconsistencias': (df_resultado[nombre_score] == 0).sum(),
        'registros_con_inconsistencias': (df_resultado[nombre_score] > 0).sum()
    }


    return df_resultado, info_score


df_vivienda_score, info_score = calcular_score_inconsistencias(df_viviendas_validado, tipo_score="percentage")


info_score


# 
# ## 2.1.3 Cálculo de indicadores derivados (35%)
# 
# Usando los datos limpios, calcula:
# 
# 1. Razón personas por dormitorio y valida contra rangos lógicos
# 2. Índice de masculinidad por hogar (hombres/mujeres * 100)
# 3. Porcentaje de población por grupo etario por hogar
# 4. Ingreso per cápita por hogar
# 5. Identificación de viviendas con características atípicas
# 
# Genera visualizaciones que muestren: - Distribución de personas por dormitorio - Relación entre ingresos del hogar y servicios básicos disponibles - Identificación de outliers en características de vivienda
# 

df_indicadores = df_vivienda_score.copy()


# 9   num_personas_hogar     31660 non-null  int64  
#  10  num_hombres            31660 non-null  int64  
#  11  num_mujeres            31660 non-null  int64  
#  12  num_menores_18         31660 non-null  int64  
#  13  num_adultos_18_64      31660 non-null  int64  
#  14  num_adultos_65_plus    31660 non-null  int64 

# A) Razón personas por dormitorio (Hacinamiento)
df_indicadores['personas_por_dormitorio'] = df_indicadores['num_personas_hogar'] / df_indicadores['num_dormitorios']
df_indicadores['personas_por_dormitorio'] = df_indicadores['personas_por_dormitorio'].replace([np.inf, -np.inf], np.nan) # Handle division by zero if any

# B) Índice de masculinidad
df_indicadores['indice_masculinidad'] = (df_indicadores['num_hombres'] / df_indicadores['num_mujeres']) * 100
df_indicadores['indice_masculinidad'] = df_indicadores['indice_masculinidad'].replace([np.inf, -np.inf], np.nan)


# C) Porcentaje de población por grupo etario por hogar
df_indicadores["porc_menores_18"] = df_indicadores["num_menores_18"] / df_indicadores["num_personas_hogar"] * 100
df_indicadores["num_adultos_18_64"] = df_indicadores["num_adultos_18_64"] / df_indicadores["num_personas_hogar"] * 100
df_indicadores["num_adultos_65_plus"] = df_indicadores["num_adultos_65_plus"] / df_indicadores["num_personas_hogar"] * 100

# D) Ingreso per cápita
df_indicadores['ingreso_per_capita'] = df_indicadores['ingreso_mensual_hogar'] / df_indicadores['num_personas_hogar']
df_indicadores['ingreso_per_capita'] = df_indicadores['ingreso_per_capita'].replace([np.inf, -np.inf], np.nan)

# E) Viviendas atípicas
# Ya identificado con score_inconsistencias
# df_indicadores["score_inconsistencias"]











plt.figure(figsize=(10, 6))
sns.histplot(df_indicadores['personas_por_dormitorio'].dropna(), bins=30, kde=True)
plt.title('Distribución de Personas por Dormitorio (Hacinamiento)')
plt.xlabel('Personas por Dormitorio')
plt.ylabel('Cantidad de Viviendas')

plt.show()


# B) Relación entre ingresos y servicios básicos
df_servicios = df_indicadores.melt(id_vars=['ingreso_per_capita'],
                              value_vars=['agua_potable', 'electricidad', 'gas_natural', 'internet', 'telefono_fijo', 'vehiculo_propio'],
                              var_name='servicio', value_name='disponible')

plt.figure(figsize=(12, 8))
sns.violinplot(data=df_servicios, x='servicio', y='ingreso_per_capita', hue='disponible',split=True)
plt.title('Ingresos per capita del Hogar vs. Disponibilidad de Servicios Básicos')
plt.ylabel('Ingreso Mensual del Hogar ')

plt.show()


df_servicios


df_indicadores['suma_servicios_basicos'] = df_indicadores[boolean_columns].sum(axis=1)



plt.figure(figsize=(12, 8))
sns.boxplot(data=df_indicadores, x='suma_servicios_basicos', y='ingreso_per_capita', hue='suma_servicios_basicos')
plt.title('Ingresos per capita del Hogar vs. Cantidad de Servicios Básicos')
plt.ylabel('Ingreso Mensual del Hogar ')

plt.show()


df_indicadores[df_indicadores['suma_servicios_basicos'] == 6]


plt.figure(figsize=(12, 8))
sns.boxplot(data=df_indicadores, x='suma_servicios_basicos', y='ingreso_per_capita', hue='suma_servicios_basicos')
plt.title('Ingresos per capita del Hogar vs. Cantidad de Servicios Básicos')
plt.ylabel('Ingreso Mensual del Hogar ')

plt.show()


# 0   id_vivienda            31660 non-null  object 
#  1   comuna                 31660 non-null  object 
#  2   tipo_vivienda          31660 non-null  object 
#  3   num_dormitorios        31660 non-null  int64  
#  4   num_banos              31660 non-null  int64  
#  5   superficie_m2          31660 non-null  float64
#  6   ano_construccion       31660 non-null  int64  
#  7   material_paredes       31660 non-null  object 
#  8   material_techo         31660 non-null  object 
#  9   num_personas_hogar     31660 non-null  int64  
#  10  num_hombres            31660 non-null  int64  
#  11  num_mujeres            31660 non-null  int64  
#  12  num_menores_18         31660 non-null  int64  
#  13  num_adultos_18_64      31660 non-null  int64  
#  14  num_adultos_65_plus    31660 non-null  int64  
#  15  ingreso_mensual_hogar  31660 non-null  int64  
#  16  jefe_hogar_edad        31660 non-null  int64  
#  17  jefe_hogar_sexo        31660 non-null  object 
#  18  jefe_hogar_educacion   31660 non-null  object 
#  19  agua_potable           31660 non-null  object 
#  20  electricidad           31660 non-null  object 
#  21  gas_natural            31660 non-null  object 
#  22  internet               31660 non-null  object 
#  23  telefono_fijo          31660 non-null  object 
#  24  vehiculo_propio        31660 non-null  object 
#  25  tenencia_vivienda      31660 non-null  object 
# dtypes: float64(1), int64(11), object(14)
# memory usage: 6.3+ MB

# Identificación de outliers en características de vivienda


for col in ["tipo_vivienda", "material_paredes", "material_techo"]:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df_indicadores, order=df_indicadores[col].value_counts().index)
    plt.title(f'Category Counts in {col}')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()


# tamaño de las viviendas segun su año de construccion


df_indicadores.to_csv(INTERIM_DATA_DIR / "indicadores_vivienda_interim.csv")

