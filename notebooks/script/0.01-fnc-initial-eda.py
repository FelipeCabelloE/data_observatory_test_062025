#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Initial Eda
# 
# Vamos a crear un eda inicial para ambos datasets para definir cuál de los dos debería elegir.

# importamos la las librerias necesarias y los modulos internos

import pandas as pd
import io
from data_observatory_test_062025.config import RAW_DATA_DIR
from data_observatory_test_062025.generic_eda_report import generic_report 


RAW_DATA_DIR


# Cargamos los datos a un dataframe

datasets = [dataset for dataset in RAW_DATA_DIR.rglob("*.csv") if dataset.is_file()]


datasets


pandas_dataframes = [pd.read_csv(dataset) for dataset in datasets]


df_viviendas = pandas_dataframes[0]
df_salud = pandas_dataframes[1]


# Hacemos el reporte de vivienda

generic_report(df_viviendas)


generic_report(df_salud)


# ## Ambos datasets tiene sus desafios
# Me parece que me iré por el de vivienda

# 
