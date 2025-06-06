import pandas as pd
from pandas import DataFrame

from data_observatory_test_062025.config import RAW_DATA_DIR, INTERIM_DATA_DIR


def load_data_viviendas() -> DataFrame:
    """helper function that  returns a loaded dataframe"""
    datasets = [dataset for dataset in RAW_DATA_DIR.rglob("*.csv") if dataset.is_file()]
    df_viviendas = pd.read_csv(datasets[0])
    return df_viviendas


def load_interim_data() -> DataFrame:
    """helper function that  returns a loaded dataframe"""
    datasets = [
        dataset for dataset in INTERIM_DATA_DIR.rglob("*.csv") if dataset.is_file()
    ]
    df_viviendas = pd.read_csv(datasets[0])
    return df_viviendas
