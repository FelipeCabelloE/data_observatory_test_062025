from pandas import DataFrame


def generate_report_section(title: str, content_dict: dict) -> None:
    """Helper function to print formatted report sections."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    for subtitle, text in content_dict.items():
        print(f"\n--- {subtitle} ---")
        print(text)
    print("=" * 60)


def generic_report(dataframe: DataFrame) -> None:
    """Generate a simple, inital report to understand the dataframes"""

    # 1.1.1: Exploración inicial y documentación de problemas
    initial_report = {
        "Dataset shape": dataframe.shape,
        "Dataset columns": dataframe.columns,
        "Dataset Info": dataframe.info(),
        "Dtypes": dataframe.dtypes,
        "Numeric Describe": dataframe.describe(),
        "Object Describe": dataframe.describe(include="object"),
        "Valores Faltantes": dataframe.isna().sum().to_string(),
        "Duplicados": f"Número de filas duplicadas: {dataframe.duplicated().sum()}",
        "Valores únicos": dataframe.nunique().to_string(),
    }
    generate_report_section("1.1.1 Exploración Inicial", initial_report)
