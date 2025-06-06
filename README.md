# data_observatory_test_062025
_A data science project for the data scientist position in Data Observatory._


## Bienvenido

Bienvenido a la postulación de Felipe Cabello :).

En la sección de "Project Organization" podras ver la organización general de esta Repo.

## __Mis respuestas están en la carpeta ./notebooks__

## Gettign started

Si quieres reproducir los resultados que obtuve debes considerar lo siguiente.

Este proyecto usa principalmente la herramienta [uv](https://docs.astral.sh/uv/getting-started/installation/) de astral para funcionar. Por lo tanto, deberías tener instalada esta herramienta antes de comenzar.

Para instalar uv sigue las instrucciones de la [página](https://docs.astral.sh/uv/getting-started/installation/)

o utiliza este comando:


```
curl -LsSf https://astral.sh/uv/install.sh | sh
``` 
Una vez instalado, todo es bastante simple.

## Instalación

simplemente corre ```uv sync```

Luego puedes correr ```uv run jupyter lab``` e ir a la carpeta [./notebooks](https://github.com/FelipeCabelloE/data_observatory_test_062025/tree/master/notebooks) y ver mis respuestas :)


## Usos alternativos

Este repositorio también hace uso de la herramienta ```make```

así que también puedes correr ```make requirements``` para instalar el proyecto, y luego ```make create_environment```

despues es solo un tema de activar python (```source .venv/bin/activate```) y correr ```jupyter lab```.

Espero esté todo claro. Si tienes dudas, me puedes contactar!

## Nota importante

Las respuestas están en la carpeta [./notebooks](https://github.com/FelipeCabelloE/data_observatory_test_062025/tree/master/notebooks). Hay una carpeta con un intento fallido de la pregunta 3, pero no me gustó el resultado ni el acercamiento, así que lo dejé como referencia, pero no correspnde a mi respuesta oficial. Las únicas respuestas válidas están en la base de la carpeta. Los otros directorios son solo de referencia.





## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         data_observatory_test_062025 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── data_observatory_test_062025   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes data_observatory_test_062025 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

--------

