from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "frecuencia_cardiaca",
    "potencia",
    "cadencia",
    "tiempo",
    "temperatura",
    "pendiente",
    "velocidad",
]
TARGET = "fatiga"


def generar_archivos_entrenamiento_prueba(
    ruta_dataset: str = "dataset_ciclismo_fatiga.csv",
    ruta_train: str = "train.csv",
    ruta_test: str = "test.csv",
    test_size: float = 0.20,
    random_state: int = 42,
):
    df = pd.read_csv(ruta_dataset).dropna().drop_duplicates()
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_path = Path(ruta_train)
    test_path = Path(ruta_test)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


def entrenar_modelos_estandarizados(ruta_train: str = "train.csv"):
    train_df = pd.read_csv(ruta_train)
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    pipeline_lr = Pipeline(
        steps=[("scaler", StandardScaler()), ("modelo", LinearRegression())]
    )
    pipeline_knn = Pipeline(
        steps=[("scaler", StandardScaler()), ("modelo", KNeighborsRegressor(n_neighbors=5))]
    )

    pipeline_lr.fit(X_train, y_train)
    pipeline_knn.fit(X_train, y_train)
    return pipeline_lr, pipeline_knn