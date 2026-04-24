import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from modelos_estandar import FEATURES, TARGET


def entrenar_modelo_arbol(ruta_train: str = "train.csv"):
    train_df = pd.read_csv(ruta_train)
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    # Los modelos de arbol no requieren estandarizacion.
    pipeline_rf = Pipeline(
        steps=[
            (
                "modelo",
                RandomForestRegressor(n_estimators=120, random_state=42),
            )
        ]
    )
    pipeline_rf.fit(X_train, y_train)
    return pipeline_rf