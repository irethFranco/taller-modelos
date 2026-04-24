from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import dump
from sklearn.metrics import mean_squared_error, r2_score

from modelos_arboles import entrenar_modelo_arbol
from modelos_estandar import (
    FEATURES,
    TARGET,
    entrenar_modelos_estandarizados,
    generar_archivos_entrenamiento_prueba,
)

DATASET_PATH = Path("dataset_ciclismo_fatiga.csv")
TRAIN_PATH = Path("train.csv")
TEST_PATH = Path("test.csv")
MODELOS_DIR = Path("modelos_guardados")

st.set_page_config(page_title="Taller Fatiga Ciclismo", page_icon="🚴", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef7f1 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .panel-principal {
        background: #ffffff;
        border: 1px solid #d8e6ff;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .panel-principal h2 {
        color: #0b3d91;
        margin-bottom: 0.3rem;
    }
    .panel-principal p {
        color: #2f3c4f;
        margin: 0;
    }
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        color: #ffffff;
        background: linear-gradient(90deg, #0061ff 0%, #00a3a3 100%);
        padding: 0.6rem 1rem;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(0, 97, 255, 0.25);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #dce8f5;
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="panel-principal">
        <h2>Taller de ML - Fatiga en Ciclismo</h2>
        <p>Flujo: separar dataset en train/test, entrenar modelos con pipeline y evaluar sobre test.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("1) Generar train.csv y test.csv"):
    try:
        train_df, test_df = generar_archivos_entrenamiento_prueba(
            ruta_dataset=str(DATASET_PATH),
            ruta_train=str(TRAIN_PATH),
            ruta_test=str(TEST_PATH),
        )
        st.success("Archivos generados correctamente.")
        st.info(f"Train: {len(train_df)} filas | Test: {len(test_df)} filas")
    except Exception as exc:
        st.error(f"No se pudo separar el dataset: {exc}")

if st.button("2) Entrenar y guardar modelos"):
    if not TRAIN_PATH.exists():
        st.warning("Primero debes generar el archivo train.csv.")
    else:
        try:
            pipeline_lr, pipeline_knn = entrenar_modelos_estandarizados(str(TRAIN_PATH))
            pipeline_rf = entrenar_modelo_arbol(str(TRAIN_PATH))

            st.session_state["modelos"] = {
                "lr": pipeline_lr,
                "knn": pipeline_knn,
                "rf": pipeline_rf,
            }

            MODELOS_DIR.mkdir(exist_ok=True)
            dump(pipeline_lr, MODELOS_DIR / "pipeline_regresion_lineal.joblib")
            dump(pipeline_knn, MODELOS_DIR / "pipeline_knn.joblib")
            dump(pipeline_rf, MODELOS_DIR / "pipeline_arbol.joblib")

            st.success("Modelos entrenados y guardados en disco.")
            st.caption("KNN y Regresión usan estandarización; Árbol se entrena sin escalar.")
        except Exception as exc:
            st.error(f"No se pudieron entrenar los modelos: {exc}")

if st.button("3) Probar con test.csv"):
    if "modelos" not in st.session_state:
        st.error("Primero debes entrenar los modelos.")
    elif not TEST_PATH.exists():
        st.error("No existe test.csv. Genera la separación primero.")
    else:
        test_df = pd.read_csv(TEST_PATH)
        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]

        pred_lr = st.session_state["modelos"]["lr"].predict(X_test)
        pred_knn = st.session_state["modelos"]["knn"].predict(X_test)
        pred_rf = st.session_state["modelos"]["rf"].predict(X_test)

        resultados = pd.DataFrame(
            {
                "Modelo": ["Regresión Lineal", "KNN", "Random Forest"],
                "MSE": [
                    f"{mean_squared_error(y_test, pred_lr):.4f}",
                    f"{mean_squared_error(y_test, pred_knn):.4f}",
                    f"{mean_squared_error(y_test, pred_rf):.4f}",
                ],
                "R2": [
                    f"{r2_score(y_test, pred_lr):.4f}",
                    f"{r2_score(y_test, pred_knn):.4f}",
                    f"{r2_score(y_test, pred_rf):.4f}",
                ],
            }
        )

        st.subheader("Resultados de evaluación")
        st.dataframe(resultados, use_container_width=True)