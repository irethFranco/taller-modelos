from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.metrics import mean_squared_error, r2_score

from modelos_arboles import entrenar_modelo_arbol
from modelos_estandar import (
    FEATURES,
    TARGET,
    entrenar_modelos_estandarizados,
    generar_archivos_entrenamiento_prueba,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_ciclismo_fatiga.csv"
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"
MODELOS_DIR = BASE_DIR / "modelos_guardados"
MODEL_PATHS = {
    "lr": MODELOS_DIR / "pipeline_regresion_lineal.joblib",
    "knn": MODELOS_DIR / "pipeline_knn.joblib",
    "rf": MODELOS_DIR / "pipeline_arbol.joblib",
}


def modelos_guardados_disponibles() -> bool:
    return all(path.exists() for path in MODEL_PATHS.values())


def cargar_modelos_guardados():
    return {
        "lr": load(MODEL_PATHS["lr"]),
        "knn": load(MODEL_PATHS["knn"]),
        "rf": load(MODEL_PATHS["rf"]),
    }


def entrenar_y_guardar_modelos():
    pipeline_lr, pipeline_knn = entrenar_modelos_estandarizados(str(TRAIN_PATH))
    pipeline_rf = entrenar_modelo_arbol(str(TRAIN_PATH))
    MODELOS_DIR.mkdir(exist_ok=True)
    dump(pipeline_lr, MODEL_PATHS["lr"])
    dump(pipeline_knn, MODEL_PATHS["knn"])
    dump(pipeline_rf, MODEL_PATHS["rf"])
    return {"lr": pipeline_lr, "knn": pipeline_knn, "rf": pipeline_rf}


def limpiar_resultados_test():
    for key in ("tabla_metricas", "tabla_manual", "pred_manual"):
        if key in st.session_state:
            del st.session_state[key]

st.set_page_config(page_title="Taller Fatiga Ciclismo", page_icon="🚴", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #1f2937;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.2rem;
    }
    .tarjeta {
        background: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.9rem;
    }
    .tarjeta h2, .tarjeta h3 {
        color: #0f172a;
        margin-bottom: 0;
        font-size: 2rem;
        letter-spacing: 0.2px;
    }
    .tarjeta p {
        color: #334155;
        margin: 0.2rem 0;
    }
    .estado-ok {
        color: #0f5132;
        font-weight: 600;
    }
    .bloque {
        background: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        font-weight: 500;
        color: #1f2937;
        background: #ffffff;
        padding: 0.6rem 0.8rem;
        transition: all 0.12s ease;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .stButton > button:hover {
        border-color: #94a3b8;
        background: #f8fafc;
    }
    div[data-testid="stTable"] table {
        background: #ffffff;
        color: #111827;
        border: 1px solid #dbe4f0;
        border-radius: 10px;
    }
    .label-seccion {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .nota {
        color: #475569;
        font-size: 0.92rem;
    }
    .divider {
        margin: 0.5rem 0 0.3rem 0;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tarjeta">
        <h2>Taller de ML - Fatiga en Ciclismo</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

if "modelos" not in st.session_state and modelos_guardados_disponibles():
    st.session_state["modelos"] = cargar_modelos_guardados()
if "entrenado_en_sesion" not in st.session_state:
    st.session_state["entrenado_en_sesion"] = False

col_izq, col_der = st.columns([1, 1.25], gap="large")

with col_izq:
    st.markdown('<div class="label-seccion">1) Preparación y entrenamiento</div>', unsafe_allow_html=True)
    st.markdown('<div class="bloque">', unsafe_allow_html=True)
    if DATASET_PATH.exists():
        st.markdown('<div class="estado-ok">Dataset cargado correctamente.</div>', unsafe_allow_html=True)
    else:
        st.error("No se encuentra el dataset base.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    modelos_ya_listos = st.session_state.get("entrenado_en_sesion", False)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Generar train/test"):
            try:
                train_df, test_df = generar_archivos_entrenamiento_prueba(
                    ruta_dataset=str(DATASET_PATH),
                    ruta_train=str(TRAIN_PATH),
                    ruta_test=str(TEST_PATH),
                )
                st.success(f"Archivos creados. Train: {len(train_df)} | Test: {len(test_df)}")
            except Exception as exc:
                st.error(f"No se pudo separar el dataset: {exc}")
    with col_btn2:
        if not modelos_ya_listos:
            if st.button("Entrenar modelos"):
                if not TRAIN_PATH.exists():
                    st.warning("Primero genera train.csv y test.csv.")
                else:
                    try:
                        st.session_state["modelos"] = entrenar_y_guardar_modelos()
                        st.session_state["entrenado_en_sesion"] = True
                        limpiar_resultados_test()
                        st.success("Modelos entrenados y guardados.")
                    except Exception as exc:
                        st.error(f"No se pudieron entrenar los modelos: {exc}")
        else:
            st.info("Modelo ya entrenado")

    col_btn3, col_btn4 = st.columns(2)
    with col_btn3:
        if st.button("Cargar modelos"):
            if modelos_guardados_disponibles():
                st.session_state["modelos"] = cargar_modelos_guardados()
                st.success("Modelos cargados desde disco.")
            else:
                st.warning("No hay modelos guardados. Entrena primero.")
    with col_btn4:
        if st.button("Nuevo test"):
            limpiar_resultados_test()
            st.session_state["manual_values"] = {}
            st.success("Formulario y resultados reiniciados.")
    if modelos_ya_listos:
        st.markdown(
            '<div class="nota">Modelos ya entrenados: puedes hacer nuevos test sin reentrenar.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="nota">Entrena una vez y luego usa "Nuevo test" para seguir probando.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_der:
    st.markdown('<div class="label-seccion">2) Test de métricas sobre test.csv</div>', unsafe_allow_html=True)
    st.markdown('<div class="bloque">', unsafe_allow_html=True)
    if st.button("Evaluar métricas comparativas"):
        if "modelos" not in st.session_state:
            if modelos_guardados_disponibles():
                st.session_state["modelos"] = cargar_modelos_guardados()
            else:
                st.error("Primero entrena o carga modelos.")
        elif not TEST_PATH.exists():
            st.error("No existe test.csv. Genera la separación primero.")

        if "modelos" in st.session_state and TEST_PATH.exists():
            test_df = pd.read_csv(TEST_PATH)
            X_test = test_df[FEATURES]
            y_test = test_df[TARGET]

            pred_lr = st.session_state["modelos"]["lr"].predict(X_test)
            pred_knn = st.session_state["modelos"]["knn"].predict(X_test)
            pred_rf = st.session_state["modelos"]["rf"].predict(X_test)

            st.session_state["tabla_metricas"] = pd.DataFrame(
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

    if "tabla_metricas" in st.session_state:
        st.table(st.session_state["tabla_metricas"])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="label-seccion">3) Predicción manual (jugar con valores)</div>', unsafe_allow_html=True)
st.markdown('<div class="bloque">', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("**Frecuencia cardiaca**")
    frecuencia_cardiaca = st.number_input(
        "frecuencia_cardiaca", min_value=40.0, max_value=220.0, value=140.0, step=1.0
    )
    st.markdown("**Potencia**")
    potencia = st.number_input(
        "potencia", min_value=0.0, max_value=2000.0, value=250.0, step=5.0
    )
    st.markdown("**Cadencia**")
    cadencia = st.number_input(
        "cadencia", min_value=0.0, max_value=220.0, value=85.0, step=1.0
    )
with col_b:
    st.markdown("**Tiempo**")
    tiempo = st.number_input(
        "tiempo", min_value=0.0, max_value=400.0, value=60.0, step=1.0
    )
    st.markdown("**Temperatura**")
    temperatura = st.number_input(
        "temperatura", min_value=-10.0, max_value=60.0, value=24.0, step=0.5
    )
with col_c:
    st.markdown("**Pendiente**")
    pendiente = st.number_input(
        "pendiente", min_value=-25.0, max_value=25.0, value=2.0, step=0.5
    )
    st.markdown("**Velocidad**")
    velocidad = st.number_input(
        "velocidad", min_value=0.0, max_value=120.0, value=30.0, step=0.5
    )

if st.button("Predecir con valores manuales"):
    if "modelos" not in st.session_state:
        if modelos_guardados_disponibles():
            st.session_state["modelos"] = cargar_modelos_guardados()
        else:
            st.error("Primero entrena o carga modelos.")

    if "modelos" in st.session_state:
        fila_manual = pd.DataFrame(
            [
                {
                    "frecuencia_cardiaca": frecuencia_cardiaca,
                    "potencia": potencia,
                    "cadencia": cadencia,
                    "tiempo": tiempo,
                    "temperatura": temperatura,
                    "pendiente": pendiente,
                    "velocidad": velocidad,
                }
            ]
        )

        pred_lr = float(st.session_state["modelos"]["lr"].predict(fila_manual)[0])
        pred_knn = float(st.session_state["modelos"]["knn"].predict(fila_manual)[0])
        pred_rf = float(st.session_state["modelos"]["rf"].predict(fila_manual)[0])
        promedio = (pred_lr + pred_knn + pred_rf) / 3

        st.session_state["tabla_manual"] = pd.DataFrame(
            {
                "Modelo": ["Regresión Lineal", "KNN", "Random Forest", "Promedio"],
                "Fatiga predicha": [f"{pred_lr:.4f}", f"{pred_knn:.4f}", f"{pred_rf:.4f}", f"{promedio:.4f}"],
            }
        )

if "tabla_manual" in st.session_state:
    st.table(st.session_state["tabla_manual"])
st.markdown("</div>", unsafe_allow_html=True)