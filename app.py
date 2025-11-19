# app.py
# -*- coding: utf-8 -*-

import os
import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="DanuCard – Churn & Risk Dashboard",
    layout="wide",
)

st.title("DanuCard – Churn & Risk Dashboard")

# ==========================================================
# CONSTANTES
# ==========================================================

AGG_TX_CSV = "transactions_by_state_month.csv"     # month, state, total_trx, total_amount, n_users
USERS_CSV  = "base_integrada_small.csv"            # tu base integrada pequeña

# Si quieres usar Drive, pon aquí SOLO el ID del archivo (no toda la URL)
DRIVE_BASE_ID = ""  # por ejemplo: "11S9-SZCMF30LGyMWz4nexjIW8Ltdi1oo" (si lo quieres usar)


FEATURES_CAT = [
    "share_tier", "antiguedad_categoria", "usertype",
    "gender", "occupation", "creationflow",
]

FEATURES_NUM = [
    "total_amount", "total_trx", "antiguedad_cliente_days",
    "llamadas_cc", "minutos_cc", "aht_promedio_cc",
    "trx_share_global",
]


# ==========================================================
# FUNCIONES DE CARGA
# ==========================================================

@st.cache_data(show_spinner=False)
def load_agg_transactions() -> pd.DataFrame:
    """Carga archivo agregado de transacciones por mes y estado."""
    if not os.path.exists(AGG_TX_CSV):
        st.error(f"No se encontró el archivo {AGG_TX_CSV} en el repositorio.")
        return pd.DataFrame()

    df = pd.read_csv(AGG_TX_CSV)
    # Normalizamos tipos
    if "month" in df.columns:
        df["month"] = df["month"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_users_base_local() -> pd.DataFrame | None:
    """Intenta cargar la base de usuarios desde un CSV local."""
    if not os.path.exists(USERS_CSV):
        return None

    try:
        base = pd.read_csv(USERS_CSV)
        return base
    except Exception as e:
        st.error(f"Error al leer {USERS_CSV}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_users_base_from_drive(drive_id: str) -> pd.DataFrame | None:
    """Carga base de usuarios desde Google Drive (si se proporciona ID)."""
    if not drive_id:
        return None

    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    try:
        base = pd.read_csv(url)
        return base
    except Exception as e:
        st.error(f"No se pudo cargar la base de usuarios desde Drive: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_model_and_transformer():
    power_tf = None
    model = None

    if os.path.exists("power_transformer.pkl"):
        try:
            with open("power_transformer.pkl", "rb") as f:
                power_tf = pickle.load(f)
        except Exception as e:
            st.warning(f"No se pudo cargar power_transformer.pkl: {e}")
    else:
        st.info("No se encontró power_transformer.pkl en el repositorio.")

    if os.path.exists("xgboost_model.pkl"):
        try:
            with open("xgboost_model.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.warning(f"No se pudo cargar xgboost_model.pkl: {e}")
    else:
        st.info("No se encontró xgboost_model.pkl en el repositorio.")

    return power_tf, model


# ==========================================================
# CARGA GLOBAL DE DATOS
# ==========================================================

agg_tx = load_agg_transactions()

# Primero intentamos local, si no existe intentamos Drive (si pusiste ID)
base_integrada = load_users_base_local()
if base_integrada is None:
    base_integrada = load_users_base_from_drive(DRIVE_BASE_ID)

power_transformer, xgb_model = load_model_and_transformer()


# ==========================================================
# PREPROCESO / MODELO DE CHURN
# ==========================================================

@st.cache_data(show_spinner=False)
def score_users_with_model(base: pd.DataFrame | None,
                           power_tf,
                           model) -> pd.DataFrame:
    """Devuelve base con columna churn_proba si el modelo funciona."""
    if base is None or base.empty:
        return pd.DataFrame()

    df = base.copy()

    # Aseguramos que exista id_user (clave)
    if "id_user" not in df.columns:
        st.warning("La base de usuarios no tiene columna 'id_user'.")
        df["churn_proba"] = np.nan
        return df

    # Verificamos columnas requeridas
    missing = [c for c in (FEATURES_CAT + FEATURES_NUM) if c not in df.columns]
    if missing:
        st.info(
            "Faltan estas columnas para el modelo, se rellenan con valores neutros: "
            + ", ".join(missing)
        )
        # Creamos columnas faltantes con valores neutros
        for c in FEATURES_NUM:
            if c not in df.columns:
                df[c] = 0.0
        for c in FEATURES_CAT:
            if c not in df.columns:
                df[c] = "Unknown"

    if model is None:
        st.info("El modelo XGBoost no se cargó. Se omiten las predicciones.")
        df["churn_proba"] = np.nan
        return df

    # Filtramos filas con NaN en las features (solo para el modelo)
    df_model = df.dropna(subset=FEATURES_CAT + FEATURES_NUM).copy()
    if df_model.empty:
        st.info("No hay suficientes datos completos para el modelo.")
        df["churn_proba"] = np.nan
        return df

    # Numéricas
    X_num = df_model[FEATURES_NUM].values
    if power_tf is not None:
        try:
            X_num = power_tf.transform(X_num)
        except Exception:
            # Si falla el transformer, seguimos con numéricas en bruto
            pass

    # Categóricas -> one-hot
    X_cat = pd.get_dummies(df_model[FEATURES_CAT].astype(str), drop_first=False)
    X = np.concatenate([X_num, X_cat.values], axis=1)

    try:
        proba = model.predict_proba(X)[:, 1]
        df_model["churn_proba"] = proba
    except Exception as e:
        st.info(f"El modelo no aceptó el formato de entrada ({e}). Se omiten predicciones.")
        df["churn_proba"] = np.nan
        return df

    # Volvemos a unir al DF completo
    df = df.merge(df_model[["id_user", "churn_proba"]], on="id_user", how="left")

    # Nivel de riesgo por días de inactividad (si existe)
    if "max_dias_inactividad" in df.columns:
        df["risk_level"] = pd.cut(
            df["max_dias_inactividad"],
            bins=[-1, 10, 20, 1000],
            labels=["Low", "Medium", "High"],
        )
    else:
        df["risk_level"] = np.nan

    return df


base_scored = score_users_with_model(base_integrada, power_transformer, xgb_model)


@st.cache_data(show_spinner=False)
def compute_churn_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta un motivo de churn simple con reglas de negocio."""
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    # Si tenemos etiqueta de churn real la usamos para subset,
    # si no, usamos todos los que tengan probabilidad alta.
    if "churn" in data.columns:
        subset = data[data["churn"] == 1].copy()
        if subset.empty:
            subset = data
    else:
        if "churn_proba" in data.columns and data["churn_proba"].notna().any():
            subset = data[data["churn_proba"] >= 0.3].copy()
        else:
            subset = data

    # Percentiles sobre subset
    for col in ["llamadas_cc", "total_amount", "total_trx"]:
        if col not in subset.columns:
            subset[col] = 0.0

    q_ll = subset["llamadas_cc"].quantile(0.75)
    q_amt = subset["total_amount"].quantile(0.25)
    q_trx = subset["total_trx"].quantile(0.25)

    def _reason(row):
        if row["llamadas_cc"] >= q_ll:
            return "LLAMADAS"
        elif row["total_amount"] <= q_amt:
            return "AMOUNT"
        elif row["total_trx"] <= q_trx:
            return "TRANSACTIONS"
        else:
            return "CREATION FLOW"

    data["churn_reason"] = data.apply(_reason, axis=1)
    return data


base_with_reasons = compute_churn_reasons(base_scored)


# ==========================================================
# FUNCIONES DE MÉTRICAS A PARTIR DE transactions_by_state_month
# ==========================================================

@st.cache_data(show_spinner=False)
def compute_monthly_metrics(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen mensual (sumando todos los estados)."""
    if agg_df.empty:
        return pd.DataFrame()

    metrics = (
        agg_df
        .groupby("month", as_index=False)
        .agg(
            n_users=("n_users", "sum"),
            total_trx=("total_trx", "sum"),
            total_amount=("total_amount", "sum"),
        )
        .sort_values("month")
    )

    metrics["amount_per_user"] = (
        metrics["total_amount"] / metrics["n_users"].replace(0, np.nan)
    )

    metrics["users_growth_pct"] = metrics["n_users"].pct_change() * 100
    metrics["trx_growth_pct"] = metrics["total_trx"].pct_change() * 100
    metrics["amount_per_user_growth_pct"] = (
        metrics["amount_per_user"].pct_change() * 100
    )

    # Proxy de churn: caída de usuarios
    metrics["churn_proxy"] = -metrics["users_growth_pct"]

    return metrics


@st.cache_data(show_spinner=False)
def compute_state_metrics(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por estado con el archivo agregado."""
    if agg_df.empty:
        return pd.DataFrame()

    df = (
        agg_df
        .groupby("state", as_index=False)
        .agg(
            n_users=("n_users", "sum"),
            total_trx=("total_trx", "sum"),
            total_amount=("total_amount", "sum"),
        )
    )

    df["trx_per_user"] = df["total_trx"] / df["n_users"].replace(0, np.nan)
    df["amount_per_user"] = df["total_amount"] / df["n_users"].replace(0, np.nan)
    df["churn_proxy"] = -df["n_users"]  # menos usuarios = más churn
    return df


monthly_metrics = compute_monthly_metrics(agg_tx)
state_metrics = compute_state_metrics(agg_tx)


# ==========================================================
# PÁGINA 1 – OVERVIEW (NUEVO DISEÑO)
# ==========================================================

def page_overview():
    st.subheader("1. Service / App Usage Overview")

    if agg_tx.empty:
        st.info("No se pudo cargar el archivo de transacciones agregadas.")
        return

    df = agg_tx.copy()

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        months_sorted = sorted(df["month"].unique())
        month_filter = st.multiselect(
            "Filter by month (YYYY-MM)",
            options=months_sorted,
            default=months_sorted,
        )

    with col_f2:
        states_sorted = sorted(df["state"].dropna().unique())
        states_sorted = [s for s in states_sorted if s != "Unknown"]
        state_filter = st.multiselect(
            "Filter by state",
            options=states_sorted,
            default=states_sorted,
        )

    mask_month = df["month"].isin(month_filter)
    mask_state = df["state"].isin(state_filter)
    df_f = df[mask_month & mask_state].copy()

    if df_f.empty:
        st.info("No hay datos para los filtros seleccionados.")
        return

    metrics_f = compute_monthly_metrics(df_f)

    # KPIs principales (diseño distinto)
    total_users = metrics_f["n_users"].iloc[-1]
    total_amount = metrics_f["total_amount"].sum()
    total_trx = metrics_f["total_trx"].sum()
    avg_amount_per_user = metrics_f["amount_per_user"].iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Current active users", f"{int(total_users):,}")
    with k2:
        st.metric("Total transactions (period)", f"{int(total_trx):,}")
    with k3:
        st.metric("Total amount (period)", f"{total_amount:,.0f}")
    with k4:
        st.metric("Avg amount per user (last month)", f"{avg_amount_per_user:,.2f}")

    st.markdown("---")

    col_top, col_bottom = st.columns([2, 1])

    # Línea de tendencia: usuarios y churn proxy
    with col_top:
        fig_line = px.line(
            metrics_f,
            x="month",
            y=["n_users", "total_trx"],
            markers=True,
            title="User base & transactions over time",
        )
        fig_line.update_layout(
            xaxis_title="Month",
            yaxis_title="Value",
            legend_title="Metric",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_bottom:
        fig_churn = px.bar(
            metrics_f,
            x="month",
            y="churn_proxy",
            title="Churn proxy – month over month",
        )
        fig_churn.update_layout(
            xaxis_title="Month",
            yaxis_title="Churn proxy (%)",
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("---")
    st.markdown("### State comparison")

    df_state = compute_state_metrics(df_f)

    metric_choice = st.selectbox(
        "Metric by state",
        ["n_users", "trx_per_user", "amount_per_user", "churn_proxy"],
        index=0,
    )

    fig_state = px.bar(
        df_state.sort_values(metric_choice, ascending=False),
        x="state",
        y=metric_choice,
        title=f"States by {metric_choice}",
    )
    st.plotly_chart(fig_state, use_container_width=True)


# ==========================================================
# PÁGINA 2 – CHURN & USER RISK (NUEVO DISEÑO)
# ==========================================================

def page_churn_risk():
    st.subheader("2. Churn & User Risk")

    if base_scored is None or base_scored.empty:
        st.info("No se pudo cargar la base de usuarios o el modelo.")
        return

    df = base_scored.copy()

    # Si no hay churn_proba, usamos etiqueta churn si existe
    if "churn_proba" in df.columns and df["churn_proba"].notna().any():
        df["churn_score"] = df["churn_proba"]
        score_col = "churn_proba"
        score_label = "Predicted churn probability"
    elif "churn" in df.columns:
        df["churn_score"] = df["churn"]
        score_col = "churn"
        score_label = "Observed churn label"
    else:
        st.info("No hay probabilidad ni etiqueta de churn para analizar.")
        return

    # Definimos bandas de riesgo
    df["risk_band"] = pd.cut(
        df["churn_score"],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
    )

    # Métricas globales
    total_users = len(df)
    high_risk = (df["risk_band"] == "High").sum()
    medium_risk = (df["risk_band"] == "Medium").sum()

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total users", f"{total_users:,}")
    with k2:
        st.metric("High-risk users", f"{high_risk:,}")
    with k3:
        st.metric("Medium-risk users", f"{medium_risk:,}")

    st.markdown("---")

    col_hist, col_pie = st.columns([2, 1])

    with col_hist:
        fig_hist = px.histogram(
            df,
            x=score_col,
            nbins=30,
            title=f"{score_label} distribution",
        )
        fig_hist.update_layout(
            xaxis_title=score_label,
            yaxis_title="Users",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_pie:
        band_counts = (
            df["risk_band"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
            .reset_index()
        )
        band_counts.columns = ["risk_band", "count"]
        fig_pie = px.pie(
            band_counts,
            names="risk_band",
            values="count",
            title="Users by risk band",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("### Segmentation by tier and risk")

    if "share_tier" in df.columns:
        seg = (
            df.groupby(["share_tier", "risk_band"])
            .size()
            .reset_index(name="users")
        )
        fig_seg = px.bar(
            seg,
            x="share_tier",
            y="users",
            color="risk_band",
            barmode="group",
            title="Users by tier and risk band",
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.info("La base no tiene columna 'share_tier' para segmentar por tier.")


# ==========================================================
# PÁGINA 3 – STRATEGY BY SEGMENT (NUEVO DISEÑO)
# ==========================================================

def page_strategy():
    st.subheader("3. Strategy for Churn Risk Segments")

    if base_with_reasons is None or base_with_reasons.empty:
        st.info("No se pudieron calcular los motivos de churn.")
        return

    df = base_with_reasons.copy()

    # Si no hay churn_proba ni churn, usamos todos
    if "churn_proba" in df.columns and df["churn_proba"].notna().any():
        df_segment = df[df["churn_proba"] >= 0.3].copy()
    elif "churn" in df.columns:
        df_segment = df[df["churn"] == 1].copy()
    else:
        df_segment = df.copy()

    if df_segment.empty:
        st.info("No hay usuarios en riesgo identificados con las reglas actuales.")
        return

    col_motivos, col_tabla = st.columns([1, 1])

    with col_motivos:
        reasons_count = (
            df_segment["churn_reason"]
            .value_counts()
            .reindex(["LLAMADAS", "AMOUNT", "TRANSACTIONS", "CREATION FLOW"])
            .fillna(0)
            .reset_index()
        )
        reasons_count.columns = ["reason", "users"]

        fig_mot = px.bar(
            reasons_count,
            x="reason",
            y="users",
            title="Users at risk by main churn motive",
        )
        st.plotly_chart(fig_mot, use_container_width=True)

    with col_tabla:
        st.markdown("**Suggested actions by motive (template)**")
        st.write(
            "- **LLAMADAS**: revisar tiempo de espera en call center, ofrecer canal digital alterno.\n"
            "- **AMOUNT**: incentivar mayor uso con cashback o promociones específicas.\n"
            "- **TRANSACTIONS**: campañas de reactivación y recordatorios personalizados.\n"
            "- **CREATION FLOW**: simplificar onboarding y mejorar comunicación inicial."
        )

    st.markdown("---")
    st.markdown("### Table of key segments (tier × motive)")

    if "share_tier" in df_segment.columns:
        seg_table = (
            df_segment
            .groupby(["share_tier", "churn_reason"])
            .size()
            .reset_index(name="users")
            .sort_values("users", ascending=False)
        )
        st.dataframe(seg_table, use_container_width=True)
    else:
        st.info("La base no tiene columna 'share_tier'; se omite la tabla por tier.")


# ==========================================================
# NAVEGACIÓN
# ==========================================================

page = st.sidebar.radio(
    "Select page",
    options=[
        "1. Overview",
        "2. Churn & risk",
        "3. Strategy",
    ],
)

if page.startswith("1"):
    page_overview()
elif page.startswith("2"):
    page_churn_risk()
else:
    page_strategy()
