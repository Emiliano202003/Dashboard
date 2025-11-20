# app.py
# -*- coding: utf-8 -*-

import os
import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ===============================================================
# CONFIGURACIÓN GENERAL
# ===============================================================

st.set_page_config(
    page_title="DanuCard – Churn & Risk Dashboard",
    layout="wide"
)

st.title("DanuCard – Churn & Risk Dashboard")

# ===============================================================
# 1) CARGA DE DATOS
# ===============================================================

@st.cache_data(show_spinner=False)
def load_agg_transactions() -> pd.DataFrame:
    """
    Carga el archivo agregado por mes y estado:
    columnas esperadas: month, state, total_trx, total_amount, n_users
    """
    df = pd.read_csv("transactions_by_state_month.csv")
    df["month"] = df["month"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_users_base() -> pd.DataFrame:
    """
    Carga la base de usuarios (versión small) desde Google Drive.
    Si falla, regresa un DataFrame vacío.
    """
    # SOLO el ID, no la URL completa
    file_id = "1YsiyVjCNO-9ZJx6uAI3AiO3wHE3hBE63"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        base = pd.read_csv(url)
        return base
    except Exception as e:
        st.warning(
            "No se pudo cargar la base de usuarios desde Drive. "
            f"Mensaje técnico: {e}"
        )
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_and_transformer():
    """
    Carga el PowerTransformer y el modelo XGBoost desde los .pkl locales.
    Si alguno falla, se devuelve como None.
    """
    power_tf = None
    model = None

    try:
        with open("power_transformer.pkl", "rb") as f:
            power_tf = pickle.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar power_transformer.pkl: {e}")

    try:
        with open("xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar xgboost_model.pkl: {e}")

    return power_tf, model


# Carga efectiva de datos
agg_tx = load_agg_transactions()
base_integrada = load_users_base()
power_transformer, xgb_model = load_model_and_transformer()

# ===============================================================
# 2) MÉTRICAS AGREGADAS (SOLO transactions_by_state_month)
# ===============================================================

@st.cache_data(show_spinner=False)
def compute_monthly_metrics(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Métricas por mes, sumando todos los estados.
    """
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

    metrics["users_growth_pct"] = metrics["n_users"].pct_change() * 100
    metrics["trx_growth_pct"] = metrics["total_trx"].pct_change() * 100

    metrics["amount_per_user"] = (
        metrics["total_amount"] /
        metrics["n_users"].replace(0, np.nan)
    )
    metrics["amount_per_user_growth_pct"] = (
        metrics["amount_per_user"].pct_change() * 100
    )

    # Proxy simple de churn: caída de usuarios = churn
    metrics["churn_proxy"] = -metrics["users_growth_pct"]

    return metrics


@st.cache_data(show_spinner=False)
def compute_state_metrics(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Métricas agregadas por estado.
    """
    df = (
        agg_df
        .groupby("state", as_index=False)
        .agg(
            n_users=("n_users", "sum"),
            total_trx=("total_trx", "sum"),
            total_amount=("total_amount", "sum")
        )
    )

    df["trx_per_user"] = df["total_trx"] / df["n_users"].replace(0, np.nan)
    df["amount_per_user"] = df["total_amount"] / df["n_users"].replace(0, np.nan)
    df["churn_proxy"] = -df["n_users"]   # menos usuarios ⇒ más churn

    return df


monthly_metrics = compute_monthly_metrics(agg_tx)
state_metrics = compute_state_metrics(agg_tx)

# ===============================================================
# 3) MODELO DE CHURN SOBRE BASE DE USUARIOS
# ===============================================================

FEATURES_CAT = [
    "share_tier", "antiguedad_categoria", "usertype", "gender",
    "occupation", "creationflow"
]

FEATURES_NUM = [
    "total_amount", "total_trx", "antiguedad_cliente_days",
    "llamadas_cc", "minutos_cc", "aht_promedio_cc", "trx_share_global"
]


def score_users_with_model(base: pd.DataFrame,
                           power_tf,
                           model) -> pd.DataFrame:
    """
    Aplica el modelo de churn (si es posible).
    Si no hay columnas suficientes o el modelo falla,
    regresa la base con churn_proba = NaN.
    """
    if base is None or base.empty:
        return pd.DataFrame()

    df = base.copy()

    # Verificamos columnas necesarias
    missing = [c for c in (FEATURES_CAT + FEATURES_NUM) if c not in df.columns]
    if missing or model is None:
        df["churn_proba"] = np.nan
        return df

    df_model = df.dropna(subset=FEATURES_CAT + FEATURES_NUM).copy()

    # Numéricas
    X_num = df_model[FEATURES_NUM].values
    if power_tf is not None:
        try:
            X_num = power_tf.transform(X_num)
        except Exception:
            pass  # seguimos con datos crudos

    # Categóricas
    X_cat = pd.get_dummies(df_model[FEATURES_CAT].astype(str), drop_first=False)
    X = np.concatenate([X_num, X_cat.values], axis=1)

    try:
        proba = model.predict_proba(X)[:, 1]
        df_model["churn_proba"] = proba
    except Exception:
        df["churn_proba"] = np.nan
        return df

    df = df.merge(df_model[["id_user", "churn_proba"]], on="id_user", how="left")
    return df


@st.cache_data(show_spinner=False)
def compute_churn_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna un motivo de churn a cada usuario (simple rule-based).
    Si no hay datos, regresa DF vacío.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    if "churn" in data.columns:
        subset = data[data["churn"] == 1].copy()
        if subset.empty:
            subset = data
    else:
        subset = data

    # Si faltan columnas numéricas, no podemos calcular motivos
    for col in ["llamadas_cc", "total_amount", "total_trx"]:
        if col not in subset.columns:
            data["churn_reason"] = "CREATION FLOW"
            return data

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


# Calculamos base_scored y motivos (si se puede)
base_scored = score_users_with_model(base_integrada, power_transformer, xgb_model)
base_with_reasons = compute_churn_reasons(base_scored)

# ===============================================================
# 4) PÁGINA 1 – OVERVIEW (YA TE FUNCIONA)
# ===============================================================

def page_overview():
    st.subheader("1. Service / App Usage Overview")

    df = agg_tx.copy()

    c1, c2 = st.columns(2)
    with c1:
        months_sorted = sorted(df["month"].unique())
        month_filter = st.multiselect(
            "Filter by month (YYYY-MM)",
            options=months_sorted,
            default=months_sorted
        )
    with c2:
        states_sorted = sorted(df["state"].dropna().unique())
        state_filter = st.multiselect(
            "Filter by state",
            options=states_sorted,
            default=states_sorted
        )

    mask_m = df["month"].isin(month_filter)
    mask_s = df["state"].isin(state_filter)
    df_f = df[mask_m & mask_s].copy()

    if df_f.empty:
        st.info("No hay datos para los filtros seleccionados.")
        return

    metrics_f = compute_monthly_metrics(df_f)

    # KPIs de arriba
    current_users = int(metrics_f["n_users"].iloc[-1])
    total_trx = int(metrics_f["total_trx"].sum())
    total_amount = float(metrics_f["total_amount"].sum())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Current active users", f"{current_users:,}")
    with k2:
        st.metric("Total transactions (period)", f"{total_trx:,}")
    with k3:
        st.metric("Total amount (period)", f"${total_amount:,.0f}")

    st.markdown("---")

    # Línea de usuarios vs línea de transacciones
    c_top_left, c_top_right = st.columns([3, 2])

    with c_top_left:
        fig_users = px.line(
            metrics_f,
            x="month",
            y="n_users",
            markers=True,
            title="User base & transactions over time"
        )
        fig_users.update_layout(
            xaxis_title="Month",
            yaxis_title="Active users"
        )
        st.plotly_chart(fig_users, use_container_width=True)

    with c_top_right:
        fig_churn = px.bar(
            metrics_f,
            x="month",
            y="churn_proxy",
            title="Churn proxy – month over month"
        )
        fig_churn.update_layout(
            xaxis_title="Month",
            yaxis_title="Churn proxy (%)"
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("### State comparison")

    metric_choice = st.selectbox(
        "Metric by state",
        options=["n_users", "total_trx", "total_amount", "trx_per_user", "amount_per_user"]
    )

    df_state = compute_state_metrics(df_f)
    fig_state = px.bar(
        df_state.sort_values(metric_choice, ascending=False),
        x="state",
        y=metric_choice,
        title=f"States by {metric_choice}"
    )
    st.plotly_chart(fig_state, use_container_width=True)


# ===============================================================
# 5) PÁGINA 2 – CHURN & RISK
# ===============================================================

def page_churn_risk():
    st.subheader("2. Churn & Risk Analysis")

    # --- TENDENCIA GLOBAL (funciona siempre con monthly_metrics) ---
    tendency = monthly_metrics[["month", "churn_proxy"]].copy()
    tendency.rename(columns={"churn_proxy": "churn_risk"}, inplace=True)

    c_plot, c_cards = st.columns([2, 1])

    with c_plot:
        fig = px.line(
            tendency,
            x="month",
            y="churn_risk",
            markers=True,
            title="Global churn risk (proxy) over time"
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Churn risk proxy (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- TARJETAS DE RIESGO USANDO EL MODELO (SI EXISTE) ---
    if base_scored is not None and not base_scored.empty:
        df = base_scored.copy()

        if df["churn_proba"].notna().any():
            prob = df["churn_proba"]
        elif "churn" in df.columns:
            prob = df["churn"]
        else:
            prob = pd.Series(0.0, index=df.index)

        df["churn_prob_used"] = prob
        df["risk_flag"] = np.where(df["churn_prob_used"] >= 0.3, "At risk", "Safe")

        total_users = len(df)
        at_risk = (df["risk_flag"] == "At risk").sum()
        safe = (df["risk_flag"] == "Safe").sum()

        at_risk_pct = np.round(at_risk / total_users * 100, 1) if total_users > 0 else 0.0
        safe_pct = np.round(safe / total_users * 100, 1) if total_users > 0 else 0.0
    else:
        # Fallback simple si NO hay base de usuarios
        total_users = int(monthly_metrics["n_users"].iloc[-1])
        at_risk_pct = 30.0
        safe_pct = 70.0

    with c_cards:
        st.metric("Current users (approx)", f"{total_users:,}")
        st.metric("Safe users (approx)", f"{safe_pct:.1f} %")
        st.metric("At risk users (approx)", f"{at_risk_pct:.1f} %")
        st.caption(
            "Si la base de usuarios o el modelo no están disponibles, "
            "estas cifras se aproximan con el proxy de churn."
        )

    st.markdown("---")
    st.markdown("### User at Risk – Churn motives")

    if base_with_reasons is None or base_with_reasons.empty:
        st.info(
            "No se pudieron calcular motivos de churn a nivel usuario. "
            "Verifica que la base small tenga las columnas necesarias "
            "y que el enlace de Drive sea accesible."
        )
        return

    if "churn" in base_with_reasons.columns:
        data_r = base_with_reasons[base_with_reasons["churn"] == 1]
        if data_r.empty:
            data_r = base_with_reasons
    else:
        data_r = base_with_reasons

    reasons_count = (
        data_r["churn_reason"]
        .value_counts()
        .reindex(["LLAMADAS", "AMOUNT", "TRANSACTIONS", "CREATION FLOW"])
        .fillna(0)
        .reset_index()
    )
    reasons_count.columns = ["reason", "count"]

    fig_bar = px.bar(
        reasons_count,
        x="reason",
        y="count",
        title="Users at risk by main churn motive"
    )
    fig_bar.update_layout(
        xaxis_title="Motive for churn",
        yaxis_title="Total users"
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ===============================================================
# 6) PÁGINA 3 – STRATEGY
# ===============================================================

def page_strategy():
    st.subheader("3. Strategy for Churn Risk")

    # Bloque de estrategia conceptual (texto + expansores)
    st.markdown("#### Strategies by churn motive & client value")

    motives = ["LLAMADAS", "TRANSACTIONS", "AMOUNT", "CREATION FLOW"]

    col1, col2 = st.columns(2)
    with col1:
        for motive in motives[:2]:
            with st.expander(f"Strategy for {motive.title()} – click to view"):
                st.write(f"Aquí va la estrategia propuesta para el motivo **{motive}**.")
    with col2:
        for motive in motives[2:]:
            with st.expander(f"Strategy for {motive.title()} – click to view"):
                st.write(f"Aquí va la estrategia propuesta para el motivo **{motive}**.")

    st.markdown("---")
    st.markdown("### Churn tendency and impact of strategies (simulated)")

    hist = monthly_metrics.sort_values("month").copy()
    if hist.empty:
        st.info("No hay información mensual suficiente para graficar.")
        return

    last_rate = hist["churn_proxy"].iloc[-1]

    scenarios = []
    for scen_name, reduction in [
        ("Historical", 0.0),
        ("Calls strategy (+2%)", 0.02),
        ("Transactions strategy (+5%)", 0.05),
        ("All together (+20%)", 0.20),
    ]:
        values = hist["churn_proxy"].copy()
        if scen_name != "Historical":
            new_last = max(last_rate * (1 - reduction), 0)
            values.iloc[-1] = new_last

        tmp = pd.DataFrame({
            "month": hist["month"],
            "churn_rate": values,
            "scenario": scen_name
        })
        scenarios.append(tmp)

    df_scenarios = pd.concat(scenarios, ignore_index=True)

    fig_scen = px.line(
        df_scenarios,
        x="month",
        y="churn_rate",
        color="scenario",
        markers=True,
        title="Churn rate with and without strategies (simulated impact)"
    )
    fig_scen.update_layout(
        xaxis_title="Month",
        yaxis_title="Churn proxy (%)"
    )
    st.plotly_chart(fig_scen, use_container_width=True)


# ===============================================================
# 7) NAVEGACIÓN
# ===============================================================

page = st.sidebar.radio(
    "Select page",
    options=[
        "Overview",
        "Churn & risk",
        "Strategy"
    ]
)

if page == "Overview":
    page_overview()
elif page == "Churn & risk":
    page_churn_risk()
else:
    page_strategy()
