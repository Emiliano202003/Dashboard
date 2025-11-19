# app.py
# -*- coding: utf-8 -*-

import os
import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Danu Churn Dashboard",
    layout="wide"
)

st.title("DanuCard – Churn & Risk Dashboard")

# ------------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_agg_transactions():
    """
    Carga el archivo ya agregado por mes y estado:
    columns esperadas: month, state, total_trx, total_amount, n_users
    """
    df = pd.read_csv("transactions_by_state_month.csv")
    df["month"] = df["month"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_users_base():
    """
    Carga la base de usuarios desde Google Drive.
    Usa la base integrada 'small' que subiste a Drive.
    """
    # Este es SOLO el ID del archivo (sale de la URL de compartir)
    file_id = "1YsiyVjCNO-9ZJx6uAI3AiO3wHE3hBE63"

    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        base = pd.read_csv(base_url)
        return base
    except Exception as e:
        st.error(f"No se pudo cargar la base de usuarios desde Drive: {e}")
        return None




@st.cache_resource(show_spinner=False)
def load_model_and_transformer():
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


agg_tx = load_agg_transactions()
base_integrada = load_users_base()
power_transformer, xgb_model = load_model_and_transformer()

# ------------------------------------------------------------------
# MÉTRICAS A PARTIR DEL ARCHIVO AGREGADO
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_monthly_metrics(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de transactions_by_state_month.csv,
    calcula métricas por mes (sumando todos los estados).
    """
    metrics = (
        agg_df
        .groupby("month", as_index=False)
        .agg(
            n_users=("n_users", "sum"),
            total_trx=("total_trx", "sum"),
            total_amount=("total_amount", "sum")
        )
        .sort_values("month")
    )

    # Crecimientos porcentuales
    metrics["users_growth_pct"] = metrics["n_users"].pct_change() * 100
    metrics["trx_growth_pct"] = metrics["total_trx"].pct_change() * 100

    metrics["amount_per_user"] = (
        metrics["total_amount"] /
        metrics["n_users"].replace(0, np.nan)
    )
    metrics["amount_per_user_growth_pct"] = (
        metrics["amount_per_user"].pct_change() * 100
    )

    # Proxy sencillo de churn: caída de usuarios = churn
    metrics["churn_proxy"] = -metrics["users_growth_pct"]

    return metrics


@st.cache_data(show_spinner=False)
def compute_state_metrics(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Métricas agregadas por estado a partir del archivo de meses/estados.
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

    df["trx_per_user"] = (
        df["total_trx"] / df["n_users"].replace(0, np.nan)
    )
    df["amount_per_user"] = (
        df["total_amount"] / df["n_users"].replace(0, np.nan)
    )

    # Proxy de churn por estado (opcional)
    # mientras más pocos usuarios => "más churn"
    df["churn_proxy"] = -df["n_users"]
    return df


monthly_metrics = compute_monthly_metrics(agg_tx)
state_metrics = compute_state_metrics(agg_tx)

# ------------------------------------------------------------------
# MODELO DE CHURN SOBRE BASE DE USUARIOS
# ------------------------------------------------------------------

FEATURES_CAT = [
    "share_tier", "antiguedad_categoria", "usertype", "gender",
    "occupation", "creationflow"
]

FEATURES_NUM = [
    "total_amount", "total_trx", "antiguedad_cliente_days",
    "llamadas_cc", "minutos_cc", "aht_promedio_cc", "trx_share_global"
]


def score_users_with_model(base, power_tf, model):
    if base is None:
        return pd.DataFrame()

    df = base.copy()

    missing = [c for c in (FEATURES_CAT + FEATURES_NUM) if c not in df.columns]
    if missing or model is None:
        st.info(
            "No se encontraron todas las columnas necesarias o el modelo "
            "no se cargó. Se omiten las predicciones."
        )
        df["churn_proba"] = np.nan
        return df

    df_model = df.dropna(subset=FEATURES_CAT + FEATURES_NUM).copy()

    # Numéricas
    X_num = df_model[FEATURES_NUM].values
    if power_tf is not None:
        try:
            X_num = power_tf.transform(X_num)
        except Exception:
            pass

    # Categóricas
    X_cat = pd.get_dummies(df_model[FEATURES_CAT].astype(str), drop_first=False)
    X = np.concatenate([X_num, X_cat.values], axis=1)

    try:
        proba = model.predict_proba(X)[:, 1]
        df_model["churn_proba"] = proba
    except Exception:
        st.info("El modelo no aceptó el formato de entrada. Se omiten predicciones.")
        df["churn_proba"] = np.nan
        return df

    df = df.merge(df_model[["id_user", "churn_proba"]], on="id_user", how="left")
    return df


base_scored = score_users_with_model(base_integrada, power_transformer, xgb_model)

# Riesgo por días de inactividad
if not base_scored.empty and "max_dias_inactividad" in base_scored.columns:
    base_scored["risk_level"] = pd.cut(
        base_scored["max_dias_inactividad"],
        bins=[-1, 10, 20, 1000],
        labels=["Low", "Medium", "High"]
    )
else:
    if not base_scored.empty:
        st.info("No se encontró max_dias_inactividad; no se calculan niveles de riesgo.")
    if "risk_level" not in base_scored.columns:
        base_scored["risk_level"] = np.nan

# ------------------------------------------------------------------
# ASIGNACIÓN DE MOTIVO DE CHURN
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_churn_reasons(df):
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    if "churn" in data.columns:
        subset = data[data["churn"] == 1].copy()
        if subset.empty:
            subset = data
    else:
        subset = data

    # Percentiles
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

# ------------------------------------------------------------------
# PÁGINA 1 – STATISTICS ON SERVICE / APP USAGE
# ------------------------------------------------------------------

def page_1():
    st.subheader("1. Statistics on Service / App Usage")

    df = agg_tx.copy()

    # Filtros
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        months_sorted = sorted(df["month"].unique())
        month_filter = st.multiselect(
            "Date filter (meses, formato YYYY-MM)",
            options=months_sorted,
            default=months_sorted
        )

    with col_f2:
        states_sorted = sorted(df["state"].dropna().unique())
        states_sorted = [s for s in states_sorted if s != "Unknown"]
        state_filter = st.multiselect(
            "State filter",
            options=states_sorted,
            default=states_sorted
        )

    mask_month = df["month"].isin(month_filter)
    mask_state = df["state"].isin(state_filter)
    df_f = df[mask_month & mask_state].copy()

    if df_f.empty:
        st.info("No hay datos para los filtros seleccionados.")
        return

    metrics_f = compute_monthly_metrics(df_f)

    # KPIs
    churn_increase = metrics_f["churn_proxy"].dropna().mean()
    trx_increase = metrics_f["trx_growth_pct"].dropna().mean()
    amount_increase = metrics_f["amount_per_user_growth_pct"].dropna().mean()

    churn_increase = np.round(churn_increase, 1) if not np.isnan(churn_increase) else 0.0
    trx_increase = np.round(trx_increase, 1) if not np.isnan(trx_increase) else 0.0
    amount_increase = np.round(amount_increase, 1) if not np.isnan(amount_increase) else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Churn Monthly Average Increase (proxy)", f"{churn_increase:.1f}%")
    with c2:
        st.metric("Monthly Transactions Increase Avg", f"{trx_increase:.1f}%")
    with c3:
        st.metric("Monthly increase in avg Amount per user", f"{amount_increase:.1f}%")

    st.markdown("---")

    view_mode = st.radio(
        "View",
        options=["Historical", "Geographical"],
        horizontal=True
    )

    if view_mode == "Historical":
        col_l, col_r = st.columns([2, 1])

        with col_l:
            fig = px.line(
                metrics_f,
                x="month",
                y="n_users",
                markers=True,
                title="User activity – monthly tendency"
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Active users")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if not base_scored.empty and "risk_level" in base_scored.columns:
                dist = (
                    base_scored["risk_level"]
                    .value_counts()
                    .reset_index()
                )
                dist.columns = ["risk_level", "count"]
                fig_pie = px.pie(
                    dist,
                    names="risk_level",
                    values="count",
                    title="Historical client distribution (risk level)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info(
                    "Aquí podrías mostrar el pie de riesgo histórico una vez que "
                    "la base de usuarios cargue correctamente."
                )

    else:
        st.markdown("### Geographical view")

        metric_choice = st.radio(
            "Metric to display",
            options=["Churn proxy", "Amount", "Transactions"],
            horizontal=True
        )

        df_state = compute_state_metrics(df_f)

        if metric_choice == "Churn proxy":
            df_state["value"] = df_state["churn_proxy"]
            value_label = "Churn proxy (lower users = more churn)"
        elif metric_choice == "Amount":
            df_state["value"] = df_state["amount_per_user"]
            value_label = "Average amount per user"
        else:
            df_state["value"] = df_state["trx_per_user"]
            value_label = "Average transactions per user"

        geojson_path = "mexico_geojson.json"
        if os.path.exists(geojson_path):
            with open(geojson_path, "r", encoding="utf-8") as f:
                mexico_geo = json.load(f)

            fig_map = px.choropleth(
                df_state,
                geojson=mexico_geo,
                locations="state",
                featureidkey="properties.state_code",
                color="value",
                projection="mercator",
                title=f"Heatmap by state – {value_label}"
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info(
                "No se encontró 'mexico_geojson.json'. "
                "Se muestra un gráfico de barras por estado."
            )
            fig_bar = px.bar(
                df_state,
                x="state",
                y="value",
                title=f"{value_label} by state"
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ------------------------------------------------------------------
# PÁGINA 2 – USER TENDENCY & USER AT RISK ANALYSIS
# ------------------------------------------------------------------

def page_2():
    st.subheader("2. User Tendency & User at Risk Analysis")

    if base_scored is None or base_scored.empty:
        st.info("No se pudo cargar la base de usuarios para el modelo.")
        return

    # Para la gráfica de tendencia usamos la métrica mensual (proxy de churn)
    tendency = monthly_metrics[["month", "churn_proxy"]].copy()
    tendency.rename(columns={"churn_proxy": "avg_churn_risk"}, inplace=True)

    col_graf, col_cards = st.columns([2, 1])

    with col_graf:
        fig = px.line(
            tendency,
            x="month",
            y="avg_churn_risk",
            markers=True,
            title="User Tendency – churn risk proxy over time"
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Churn risk proxy (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cards:
        total_users = len(base_scored)
        if base_scored["churn_proba"].notna().any():
            prob = base_scored["churn_proba"]
        elif "churn" in base_scored.columns:
            prob = base_scored["churn"]
        else:
            prob = pd.Series(0.0, index=base_scored.index)

        risk_flag = np.where(prob >= 0.3, "At risk", "Safe")
        at_risk = (risk_flag == "At risk").sum()
        safe = (risk_flag == "Safe").sum()

        at_risk_pct = np.round(at_risk / total_users * 100, 1) if total_users > 0 else 0
        safe_pct = np.round(safe / total_users * 100, 1) if total_users > 0 else 0

        st.metric("Current users", total_users)
        st.metric("Safe users", f"{safe_pct} %")
        st.metric("At risk users", f"{at_risk_pct} %")
        st.caption("Rojo: usuarios perdidos, Verde: usuarios ganados (conceptual).")

    st.markdown("---")
    st.markdown("### User at Risk Analysis")

    col_left, col_mid, col_right = st.columns([1, 1, 2])

    with col_left:
        st.markdown("**Client value**")
        if "share_tier" in base_scored.columns:
            tiers = sorted(base_scored["share_tier"].dropna().unique())
            for t in tiers:
                st.markdown(f"- {t}")

    with col_mid:
        st.markdown("**Client risk (por días de inactividad)**")
        st.markdown("- High: 20–42 días")
        st.markdown("- Medium: 10–20 días")
        st.markdown("- Low: 0–10 días")

    with col_right:
        if base_with_reasons is None or base_with_reasons.empty:
            st.info("No se pudieron calcular los motivos de churn.")
        else:
            if "churn" in base_with_reasons.columns:
                data_r = base_with_reasons[base_with_reasons["churn"] == 1]
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


# ------------------------------------------------------------------
# PÁGINA 3 – STRATEGY FOR CHURN RISK BY TIER
# ------------------------------------------------------------------

def page_3():
    st.subheader("3. Strategy for Churn Risk by Tier")

    if base_scored is None or base_scored.empty:
        st.info("No se pudo cargar la base de usuarios para el modelo.")
        return

    if "share_tier" not in base_scored.columns:
        st.info("No existe la columna share_tier en la base.")
        return

    tiers = sorted(base_scored["share_tier"].dropna().unique())
    selected_tier = st.selectbox("Tier filter", options=tiers)

    st.markdown("#### Strategies by churn motive")

    col1, col2 = st.columns(2)
    motives = ["LLAMADAS", "TRANSACTIONS", "AMOUNT", "CREATION FLOW"]

    with col1:
        for motive in motives[:2]:
            with st.expander(f"Strategy for {motive.title()} – click to view"):
                st.write(
                    f"Texto futuro a insertar para **Tier {selected_tier}** "
                    f"y motivo **{motive}**."
                )

    with col2:
        for motive in motives[2:]:
            with st.expander(f"Strategy for {motive.title()} – click to view"):
                st.write(
                    f"Texto futuro a insertar para **Tier {selected_tier}** "
                    f"y motivo **{motive}**."
                )

    st.markdown("#### All together")
    with st.expander("All motives and all tiers"):
        st.write(
            "Texto futuro a insertar con las 12 soluciones "
            "(4 motivos × 3 tiers)."
        )

    st.markdown("---")
    st.markdown("### Churn tendency and impact of strategies")

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


# ------------------------------------------------------------------
# NAVEGACIÓN
# ------------------------------------------------------------------

page = st.sidebar.radio(
    "Selecciona la página",
    options=[
        "1. Service / App usage",
        "2. User tendency & risk",
        "3. Strategy by tier"
    ]
)

if page.startswith("1"):
    page_1()
elif page.startswith("2"):
    page_2()
else:
    page_3()





