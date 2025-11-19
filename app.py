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
def load_page1_data():
    """
    Datos agregados por mes y estado para la página 1.
    Debe existir el archivo:
      - transactions_by_state_month.csv
    con columnas: state, month, total_trx, total_amount, n_users
    """
    df = pd.read_csv("transactions_by_state_month.csv")
    df["month"] = df["month"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_user_base():
    """
    Base integrada reducida para el modelo (páginas 2 y 3).
    Debe existir el archivo:
      - base_integrada_small.csv
    """
    try:
        base = pd.read_csv("base_integrada_small.csv")
        return base
    except Exception as e:
        st.error(f"No se pudo cargar 'base_integrada_small.csv': {e}")
        return pd.DataFrame()


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

# ------------------------------------------------------------------
# FUNCIONES DE MODELO
# ------------------------------------------------------------------

FEATURES_CAT = [
    "share_tier", "antiguedad_categoria", "usertype", "gender",
    "occupation", "creationflow"
]

FEATURES_NUM = [
    "total_amount", "total_trx", "antiguedad_cliente_days",
    "llamadas_cc", "minutos_cc", "aht_promedio_cc", "trx_share_global"
]


@st.cache_data(show_spinner=False)
def score_users_with_model(base, power_tf, model):
    df = base.copy()

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
            pass  # si truena el transformer, seguimos con datos crudos

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
def compute_churn_reasons(df):
    data = df.copy()

    if "churn" not in data.columns:
        subset = data
    else:
        subset = data[data["churn"] == 1].copy()

    # Por si faltan columnas numéricas
    for col in ["llamadas_cc", "total_amount", "total_trx"]:
        if col not in subset.columns:
            subset[col] = 0

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


@st.cache_data(show_spinner=False)
def get_scored_base():
    """
    Carga base_integrada_small, aplica modelo, asigna risk_level
    y motivos de churn.
    """
    base = load_user_base()
    if base.empty:
        return base, base  # dos vacíos

    power_tf, model = load_model_and_transformer()
    base_scored = score_users_with_model(base, power_tf, model)

    # Risk level por días de inactividad
    if "max_dias_inactividad" in base_scored.columns:
        base_scored["risk_level"] = pd.cut(
            base_scored["max_dias_inactividad"],
            bins=[-1, 10, 20, 1000],
            labels=["Low", "Medium", "High"]
        )
    else:
        base_scored["risk_level"] = np.nan

    base_with_reasons = compute_churn_reasons(base_scored)

    return base_scored, base_with_reasons

# ------------------------------------------------------------------
# FUNCIONES PARA RESÚMENES MENSUALES (P1 y P3)
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_monthly_from_agg(df):
    """
    A partir de transactions_by_state_month.csv (state, month, total_trx, total_amount, n_users)
    devuelve un resumen mensual global.
    """
    monthly = (
        df.groupby("month", as_index=False)
          .agg(
              n_users=("n_users", "sum"),
              total_trx=("total_trx", "sum"),
              total_amount=("total_amount", "sum")
          )
          .sort_values("month")
    )

    if monthly.empty:
        return monthly

    # Aproximación de churn: caída de usuarios como proxy
    monthly["users_growth_pct"] = monthly["n_users"].pct_change() * 100
    # Lo usamos como "churn_rate_proxy": entre 0 y 100 aprox
    churn_rate = -monthly["users_growth_pct"].clip(lower=-100, upper=100).fillna(0)
    monthly["churn_rate"] = churn_rate

    return monthly

# ------------------------------------------------------------------
# PÁGINA 1 – STATISTICS ON SERVICE / APP USAGE
# ------------------------------------------------------------------

def page_1():
    st.subheader("1. Statistics on Service / App Usage")

    # 1) Datos agregados por mes y estado
    df = load_page1_data()  # transactions_by_state_month.csv

    # 2) Filtros
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

    # 3) Métricas por mes
    metrics_f = (
        df_f.groupby("month", as_index=False)
            .agg(
                n_users=("n_users", "sum"),
                total_trx=("total_trx", "sum"),
                total_amount=("total_amount", "sum")
            )
            .sort_values("month")
    )

    if metrics_f.empty:
        st.info("No hay datos para los filtros seleccionados.")
        return

    metrics_f["users_growth_pct"] = metrics_f["n_users"].pct_change() * 100
    churn_increase = (-metrics_f["users_growth_pct"]).dropna().mean()

    metrics_f["trx_growth_pct"] = metrics_f["total_trx"].pct_change() * 100
    trx_increase = metrics_f["trx_growth_pct"].dropna().mean()

    metrics_f["amount_per_user"] = metrics_f["total_amount"] / metrics_f["n_users"].replace(0, np.nan)
    metrics_f["amount_per_user_growth_pct"] = metrics_f["amount_per_user"].pct_change() * 100
    amount_increase = metrics_f["amount_per_user_growth_pct"].dropna().mean()

    churn_increase = np.round(churn_increase, 1) if not np.isnan(churn_increase) else 0.0
    trx_increase = np.round(trx_increase, 1) if not np.isnan(trx_increase) else 0.0
    amount_increase = np.round(amount_increase, 1) if not np.isnan(amount_increase) else 0.0

    # 4) Cards KPI
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Churn Monthly Average Increase", f"{churn_increase}%")
    with c2:
        st.metric("Monthly Transactions Increase Avg", f"{trx_increase}%")
    with c3:
        st.metric("Monthly increase in avg Amount per user", f"{amount_increase}%")

    st.markdown("---")

    # 5) Historical vs Geographical
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
            st.info(
                "Aquí podría ir el pie chart de riesgo histórico usando la base de usuarios "
                "(se calcula en la página 2 con el modelo)."
            )

    else:
        st.markdown("### Geographical view")

        metric_choice = st.radio(
            "Metric to display",
            options=["Churn proxy", "Amount", "Transactions"],
            horizontal=True
        )

        df_state = (
            df_f.groupby("state", as_index=False)
                .agg(
                    n_users=("n_users", "sum"),
                    total_trx=("total_trx", "sum"),
                    total_amount=("total_amount", "sum")
                )
        )

        df_state["trx_per_user"] = df_state["total_trx"] / df_state["n_users"].replace(0, np.nan)
        df_state["amount_per_user"] = df_state["total_amount"] / df_state["n_users"].replace(0, np.nan)

        if metric_choice == "Churn proxy":
            df_state["value"] = -df_state["n_users"]
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
                "No se encontró el archivo 'mexico_geojson.json'. "
                "Se muestra un gráfico de barras por estado en lugar del mapa."
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

    base_scored, base_with_reasons = get_scored_base()
    if base_scored.empty:
        st.info("No se pudo cargar la base de usuarios para el modelo.")
        return

    # Si no tenemos probabilidad, usamos churn real o ceros
    if base_scored["churn_proba"].notna().any():
        prob = base_scored["churn_proba"]
    elif "churn" in base_scored.columns:
        prob = base_scored["churn"]
    else:
        prob = pd.Series(0.0, index=base_scored.index)

    base_scored["churn_prob_used"] = prob

    # "Mes" proxy basado en antigüedad del cliente
    if "antiguedad_cliente_days" in base_scored.columns:
        base_scored["tenure_month"] = (
            (base_scored["antiguedad_cliente_days"] / 30)
            .round()
            .astype(int)
        )
    else:
        base_scored["tenure_month"] = 0

    scored_valid = base_scored.dropna(subset=["tenure_month"])

    tendency = (
        scored_valid
        .groupby("tenure_month")["churn_prob_used"]
        .mean()
        .reset_index()
        .rename(columns={"churn_prob_used": "avg_churn_prob"})
        .sort_values("tenure_month")
    )

    col_graf, col_cards = st.columns([2, 1])

    with col_graf:
        fig = px.line(
            tendency,
            x="tenure_month",
            y="avg_churn_prob",
            markers=True,
            title="User Tendency – average churn probability vs. client tenure (months)"
        )
        fig.update_layout(
            xaxis_title="Client tenure (months, approx.)",
            yaxis_title="Average churn probability"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cards:
        total_users = len(scored_valid)
        current_users = total_users
        at_risk = (scored_valid["churn_prob_used"] >= 0.3).sum()
        safe = (scored_valid["churn_prob_used"] < 0.3).sum()

        at_risk_pct = np.round(at_risk / total_users * 100, 1) if total_users > 0 else 0
        safe_pct = np.round(safe / total_users * 100, 1) if total_users > 0 else 0

        st.metric("Current users", current_users)
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
        else:
            st.info("No se encontró la columna 'share_tier'.")

    with col_mid:
        st.markdown("**Client risk (por días de inactividad)**")
        st.markdown("- High: 20–42 días")
        st.markdown("- Medium: 10–20 días")
        st.markdown("- Low: 0–10 días")

    with col_right:
        # Bar de motivos de churn
        if "churn" in base_with_reasons.columns:
            data_r = base_with_reasons[base_with_reasons["churn"] == 1]
        else:
            data_r = base_with_reasons

        if "churn_reason" not in data_r.columns:
            st.info("No se pudo calcular churn_reason.")
        else:
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

    base_scored, base_with_reasons = get_scored_base()
    if base_scored.empty:
        st.info("No se pudo cargar la base de usuarios para el modelo.")
        return

    # Filtro de Tier
    if "share_tier" not in base_scored.columns:
        st.info("No se encontró la columna 'share_tier' en la base de usuarios.")
        return

    tiers = sorted(base_scored["share_tier"].dropna().unique())
    selected_tier = st.selectbox("Tier filter", options=tiers)

    st.markdown("#### Strategies by churn motive")

    col1, col2 = st.columns(2)
    motives = ["LLAMADAS", "TRANSACTIONS", "AMOUNT", "CREATION FLOW"]

    with col1:
        for motive in motives[:2]:
            with st.expander(f"Strategy for {motive.title()} – click to view"):
                st.write(f"Texto futuro a insertar para **Tier {selected_tier}** y motivo **{motive}**.")

    with col2:
        for motive in motives[2:]:
            with st.expander(f"Strategy for {motive.title()} – click to view"):
                st.write(f"Texto futuro a insertar para **Tier {selected_tier}** y motivo **{motive}**.")

    st.markdown("#### All together")
    with st.expander("All motives and all tiers"):
        st.write(
            "Texto futuro a insertar con las 12 soluciones "
            "(4 motivos × 3 tiers)."
        )

    st.markdown("---")
    st.markdown("### Churn tendency and impact of strategies")

    # Usamos el resumen mensual global de la página 1 como base
    df_agg = load_page1_data()
    hist = compute_monthly_from_agg(df_agg)
    if hist.empty:
        st.info("No hay información mensual suficiente para graficar.")
        return

    last_rate = hist["churn_rate"].iloc[-1]

    scenarios = []
    for scen_name, reduction in [
        ("Historical", 0.0),
        ("Calls strategy (+2%)", 0.02),
        ("Transactions strategy (+5%)", 0.05),
        ("All together (+20%)", 0.20)
    ]:
        values = hist["churn_rate"].copy()
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
        yaxis_title="Churn rate (proxy %)"
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
