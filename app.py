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
# CARGA DE DATOS Y MODELOS
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    base_url = "https://drive.google.com/uc?export=download&id=14a3S4LtFiG7j6pw1QtFWGxg1hx4bQvnz"
    trans_url = "https://drive.google.com/uc?export=download&id=11S9-SZCMF30LGyMWz4nexjIW8Ltdi1oo"

    base = pd.read_csv(base_url)
    trans = pd.read_csv(trans_url, parse_dates=["fechaf"])

    return base, trans



@st.cache_resource(show_spinner=False)
def load_model_and_transformer():
    power_tf = None
    model = None
    # Estos try/except son para que la app no se caiga si algo falla
    try:
        with open("power_transformer.pkl", "rb") as f:
            power_tf = pickle.load(f)
    except Exception as e:
        print("No se pudo cargar power_transformer.pkl:", e)

    try:
        with open("xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print("No se pudo cargar xgboost_model.pkl:", e)

    return power_tf, model


base_integrada, transacciones = load_data()
power_transformer, xgb_model = load_model_and_transformer()

# ------------------------------------------------------------------
# PREPARACIÓN DE DATOS COMUNES
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_transactions_with_month_and_state(base, trans):
    df = trans.copy()
    df["fechaf"] = pd.to_datetime(df["fechaf"])
    df["month"] = df["fechaf"].dt.to_period("M").astype(str)

    # Unimos info de estado (y churn si existe) por usuario
    merge_cols = ["id_user", "state"]
    if "churn" in base.columns:
        merge_cols.append("churn")

    df = df.merge(base[merge_cols].drop_duplicates("id_user"),
                  on="id_user", how="left")
    return df


@st.cache_data(show_spinner=False)
def compute_monthly_metrics(trans):
    # Usuarios únicos por mes
    users_month = trans.groupby("month")["id_user"].nunique()

    # Churn por mes (si existe la columna)
    if "churn" in trans.columns:
        churned = (
            trans[trans["churn"] == 1]
            .groupby("month")["id_user"]
            .nunique()
        )
        churn_rate = (churned / users_month * 100).reindex(users_month.index).fillna(0)
    else:
        churn_rate = pd.Series(0, index=users_month.index)

    monthly_trx = trans.groupby("month")["trnx"].sum()
    monthly_amount = trans.groupby("month")["amount"].sum()

    df = pd.DataFrame({
        "month": users_month.index,
        "active_users": users_month.values,
        "churn_rate": churn_rate.values,
        "total_trx": monthly_trx.reindex(users_month.index, fill_value=0).values,
        "total_amount": monthly_amount.reindex(users_month.index, fill_value=0).values
    })
    return df


@st.cache_data(show_spinner=False)
def compute_state_metrics(trans):
    # Métricas a nivel estado
    group = trans.groupby("state")
    users_state = group["id_user"].nunique()
    total_trx_state = group["trnx"].sum()
    total_amount_state = group["amount"].sum()

    if "churn" in trans.columns:
        churned_state = (
            trans[trans["churn"] == 1]
            .groupby("state")["id_user"]
            .nunique()
        )
        churn_rate_state = (churned_state / users_state * 100).reindex(users_state.index).fillna(0)
    else:
        churn_rate_state = pd.Series(0, index=users_state.index)

    df = pd.DataFrame({
        "state": users_state.index,
        "users": users_state.values,
        "total_trx": total_trx_state.reindex(users_state.index, fill_value=0).values,
        "total_amount": total_amount_state.reindex(users_state.index, fill_value=0).values,
        "churn_rate": churn_rate_state.values
    })

    # Promedios por usuario
    df["trx_per_user"] = df["total_trx"] / df["users"]
    df["amount_per_user"] = df["total_amount"] / df["users"]
    return df


trans_full = prepare_transactions_with_month_and_state(base_integrada, transacciones)
monthly_metrics = compute_monthly_metrics(trans_full)
state_metrics = compute_state_metrics(trans_full)

# ------------------------------------------------------------------
# FUNCIONES PARA MODELO (PARA PÁGINA 2)
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

    # Comprobamos que estén todas las columnas
    missing = [c for c in (FEATURES_CAT + FEATURES_NUM) if c not in df.columns]
    if missing or model is None:
        # Si faltan columnas o el modelo no cargó, devolvemos el DF sin predicción
        df["churn_proba"] = np.nan
        return df

    # Quitamos filas con NA en las features
    df_model = df.dropna(subset=FEATURES_CAT + FEATURES_NUM).copy()

    # Numéricas
    X_num = df_model[FEATURES_NUM].values
    if power_tf is not None:
        try:
            X_num = power_tf.transform(X_num)
        except Exception:
            # Si el transformer no funciona, seguimos con los datos crudos
            pass

    # Categóricas – one-hot simple
    X_cat = pd.get_dummies(df_model[FEATURES_CAT].astype(str), drop_first=False)

    # Unimos numéricas + categóricas
    X = np.concatenate([X_num, X_cat.values], axis=1)

    # Predicción de probabilidad de churn (clase positiva)
    try:
        proba = model.predict_proba(X)[:, 1]
        df_model["churn_proba"] = proba
    except Exception:
        # Si el modelo no acepta este formato, mejor no romper la app
        df["churn_proba"] = np.nan
        return df

    # Unimos de vuelta al DF original
    df = df.merge(df_model[["id_user", "churn_proba"]], on="id_user", how="left")
    return df


base_scored = score_users_with_model(base_integrada, power_transformer, xgb_model)

# Categoría de riesgo por días de inactividad
if "max_dias_inactividad" in base_scored.columns:
    base_scored["risk_level"] = pd.cut(
        base_scored["max_dias_inactividad"],
        bins=[-1, 10, 20, 1000],
        labels=["Low", "Medium", "High"]
    )
else:
    base_scored["risk_level"] = np.nan

# ------------------------------------------------------------------
# ASIGNACIÓN DE MOTIVO DE CHURN (PÁGINA 2 Y 3)
# ------------------------------------------------------------------

def assign_churn_reason(row,
                        p_llamadas=0.75,
                        p_amount=0.25,
                        p_trx=0.25):
    """
    Asigna un motivo principal de churn para cada usuario.
    Regla sencilla:
    - "LLAMADAS": muchas llamadas a call center (por encima del p_llamadas percentil)
    - "AMOUNT": mueve poco dinero (por debajo del p_amount percentil)
    - "TRANSACTIONS": pocas transacciones (por debajo del p_trx percentil)
    - "CREATION FLOW": caso residual
    """
    return row["reason_tmp"]  # placeholder – se sobreescribe abajo


@st.cache_data(show_spinner=False)
def compute_churn_reasons(df):
    data = df.copy()

    if "churn" not in data.columns:
        # Si no tenemos etiqueta de churn, tomamos todos los usuarios como referencia
        subset = data
    else:
        subset = data[data["churn"] == 1].copy()

    # Percentiles para las reglas
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

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        months_sorted = sorted(monthly_metrics["month"].unique())
        month_filter = st.multiselect(
            "Date filter (meses, formato YYYY-MM)",
            options=months_sorted,
            default=months_sorted
        )

    with col_f2:
        states_sorted = sorted(base_integrada["state"].dropna().unique())
        state_filter = st.multiselect(
            "State filter",
            options=states_sorted,
            default=states_sorted
        )

    # Aplicamos filtros a transacciones
    mask_month = trans_full["month"].isin(month_filter)
    mask_state = trans_full["state"].isin(state_filter)
    trans_f = trans_full[mask_month & mask_state]

    metrics_f = compute_monthly_metrics(trans_f)

    # KPIs para las tarjetas
    metrics_f = metrics_f.sort_values("month")
    # Evitamos divisiones por cero / NaN
    churn_increase = metrics_f["churn_rate"].pct_change().dropna().mean() * 100
    trx_increase = metrics_f["total_trx"].pct_change().dropna().mean() * 100
    amount_increase = (
        (metrics_f["total_amount"] / metrics_f["active_users"])
        .replace([np.inf, -np.inf], np.nan)
        .pct_change()
        .dropna()
        .mean() * 100
    )

    churn_increase = np.round(churn_increase, 1) if not np.isnan(churn_increase) else 0
    trx_increase = np.round(trx_increase, 1) if not np.isnan(trx_increase) else 0
    amount_increase = np.round(amount_increase, 1) if not np.isnan(amount_increase) else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Churn Monthly Average Increase", f"{churn_increase}%")
    with c2:
        st.metric("Monthly Transactions Increase Avg", f"{trx_increase}%")
    with c3:
        st.metric("Monthly increase in avg Amount per user", f"{amount_increase}%")

    st.markdown("---")

    view_mode = st.radio(
        "View",
        options=["Historical", "Geographical"],
        horizontal=True
    )

    if view_mode == "Historical":
        # Parte izquierda: línea de churn
        col_l, col_r = st.columns([2, 1])

        with col_l:
            fig = px.line(
                metrics_f,
                x="month",
                y="churn_rate",
                markers=True,
                title="Churn – Monthly tendency"
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # Pie de distribución histórica de clientes por nivel de riesgo
            if "risk_level" in base_scored.columns:
                dist = base_scored["risk_level"].value_counts().reset_index()
                dist.columns = ["risk_level", "count"]
                fig_pie = px.pie(
                    dist,
                    names="risk_level",
                    values="count",
                    title="Historical client distribution (risk level)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No se encontró la columna 'risk_level' para el pie de distribución.")

    else:
        # Vista geográfica / por estado
        st.markdown("### Geographical view")

        metric_choice = st.radio(
            "Metric to display on the map",
            options=["Churn", "Amount", "Transactions"],
            horizontal=True
        )

        df_state = state_metrics.copy()
        if metric_choice == "Churn":
            df_state["value"] = df_state["churn_rate"]
            value_label = "Churn rate (%)"
        elif metric_choice == "Amount":
            df_state["value"] = df_state["amount_per_user"]
            value_label = "Average amount per user"
        else:
            df_state["value"] = df_state["trx_per_user"]
            value_label = "Average transactions per user"

        # Intentamos usar un geojson de México si existe
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
                "De momento se muestra un gráfico de barras por estado en lugar del mapa."
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

    # Para la parte de User Tendency vamos a usar la predicción del modelo (si existe)
    # y el mes de última transacción por usuario.
    last_trx = (
        transacciones
        .groupby("id_user")["fechaf"]
        .max()
        .reset_index()
        .rename(columns={"fechaf": "last_trx_date"})
    )
    last_trx["month"] = last_trx["last_trx_date"].dt.to_period("M").astype(str)

    scored = base_scored.merge(last_trx, on="id_user", how="left")

    # Si no tenemos probabilidad, usamos churn real (0/1)
    if scored["churn_proba"].notna().any():
        prob = scored["churn_proba"]
    elif "churn" in scored.columns:
        prob = scored["churn"]
    else:
        prob = pd.Series(0.0, index=scored.index)

    scored["churn_prob_used"] = prob

    # Definimos categorías: safe (<0.3), at risk (>=0.3)
    scored["risk_flag"] = np.where(scored["churn_prob_used"] >= 0.3, "At risk", "Safe")

    # Métrica general (usando todos los usuarios con mes conocido)
    scored_valid = scored.dropna(subset=["month"])
    tendency = (
        scored_valid.groupby("month")["churn_prob_used"]
        .mean()
        .reset_index()
        .rename(columns={"churn_prob_used": "avg_churn_prob"})
    )

    col_graf, col_cards = st.columns([2, 1])

    with col_graf:
        fig = px.line(
            tendency,
            x="month",
            y="avg_churn_prob",
            markers=True,
            title="User Tendency – average churn probability over time"
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average churn probability"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cards:
        total_users = len(scored_valid)
        current_users = total_users  # aquí lo tomamos igual; podrías refinarlo
        at_risk = (scored_valid["risk_flag"] == "At risk").sum()
        safe = (scored_valid["risk_flag"] == "Safe").sum()

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
        tiers = sorted(base_scored["share_tier"].dropna().unique())
        for t in tiers:
            st.markdown(f"- {t}")

    with col_mid:
        st.markdown("**Client risk (por días de inactividad)**")
        st.markdown("- High: 20–42 días")
        st.markdown("- Medium: 10–20 días")
        st.markdown("- Low: 0–10 días")

    with col_right:
        # Bar de motivos de churn (usuarios churned solamente si existe la columna)
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

    # Filtro de Tier
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

    # Usamos la misma serie de churn histórico que en la página 1
    hist = monthly_metrics.sort_values("month").copy()
    if hist.empty:
        st.info("No hay información mensual suficiente para graficar.")
        return

    last_rate = hist["churn_rate"].iloc[-1]

    # Creamos escenarios: histórico, -2%, -5%, -20% sobre el último punto
    scenarios = []
    for scen_name, reduction in [
        ("Historical", 0.0),
        ("Calls strategy (+2%)", 0.02),
        ("Transactions strategy (+5%)", 0.05),
        ("All together (+20%)", 0.20)
    ]:
        # Copiamos la serie histórica
        values = hist["churn_rate"].copy()
        if scen_name != "Historical":
            # Reducimos el último valor de forma creíble (no a cero)
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
        yaxis_title="Churn rate (%)"
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

