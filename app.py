"""
Flood Risk in Europe — Interactive Dashboard
Built with Streamlit + Plotly. Data source: EM-DAT (2000-2026) + World Bank GDP.

Run locally:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

st.set_page_config(
    page_title="Flood Risk in Europe",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 8px;
        padding: 1rem 1.25rem; border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 2px; white-space: nowrap; }
    .metric-value { font-size: clamp(16px, 1.7vw, 26px); font-weight: 600; color: #212529;
                    white-space: nowrap; overflow: visible; }
    .metric-sub   { font-size: 11px; color: #adb5bd; margin-top: 2px; }
    .section-note {
        font-size: 12px; color: #868e96;
        border-left: 3px solid #dee2e6;
        padding-left: 10px; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Country colour palette (consistent across all tabs)
COUNTRY_COLORS = {
    "Germany": "#2a78d6", "UK": "#eda100", "Italy": "#1baf7a",
    "Spain": "#e34948", "France": "#7c5cbf", "Russia": "#e87ba4",
    "Romania": "#eb6834", "Austria": "#1baf7a", "Czechia": "#4a3aa7",
    "Poland": "#008300", "Switzerland": "#BA7517", "Slovenia": "#d03b3b",
    "Serbia": "#185FA5", "Belgium": "#639922", "Portugal": "#e05252",
    "Bosnia": "#fab219", "Bulgaria": "#3987e5", "Greece": "#d55181",
    "Ukraine": "#c98500", "Belarus": "#9085e9",
}
FLOOD_TYPE_COLORS = {
    "Coastal flood": "#2a78d6",
    "Flash flood":   "#e34948",
    "Flood (General)": "#1baf7a",
    "Riverine flood":  "#eda100",
}

def country_color(name):
    return COUNTRY_COLORS.get(name, "#888780")


@st.cache_data
def load_data():
    """Loads pre-aggregated flood statistics (country x flood_type x year) and a
    curated list of Europe's costliest individual events, instead of the raw
    EM-DAT export. EM-DAT's terms of use prohibit redistributing the database
    (or a substantial part of it) publicly, so this repo never contains the raw
    per-event file. If you want to reproduce the aggregation yourself, register
    for a free EM-DAT account at public.emdat.be and run
    scripts/generate_agg_data.py against your own copy.

    Loading order: local file (data/) first, for local development where you
    may have generated your own copy; otherwise falls back to a private URL
    configured in Streamlit secrets (used for the deployed app, so visitors
    still see real data without the source file ever being committed to git).
    """
    base = os.path.dirname(os.path.abspath(__file__))
    agg_path    = os.path.join(base, "data", "flood_agg.csv")
    events_path = os.path.join(base, "data", "top_events.csv")
    gdp_path    = os.path.join(base, "data", "external",
                               "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4569.csv")

    def _load_csv(local_path, secret_key):
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        url = st.secrets.get(secret_key, None)
        if not url:
            st.error(
                f"Data file not found locally and no '{secret_key}' configured "
                "in Streamlit secrets. See README for setup instructions."
            )
            st.stop()
        return pd.read_csv(url)

    agg = _load_csv(agg_path, "FLOOD_AGG_URL")
    events = _load_csv(events_path, "TOP_EVENTS_URL")

    gdp_raw  = pd.read_csv(gdp_path, skiprows=4)
    years    = [str(y) for y in range(2000, 2025)]
    gdp_long = gdp_raw[["Country Code"] + years].melt(
        id_vars="Country Code", var_name="year", value_name="gdp"
    )
    gdp_long.columns = ["iso", "year", "gdp"]
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp"]  = pd.to_numeric(gdp_long["gdp"], errors="coerce")
    gdp_avg = gdp_long.groupby("iso")["gdp"].mean().reset_index()
    gdp_avg.columns = ["iso", "gdp_avg_usd"]
    gdp_avg["gdp_avg_bn"] = (gdp_avg["gdp_avg_usd"] / 1e9).round(1)

    return agg, events, gdp_avg


agg, events, gdp_avg = load_data()
ALL_COUNTRIES  = sorted(agg["country"].dropna().unique())
ALL_TYPES      = sorted(agg["flood_type"].dropna().unique())
YEAR_MIN       = int(agg["year"].min())
YEAR_MAX       = int(agg["year"].max())
ISO_MAP        = agg[["country", "iso"]].drop_duplicates()

# Rango de daño del dataset COMPLETO (sin filtrar), usado para fijar la escala
# de color del mapa. Si el rango se recalculara sobre la selección filtrada,
# el mismo tono de rojo pasaría a significar cosas distintas según el filtro
# (p.ej. Bélgica se vería "catastrófica" solo por ser la más alta de un grupo
# de países pequeños) — la práctica estándar en reportes de riesgo (Munich Re,
# Swiss Re, Copernicus) es una escala de color fija y comparable en todo momento.
_map_global = agg.groupby("iso")["total_damage_adj"].sum(min_count=1).reset_index()
_map_global["total_damage_b"] = _map_global["total_damage_adj"] / 1e6
_map_global = _map_global.dropna(subset=["total_damage_b"])
_map_global = _map_global[_map_global["total_damage_b"] > 0]
MAP_GLOBAL_MIN = _map_global["total_damage_b"].min()
MAP_GLOBAL_MAX = _map_global["total_damage_b"].max()


def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>', unsafe_allow_html=True,
    )


def decade_ticks(lo, hi, prefix="", suffix=""):
    """Solo potencias de 10 como marcas, con el valor completo escrito.
    En log-scale los dígitos 2-9 se repiten en cada década (20,30..90,
    200,300..900), lo que confunde si no está acostumbrado a leerlo.
    Se calcula sobre el rango real de datos (lo/hi), nunca fijo, para que
    los ticks caigan siempre dentro del rango visible aunque el usuario
    filtre a valores muy pequeños o muy grandes."""
    p_lo = int(np.floor(np.log10(lo)))
    p_hi = int(np.ceil(np.log10(hi)))
    vals = [10 ** p for p in range(p_lo, p_hi + 1)]
    text = [f"{prefix}{v:,}{suffix}" for v in vals]
    return vals, text


def barh(df_plot, x_col, y_col, title, color_col=None,
         color_scale="YlOrRd", x_prefix="", x_suffix="", text_fmt=None,
         ascending=False):
    """Horizontal bar chart. ascending=False (default) puts largest value at top."""
    if df_plot.empty:
        st.info("No data available for the current selection.")
        return
    if text_fmt is None:
        text_fmt = lambda v: f"{x_prefix}{v:.1f}{x_suffix}"

    # category_orders coloca el primer elemento de la lista arriba del todo,
    # así que ascending=False (mayor primero) deja el valor más alto arriba.
    df_sorted = df_plot.sort_values(x_col, ascending=ascending).copy()

    # Si el color codifica lo mismo que el eje Y (p.ej. país en ambos), la
    # leyenda es redundante con las etiquetas del eje y solo resta espacio.
    redundant_legend = color_col == y_col

    if color_col and color_col in df_sorted.columns:
        fig = px.bar(df_sorted, x=x_col, y=y_col, orientation="h",
                     labels={x_col: "", y_col: ""},
                     title=title,
                     color=color_col,
                     color_discrete_map={**COUNTRY_COLORS, **FLOOD_TYPE_COLORS},
                     category_orders={y_col: df_sorted[y_col].tolist()})
    else:
        fig = px.bar(df_sorted, x=x_col, y=y_col, orientation="h",
                     color=x_col, color_continuous_scale=color_scale,
                     labels={x_col: "", y_col: ""},
                     title=title,
                     category_orders={y_col: df_sorted[y_col].tolist()})

    # Fix text labels — must be set after fig creation to avoid Plotly overwriting them
    for i, trace in enumerate(fig.data):
        trace_texts = []
        for y_val in trace.y:
            match = df_sorted[df_sorted[y_col] == y_val]
            if not match.empty:
                trace_texts.append(text_fmt(match[x_col].values[0]))
            else:
                trace_texts.append("")
        trace.text = trace_texts
        trace.textposition = "outside"
        trace.cliponaxis = False
        trace.hovertemplate = f"<b>%{{y}}</b><br>{x_prefix}%{{x:.2f}}{x_suffix}<extra></extra>"

    max_val = df_sorted[x_col].max()
    fig.update_layout(
        showlegend=bool(color_col) and not redundant_legend,
        coloraxis_showscale=False,
        margin=dict(l=0, r=100, t=40, b=0),
        height=max(280, len(df_sorted) * 44),
        xaxis_tickprefix=x_prefix,
        xaxis_range=[0, max_val * 1.2] if pd.notna(max_val) else None,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Flood Risk in Europe")
    st.markdown("*EM-DAT · 2000-2026 · 36 countries*")
    st.divider()
    st.markdown("### Filters")
    sel_countries = st.multiselect("Countries", ALL_COUNTRIES, placeholder="All countries")
    sel_types     = st.multiselect("Flood type", ALL_TYPES, placeholder="All types")
    year_range    = st.slider("Year range", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))
    st.divider()
    st.markdown(
        "<small>Data: [EM-DAT](https://public.emdat.be) · "
        "[World Bank](https://data.worldbank.org) · "
        "[Natural Earth](https://naturalearthdata.com)</small>",
        unsafe_allow_html=True,
    )

# ── Filter ────────────────────────────────────────────────────────────────────
df_f = agg.copy()
if sel_countries:
    df_f = df_f[df_f["country"].isin(sel_countries)]
if sel_types:
    df_f = df_f[df_f["flood_type"].isin(sel_types)]
df_f = df_f[(df_f["year"] >= year_range[0]) & (df_f["year"] <= year_range[1])]

# ── Header + KPIs ─────────────────────────────────────────────────────────────
st.title("🌊 Flood Risk in Europe")
st.markdown(
    "The full dataset covers **493 flood events** across **36 European countries** "
    "(2000-2026) · EM-DAT + World Bank GDP. The numbers below reflect your current "
    "filter selection — use the sidebar to narrow them down."
)

total_dmg  = df_f["total_damage_adj"].sum(min_count=1)
total_ins  = df_f["insured_damage_adj"].sum(min_count=1)
total_dead = df_f["total_deaths"].sum(min_count=1)
total_aff  = df_f["total_affected"].sum(min_count=1)
n_events   = int(df_f["n_events"].sum())
cov_pct    = (total_ins / total_dmg * 100) if (pd.notna(total_dmg) and total_dmg > 0) else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1: kpi("Flood events", f"{n_events:,}", f"{year_range[0]}–{year_range[1]}")
with c2: kpi("Total damage", f"${total_dmg/1e6:.1f}B" if pd.notna(total_dmg) else "N/A", "Adjusted USD")
with c3: kpi("Insured losses", f"${total_ins/1e6:.1f}B" if pd.notna(total_ins) else "N/A",
             f"{cov_pct:.0f}% coverage" if pd.notna(total_ins) else "")
with c4: kpi("Total deaths", f"{int(total_dead):,}" if pd.notna(total_dead) else "N/A", "Recorded fatalities")
with c5: kpi("People affected", f"{int(total_aff):,}" if pd.notna(total_aff) else "N/A", "Injured + affected + homeless")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🗺️ Map", "💶 Losses", "📈 Trends", "🛡️ Insurance",
    "💀 Mortality", "⚠️ Vulnerability", "🌊 Flood Types", "🔥 Extremes",
    "📋 Key Findings",
])

# ─── TAB 1: MAP ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Geographic distribution of flood economic damage")
    st.markdown('<p class="section-note">Choropleth of cumulative inflation-adjusted losses '
                '(log scale). Grey = no data recorded (data gap, not zero risk). Color scale is '
                'fixed to the full dataset range ($'
                + f"{MAP_GLOBAL_MIN:.1f}B–${MAP_GLOBAL_MAX:.0f}B" +
                ') so colors stay comparable across filters — a filtered view may look mostly '
                'pale if the selected countries are small relative to the top of that range.</p>',
                unsafe_allow_html=True)

    map_data = (
        df_f.groupby("iso")["total_damage_adj"].sum(min_count=1).reset_index()
    )
    map_data.columns = ["iso", "total_damage_adj"]
    map_data["total_damage_b"] = (map_data["total_damage_adj"] / 1e6).round(2)
    map_data = map_data.dropna(subset=["total_damage_b"])
    map_data = map_data[map_data["total_damage_b"] > 0]
    map_data["log_damage"] = np.log10(map_data["total_damage_b"])
    iso_name = ISO_MAP.set_index("iso")["country"]
    map_data["country_name"] = map_data["iso"].map(iso_name)

    if map_data.empty:
        st.info("No damage data available for the current selection.")
    else:
        cb_vals, cb_text = decade_ticks(MAP_GLOBAL_MIN, MAP_GLOBAL_MAX, prefix="$", suffix="B")
        cb_tickvals = [np.log10(v) for v in cb_vals]
        # El rango de color debe cubrir los mismos límites redondeados que los
        # ticks (no los valores exactos min/max), o el tick de la punta
        # (p.ej. $100B) cae justo fuera del rango visible y desaparece.
        log_min, log_max = cb_tickvals[0], cb_tickvals[-1]

        fig_map = px.choropleth(
            map_data, locations="iso", color="log_damage",
            hover_name="country_name",
            hover_data={"total_damage_b": ":.2f", "log_damage": False, "iso": False},
            color_continuous_scale="YlOrRd", scope="europe",
            range_color=[log_min, log_max],
            labels={"total_damage_b": "Damage (B USD)", "log_damage": "Log₁₀ Damage"},
        )
        fig_map.update_layout(
            coloraxis_colorbar=dict(
                title="Damage (B USD)<br><sup>log scale</sup>",
                tickvals=cb_tickvals, ticktext=cb_text,
            ),
            margin=dict(l=0, r=0, t=10, b=0), height=520,
            geo=dict(
                showcoastlines=True, coastlinecolor="lightgrey",
                showland=True, landcolor="#f5f5f0",
                showocean=True, oceancolor="#eaf4fb",
                showframe=False, projection_type="natural earth",
                center=dict(lat=54, lon=15),
                lataxis_range=[34, 72], lonaxis_range=[-25, 45],
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    map_cov_pct = (df_f["n_damage_events"].sum() / df_f["n_events"].sum() * 100
                   if df_f["n_events"].sum() else 0)
    st.markdown(
        f"<small>⚠️ {map_cov_pct:.0f}% of events in the current selection have recorded "
        "damage data. Countries with events but no damage records (e.g. Norway) appear "
        "grey or are absent from damage charts.</small>",
        unsafe_allow_html=True)

# ─── TAB 2: ECONOMIC LOSSES ───────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Which countries have suffered the greatest flood losses?")
    col_a, col_b = st.columns(2)

    dmg_by_country = (
        df_f.groupby("country")["total_damage_adj"]
        .sum(min_count=1).dropna()
        .sort_values(ascending=False).head(15)
    )
    dmg_df = (dmg_by_country / 1e6).round(1).reset_index()
    dmg_df.columns = ["country", "damage_b"]

    dmg_title = ("Top 15 countries — cumulative damage (B USD)" if not sel_countries
                 else f"Selected countries ({len(dmg_df)}) — cumulative damage (B USD)")
    gdp_title = ("Damage as % of avg GDP (2000-2024)" if not sel_countries
                 else f"Damage as % of avg GDP — selected countries")

    with col_a:
        barh(dmg_df, "damage_b", "country",
             dmg_title,
             color_col="country",
             text_fmt=lambda v: f"${v:.1f}B")
        if sel_countries:
            missing_dmg = sorted(set(sel_countries) - set(dmg_df["country"]))
            if missing_dmg:
                st.markdown(
                    f'<p class="section-note">⚠️ Excluded (no damage data recorded): '
                    f'<b>{", ".join(missing_dmg)}</b>.</p>', unsafe_allow_html=True)

    with col_b:
        dmg_gdp = dmg_df.merge(ISO_MAP, on="country")
        dmg_gdp = dmg_gdp.merge(gdp_avg[["iso", "gdp_avg_bn"]], on="iso", how="left")
        dmg_gdp["pct_gdp"] = (dmg_gdp["damage_b"] / dmg_gdp["gdp_avg_bn"] * 100).round(2)
        dmg_gdp = dmg_gdp.dropna(subset=["pct_gdp"])
        barh(dmg_gdp, "pct_gdp", "country",
             gdp_title,
             color_col="country",
             text_fmt=lambda v: f"{v:.2f}%")
        if sel_countries:
            # Puede excluir países distintos al gráfico de la izquierda: aquí
            # también hace falta tener PIB, no solo daño.
            missing_gdp = sorted(set(sel_countries) - set(dmg_gdp["country"]))
            if missing_gdp:
                st.markdown(
                    f'<p class="section-note">⚠️ Excluded (no damage or GDP data recorded): '
                    f'<b>{", ".join(missing_gdp)}</b>.</p>', unsafe_allow_html=True)

    st.markdown(
        '<p class="section-note">Left: absolute cumulative losses. Right: same losses '
        'normalised by average GDP — smaller economies can rank far higher in relative terms.'
        + (f' Showing {len(dmg_df)} selected countries.' if sel_countries else ' Showing top 15.')
        + '</p>',
        unsafe_allow_html=True)

# ─── TAB 3: TRENDS ────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### How have flood frequency and severity evolved over time?")
    df_trend = df_f[df_f["year"] <= 2024].copy()
    col_a, col_b = st.columns(2)

    with col_a:
        # Events per year: line chart per country if filtered, bars+trend if not
        if sel_countries:
            events_yr = (
                df_trend.groupby(["year", "country"])["n_events"].sum().reset_index(name="events")
            )
            # Rellenar años sin eventos con 0 — si no, plotly conecta el último
            # año con datos directamente con el siguiente, dibujando una
            # diagonal que sugiere una subida/bajada gradual inexistente.
            full_years = range(df_trend["year"].min(), df_trend["year"].max() + 1)
            full_grid = pd.MultiIndex.from_product(
                [full_years, sel_countries], names=["year", "country"]
            ).to_frame(index=False)
            events_yr = full_grid.merge(events_yr, on=["year", "country"], how="left")
            events_yr["events"] = events_yr["events"].fillna(0).astype(int)

            fig_ev = px.bar(
                events_yr, x="year", y="events", color="country",
                color_discrete_map=COUNTRY_COLORS,
                barmode="group",
                labels={"events": "Events", "year": "Year", "country": "Country"},
                title="Flood events per year by country (2000-2024)",
            )
        else:
            events_yr_all = df_trend.groupby("year")["n_events"].sum().reset_index(name="events")
            events_yr_all["rolling"] = events_yr_all["events"].rolling(3, center=True).mean()
            if len(events_yr_all) >= 2:
                slope, intercept, r2, p_val, _ = stats.linregress(
                    events_yr_all["year"], events_yr_all["events"])
                events_yr_all["trend"] = slope * events_yr_all["year"] + intercept
            fig_ev = go.Figure()
            fig_ev.add_bar(x=events_yr_all["year"], y=events_yr_all["events"],
                           name="Events / year", marker_color="#5b9bd5", opacity=0.7)
            if len(events_yr_all) >= 2:
                fig_ev.add_scatter(x=events_yr_all["year"], y=events_yr_all["rolling"],
                                   name="3-yr rolling mean",
                                   line=dict(color="#1a3a5c", width=2))
                fig_ev.add_scatter(x=events_yr_all["year"], y=events_yr_all["trend"],
                                   name=f"Trend (R²={r2**2:.2f}, p={p_val:.3f})",
                                   line=dict(color="#e05252", width=2, dash="dash"))
            fig_ev = go.Figure(fig_ev)

        fig_ev.update_layout(
            title=fig_ev.layout.title.text if fig_ev.layout.title.text else "Flood events per year (2000-2024)",
            xaxis_title="Year", yaxis_title="Events",
            legend=dict(orientation="h", yanchor="top", y=-0.25),
            margin=dict(l=0, r=0, t=40, b=80), height=380)
        st.plotly_chart(fig_ev, use_container_width=True)

    with col_b:
        if sel_countries:
            dmg_yr = (
                df_trend.groupby(["year", "country"])["total_damage_adj"]
                .sum(min_count=1).reset_index()
            )
            full_years = range(df_trend["year"].min(), df_trend["year"].max() + 1)
            full_grid = pd.MultiIndex.from_product(
                [full_years, sel_countries], names=["year", "country"]
            ).to_frame(index=False)
            dmg_yr = full_grid.merge(dmg_yr, on=["year", "country"], how="left")
            # 0 = sin daño registrado ese año (incluye años sin eventos),
            # así la línea no salta en diagonal sobre huecos.
            dmg_yr["total_damage_adj"] = dmg_yr["total_damage_adj"].fillna(0)
            dmg_yr["damage_b"] = (dmg_yr["total_damage_adj"] / 1e6).round(1)
            fig_dmg_yr = px.bar(
                dmg_yr, x="year", y="damage_b", color="country",
                color_discrete_map=COUNTRY_COLORS,
                barmode="group",
                labels={"damage_b": "Damage (B USD)", "year": "Year", "country": "Country"},
                title="Total flood damage per year by country (2000-2024)",
            )
            fig_dmg_yr.update_layout(yaxis_tickprefix="$", yaxis_ticksuffix="B")
        else:
            exclude_outlier = st.checkbox(
                "Exclude outlier year (2021 — Ahr Valley floods, Germany)",
                key="exclude_2021_outlier",
                help="2021 is dominated by a single 47.5B USD event, which compresses "
                     "the scale for every other year. Toggle this to compare the rest "
                     "of the series more clearly.",
            )
            dmg_yr_all = df_trend.groupby("year")["total_damage_adj"].sum().reset_index()
            dmg_yr_all["damage_b"] = (dmg_yr_all["total_damage_adj"] / 1e6).round(1)
            peak_row = dmg_yr_all.loc[dmg_yr_all["damage_b"].idxmax()]
            plot_df = dmg_yr_all[dmg_yr_all["year"] != 2021] if exclude_outlier else dmg_yr_all

            fig_dmg_yr = px.bar(
                plot_df, x="year", y="damage_b",
                labels={"damage_b": "Damage (B USD)", "year": "Year"},
                title="Total flood damage per year (2000-2024)"
                      + (" — 2021 excluded" if exclude_outlier else ""),
            )
            fig_dmg_yr.update_traces(marker_color="#d9534f")
            if not exclude_outlier and peak_row["year"] == 2021:
                fig_dmg_yr.add_annotation(
                    x=2021, y=peak_row["damage_b"],
                    text="2021: Ahr Valley floods (Germany)",
                    showarrow=True, arrowhead=2, ax=0, ay=-40,
                    bgcolor="white", bordercolor="#d9534f",
                )

        fig_dmg_yr.update_traces(
            hovertemplate="<b>%{x}</b><br>$%{y:.1f}B<extra></extra>")
        fig_dmg_yr.update_layout(
            margin=dict(l=0, r=0, t=40, b=0), height=380,
            yaxis_tickprefix="$", yaxis_ticksuffix="B")
        st.plotly_chart(fig_dmg_yr, use_container_width=True)

    if not sel_countries:
        st.markdown(
            '<p class="section-note">Trend in event frequency is statistically significant '
            '(p=0.016) but explains only 22.7% of variance (R²=0.227). '
            'Recent years may be underreported. 2026 excluded (incomplete year).</p>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="section-note">Grouped bars show each selected country per year. '
            'Years with no recorded events show as 0 (not omitted). '
            '2026 excluded (incomplete year).</p>',
            unsafe_allow_html=True)

# ─── TAB 4: INSURANCE GAP ─────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### How much of the total flood damage is covered by insurance?")
    st.markdown(
        '<p class="section-note">⚠️ Only includes events where both total damage and '
        'insured damage are recorded — a stricter subset than the Economic Losses tab.</p>',
        unsafe_allow_html=True)

    gap_data = (
        df_f.groupby("country")[["matched_damage_adj", "matched_insured_adj"]]
        .sum(min_count=1).dropna(subset=["matched_damage_adj"])
        .sort_values("matched_damage_adj", ascending=False).head(10) / 1e6
    )
    gap_data.columns = ["total_damage_adj", "insured_damage_adj"]
    gap_data["insurance_gap"] = gap_data["total_damage_adj"] - gap_data["insured_damage_adj"]
    gap_data["coverage_pct"]  = (gap_data["insured_damage_adj"] / gap_data["total_damage_adj"] * 100).round(1)

    if gap_data.empty:
        st.info("No insurance gap data available for the current selection.")
    else:
        if sel_countries:
            missing = sorted(set(sel_countries) - set(gap_data.index))
            if missing:
                st.markdown(
                    f'<p class="section-note">⚠️ Excluded (no insured damage recorded '
                    f'for any event): <b>{", ".join(missing)}</b>.</p>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            # Mismo orden que la tabla de la derecha (daño total, mayor arriba),
            # así las barras y las filas de la tabla se corresponden 1 a 1.
            countries_sorted = gap_data.sort_values("total_damage_adj", ascending=True).index.tolist()
            fig_gap = go.Figure()
            fig_gap.add_bar(
                y=countries_sorted,
                x=gap_data.loc[countries_sorted, "insured_damage_adj"],
                name="Insured", orientation="h", marker_color="#2a78d6",
                hovertemplate="<b>%{y}</b><br>Insured: $%{x:.1f}B<extra></extra>")
            fig_gap.add_bar(
                y=countries_sorted,
                x=gap_data.loc[countries_sorted, "insurance_gap"],
                name="Uninsured (gap)", orientation="h", marker_color="#e34948",
                hovertemplate="<b>%{y}</b><br>Gap: $%{x:.1f}B<extra></extra>")
            fig_gap.update_layout(
                barmode="stack", title="Insurance gap by country (B USD)",
                xaxis_title="Billion USD (adjusted)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                margin=dict(l=0, r=0, t=70, b=40),
                height=max(280, len(gap_data) * 44))
            st.plotly_chart(fig_gap, use_container_width=True)

        with col_b:
            st.markdown("**Coverage rate by country**")
            tbl = gap_data[["total_damage_adj", "insured_damage_adj", "insurance_gap", "coverage_pct"]].copy()
            tbl.columns = ["Total ($B)", "Insured ($B)", "Gap ($B)", "Coverage %"]
            st.dataframe(
                tbl.round(1), use_container_width=True,
                column_config={
                    "Total ($B)": st.column_config.NumberColumn(width="small"),
                    "Insured ($B)": st.column_config.NumberColumn(width="small"),
                    "Gap ($B)": st.column_config.NumberColumn(width="small"),
                    "Coverage %": st.column_config.NumberColumn(width="small"),
                })

# ─── TAB 5: MORTALITY & POPULATION ────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Which countries have suffered the highest human impact?")
    col_a, col_b = st.columns(2)

    with col_a:
        deaths_df = (
            df_f.groupby("country")["total_deaths"]
            .sum(min_count=1).dropna()
            .sort_values(ascending=False).head(15).reset_index()
        )
        deaths_df.columns = ["country", "deaths"]
        barh(deaths_df, "deaths", "country",
             "Total deaths by country (top 15)",
             color_col="country",
             text_fmt=lambda v: f"{int(v):,}")
        if sel_countries:
            missing_deaths = sorted(set(sel_countries) - set(deaths_df["country"]))
            if missing_deaths:
                st.markdown(
                    f'<p class="section-note">⚠️ Excluded (no deaths recorded): '
                    f'<b>{", ".join(missing_deaths)}</b>.</p>', unsafe_allow_html=True)

    with col_b:
        aff_df = (
            df_f.groupby("country")["total_affected"]
            .sum(min_count=1).dropna()
            .sort_values(ascending=False).head(15).reset_index()
        )
        aff_df.columns = ["country", "affected"]
        barh(aff_df, "affected", "country",
             "Total people affected by country (top 15)",
             color_col="country",
             text_fmt=lambda v: f"{int(v):,}")
        if sel_countries:
            missing_aff = sorted(set(sel_countries) - set(aff_df["country"]))
            if missing_aff:
                st.markdown(
                    f'<p class="section-note">⚠️ Excluded (no affected-population data '
                    f'recorded): <b>{", ".join(missing_aff)}</b>.</p>', unsafe_allow_html=True)

    if not sel_countries:
        st.markdown(
            '<p class="section-note">Deaths and people affected do not always correlate '
            'with economic damage. Bosnia ranks 2nd in people affected despite not appearing '
            'in the top economic loss tables.</p>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="section-note">Deaths and people affected do not always correlate '
            'with economic damage. Results reflect only the countries selected in the filter.</p>',
            unsafe_allow_html=True)

# ─── TAB 6: VULNERABILITY ─────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Do countries with higher losses also record higher mortality?")
    st.markdown(
        '<p class="section-note">Scatter plot: economic damage vs deaths. Axes switch to '
        'log scale automatically when values span more than 10x, to keep small and large '
        'countries readable on the same chart. Dashed lines = medians. Countries shown have '
        'at least 3 recorded events for both damage and deaths — a data-reliability filter, '
        'not a magnitude cutoff, so inclusion isn\'t biased toward countries already known '
        'to be large.</p>',
        unsafe_allow_html=True)

    vuln = df_f.groupby("country").agg(
        total_damage_b=("total_damage_adj", lambda x: x.sum(min_count=1) / 1e6),
        total_deaths=("total_deaths", "sum"),
        n_damage_events=("total_damage_adj", lambda x: x.notna().sum()),
        n_death_events=("total_deaths", lambda x: x.notna().sum()),
    ).reset_index()

    # Remove rows where deaths or damage are 0, NaN, or negative (breaks log scale)
    vuln = vuln.dropna(subset=["total_damage_b", "total_deaths"])
    vuln = vuln[(vuln["total_damage_b"] > 0) & (vuln["total_deaths"] > 0)]
    # Filtrar por VOLUMEN de eventos (>=3 de cada tipo), no por magnitud del
    # resultado (daño/muertes grandes). Un umbral de magnitud es circular:
    # decide qué países se muestran usando las mismas variables que el
    # gráfico pretende explorar, sesgando hacia los países ya conocidos como
    # grandes. Filtrar por número de eventos selecciona países según cuán
    # fiable es su cifra agregada, sin importar si el total es grande o
    # pequeño — mismo criterio usado en analysis.ipynb.
    vuln_f = vuln[(vuln["n_damage_events"] >= 3) & (vuln["n_death_events"] >= 3)].copy()

    if vuln_f.empty:
        st.info("Not enough data for vulnerability analysis with the current selection.")
    else:
        med_dmg    = vuln_f["total_damage_b"].median()
        med_deaths = vuln_f["total_deaths"].median()

        vuln_f["color"] = vuln_f["country"].map(country_color)

        # Use log scale on each axis only when its own range justifies it
        # (avoids the axis distortion a log scale causes on small/narrow datasets)
        use_log_x = vuln_f["total_damage_b"].max() / max(vuln_f["total_damage_b"].min(), 0.01) > 10
        use_log_y = vuln_f["total_deaths"].max() / max(vuln_f["total_deaths"].min(), 1) > 10

        fig_vuln = px.scatter(
            vuln_f, x="total_damage_b", y="total_deaths",
            text="country", color="country",
            color_discrete_map=COUNTRY_COLORS,
            log_x=use_log_x, log_y=use_log_y,
            labels={"total_damage_b": "Total Damage (B USD)",
                    "total_deaths": "Total Deaths",
                    "country": "Country"},
            title="Economic damage vs mortality — vulnerability profile",
        )
        fig_vuln.update_traces(
            textposition="top center",
            marker=dict(size=14, line=dict(width=0.5, color="gray")),
            hovertemplate="<b>%{text}</b><br>Damage: $%{x:.1f}B<br>Deaths: %{y:.0f}<extra></extra>",
        )
        # Median lines only when 5+ countries
        if len(vuln_f) >= 5:
            fig_vuln.add_vline(x=med_dmg, line_dash="dot", line_color="gray",
                               annotation_text=f"Median ${med_dmg:.1f}B",
                               annotation_position="top right")
            fig_vuln.add_hline(y=med_deaths, line_dash="dot", line_color="gray",
                               annotation_text=f"Median {int(med_deaths)} deaths",
                               annotation_position="bottom left")

        # Fijar el rango de cada eje explícitamente. Dejarlo en automático con
        # log_y + rangemode disparaba el eje a 10^64, aplastando todos los
        # puntos en una franja diminuta y provocando que las etiquetas de país
        # se amontonaran unas sobre otras.
        x_min, x_max = vuln_f["total_damage_b"].min(), vuln_f["total_damage_b"].max()
        y_min, y_max = vuln_f["total_deaths"].min(), vuln_f["total_deaths"].max()

        if use_log_x:
            xt_vals, xt_text = decade_ticks(x_min * 0.6, x_max * 1.6, prefix="$", suffix="B")
            xaxis_cfg = dict(range=[np.log10(x_min * 0.6), np.log10(x_max * 1.6)],
                              tickvals=xt_vals, ticktext=xt_text)
        else:
            xaxis_cfg = dict(rangemode="tozero")
        if use_log_y:
            yt_vals, yt_text = decade_ticks(max(y_min * 0.6, 0.5), y_max * 1.6)
            yaxis_cfg = dict(range=[np.log10(max(y_min * 0.6, 0.5)), np.log10(y_max * 1.6)],
                              tickvals=yt_vals, ticktext=yt_text)
        else:
            yaxis_cfg = dict(rangemode="tozero")

        fig_vuln.update_layout(
            margin=dict(l=0, r=0, t=40, b=60), height=500,
            showlegend=True,
            xaxis=xaxis_cfg,
            yaxis=yaxis_cfg,
        )
        st.plotly_chart(fig_vuln, use_container_width=True)

# ─── TAB 7: FLOOD TYPES ───────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("### Which type of flood causes the most damage and mortality?")

    if sel_countries:
        # Con filtro de país activo, desglosamos cada barra por país
        # para que el gráfico siga reflejando la selección.
        ft_c = df_f.groupby(["flood_type", "country"]).agg(
            total_events=("n_events", "sum"),
            total_damage_b=("total_damage_adj", lambda x: x.sum(min_count=1) / 1e6),
            total_deaths=("total_deaths", lambda x: x.sum(min_count=1)),
        ).reset_index()
        order = (ft_c.groupby("flood_type")["total_events"].sum()
                 .sort_values(ascending=False).index.tolist())

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fig1 = px.bar(ft_c, x="flood_type", y="total_events", color="country",
                          color_discrete_map=COUNTRY_COLORS,
                          category_orders={"flood_type": order},
                          title="Number of events",
                          labels={"flood_type": "", "total_events": "Events"})
            fig1.update_layout(margin=dict(l=0,r=0,t=40,b=100), height=380, xaxis_tickangle=35)
            st.plotly_chart(fig1, use_container_width=True)
        with col_b:
            fig2 = px.bar(ft_c, x="flood_type", y="total_damage_b", color="country",
                          color_discrete_map=COUNTRY_COLORS,
                          category_orders={"flood_type": order},
                          title="Total damage (B USD)",
                          labels={"flood_type": "", "total_damage_b": "B USD"})
            fig2.update_layout(margin=dict(l=0,r=0,t=40,b=100), height=380,
                               yaxis_tickprefix="$", yaxis_ticksuffix="B", xaxis_tickangle=35)
            st.plotly_chart(fig2, use_container_width=True)
        with col_c:
            fig3 = px.bar(ft_c, x="flood_type", y="total_deaths", color="country",
                          color_discrete_map=COUNTRY_COLORS,
                          category_orders={"flood_type": order},
                          title="Total deaths",
                          labels={"flood_type": "", "total_deaths": "Deaths"})
            fig3.update_layout(margin=dict(l=0,r=0,t=40,b=100), height=380, xaxis_tickangle=35)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            f'<p class="section-note">Data filtered to: <b>{", ".join(sel_countries)}</b>. '
            f'Each bar is broken down by country to reflect the current selection.</p>',
            unsafe_allow_html=True)

    else:
        ft = df_f.groupby("flood_type").agg(
            total_events=("n_events", "sum"),
            total_damage_b=("total_damage_adj", lambda x: x.sum(min_count=1) / 1e6),
            total_deaths=("total_deaths", lambda x: x.sum(min_count=1)),
            events_with_damage=("n_damage_events", "sum"),
            events_with_deaths=("n_death_events", "sum"),
        ).round(1).reset_index()
        ft["avg_deaths"] = (ft["total_deaths"] / ft["total_events"]).round(2)
        ft["damage_cov"] = (ft["events_with_damage"] / ft["total_events"] * 100).round(0).astype(int).astype(str) + "%"

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fig1 = px.bar(ft.sort_values("total_events", ascending=False),
                          x="flood_type", y="total_events",
                          color="flood_type", color_discrete_map=FLOOD_TYPE_COLORS,
                          title="Number of events",
                          labels={"flood_type": "", "total_events": "Events"})
            fig1.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=100), height=380,
                               xaxis_tickangle=35)
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            fig2 = px.bar(ft.sort_values("total_damage_b", ascending=False),
                          x="flood_type", y="total_damage_b",
                          color="flood_type", color_discrete_map=FLOOD_TYPE_COLORS,
                          title="Total damage (B USD)",
                          labels={"flood_type": "", "total_damage_b": "B USD"})
            fig2.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=100), height=380,
                               yaxis_tickprefix="$", yaxis_ticksuffix="B", xaxis_tickangle=35)
            st.plotly_chart(fig2, use_container_width=True)

        with col_c:
            fig3 = px.bar(ft.sort_values("avg_deaths", ascending=False),
                          x="flood_type", y="avg_deaths",
                          color="flood_type", color_discrete_map=FLOOD_TYPE_COLORS,
                          title="Avg deaths per event",
                          labels={"flood_type": "", "avg_deaths": "Deaths/event"})
            fig3.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=100), height=380,
                               xaxis_tickangle=35)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Data coverage by flood type**")
        ft_tbl = ft[["flood_type","total_events","total_damage_b","total_deaths","avg_deaths","damage_cov"]].copy()
        ft_tbl.columns = ["Flood Type","Events","Damage (B USD)","Deaths","Avg Deaths/Event","Damage Coverage"]
        st.dataframe(ft_tbl.set_index("Flood Type"), use_container_width=True)

    if not sel_countries or "Germany" in sel_countries:
        st.markdown(
            '<p class="section-note">"Flood (General)" dominates damage because it contains '
            'the 2021 Ahr Valley event ($47.5B) in Germany. Flash floods show the highest '
            'mortality rate per event despite lower total damage.</p>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="section-note">Flash floods show the highest mortality rate per event '
            'despite lower total damage. Results reflect only the countries selected in the filter.</p>',
            unsafe_allow_html=True)

# ─── TAB 8: EXTREME EVENTS ────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### Which individual events caused the greatest losses?")
    st.markdown(
        '<p class="section-note">⚠️ Individual event detail is limited to Europe\'s 30 '
        'costliest recorded flood events (a curated list — see README on data licensing). '
        'Every other tab in this dashboard uses full aggregated data across all 493 events; '
        'only this tab is restricted, since ranking individual events needs event-level '
        'detail rather than country/year sums. If your filter selects a country or period '
        'with no events in this list, that does not mean it had no flood damage — just none '
        'among Europe\'s top 30.</p>',
        unsafe_allow_html=True)
    top_n = st.slider("Show top N events", 5, 20, 10)

    ev_f = events.copy()
    if sel_countries:
        ev_f = ev_f[ev_f["country"].isin(sel_countries)]
    if sel_types:
        ev_f = ev_f[ev_f["flood_type"].isin(sel_types)]
    ev_f = ev_f[(ev_f["year"] >= year_range[0]) & (ev_f["year"] <= year_range[1])]

    top_ev = (
        ev_f[ev_f["total_damage_adj"].notna()]
        .nlargest(top_n, "total_damage_adj")
        [["country","year","month","flood_type","total_damage_adj","insured_damage_adj","total_deaths"]]
        .copy()
    )
    if top_ev.empty:
        st.info("None of Europe's 30 costliest recorded flood events match the current "
                "filter. Try a broader selection, or see the Economic Losses / Flood Types "
                "tabs for full aggregated figures for this selection.")
    else:
        top_ev["total_damage_b"]   = (top_ev["total_damage_adj"] / 1e6).round(1)
        top_ev["insured_damage_b"] = (top_ev["insured_damage_adj"] / 1e6).round(1)
        top_ev["total_deaths"]     = top_ev["total_deaths"].fillna(0).astype(int)
        top_ev["label"]            = (top_ev["country"] + " " +
                                      top_ev["year"].astype(str) + "/" +
                                      top_ev["month"].astype(str))

        # Sort ascending so largest bar appears at top in horizontal chart
        top_ev_sorted = top_ev.sort_values("total_damage_b", ascending=True).copy()
        top_ev_sorted["label"] = (top_ev_sorted["country"] + " " +
                                  top_ev_sorted["year"].astype(str) + "/" +
                                  top_ev_sorted["month"].astype(str))
        ordered_labels = top_ev_sorted["label"].tolist()

        fig_top = go.Figure()
        for country in top_ev_sorted["country"].unique():
            cdf = top_ev_sorted[top_ev_sorted["country"] == country]
            fig_top.add_trace(go.Bar(
                x=cdf["total_damage_b"],
                y=cdf["label"],
                orientation="h",
                name=country,
                marker_color=COUNTRY_COLORS.get(country, "#888780"),
                customdata=cdf[["total_deaths", "insured_damage_b", "flood_type"]].values,
                hovertemplate="<b>%{y}</b><br>Damage: $%{x:.1f}B<br>"
                              "Insured: $%{customdata[1]:.1f}B<br>"
                              "Deaths: %{customdata[0]}<br>"
                              "Type: %{customdata[2]}<extra></extra>",
            ))
        fig_top.update_layout(
            barmode="overlay",
            yaxis=dict(categoryorder="array", categoryarray=ordered_labels),
            margin=dict(l=0, r=60, t=40, b=0),
            height=max(300, top_n*40),
            xaxis_tickprefix="$", xaxis_ticksuffix="B",
            title=f"Top {top_n} most costly flood events",
        )
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("**Full event table**")
        tbl_ev = top_ev[["country","year","month","flood_type",
                          "total_damage_b","insured_damage_b","total_deaths"]].copy()
        tbl_ev.columns = ["Country","Year","Month","Flood Type",
                          "Damage (B USD)","Insured (B USD)","Deaths"]
        tbl_ev["Insured (B USD)"] = tbl_ev["Insured (B USD)"].apply(
            lambda v: f"${v:.1f}B" if pd.notna(v) else "N/A")
        st.dataframe(
            tbl_ev.reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Country": st.column_config.TextColumn("Country", width="medium"),
                "Flood Type": st.column_config.TextColumn("Flood Type", width="medium"),
            },
        )

# ─── TAB 9: KEY FINDINGS ───────────────────────────────────────────────────────
with tabs[8]:
    st.markdown("### What does this analysis mean?")
    st.markdown(
        '<p class="section-note">This tab is a fixed summary of the full 2000-2026 '
        'dataset (unfiltered) — it does not react to the sidebar filters, unlike the '
        'other tabs.</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Economic losses & insurance gap")
        st.markdown(
            "- Germany accounts for the largest cumulative flood losses in Europe "
            "(**93.8B USD** adjusted), driven mainly by the 2021 Ahr Valley floods "
            "(47.5B USD) — one of the costliest natural disasters in European history.\n"
            "- The top 3 countries (Germany, UK, Italy) represent **~66%** of all "
            "recorded flood damage across the 27 countries with damage data.\n"
            "- Even in Germany, **78% of flood losses were uninsured** (73.3B USD gap). "
            "Italy's insured share is only 5% within the matched subset. This is a "
            "systemic protection gap, not an isolated case.\n"
            "- Insurance data coverage in Eastern Europe is very low (Romania 0%, "
            "Russia 4%, Poland 10%), so apparent underinsurance there may partly "
            "reflect reporting gaps rather than confirmed absence of coverage."
        )

    with col_b:
        st.markdown("#### Mortality, vulnerability & flood type")
        st.markdown(
            "- Russia and Spain recorded the highest flood mortality (580 and 321 "
            "deaths) despite lower economic losses than Germany or the UK — a sign "
            "of lower resilience and adaptive capacity, not just hazard exposure.\n"
            "- The Spain 2024 DANA event (232 deaths, 11.3B USD damage) shows the "
            "growing lethality of flash floods under climate change.\n"
            "- Slovenia shows the highest damage relative to GDP of any country "
            "(8.42%) and ranks 4th in population affected (1.52M) — disproportionate "
            "for a country that ranks only 12th in absolute damage ($3.9B). Smaller "
            "economies can be hit hardest in relative terms even when absolute "
            "losses look modest.\n"
            "- Riverine floods dominate frequency and total damage. Among named "
            "hazard types, flash floods have the highest mortality rate per event "
            "(5.94 deaths/event), consistent with IPCC AR6 projections."
        )

    st.markdown("#### Business & policy implications")
    imp_a, imp_b, imp_c = st.columns(3)
    with imp_a:
        st.markdown(
            "**Insurers & reinsurers**\n"
            "- Germany, Italy and the UK warrant priority for flood risk capital "
            "allocation, read alongside data coverage rates.\n"
            "- Flash floods need separate modelling frameworks given their high "
            "mortality rate per event."
        )
    with imp_b:
        st.markdown(
            "**Governments & regulators**\n"
            "- The scale of the protection gap supports the case for mandatory "
            "flood insurance schemes.\n"
            "- Higher mortality relative to losses in Eastern Europe points to a "
            "need for targeted adaptation investment."
        )
    with imp_c:
        st.markdown(
            "**Investors & asset managers**\n"
            "- Losses are concentrated in a small number of catastrophic events, "
            "making tail-risk assessment critical.\n"
            "- Climate change is expected to intensify flash and coastal floods, "
            "beyond what historical data alone captures."
        )

    with st.expander("Limitations of this dataset"):
        st.markdown(
            "- EM-DAT relies on reported data: only **37%** of events have recorded "
            "economic damage, and only **10%** have recorded insured damage — "
            "figures here reflect that reported subset, not total flood impact.\n"
            "- Recent years are likely under-reported, since disaster data entry "
            "and validation take time after an event.\n"
            "- \"Damage as % of GDP\" divides 27 years of cumulative damage by a "
            "single-year average GDP. It is a rough exposure indicator, **not** an "
            "annual loss rate.\n"
            "- The declining trend in event frequency (Trends tab) has a low R² "
            "(0.227): year explains only ~23% of the variance in event counts. "
            "Statistically significant, but a weak signal — not strong evidence of "
            "declining flood frequency.\n"
            "- \"Flood (General)\" is an EM-DAT catch-all category, not a distinct "
            "physical hazard — its high mortality rate should be read with that "
            "caveat."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>**Data:** EM-DAT (CRED, UCLouvain) · World Bank GDP · Natural Earth · "
    "**Author:** Sheila Alonso · "
    "[GitHub](https://github.com/AlonsoSheila/flood-risk-europe) · "
    "[Notebook](https://github.com/AlonsoSheila/flood-risk-europe/blob/main/analysis.ipynb) · "
    "[Executive Report](https://github.com/AlonsoSheila/flood-risk-europe/blob/main/docs/executive_report_en.pdf)"
    "</small>",
    unsafe_allow_html=True,
)
