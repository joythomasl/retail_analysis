import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import os

# Resolve paths relative to the repo root regardless of the process's cwd
# (this file lives in app/, so plain "outputs/..." only worked when launched
# from the repo root -- running `streamlit run app.py` from inside app/
# silently missed the real data and fell back to mock data).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# 1. Page Configuration
st.set_page_config(
    page_title="ISCDF Control Tower", 
    layout="wide", 
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# 2. Header
st.title("🌐 Intelligent Supply Chain Disruption Forecaster")
st.markdown("### Proactive AI Control Tower")

# 3. Data Loading
@st.cache_data
def load_data():
    file_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    st.error(
        "🚨 `outputs/predictions.csv` not found! Run `python notebooks/rebuild_pipeline.py` "
        "from the repo root first to train the model and generate real predictions."
    )
    st.stop()

df = load_data()

# 4. Top KPI Executive Dashboard
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Active Shipments", f"{len(df):,}")
with col2:
    high_risk_count = len(df[df['risk_band'] == 'High'])
    st.metric("High-Risk Alerts", high_risk_count, delta="Action Required", delta_color="inverse")
with col3:
    var_usd = df[df['risk_band'] == 'High']['order_value_usd'].sum()
    st.metric("Value at Risk (VaR)", f"${var_usd:,.2f}")
with col4:
    network_health = 100 - (high_risk_count / len(df) * 100)
    st.metric("Network Health Index", f"{network_health:.1f}%")

# 5. Main Workflow Tabs
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Global Map", "⚠️ Network Cascade", "⚙️ What-If Simulator", "📋 Shipment Register"]
)

# ==========================================
# TAB 1: 3D PYDECK MAP
# ==========================================
with tab1:
    st.subheader("Live Network Cascade")
    
    # Covers every destination_city in the real dataset (outputs/predictions.csv);
    # "Shanghai" is kept as the illustrative origin hub -- predictions.csv
    # doesn't carry a per-shipment origin city to plot from.
    geo_map = {
        "Atlanta": [-84.3880, 33.7490],
        "Bengaluru": [77.5946, 12.9716],
        "Berlin": [13.4050, 52.5200],
        "Birmingham": [-1.8904, 52.4862],
        "Chennai": [80.2707, 13.0827],
        "Chicago": [-87.6298, 41.8781],
        "Delhi": [77.1025, 28.7041],
        "Dubai": [55.2708, 25.2048],
        "Hamburg": [9.9937, 53.5511],
        "Houston": [-95.3698, 29.7604],
        "London": [-0.1278, 51.5074],
        "Los Angeles": [-118.2437, 34.0522],
        "Manchester": [-2.2426, 53.4808],
        "Marseille": [5.3698, 43.2965],
        "Melbourne": [144.9631, -37.8136],
        "Mumbai": [72.8777, 19.0760],
        "Munich": [11.5820, 48.1351],
        "New York": [-74.0060, 40.7128],
        "Osaka": [135.5023, 34.6937],
        "Paris": [2.3522, 48.8566],
        "Singapore": [103.8198, 1.3521],
        "Sydney": [151.2093, -33.8688],
        "Tokyo": [139.6503, 35.6762],
        "Toronto": [-79.3832, 43.6532],
        "Vancouver": [-123.1207, 49.2827],
        "Frankfurt": [8.6821, 50.1109],
        "Shanghai": [121.4737, 31.2304],
    }

    map_data = []
    for _, row in df.iterrows():
        dest = row['destination_city']
        if dest in geo_map:
            color = [255, 50, 50, 200] if row['risk_band'] == 'High' else [50, 255, 50, 200]
            map_data.append({
                "origin_name": "Shanghai",
                "dest_name": dest,
                "start_lon": geo_map["Shanghai"][0],
                "start_lat": geo_map["Shanghai"][1],
                "end_lon": geo_map[dest][0],
                "end_lat": geo_map[dest][1],
                "color": color,
                "risk": row['risk_band']
            })
            
    if map_data:
        map_df = pd.DataFrame(map_data)
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=map_df,
            get_source_position="[start_lon, start_lat]",
            get_target_position="[end_lon, end_lat]",
            get_source_color="color",
            get_target_color="color",
            get_width=5,
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(latitude=40.0, longitude=-40.0, zoom=1.5, pitch=45, bearing=0)
        st.pydeck_chart(pdk.Deck(
            map_style="dark", # Uses the built-in, token-free dark map
            initial_view_state=view_state,
            layers=[arc_layer],
            tooltip={"text": "{origin_name} to {dest_name}\nRisk Level: {risk}"}
        ))
    else:
        st.info("No mapped destinations found in current data.")

# ==========================================
# TAB 2: NETWORK CASCADE (GRAPHVIZ)
# ==========================================
with tab2:
    st.subheader("⚠️ Network Cascade Simulator")
    st.markdown("Visualizing how upstream delays impact downstream safety stock.")
    
    high_risk_df = df[df['risk_band'] == 'High']
    if not high_risk_df.empty:
        sim_shipment = st.selectbox("Select High-Risk Shipment to Trace:", high_risk_df['shipment_id'])
        
        if sim_shipment:
            row = high_risk_df[high_risk_df['shipment_id'] == sim_shipment].iloc[0]
            delay_days = max(1, int(row['risk_score'] / 10))
            warehouse_buffer = 4
            net_delay = delay_days - warehouse_buffer
            customer_status = "Stockout" if net_delay > 0 else "Safe"
            
            supplier_color = "#ff4b4b"
            wh_color = "#ffa500" if net_delay > 0 else "#00c04b"
            cust_color = "#ff4b4b" if net_delay > 0 else "#00c04b"
            
            graph_code = f"""
            digraph G {{
                rankdir=LR;
                node [shape=box, style=filled, fontname="Helvetica", rounded=true];
                Supplier [label="Supplier: Origin\\nDelay: +{delay_days} Days", fillcolor="{supplier_color}", fontcolor="white"];
                Warehouse [label="Hub: {row['destination_city']}\\nSafety Buffer: {warehouse_buffer} Days", fillcolor="{wh_color}", fontcolor="white"];
                Customer [label="Customer: Final\\nStatus: {customer_status}", fillcolor="{cust_color}", fontcolor="white"];
                Supplier -> Warehouse [label=" transit ", fontcolor="gray"];
                Warehouse -> Customer [label=" net impact: {net_delay} days ", fontcolor="gray"];
            }}
            """
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.graphviz_chart(graph_code)
            with c2:
                st.markdown("### Cascade Analysis")
                if "driver_1" in row and pd.notna(row.get("driver_1")):
                    st.caption(f"Top risk driver (SHAP): `{row['driver_1']}`")
                st.error(f"**Primary Shock:** {delay_days} days late from origin.")
                if net_delay > 0:
                    st.warning(f"**Buffer Exhausted:** Warehouse safety stock ({warehouse_buffer} days) is insufficient.")
                    st.error(f"**Cascading Impact:** Customer will experience a {net_delay}-day stockout.")
                    action = row.get("recommended_action") or f"Expedite {net_delay} days of inventory via Air Freight."
                    st.success(f"**Suggested Mitigation:** {action}")
                else:
                    st.success(f"**Shock Absorbed:** Warehouse safety stock ({warehouse_buffer} days) covers the delay. No customer impact.")
    else:
        st.success("No high risk shipments to simulate!")

# ==========================================
# TAB 3: LIVE WHAT-IF SIMULATOR
# ==========================================
with tab3:
    st.subheader("Smart Order Entry (What-If Simulator)")
    
    # Safely load the ML artifacts (prefer tuned model, fallback to original)
    try:
        tuned_path = os.path.join(OUTPUTS_DIR, "model_tuned.pkl")
        if os.path.exists(tuned_path):
            model = joblib.load(tuned_path)
        else:
            model = joblib.load(os.path.join(OUTPUTS_DIR, "model.pkl"))

        encoder = joblib.load(os.path.join(OUTPUTS_DIR, "encoder.pkl"))
        features = joblib.load(os.path.join(OUTPUTS_DIR, "features.pkl"))
        # Per-feature defaults/ranges/options, built from whatever schema the
        # model was actually trained on -- lets this form stay correct across
        # both the real-data schema and the synthetic fallback schema without
        # hardcoding column names (see notebooks/rebuild_pipeline.py).
        feature_stats = joblib.load(os.path.join(OUTPUTS_DIR, "feature_stats.pkl"))
        ml_ready = True
    except FileNotFoundError:
        st.error(
            "🚨 ML models not found or out of date! Run `python notebooks/rebuild_pipeline.py` "
            "from the repo root first."
        )
        ml_ready = False

    # Curated priority list of the most decision-relevant fields, across both
    # the real and synthetic schemas -- only ones present in `features` are
    # actually rendered as inputs. Everything else in `features` falls back
    # to its training-data median/mode from feature_stats.
    PRIORITY_FIELDS = [
        "supplier_id", "destination_city", "transportation_mode", "order_value_usd",
        "port_congestion_level", "port_congestion_level_e",
        "weather_risk_index", "ext_risk",
        "supplier_reliability_score", "sourcing_fragility",
        "historical_disruption_count", "customs_clearance_hours",
        "num_alternate_suppliers", "sea_x_congestion",
    ]

    if ml_ready:
        with st.form("order_simulator"):
            st.markdown("Adjust parameters to see how the ML model changes its risk prediction.")

            fields_to_render = [f for f in PRIORITY_FIELDS if f in features]
            live_inputs = {}
            cols = st.columns(3)
            for i, fname in enumerate(fields_to_render):
                stat = feature_stats[fname]
                col = cols[i % 3]
                label = fname.replace("_", " ").title()
                if stat["kind"] == "categorical":
                    options = stat["options"]
                    default_idx = options.index(stat["default"]) if stat["default"] in options else 0
                    live_inputs[fname] = col.selectbox(label, options, index=default_idx)
                else:
                    live_inputs[fname] = col.number_input(
                        label,
                        min_value=stat["min"],
                        max_value=stat["max"],
                        value=stat["default"],
                    )

            submit = st.form_submit_button("Simulate Risk")

            if submit:
                # Start from each feature's training-data median/mode, then
                # overwrite with whatever the user actually set above.
                input_dict = {f: feature_stats[f]["default"] for f in features}
                input_dict.update(live_inputs)

                input_df = pd.DataFrame([input_dict])

                # Ask the encoder exactly which columns it was trained on
                trained_cat_cols = encoder.feature_names_in_
                input_df[trained_cat_cols] = encoder.transform(input_df[trained_cat_cols].astype(str))

                # Run Inference
                risk_prob = model.predict_proba(input_df[features])[0, 1]
                risk_score = int(risk_prob * 100)
                
                st.markdown("---")
                st.markdown("### 🧠 Live AI Prediction")
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric("Disruption Probability", f"{risk_score}%")
                with res_col2:
                    if risk_score > 66:
                        st.error("🚨 **High Risk.** The AI predicts a severe disruption. Recommended Action: Shift to Air Freight or trigger backup supplier.")
                    elif risk_score > 33:
                        st.warning("⚠️ **Medium Risk.** Buffer stocks may be consumed. Monitor closely.")
                    else:
                        st.success("✅ **Low Risk.** Route is clear. Standard operating procedures apply.")

# ==========================================
# TAB 4: SHIPMENT REGISTER (full dataset table)
# ==========================================
with tab4:
    st.subheader("📋 Shipment Risk Register")
    st.markdown("Every scored shipment, with its top ML-identified risk driver and recommended action.")

    display_cols = [
        c for c in [
            "shipment_id", "supplier_id", "destination_city", "transportation_mode",
            "order_value_usd", "risk_score", "risk_band", "driver_1", "recommended_action",
        ]
        if c in df.columns
    ]

    f1, f2 = st.columns([1, 2])
    with f1:
        band_options = sorted(df["risk_band"].dropna().unique()) if "risk_band" in df.columns else []
        band_filter = st.multiselect("Filter by Risk Band", band_options, default=band_options)
    with f2:
        search = st.text_input("Search Shipment ID / Supplier", "")

    filtered = df.copy()
    if band_filter and "risk_band" in filtered.columns:
        filtered = filtered[filtered["risk_band"].isin(band_filter)]
    if search:
        mask = pd.Series(False, index=filtered.index)
        for col in ["shipment_id", "supplier_id"]:
            if col in filtered.columns:
                mask |= filtered[col].astype(str).str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    table = filtered[display_cols].sort_values("risk_score", ascending=False) if "risk_score" in display_cols else filtered[display_cols]

    # Pandas Styler has a hard cell-render cap (default 262,144); with a
    # large real dataset a full unfiltered table can exceed it. Cap the
    # styled view to the top-N by risk (already sorted above) instead of
    # rendering (and styling) tens of thousands of rows at once.
    MAX_STYLED_ROWS = 1000
    display_table = table.head(MAX_STYLED_ROWS)
    if len(table) > MAX_STYLED_ROWS:
        st.caption(
            f"Showing top {MAX_STYLED_ROWS:,} by risk score, out of {len(filtered):,} matching "
            f"({len(df):,} total). Narrow the filters above to see more specific shipments."
        )
    else:
        st.caption(f"Showing {len(filtered):,} of {len(df):,} shipments")

    def _highlight_risk(row):
        color = {"High": "#ff4b4b33", "Medium": "#ffa50033", "Low": "#00c04b33"}.get(row.get("risk_band"), "")
        return [f"background-color: {color}"] * len(row)

    if "risk_band" in display_table.columns:
        st.dataframe(display_table.style.apply(_highlight_risk, axis=1), use_container_width=True, height=480)
    else:
        st.dataframe(display_table, use_container_width=True, height=480)