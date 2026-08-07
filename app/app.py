import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import os

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

# 3. Data Loading (With Hackathon-Safe Mocking)
@st.cache_data
def load_data():
    file_path = "outputs/predictions.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.warning("⚠️ `outputs/predictions.csv` not found. Using UI mock data.")
        return pd.DataFrame({
            "shipment_id": ["SC-1001", "SC-1002", "SC-1003", "SC-1004"],
            "risk_band": ["High", "Medium", "Low", "High"],
            "risk_score": [88, 45, 12, 92],
            "order_value_usd": [45000, 12000, 5000, 78000],
            "destination_city": ["Los Angeles", "New York", "Chicago", "Frankfurt"],
            "driver_1": ["port_congestion_level_e", "weather_risk", "None", "sourcing_fragility"],
            "recommended_action": ["Pre-position stock; consider air freight", "Monitor", "None", "Dual-source this lane"]
        })

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
tab1, tab2, tab3 = st.tabs(["🗺️ Global Map", "⚠️ Network Cascade", "⚙️ What-If Simulator"])

# ==========================================
# TAB 1: 3D PYDECK MAP
# ==========================================
with tab1:
    st.subheader("Live Network Cascade")
    
    geo_map = {
        "Los Angeles": [-118.2437, 34.0522],
        "New York": [-74.0060, 40.7128],
        "Chicago": [-87.6298, 41.8781],
        "Frankfurt": [8.6821, 50.1109],
        "Shanghai": [121.4737, 31.2304] 
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
                st.error(f"**Primary Shock:** {delay_days} days late from origin.")
                if net_delay > 0:
                    st.warning(f"**Buffer Exhausted:** Warehouse safety stock ({warehouse_buffer} days) is insufficient.")
                    st.error(f"**Cascading Impact:** Customer will experience a {net_delay}-day stockout.")
                    st.success(f"**Suggested Mitigation:** Expedite {net_delay} days of inventory via Air Freight.")
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
        if os.path.exists("outputs/model_tuned.pkl"):
            model = joblib.load("outputs/model_tuned.pkl")
        else:
            model = joblib.load("outputs/model.pkl")
            
        encoder = joblib.load("outputs/encoder.pkl")
        features = joblib.load("outputs/features.pkl")
        ml_ready = True
    except FileNotFoundError:
        st.error("🚨 ML models not found! Run the ML Pipeline notebook first.")
        ml_ready = False

    if ml_ready:
        with st.form("order_simulator"):
            st.markdown("Adjust parameters to see how the ML model changes its risk prediction.")
            
            c1, c2, c3 = st.columns(3)
            supplier = c1.selectbox("Supplier ID", ["S1", "S2", "S3"])
            destination = c2.selectbox("Destination City", ["Los Angeles", "New York", "Frankfurt"])
            mode = c3.selectbox("Transport Mode", ["Sea", "Air", "Rail"])
            
            c4, c5, c6 = st.columns(3)
            value_usd = c4.number_input("Order Value ($)", value=55000, step=5000)
            congestion = c5.selectbox("Port Congestion (External)", ["Low", "Medium", "High"])
            fragility = c6.selectbox("Supplier Fragility", ["Low", "High"])
            
            submit = st.form_submit_button("Simulate Risk")
            
            if submit:
                # Build baseline dictionary of zeros for all numeric features
                input_dict = {f: 0 for f in features} 
                
                # Overwrite with live inputs
                input_dict['supplier_id'] = supplier
                input_dict['destination_city'] = destination
                input_dict['transportation_mode'] = mode
                input_dict['order_value_usd'] = value_usd
                input_dict['port_congestion_level_e'] = congestion
                input_dict['sourcing_fragility'] = fragility
                
                # Fill mandatory categorical defaults
                input_dict['warehouse_id'] = 'WH_A'
                input_dict['sea_x_congestion'] = 'Low'
                input_dict['customs_clearance_hours'] = 'Low'
                input_dict['ext_risk'] = 'Low'
                
                input_df = pd.DataFrame([input_dict])
                
                # Bulletproof encoding: Ask the encoder exactly which columns it was trained on
                trained_cat_cols = encoder.feature_names_in_
                input_df[trained_cat_cols] = encoder.transform(input_df[trained_cat_cols])
                
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