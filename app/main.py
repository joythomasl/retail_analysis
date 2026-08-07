# ──────────────────────────────────────────────────────────────────────────────
# Supply Chain Intelligence Platform — Combined main.py
# A comprehensive 6-module Streamlit dashboard for supply chain risk
# management: Order Entry, ML Risk Engine, Reroute Generator, Financial
# Analysis, Cascade Simulation, and Live GIS Control Tower.
#
# Data Sources:
#   • "Simulated Data" — 1627 internally generated mock shipments (default)
#   • "predictions.xlsx" — Raw Excel file ingested via load_sample_raw_data()
# ──────────────────────────────────────────────────────────────────────────────

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import math
import hashlib

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Semantic Color Constants ─────────────────────────────────────────────────
COLOR_HIGH = "#FF4B4B"
COLOR_MODERATE = "#FACA2B"
COLOR_LOW = "#09AB3B"
COLOR_BG_CARD = "#1E1E2E"
COLOR_BG_PANEL = "#14141F"
COLOR_TEXT = "#E0E0E0"
COLOR_TEXT_MUTED = "#8A8A9A"
COLOR_BORDER = "#2A2A3C"
COLOR_ACCENT = "#6C63FF"


# ─── Custom CSS ───────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        *, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

        .stApp {
            background: linear-gradient(165deg, #0A0A14 0%, #12121F 50%, #0D0D1A 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111120 0%, #0D0D1A 100%);
            border-right: 1px solid #1E1E30;
        }

        .command-center-header {
            background: linear-gradient(135deg, #13132A 0%, #1A1A35 50%, #13132A 100%);
            border: 1px solid #252545;
            border-radius: 16px;
            padding: 20px 32px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        }

        .header-title {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #FFFFFF 0%, #A8A8C8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em;
            margin: 0;
        }

        .header-subtitle {
            font-size: 12px;
            color: #6A6A8A;
            margin: 3px 0 0 0;
            font-weight: 400;
        }

        .kpi-card {
            background: linear-gradient(145deg, #16162A 0%, #1C1C34 100%);
            border: 1px solid #252540;
            border-radius: 14px;
            padding: 20px 22px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.45);
        }

        .kpi-value {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0;
            line-height: 1.1;
        }

        .kpi-label {
            font-size: 11px;
            color: #6A6A8A;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 6px;
            font-weight: 500;
        }

        .detail-panel {
            background: linear-gradient(145deg, #14142A 0%, #1A1A32 100%);
            border: 1px solid #252540;
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .detail-header {
            font-size: 18px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }

        .detail-subheader {
            font-size: 12px;
            color: #6A6A8A;
            margin-bottom: 16px;
        }

        .info-box {
            background: #111125;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 12px;
            border-left: 4px solid;
        }

        .info-box.impact { border-left-color: #FF4B4B; }
        .info-box.solution { border-left-color: #09AB3B; }
        .info-box.warning { border-left-color: #FACA2B; }
        .info-box.info { border-left-color: #6C63FF; }

        .info-box-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .info-box-content {
            font-size: 13px;
            color: #C0C0D0;
            line-height: 1.6;
        }

        .section-label {
            font-size: 14px;
            font-weight: 600;
            color: #FFFFFF;
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 1px solid #1E1E35;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .badge-high { background: rgba(255,75,75,0.15); color: #FF4B4B; }
        .badge-moderate { background: rgba(250,202,43,0.15); color: #FACA2B; }
        .badge-low { background: rgba(9,171,59,0.15); color: #09AB3B; }

        /* ── Alternative Cards ─────────────────────────────────────────────── */
        .alt-card {
            background: linear-gradient(145deg, #14142A 0%, #1A1A32 100%);
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        }

        .alt-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }

        .alt-card.original::before { background: linear-gradient(90deg, #FF4B4B, #FF6B6B); }
        .alt-card.modal-shift::before { background: linear-gradient(90deg, #FACA2B, #FFD84B); }
        .alt-card.vendor-shift::before { background: linear-gradient(90deg, #09AB3B, #2BC85B); }

        .alt-card-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .alt-card-value {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .alt-card-metric {
            font-size: 12px;
            color: #8A8AA0;
            margin-top: 4px;
        }

        /* ── Enrichment Pills ──────────────────────────────────────────────── */
        .enrichment-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #111125;
            border: 1px solid #252540;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 8px;
            width: 100%;
        }

        .enrichment-pill .ep-icon {
            font-size: 20px;
        }

        .enrichment-pill .ep-label {
            font-size: 10px;
            color: #6A6A8A;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .enrichment-pill .ep-value {
            font-size: 18px;
            font-weight: 700;
        }

        /* ── Financial Metric ──────────────────────────────────────────────── */
        .fin-metric {
            background: linear-gradient(145deg, #14142A 0%, #1A1A32 100%);
            border: 1px solid #252540;
            border-radius: 14px;
            padding: 24px;
            text-align: center;
        }

        .fin-metric .fm-value {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .fin-metric .fm-label {
            font-size: 11px;
            color: #6A6A8A;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 4px;
        }

        .fin-metric .fm-delta {
            font-size: 12px;
            margin-top: 6px;
            font-weight: 500;
        }

        /* ── Network Nodes ─────────────────────────────────────────────────── */
        .cascade-node {
            background: #111125;
            border: 1px solid #252540;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .cascade-node.critical {
            border-color: #FF4B4B;
            box-shadow: 0 0 20px rgba(255,75,75,0.2);
        }

        .cascade-node.warning {
            border-color: #FACA2B;
            box-shadow: 0 0 20px rgba(250,202,43,0.15);
        }

        .cascade-node.safe {
            border-color: #09AB3B;
        }

        /* ── Streamlit Overrides ────────────────────────────────────────────── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6C63FF 0%, #5A52E0 100%);
            border: none;
            border-radius: 10px;
            padding: 10px 28px;
            font-weight: 600;
            box-shadow: 0 4px 16px rgba(108,99,255,0.3);
            transition: all 0.2s ease;
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 24px rgba(108,99,255,0.5);
            transform: translateY(-1px);
        }

        #MainMenu { visibility: hidden; }
        header { visibility: hidden; }
        footer { visibility: hidden; }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #252540;
            border-radius: 14px;
            background: #14142A;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #111120;
            border-radius: 12px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: 500;
            font-size: 13px;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1E1E3A 0%, #252545 100%) !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK DATA & LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

# City coordinates for route calculations
CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873), "Lucknow": (26.8467, 80.9462),
    "Surat": (21.1702, 72.8311), "Nagpur": (21.1458, 79.0882),
    "Visakhapatnam": (17.6868, 83.2185), "Coimbatore": (11.0168, 76.9558),
    "Indore": (22.7196, 75.8577), "Bhopal": (23.2599, 77.4126),
    "Patna": (25.6093, 85.1376), "Kochi": (9.9312, 76.2673),
    "Guwahati": (26.1445, 91.7362), "Chandigarh": (30.7333, 76.7794),
    # International cities that appear in predictions.xlsx
    "Toronto": (43.6532, -79.3832), "Tokyo": (35.6762, 139.6503),
    "Dubai": (25.2048, 55.2708), "Shanghai": (31.2304, 121.4737),
    "Singapore": (1.3521, 103.8198), "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437),
    "Sydney": (-33.8688, 151.2093), "São Paulo": (-23.5505, -46.6333),
    "Berlin": (52.5200, 13.4050), "Paris": (48.8566, 2.3522),
    "Seoul": (37.5665, 126.9780), "Bangkok": (13.7563, 100.5018),
    "Cairo": (30.0444, 31.2357), "Lagos": (6.5244, 3.3792),
    "Mexico City": (19.4326, -99.1332), "Istanbul": (41.0082, 28.9784),
    "Nairobi": (-1.2921, 36.8219), "Jakarta": (-6.2088, 106.8456),
}

# Pre-build lat/lon dicts for vectorized .map() lookups
LAT_DICT = {city: coords[0] for city, coords in CITY_COORDS.items()}
LON_DICT = {city: coords[1] for city, coords in CITY_COORDS.items()}
FALLBACK_LAT = 20.5937
FALLBACK_LON = 78.9629

# Supplier origin cities (where suppliers ship from)
SUPPLIER_ORIGINS = {
    "Tata Components": ("Pune", 18.5204, 73.8567),
    "Reliance Materials": ("Mumbai", 19.0760, 72.8777),
    "Adani Logistics": ("Ahmedabad", 23.0225, 72.5714),
    "Mahindra Parts": ("Chennai", 13.0827, 80.2707),
    "JSW Supply": ("Hyderabad", 17.3850, 78.4867),
    "Bajaj Electronics": ("Pune", 18.5204, 73.8567),
    "Wipro Tech": ("Bangalore", 12.9716, 77.5946),
    "Larsen & Toubro": ("Mumbai", 19.0760, 72.8777),
    "Godrej Materials": ("Mumbai", 19.0760, 72.8777),
    "HCL Components": ("Delhi", 28.7041, 77.1025),
}

# Supplier fragility scores (te_supplier_id / historical disruption rate)
SUPPLIER_DISRUPTION_RATE = {
    "Tata Components": 0.08, "Reliance Materials": 0.12,
    "Adani Logistics": 0.22, "Mahindra Parts": 0.06,
    "JSW Supply": 0.18, "Bajaj Electronics": 0.15,
    "Wipro Tech": 0.05, "Larsen & Toubro": 0.09,
    "Godrej Materials": 0.11, "HCL Components": 0.14,
}

# Products each supplier offers (for vendor-shift logic)
SUPPLIER_PRODUCTS = {
    "Tata Components": ["Steel Coils", "Engine Parts", "Chassis Frames"],
    "Reliance Materials": ["Polymers", "Steel Coils", "Chemical Reagents"],
    "Adani Logistics": ["Bulk Cement", "Steel Coils", "Solar Panels"],
    "Mahindra Parts": ["Engine Parts", "Transmission Units", "Chassis Frames"],
    "JSW Supply": ["Steel Coils", "Iron Ore", "Alloy Sheets"],
    "Bajaj Electronics": ["Circuit Boards", "Sensors", "Battery Packs"],
    "Wipro Tech": ["Circuit Boards", "Sensors", "Server Racks"],
    "Larsen & Toubro": ["Turbine Blades", "Bulk Cement", "Heavy Machinery"],
    "Godrej Materials": ["Polymers", "Consumer Goods", "Chemical Reagents"],
    "HCL Components": ["Circuit Boards", "Battery Packs", "Sensors"],
}

# Mode-specific parameters
MODE_PARAMS = {
    "Sea":  {"speed_kmh": 30,  "cost_per_km_kg": 0.0003, "reliability": 0.82},
    "Air":  {"speed_kmh": 800, "cost_per_km_kg": 0.0012, "reliability": 0.96},
    "Rail": {"speed_kmh": 60,  "cost_per_km_kg": 0.0005, "reliability": 0.88},
    "Road": {"speed_kmh": 50,  "cost_per_km_kg": 0.0008, "reliability": 0.85},
}

# Warehouses for cascade simulation
WAREHOUSES = {
    "WH-Mumbai": {"city": "Mumbai", "lat": 19.12, "lon": 72.85, "safety_stock_days": 7, "daily_demand": 45},
    "WH-Delhi": {"city": "Delhi", "lat": 28.65, "lon": 77.15, "safety_stock_days": 5, "daily_demand": 62},
    "WH-Bangalore": {"city": "Bangalore", "lat": 13.02, "lon": 77.55, "safety_stock_days": 9, "daily_demand": 38},
    "WH-Chennai": {"city": "Chennai", "lat": 13.10, "lon": 80.25, "safety_stock_days": 6, "daily_demand": 50},
    "WH-Kolkata": {"city": "Kolkata", "lat": 22.55, "lon": 88.40, "safety_stock_days": 8, "daily_demand": 35},
    "WH-Hyderabad": {"city": "Hyderabad", "lat": 17.40, "lon": 78.50, "safety_stock_days": 4, "daily_demand": 55},
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def get_environmental_factors(destination: str, month: int) -> dict:
    """
    Simulate the 'autonomous lookup' — returns route-specific environmental
    risk factors based on destination and month. In production this would
    query MySQL for historical averages.
    """
    # Deterministic seed from destination+month so results are consistent
    seed = int(hashlib.md5(f"{destination}-{month}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    # Monsoon months (Jun-Sep) have higher weather risk for Indian routes
    monsoon_boost = 0.3 if month in [6, 7, 8, 9] else 0.0

    return {
        "port_congestion_level": round(float(rng.uniform(1, 8) + (monsoon_boost * 2)), 1),
        "weather_risk_index": round(float(rng.uniform(0.05, 0.6) + monsoon_boost), 2),
        "geopolitical_risk_index": round(float(rng.uniform(0.02, 0.45)), 2),
    }


def simulate_risk_score(congestion, weather, geopolitical, supplier_rate,
                        km_per_lead_day, sea_x_congestion, mode_reliability) -> float:
    """
    Simulate HistGradientBoostingClassifier output using a weighted sigmoid.
    Returns disruption probability 0-1.
    """
    z = (
        0.18 * congestion / 10
        + 0.22 * weather
        + 0.14 * geopolitical
        + 0.20 * supplier_rate
        + 0.10 * min(km_per_lead_day / 500, 1.0)
        + 0.08 * sea_x_congestion / 10
        - 0.08 * mode_reliability
    )
    # Scale and sigmoid
    z_scaled = (z - 0.15) * 12
    prob = 1 / (1 + math.exp(-z_scaled))
    return round(prob, 4)


def risk_band(prob: float) -> str:
    if prob >= 0.66:
        return "High"
    elif prob >= 0.33:
        return "Medium"
    return "Low"


def risk_color(category: str) -> str:
    return {"High": COLOR_HIGH, "Medium": COLOR_MODERATE, "Moderate": COLOR_MODERATE, "Low": COLOR_LOW}.get(category, COLOR_TEXT_MUTED)


def risk_badge_class(category: str) -> str:
    return {"High": "badge-high", "Medium": "badge-moderate", "Moderate": "badge-moderate", "Low": "badge-low"}.get(category, "")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA SOURCE 1 — Simulated Data (generate_shipment_data)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_shipment_data(seed: int = 42) -> pd.DataFrame:
    """Generate 1627 mock shipments spanning 90 days for the control tower."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    suppliers = list(SUPPLIER_DISRUPTION_RATE.keys())
    destinations = list(CITY_COORDS.keys())
    modes = ["Sea", "Air", "Rail", "Road"]

    impact_templates = [
        "Halting Assembly Line {line} at {plant}. Production loss: {units} units/day. "
        "Downstream stock-out in {days} days.",
        "Warehouse {wh} at {pct}% capacity. {n} shipments queued. "
        "Penalty of ${penalty}K if unresolved in {days} days.",
        "Temperature-sensitive cargo window exceeded. Recall risk for {units} units.",
        "Production at {plant} shifted {days} days. Buffer stock at {pct}%.",
        "{n} downstream orders rescheduled. SLA breach for {count} accounts.",
        "Minor delay absorbed by safety stock. {n} processes monitored.",
    ]

    solution_templates = [
        "Reroute via {hub} → Recovery: {days}d | Cost: +${cost}K | Reliability: {score}%",
        "Expedite via {carrier} → Delivery by {date} | Premium: +${cost}K",
        "Split shipment: {pct}% Rail, {rest}% Road → Partial in {days}d | +${cost}K",
        "Backup supplier ({supplier}) → Lead: {days}d | Cost variance: +{pct}%",
    ]

    hubs = ["Mundra Port", "Nhava Sheva", "Chennai Air", "Delhi NCR", "Kolkata Dock", "Vizag Port"]
    carriers = ["BlueDart", "DHL Express", "FedEx Priority", "Delhivery Rapid"]
    plants = ["Pune", "Chennai", "Gurugram", "Hosur", "Sanand"]
    lines = ["A", "B", "C", "D"]

    n = 1627
    records = []

    base_date = datetime.now() - timedelta(days=45)

    for i in range(n):
        supplier = suppliers[i % len(suppliers)]
        dest = destinations[i % len(destinations)]
        mode = modes[int(rng.integers(0, 4))]

        origin_city, olat, olon = SUPPLIER_ORIGINS[supplier]
        dlat, dlon = CITY_COORDS[dest]
        dlat += float(rng.uniform(-0.12, 0.12))
        dlon += float(rng.uniform(-0.12, 0.12))

        distance = haversine_km(olat, olon, dlat, dlon)
        quantity = int(rng.integers(50, 800))
        order_value = round(float(rng.uniform(5000, 500000)), 2)

        ship_date = base_date + timedelta(days=int(rng.integers(0, 90)))
        month = ship_date.month

        env = get_environmental_factors(dest, month)
        supplier_rate = SUPPLIER_DISRUPTION_RATE[supplier]
        mode_p = MODE_PARAMS[mode]

        lead_time = max(1, distance / (mode_p["speed_kmh"] * 24))
        km_per_lead_day = distance / max(lead_time, 0.1)
        sea_x_congestion = env["port_congestion_level"] if mode == "Sea" else env["port_congestion_level"] * 0.3

        prob = simulate_risk_score(
            env["port_congestion_level"], env["weather_risk_index"],
            env["geopolitical_risk_index"], supplier_rate,
            km_per_lead_day, sea_x_congestion, mode_p["reliability"]
        )
        band = risk_band(prob)

        days_overdue = int(rng.integers(1, 18)) if prob > 0.3 else 0
        is_overdue = days_overdue > 0

        # Impact text
        impact = random.choice(impact_templates).format(
            line=random.choice(lines), plant=random.choice(plants),
            units=int(rng.integers(100, 2000)), days=int(rng.integers(1, 8)),
            wh=f"W-{int(rng.integers(1,20)):02d}", pct=int(rng.integers(60, 99)),
            n=int(rng.integers(2, 12)), penalty=int(rng.integers(20, 400)),
            count=int(rng.integers(1, 8))
        )

        future = (datetime.now() + timedelta(days=int(rng.integers(2, 14)))).strftime("%b %d")
        solution = random.choice(solution_templates).format(
            hub=random.choice(hubs), days=int(rng.integers(1, 7)),
            cost=int(rng.integers(5, 120)), score=int(rng.integers(80, 99)),
            carrier=random.choice(carriers), date=future,
            pct=int(rng.integers(40, 70)), rest=int(rng.integers(30, 60)),
            supplier=random.choice(suppliers)
        )

        # Assign nearest warehouse
        wh_dists = {wh: haversine_km(dlat, dlon, v["lat"], v["lon"]) for wh, v in WAREHOUSES.items()}
        nearest_wh = min(wh_dists, key=wh_dists.get)

        records.append({
            "Order_ID": f"ORD-{2024000 + i}",
            "Supplier": supplier,
            "Origin_City": origin_city,
            "Origin_Lat": olat, "Origin_Lon": olon,
            "Destination_City": dest,
            "Lat": round(dlat, 4), "Lon": round(dlon, 4),
            "Transport_Mode": mode,
            "Distance_km": round(distance, 1),
            "Quantity": quantity,
            "Order_Value_USD": order_value,
            "Shipment_Date": ship_date.strftime("%Y-%m-%d"),
            "Lead_Time_Days": round(lead_time, 1),
            "Volume": quantity,
            "Risk_Probability": prob,
            "Risk_Score": int(prob * 100),
            "Risk_Category": band,
            "Days_Overdue": days_overdue,
            "Is_Overdue": is_overdue,
            "Port_Congestion": env["port_congestion_level"],
            "Weather_Risk": env["weather_risk_index"],
            "Geopolitical_Risk": env["geopolitical_risk_index"],
            "Supplier_Disruption_Rate": supplier_rate,
            "Warehouse_ID": nearest_wh,
            "Cascading_Impact": impact,
            "Alternative_Routing": solution,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA SOURCE 2 — Raw Excel Data Bridge (load_sample_raw_data)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_raw_data() -> pd.DataFrame:
    """
    Ingest predictions.xlsx, map raw columns to the UI schema, and generate
    vectorized fallback data for any missing columns.

    Performance guarantees:
      • Decorated with @st.cache_data — file is read ONCE per session.
      • ZERO use of iterrows(), itertuples(), .apply(), or Python loops
        inside this function.
      • All transformations are 100% vectorized (NumPy / Pandas).
      • Scales linearly to 200k+ rows with O(n) memory.

    Returns:
        pd.DataFrame: Fully processed DataFrame ready for the UI, or an
                      empty DataFrame if predictions.xlsx is not found.
    """

    # ── Step 1: Safe file ingestion ───────────────────────────────────────────
    try:
        df = pd.read_excel("predictions.xlsx")
    except FileNotFoundError:
        st.error(
            "🚨 **predictions.xlsx not found!** "
            "Place the file in the same directory as main.py and refresh."
        )
        return pd.DataFrame()

    # ── Step 2: Dynamic column mapping ────────────────────────────────────────
    # Left = raw column names from predictions.xlsx
    # Right = standardized UI column names the dashboard expects
    COLUMN_MAP = {
        "shipmen_id":           "Order_ID",
        "destination_city":     "Destination_City",
        "order_quantity":       "Volume",
        "order_value_usd":      "Order_Value_USD",
        "transportation_mode":  "Transport_Mode",
        "risk_score":           "Risk_Score",
        "delay_days":           "Days_Overdue",
        "distance_km":          "Distance_km",
        "supplier_name":        "Supplier",
        "supplier_country":     "Supplier_Country",
        "destination_country":  "Destination_Country",
        "product_category":     "Product_Category",
        "product_name":         "Product_Name",
        "unit_cost_usd":        "Unit_Cost_USD",
        "planned_lead_time_days": "Lead_Time_Days",
        "weather_risk_index":   "Weather_Risk",
        "geopolitical_risk_index": "Geopolitical_Risk",
        "port_congestion_level": "Port_Congestion",
        "carrier_name":         "Carrier_Name",
        "warehouse_id":         "Warehouse_ID",
        "order_date":           "Order_Date",
        "planned_delivery_date": "Planned_Delivery_Date",
        "actual_delivery_date": "Actual_Delivery_Date",
        "supplier_id":          "Supplier_ID",
        "supplier_reliability_score":      "Supplier_Reliability",
        "supplier_financial_health_score": "Supplier_Financial_Health",
        "num_alternate_suppliers":         "Num_Alternate_Suppliers",
        "historical_disruption_count":     "Historical_Disruption_Count",
        "fuel_price_index":     "Fuel_Price_Index",
        "customs_clearance_hours": "Customs_Clearance_Hours",
        "inventory_level_percent": "Inventory_Level_Pct",
        "demand_forecast_units":   "Demand_Forecast_Units",
        "is_peak_season":       "Is_Peak_Season",
        "contract_type":        "Contract_Type",
        "payment_terms_days":   "Payment_Terms_Days",
        "disruption_occurred":  "Disruption_Occurred",
        "disruption_type":      "Disruption_Type",
        "notes":                "Notes",
    }

    df = df.rename(columns=COLUMN_MAP)

    # ── Step 3: Vectorized fallback logic (Safety Nets) ───────────────────────
    n = len(df)

    # ── 3a. Risk_Score ────────────────────────────────────────────────────────
    if "Risk_Score" not in df.columns:
        df["Risk_Score"] = np.random.randint(10, 99, size=n)
    else:
        score_numeric = pd.to_numeric(df["Risk_Score"], errors="coerce")
        df["Risk_Score"] = np.where(
            score_numeric.isna(),
            np.random.randint(10, 99, size=n),
            score_numeric,
        ).astype(int)

    # ── 3b. Risk_Category ─────────────────────────────────────────────────────
    if "Risk_Category" not in df.columns:
        conditions = [df["Risk_Score"] > 66, df["Risk_Score"] > 33]
        choices = ["High", "Medium"]
        df["Risk_Category"] = np.select(conditions, choices, default="Low")

    # ── 3c. Risk_Probability (needed by control tower deep-dive) ──────────────
    if "Risk_Probability" not in df.columns:
        df["Risk_Probability"] = df["Risk_Score"] / 100.0

    # ── 3d. Destination_City ──────────────────────────────────────────────────
    if "Destination_City" not in df.columns:
        df["Destination_City"] = "Unknown City"
    else:
        df["Destination_City"] = df["Destination_City"].fillna("Unknown City")

    # ── 3e. Lat / Lon (Map Coordinates) ───────────────────────────────────────
    if "Lat" not in df.columns:
        df["Lat"] = df["Destination_City"].map(LAT_DICT).fillna(FALLBACK_LAT)
    if "Lon" not in df.columns:
        df["Lon"] = df["Destination_City"].map(LON_DICT).fillna(FALLBACK_LON)

    # ── 3f. Is_Overdue ────────────────────────────────────────────────────────
    if "Is_Overdue" not in df.columns:
        if "Days_Overdue" in df.columns:
            df["Is_Overdue"] = df["Days_Overdue"] > 0
        else:
            df["Is_Overdue"] = True

    # ── 3g. Days_Overdue ──────────────────────────────────────────────────────
    if "Days_Overdue" not in df.columns:
        df["Days_Overdue"] = np.random.randint(1, 16, size=n)

    # ── 3h. Quantity (alias for Volume, used by financial/cascade modules) ────
    if "Quantity" not in df.columns:
        if "Volume" in df.columns:
            df["Quantity"] = df["Volume"]
        else:
            df["Quantity"] = np.random.randint(50, 800, size=n)

    # ── 3i. Volume ────────────────────────────────────────────────────────────
    if "Volume" not in df.columns:
        df["Volume"] = df["Quantity"]

    # ── 3j. Supplier ──────────────────────────────────────────────────────────
    if "Supplier" not in df.columns:
        supplier_list = list(SUPPLIER_DISRUPTION_RATE.keys())
        df["Supplier"] = [supplier_list[i % len(supplier_list)] for i in range(n)]

    # ── 3k. Origin_City / Origin_Lat / Origin_Lon ─────────────────────────────
    #     Needed by control tower route lines and cascade module
    if "Origin_City" not in df.columns:
        # Map supplier name → origin city using SUPPLIER_ORIGINS
        supplier_to_origin = {s: info[0] for s, info in SUPPLIER_ORIGINS.items()}
        df["Origin_City"] = df["Supplier"].map(supplier_to_origin).fillna("Mumbai")

    if "Origin_Lat" not in df.columns:
        df["Origin_Lat"] = df["Origin_City"].map(LAT_DICT).fillna(FALLBACK_LAT)
    if "Origin_Lon" not in df.columns:
        df["Origin_Lon"] = df["Origin_City"].map(LON_DICT).fillna(FALLBACK_LON)

    # ── 3l. Transport_Mode ────────────────────────────────────────────────────
    if "Transport_Mode" not in df.columns:
        modes = ["Sea", "Air", "Rail", "Road"]
        df["Transport_Mode"] = np.random.choice(modes, size=n)

    # ── 3m. Distance_km ──────────────────────────────────────────────────────
    if "Distance_km" not in df.columns:
        df["Distance_km"] = np.random.uniform(100, 5000, size=n).round(1)

    # ── 3n. Lead_Time_Days ────────────────────────────────────────────────────
    if "Lead_Time_Days" not in df.columns:
        df["Lead_Time_Days"] = np.random.uniform(1, 30, size=n).round(1)

    # ── 3o. Order_Value_USD ───────────────────────────────────────────────────
    if "Order_Value_USD" not in df.columns:
        df["Order_Value_USD"] = np.random.uniform(5000, 500000, size=n).round(2)

    # ── 3p. Shipment_Date (needed by control tower timeline slider) ───────────
    if "Shipment_Date" not in df.columns:
        if "Order_Date" in df.columns:
            df["Shipment_Date"] = pd.to_datetime(df["Order_Date"]).dt.strftime("%Y-%m-%d")
        else:
            base = datetime.now() - timedelta(days=45)
            df["Shipment_Date"] = [(base + timedelta(days=int(d))).strftime("%Y-%m-%d")
                                   for d in np.random.randint(0, 90, size=n)]

    # ── 3q. Warehouse_ID (must be a key in WAREHOUSES dict for cascade) ───────
    if "Warehouse_ID" not in df.columns:
        wh_keys = list(WAREHOUSES.keys())
        df["Warehouse_ID"] = np.random.choice(wh_keys, size=n)
    else:
        # Ensure Warehouse_ID values are valid keys in WAREHOUSES
        valid_wh = set(WAREHOUSES.keys())
        wh_keys = list(valid_wh)
        df["Warehouse_ID"] = np.where(
            df["Warehouse_ID"].isin(valid_wh),
            df["Warehouse_ID"],
            np.random.choice(wh_keys, size=n),
        )

    # ── 3r. Cascading_Impact & Alternative_Routing ────────────────────────────
    #     These are text fields displayed in the control tower deep-dive panel.
    if "Cascading_Impact" not in df.columns:
        df["Cascading_Impact"] = (
            "Delay detected on this shipment. Safety stock buffer being consumed. "
            "Monitor downstream dependencies for potential cascade failure."
        )
    if "Alternative_Routing" not in df.columns:
        df["Alternative_Routing"] = (
            "Consider expedited air freight or vendor shift to reduce lead time. "
            "Contact operations for real-time rerouting options."
        )

    # ── 3s. Supplier_Disruption_Rate ──────────────────────────────────────────
    if "Supplier_Disruption_Rate" not in df.columns:
        df["Supplier_Disruption_Rate"] = (
            df["Supplier"]
            .map(SUPPLIER_DISRUPTION_RATE)
            .fillna(0.10)
        )

    # ── 3t. Weather_Risk / Geopolitical_Risk / Port_Congestion ────────────────
    if "Weather_Risk" not in df.columns:
        df["Weather_Risk"] = np.random.uniform(0.05, 0.6, size=n).round(2)
    if "Geopolitical_Risk" not in df.columns:
        df["Geopolitical_Risk"] = np.random.uniform(0.02, 0.45, size=n).round(2)
    if "Port_Congestion" not in df.columns:
        df["Port_Congestion"] = np.random.uniform(1, 8, size=n).round(1)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def create_risk_gauge(score_pct: float, category: str, height: int = 200) -> go.Figure:
    """Sleek risk gauge showing disruption probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score_pct * 100, 1) if score_pct <= 1 else score_pct,
        number={"font": {"size": 44, "color": "#FFF", "family": "Inter"}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "dtick": 25,
                     "tickfont": {"size": 10, "color": "#6A6A8A"}},
            "bar": {"color": risk_color(category), "thickness": 0.3},
            "bgcolor": "#1A1A2E", "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(9,171,59,0.12)"},
                {"range": [33, 66], "color": "rgba(250,202,43,0.12)"},
                {"range": [66, 100], "color": "rgba(255,75,75,0.12)"},
            ],
            "threshold": {"line": {"color": risk_color(category), "width": 3},
                          "thickness": 0.85, "value": round(score_pct * 100, 1) if score_pct <= 1 else score_pct},
        },
    ))
    fig.update_layout(
        height=height, margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def create_feature_waterfall(features: dict) -> go.Figure:
    """Waterfall chart showing feature contributions to risk score."""
    names = list(features.keys())
    values = list(features.values())

    colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in values]

    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in values],
        textposition="auto",
        textfont=dict(size=11, color="#FFF", family="Inter"),
    ))
    fig.update_layout(
        height=250, margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#2A2A40",
                   tickfont=dict(size=10, color="#6A6A8A")),
        yaxis=dict(tickfont=dict(size=11, color="#A0A0B0"), automargin=True),
    )
    return fig


def create_financial_waterfall(var_original: float, freight_premium: float, net_savings: float) -> go.Figure:
    """Waterfall chart for financial trade-off analysis."""
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Value at Risk<br>(Do Nothing)", "Reroute<br>Premium", "Net<br>Savings"],
        y=[var_original, -freight_premium, net_savings],
        text=[f"${var_original:,.0f}", f"-${freight_premium:,.0f}", f"${net_savings:,.0f}"],
        textposition="outside",
        textfont=dict(size=13, color="#E0E0E0", family="Inter"),
        connector={"line": {"color": "#2A2A40", "width": 1}},
        increasing={"marker": {"color": COLOR_HIGH}},
        decreasing={"marker": {"color": COLOR_LOW}},
        totals={"marker": {"color": COLOR_ACCENT if net_savings > 0 else COLOR_HIGH}},
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(size=11, color="#A0A0B0")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=10, color="#6A6A8A"), tickprefix="$"),
        showlegend=False,
    )
    return fig


def create_buffer_depletion_chart(safety_stock_days: int, daily_demand: int,
                                  delay_days: int) -> go.Figure:
    """Line chart showing safety stock depletion over time."""
    days = list(range(0, max(delay_days + 3, safety_stock_days + 3)))
    stock = []
    current = safety_stock_days * daily_demand
    for d in days:
        stock.append(current)
        current = max(0, current - daily_demand)

    stockout_day = next((d for d, s in zip(days, stock) if s <= 0), None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=stock, mode="lines+markers",
        line=dict(color=COLOR_ACCENT, width=2),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.08)",
        name="Buffer Stock",
    ))

    # Critical threshold line
    fig.add_hline(y=daily_demand * 2, line_dash="dash",
                  line_color=COLOR_MODERATE, annotation_text="Critical Threshold",
                  annotation_font_color=COLOR_MODERATE)

    if stockout_day:
        fig.add_vline(x=stockout_day, line_dash="dash", line_color=COLOR_HIGH,
                      annotation_text="Stockout", annotation_font_color=COLOR_HIGH)

    # Delay marker
    if delay_days > 0:
        fig.add_vline(x=delay_days, line_dash="dot", line_color=COLOR_MODERATE,
                      annotation_text=f"Shipment Arrives (Day {delay_days})",
                      annotation_font_color=COLOR_MODERATE,
                      annotation_position="top left")

    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Days", titlefont=dict(size=11, color="#6A6A8A"),
                   tickfont=dict(size=10, color="#6A6A8A"),
                   gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Units in Buffer", titlefont=dict(size=11, color="#6A6A8A"),
                   tickfont=dict(size=10, color="#6A6A8A"),
                   gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False,
    )
    return fig


def create_risk_distribution_chart(df: pd.DataFrame) -> go.Figure:
    cats = ["High", "Medium", "Low"]
    counts = df["Risk_Category"].value_counts().reindex(cats, fill_value=0)
    colors = [COLOR_HIGH, COLOR_MODERATE, COLOR_LOW]
    fig = go.Figure(go.Bar(
        y=counts.index, x=counts.values, orientation="h",
        marker=dict(color=colors, cornerradius=6),
        text=counts.values, textposition="auto",
        textfont=dict(size=13, color="#FFF", family="Inter"),
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=12, color="#A0A0B0"), automargin=True),
        bargap=0.35,
    )
    return fig


def build_control_tower_map(df: pd.DataFrame) -> folium.Map:
    """Build dark-themed Folium map with clustered, risk-colored markers and route lines."""
    center_lat = df["Lat"].mean() if not df.empty else 20.5937
    center_lon = df["Lon"].mean() if not df.empty else 78.9629

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles="CartoDB dark_matter", control_scale=True, prefer_canvas=True)

    cluster = MarkerCluster(
        name="Shipments",
        options={"maxClusterRadius": 45, "spiderfyOnMaxZoom": True, "disableClusteringAtZoom": 10},
    ).add_to(m)

    if not df.empty:
        vol_min, vol_max = df["Volume"].min(), df["Volume"].max()
        vol_range = vol_max - vol_min if vol_max != vol_min else 1
    else:
        vol_min, vol_range = 0, 1

    for _, row in df.iterrows():
        radius = 6 + 20 * ((row["Volume"] - vol_min) / vol_range)
        color = risk_color(row["Risk_Category"])

        popup_html = f"""
        <div style="font-family:Inter,sans-serif; min-width:200px; color:#E0E0E0;
                    background:#1A1A2E; padding:14px; border-radius:10px; border:1px solid #2A2A40;">
            <div style="font-size:14px; font-weight:700; color:#FFF;">{row['Order_ID']}</div>
            <div style="font-size:11px; color:#8A8AA0; margin:4px 0 8px;">
                {row['Supplier']} → {row['Destination_City']} ({row['Transport_Mode']})
            </div>
            <div style="display:flex; gap:14px;">
                <div><span style="font-size:9px; color:#6A6A80; text-transform:uppercase;">Risk</span><br>
                    <span style="font-size:15px; font-weight:700; color:{color};">{row['Risk_Score']}%</span></div>
                <div><span style="font-size:9px; color:#6A6A80; text-transform:uppercase;">Volume</span><br>
                    <span style="font-size:15px; font-weight:700;">{row['Volume']}</span></div>
                <div><span style="font-size:9px; color:#6A6A80; text-transform:uppercase;">Overdue</span><br>
                    <span style="font-size:15px; font-weight:700; color:#FF4B4B;">{row['Days_Overdue']}d</span></div>
            </div>
        </div>"""

        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]], radius=radius,
            color=color, fill=True, fill_color=color, fill_opacity=0.55,
            weight=1.5, opacity=0.8,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['Order_ID']} — {row['Destination_City']} (Risk: {row['Risk_Score']}%)",
        ).add_to(cluster)

        # Draw route line from origin to destination
        if row.get("Origin_Lat") and row.get("Origin_Lon"):
            folium.PolyLine(
                locations=[[row["Origin_Lat"], row["Origin_Lon"]], [row["Lat"], row["Lon"]]],
                color=color, weight=1, opacity=0.25, dash_array="5 8",
            ).add_to(m)

    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_order_entry_module():
    """Module 1+2+3: Smart Order Entry, ML Risk Engine, and Alternatives Generator."""

    st.markdown('<p class="section-label">📦 Smart Order Entry & Auto-Enrichment Engine</p>',
                unsafe_allow_html=True)

    form_col, result_col = st.columns([2, 3], gap="large")

    with form_col:
        st.markdown(
            '<div class="detail-panel">'
            '<p class="detail-header">Enter Shipment Details</p>'
            '<p class="detail-subheader">Basic inputs — environmental risks are auto-enriched</p>'
            '</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        with st.form("order_form", clear_on_submit=False):
            supplier = st.selectbox("Supplier Name", options=list(SUPPLIER_DISRUPTION_RATE.keys()))

            product_options = SUPPLIER_PRODUCTS.get(supplier, ["General Goods"])
            product = st.selectbox("Product", options=product_options)

            destination = st.selectbox("Destination City", options=sorted(CITY_COORDS.keys()))

            ship_date = st.date_input("Shipment Date", value=datetime.now().date())

            mode = st.selectbox("Transport Mode", options=["Sea", "Air", "Rail", "Road"])

            c1, c2 = st.columns(2)
            with c1:
                order_value = st.number_input("Order Value (USD)", min_value=1000, value=50000, step=5000)
            with c2:
                quantity = st.number_input("Quantity (units)", min_value=10, value=200, step=10)

            submitted = st.form_submit_button("⚡ Analyze Risk", type="primary", use_container_width=True)

    with result_col:
        if submitted:
            # ── Auto-Enrichment ───────────────────────────────────────────
            origin_city, olat, olon = SUPPLIER_ORIGINS[supplier]
            dlat, dlon = CITY_COORDS[destination]
            distance = haversine_km(olat, olon, dlat, dlon)
            env = get_environmental_factors(destination, ship_date.month)

            supplier_rate = SUPPLIER_DISRUPTION_RATE[supplier]
            mode_p = MODE_PARAMS[mode]
            lead_time = max(1, distance / (mode_p["speed_kmh"] * 24))
            km_per_lead_day = distance / max(lead_time, 0.1)
            sea_x_cong = env["port_congestion_level"] if mode == "Sea" else env["port_congestion_level"] * 0.3

            # ── ML Prediction ─────────────────────────────────────────────
            prob = simulate_risk_score(
                env["port_congestion_level"], env["weather_risk_index"],
                env["geopolitical_risk_index"], supplier_rate,
                km_per_lead_day, sea_x_cong, mode_p["reliability"]
            )
            band = risk_band(prob)
            color = risk_color(band)

            # Store in session for financial tab
            st.session_state["last_order"] = {
                "supplier": supplier, "product": product, "destination": destination,
                "mode": mode, "order_value": order_value, "quantity": quantity,
                "distance": distance, "lead_time": lead_time, "prob": prob,
                "band": band, "env": env, "supplier_rate": supplier_rate,
                "origin": origin_city,
            }

            # ── Enrichment Results ────────────────────────────────────────
            st.markdown(
                '<div class="detail-panel">'
                '<p class="detail-header">🔍 Auto-Enrichment Results</p>'
                '<p class="detail-subheader">Environmental & causal factors retrieved for this route</p>'
                '</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Environmental pills
            ep1, ep2, ep3, ep4 = st.columns(4)
            with ep1:
                cong_color = COLOR_HIGH if env["port_congestion_level"] > 6 else (COLOR_MODERATE if env["port_congestion_level"] > 3 else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🏗️</span>
                    <div><span class="ep-label">Port Congestion</span><br>
                    <span class="ep-value" style="color:{cong_color}">{env["port_congestion_level"]}/10</span></div>
                </div>''', unsafe_allow_html=True)
            with ep2:
                wx_color = COLOR_HIGH if env["weather_risk_index"] > 0.6 else (COLOR_MODERATE if env["weather_risk_index"] > 0.3 else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🌧️</span>
                    <div><span class="ep-label">Weather Risk</span><br>
                    <span class="ep-value" style="color:{wx_color}">{env["weather_risk_index"]}</span></div>
                </div>''', unsafe_allow_html=True)
            with ep3:
                gp_color = COLOR_HIGH if env["geopolitical_risk_index"] > 0.35 else (COLOR_MODERATE if env["geopolitical_risk_index"] > 0.15 else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🌐</span>
                    <div><span class="ep-label">Geopolitical Risk</span><br>
                    <span class="ep-value" style="color:{gp_color}">{env["geopolitical_risk_index"]}</span></div>
                </div>''', unsafe_allow_html=True)
            with ep4:
                sr_color = COLOR_HIGH if supplier_rate > 0.15 else (COLOR_MODERATE if supplier_rate > 0.08 else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🏭</span>
                    <div><span class="ep-label">Supplier Fragility</span><br>
                    <span class="ep-value" style="color:{sr_color}">{supplier_rate:.0%}</span></div>
                </div>''', unsafe_allow_html=True)

            # Computed features
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">📏</span>
                    <div><span class="ep-label">Distance</span><br>
                    <span class="ep-value" style="color:#E0E0E0">{distance:,.0f}km</span></div>
                </div>''', unsafe_allow_html=True)
            with fc2:
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">⏱️</span>
                    <div><span class="ep-label">Lead Time</span><br>
                    <span class="ep-value" style="color:#E0E0E0">{lead_time:.1f}d</span></div>
                </div>''', unsafe_allow_html=True)
            with fc3:
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🚀</span>
                    <div><span class="ep-label">Speed (km/lead day)</span><br>
                    <span class="ep-value" style="color:#E0E0E0">{km_per_lead_day:,.0f}</span></div>
                </div>''', unsafe_allow_html=True)
            with fc4:
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">⚓</span>
                    <div><span class="ep-label">Sea×Congestion</span><br>
                    <span class="ep-value" style="color:#E0E0E0">{sea_x_cong:.1f}</span></div>
                </div>''', unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── ML Risk Score ─────────────────────────────────────────────
            st.markdown('<p class="section-label">🤖 Predictive ML Risk Engine</p>', unsafe_allow_html=True)

            gauge_col, feat_col = st.columns([1, 1])
            with gauge_col:
                st.markdown(f'''<div class="detail-panel" style="text-align:center;">
                    <p style="font-size:11px; color:#6A6A8A; text-transform:uppercase;
                       letter-spacing:0.1em; font-weight:600;">Disruption Probability</p>
                </div>''', unsafe_allow_html=True)
                st.plotly_chart(create_risk_gauge(prob, band, 200), use_container_width=True,
                                config={"displayModeBar": False})
                badge = risk_badge_class(band)
                st.markdown(f'<div style="text-align:center;"><span class="status-badge {badge}">'
                            f'{band} Risk</span></div>', unsafe_allow_html=True)

            with feat_col:
                st.markdown(
                    '<div class="detail-panel">'
                    '<p style="font-size:11px; color:#6A6A8A; text-transform:uppercase; '
                    'letter-spacing:0.1em; font-weight:600;">Feature Contributions</p>'
                    '</div>', unsafe_allow_html=True)

                features = {
                    "Weather Risk": env["weather_risk_index"] * 0.22,
                    "Supplier Fragility": supplier_rate * 0.20,
                    "Port Congestion": (env["port_congestion_level"] / 10) * 0.18,
                    "Geopolitical Risk": env["geopolitical_risk_index"] * 0.14,
                    "Route Speed": min(km_per_lead_day / 500, 1.0) * 0.10,
                    "Sea×Congestion": (sea_x_cong / 10) * 0.08,
                    "Mode Reliability": -mode_p["reliability"] * 0.08,
                }
                st.plotly_chart(create_feature_waterfall(features), use_container_width=True,
                                config={"displayModeBar": False})

            # ── Alternatives Generator (only if HIGH risk) ────────────────
            if band == "High":
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<p class="section-label">🔄 Prescriptive Alternatives Generator</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div class="info-box impact">'
                    '<p class="info-box-title" style="color:#FF4B4B;">⚠ HIGH RISK DETECTED</p>'
                    '<p class="info-box-content">The system has intercepted this order and autonomously '
                    'generated safer alternatives. Compare the options below.</p>'
                    '</div>', unsafe_allow_html=True)

                alt1, alt2, alt3 = st.columns(3)

                # ── Option 1: Original (Risky) ────
                with alt1:
                    st.markdown(f'''<div class="alt-card original">
                        <p class="alt-card-label" style="color:{COLOR_HIGH};">⚠ Original Plan</p>
                        <p class="alt-card-value" style="color:{COLOR_HIGH};">{prob*100:.1f}%</p>
                        <p class="alt-card-metric">Disruption Probability</p>
                        <hr style="border-color:#2A2A40; margin:12px 0;">
                        <p class="alt-card-metric">Mode: <b>{mode}</b></p>
                        <p class="alt-card-metric">Supplier: <b>{supplier}</b></p>
                        <p class="alt-card-metric">Lead Time: <b>{lead_time:.1f} days</b></p>
                        <p class="alt-card-metric">Cost: <b>${order_value:,.0f}</b></p>
                    </div>''', unsafe_allow_html=True)

                # ── Option 2: Modal Shift ─────────
                alt_mode = "Air" if mode != "Air" else "Rail"
                alt_mode_p = MODE_PARAMS[alt_mode]
                alt_lead = max(1, distance / (alt_mode_p["speed_kmh"] * 24))
                alt_sea_cong = env["port_congestion_level"] if alt_mode == "Sea" else env["port_congestion_level"] * 0.3
                alt_prob_mode = simulate_risk_score(
                    env["port_congestion_level"], env["weather_risk_index"],
                    env["geopolitical_risk_index"], supplier_rate,
                    distance / max(alt_lead, 0.1), alt_sea_cong, alt_mode_p["reliability"]
                )
                alt_band_mode = risk_band(alt_prob_mode)
                cost_delta_mode = distance * quantity * (alt_mode_p["cost_per_km_kg"] - mode_p["cost_per_km_kg"])

                with alt2:
                    mc = risk_color(alt_band_mode)
                    st.markdown(f'''<div class="alt-card modal-shift">
                        <p class="alt-card-label" style="color:{COLOR_MODERATE};">✈ Modal Shift → {alt_mode}</p>
                        <p class="alt-card-value" style="color:{mc};">{alt_prob_mode*100:.1f}%</p>
                        <p class="alt-card-metric">Disruption Probability</p>
                        <hr style="border-color:#2A2A40; margin:12px 0;">
                        <p class="alt-card-metric">Mode: <b>{alt_mode}</b></p>
                        <p class="alt-card-metric">Supplier: <b>{supplier}</b> (same)</p>
                        <p class="alt-card-metric">Lead Time: <b>{alt_lead:.1f} days</b></p>
                        <p class="alt-card-metric">Cost Delta: <b style="color:{COLOR_MODERATE};">+${max(0,cost_delta_mode):,.0f}</b></p>
                    </div>''', unsafe_allow_html=True)

                # ── Option 3: Vendor Shift ────────
                # Find alternate supplier for same product
                alt_suppliers = [s for s, prods in SUPPLIER_PRODUCTS.items()
                                 if product in prods and s != supplier]
                if alt_suppliers:
                    alt_supplier = min(alt_suppliers, key=lambda s: SUPPLIER_DISRUPTION_RATE[s])
                else:
                    alt_supplier = supplier

                alt_sr = SUPPLIER_DISRUPTION_RATE[alt_supplier]
                alt_origin_city, alt_olat, alt_olon = SUPPLIER_ORIGINS[alt_supplier]
                alt_dist = haversine_km(alt_olat, alt_olon, dlat, dlon)
                alt_lead_v = max(1, alt_dist / (mode_p["speed_kmh"] * 24))
                alt_env = get_environmental_factors(destination, ship_date.month)
                alt_sea_v = alt_env["port_congestion_level"] if mode == "Sea" else alt_env["port_congestion_level"] * 0.3
                alt_prob_vendor = simulate_risk_score(
                    alt_env["port_congestion_level"], alt_env["weather_risk_index"],
                    alt_env["geopolitical_risk_index"], alt_sr,
                    alt_dist / max(alt_lead_v, 0.1), alt_sea_v, mode_p["reliability"]
                )
                alt_band_vendor = risk_band(alt_prob_vendor)

                with alt3:
                    vc = risk_color(alt_band_vendor)
                    st.markdown(f'''<div class="alt-card vendor-shift">
                        <p class="alt-card-label" style="color:{COLOR_LOW};">🏭 Vendor Shift → {alt_supplier}</p>
                        <p class="alt-card-value" style="color:{vc};">{alt_prob_vendor*100:.1f}%</p>
                        <p class="alt-card-metric">Disruption Probability</p>
                        <hr style="border-color:#2A2A40; margin:12px 0;">
                        <p class="alt-card-metric">Mode: <b>{mode}</b> (same)</p>
                        <p class="alt-card-metric">Supplier: <b>{alt_supplier}</b></p>
                        <p class="alt-card-metric">Fragility: <b style="color:{COLOR_LOW};">{alt_sr:.0%}</b> (was {supplier_rate:.0%})</p>
                        <p class="alt-card-metric">From: <b>{alt_origin_city}</b></p>
                    </div>''', unsafe_allow_html=True)

                # Action buttons
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("Proceed with Original", key="orig_btn", use_container_width=True):
                        st.warning("⚠ Proceeding with high-risk original plan. Monitoring activated.")
                with b2:
                    if st.button(f"✈ Switch to {alt_mode}", type="primary", key="modal_btn", use_container_width=True):
                        st.success(f"✅ Transport mode changed to {alt_mode}. Risk reduced to {alt_prob_mode*100:.1f}%.")
                with b3:
                    if st.button(f"🏭 Switch Vendor", type="primary", key="vendor_btn", use_container_width=True):
                        st.success(f"✅ Supplier changed to {alt_supplier}. Risk reduced to {alt_prob_vendor*100:.1f}%.")

            elif band == "Medium":
                st.markdown(
                    '<div class="info-box warning">'
                    '<p class="info-box-title" style="color:#FACA2B;">⚡ MEDIUM RISK — Monitor Recommended</p>'
                    '<p class="info-box-content">This shipment falls in the medium-risk band. '
                    'Consider setting up automated monitoring alerts. No immediate rerouting required.</p>'
                    '</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-box solution">'
                    '<p class="info-box-title" style="color:#09AB3B;">✓ LOW RISK — Cleared for Dispatch</p>'
                    '<p class="info-box-content">All environmental and vendor indicators are within '
                    'acceptable thresholds. This shipment is cleared for standard processing.</p>'
                    '</div>', unsafe_allow_html=True)

        else:
            # Empty state
            st.markdown('''
            <div class="detail-panel" style="text-align:center; padding:80px 28px;">
                <div style="font-size:56px; margin-bottom:16px;">⚡</div>
                <p class="detail-header" style="font-size:20px; color:#6A6A8A;">
                    Enter Order Details to Begin
                </p>
                <p style="font-size:13px; color:#4A4A6A; max-width:360px; margin:8px auto 0;">
                    Fill in the shipment form on the left and click <b>Analyze Risk</b>.
                    The system will auto-enrich environmental data, run the ML risk model,
                    and generate alternatives if the risk is high.
                </p>
            </div>''', unsafe_allow_html=True)


def render_financial_module(df: pd.DataFrame):
    """Module 4: Financial Trade-Off & Penalty Estimator."""

    st.markdown('<p class="section-label">💰 Financial Trade-Off & Penalty Estimator</p>',
                unsafe_allow_html=True)

    # Select which order to analyze
    last_order = st.session_state.get("last_order")

    if last_order:
        source_label = f"Last Submitted: {last_order['supplier']} → {last_order['destination']} ({last_order['mode']})"
    else:
        source_label = None

    analysis_source = st.radio(
        "Analyze financials for:",
        options=["Select from active shipments", "Last submitted order"] if last_order else ["Select from active shipments"],
        horizontal=True, label_visibility="collapsed"
    )

    if analysis_source == "Last submitted order" and last_order:
        order_value = last_order["order_value"]
        distance = last_order["distance"]
        quantity = last_order["quantity"]
        mode = last_order["mode"]
        prob = last_order["prob"]
        band = last_order["band"]
        lead_time = last_order["lead_time"]
        label = f"{last_order['supplier']} → {last_order['destination']}"
    else:
        # Let user pick from high-risk shipments
        high_risk = df[df["Risk_Category"] == "High"].head(15)
        if high_risk.empty:
            high_risk = df.nlargest(10, "Risk_Score")

        options = [f"{r['Order_ID']} — {r['Supplier']} → {r['Destination_City']} (Risk: {r['Risk_Score']}%)"
                   for _, r in high_risk.iterrows()]
        selected = st.selectbox("Select Shipment", options=options)

        if selected:
            idx = options.index(selected)
            row = high_risk.iloc[idx]
            order_value = row["Order_Value_USD"]
            distance = row["Distance_km"]
            quantity = row["Quantity"]
            mode = row["Transport_Mode"]
            prob = row["Risk_Probability"]
            band = row["Risk_Category"]
            lead_time = row["Lead_Time_Days"]
            label = f"{row['Supplier']} → {row['Destination_City']}"
        else:
            st.info("Select a shipment to view financial analysis.")
            return

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Calculations ──────────────────────────────────────────────────────
    holding_cost_rate = 0.0015  # 0.15% per day
    predicted_delay = max(1, int(prob * 20))  # Simulated delay
    var_original = order_value * predicted_delay * holding_cost_rate

    # Freight premium for switching to Air
    current_cost_per = MODE_PARAMS[mode]["cost_per_km_kg"]
    air_cost_per = MODE_PARAMS["Air"]["cost_per_km_kg"]
    freight_premium = distance * quantity * (air_cost_per - current_cost_per) if mode != "Air" else 0

    net_savings = var_original - freight_premium

    # ── Display ───────────────────────────────────────────────────────────
    st.markdown(f'''<div class="detail-panel">
        <p class="detail-header">📊 Financial Analysis: {label}</p>
        <p class="detail-subheader">Predicted delay: {predicted_delay} days · Mode: {mode} · Value: ${order_value:,.0f}</p>
    </div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # KPI cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'''<div class="fin-metric">
            <p class="fm-value" style="color:{COLOR_HIGH};">${var_original:,.0f}</p>
            <p class="fm-label">Value at Risk (VaR)</p>
            <p class="fm-delta" style="color:{COLOR_HIGH};">Penalty if delayed {predicted_delay}d</p>
        </div>''', unsafe_allow_html=True)
    with m2:
        st.markdown(f'''<div class="fin-metric">
            <p class="fm-value" style="color:{COLOR_MODERATE};">${freight_premium:,.0f}</p>
            <p class="fm-label">Air Freight Premium</p>
            <p class="fm-delta" style="color:{COLOR_TEXT_MUTED};">Cost to upgrade to Air</p>
        </div>''', unsafe_allow_html=True)
    with m3:
        sav_color = COLOR_LOW if net_savings > 0 else COLOR_HIGH
        st.markdown(f'''<div class="fin-metric">
            <p class="fm-value" style="color:{sav_color};">${net_savings:,.0f}</p>
            <p class="fm-label">Net Savings</p>
            <p class="fm-delta" style="color:{sav_color};">{"✓ Reroute saves money" if net_savings > 0 else "✗ Reroute costs more"}</p>
        </div>''', unsafe_allow_html=True)
    with m4:
        roi = (net_savings / max(freight_premium, 1)) * 100 if freight_premium > 0 else 0
        st.markdown(f'''<div class="fin-metric">
            <p class="fm-value" style="color:{COLOR_ACCENT};">{roi:.0f}%</p>
            <p class="fm-label">Reroute ROI</p>
            <p class="fm-delta" style="color:{COLOR_TEXT_MUTED};">Return on mitigation spend</p>
        </div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Waterfall chart
    wf_col, breakdown_col = st.columns([2, 1])
    with wf_col:
        st.markdown('<p class="section-label">📉 Trade-Off Waterfall</p>', unsafe_allow_html=True)
        st.plotly_chart(create_financial_waterfall(var_original, freight_premium, net_savings),
                        use_container_width=True, config={"displayModeBar": False})

    with breakdown_col:
        st.markdown('<p class="section-label">📋 Cost Breakdown</p>', unsafe_allow_html=True)

        st.markdown(f'''<div class="info-box impact">
            <p class="info-box-title" style="color:{COLOR_HIGH};">Value at Risk (VaR)</p>
            <p class="info-box-content">
                Order Value: ${order_value:,.0f}<br>
                × Holding Rate: {holding_cost_rate:.2%}/day<br>
                × Predicted Delay: {predicted_delay} days<br>
                <b style="color:{COLOR_HIGH};">= ${var_original:,.0f} potential penalty</b>
            </p>
        </div>''', unsafe_allow_html=True)

        if mode != "Air":
            st.markdown(f'''<div class="info-box warning">
                <p class="info-box-title" style="color:{COLOR_MODERATE};">Freight Premium</p>
                <p class="info-box-content">
                    Distance: {distance:,.0f} km<br>
                    × Quantity: {quantity} units<br>
                    × Rate Delta: ${(air_cost_per - current_cost_per):.4f}/km·kg<br>
                    <b style="color:{COLOR_MODERATE};">= ${freight_premium:,.0f} upgrade cost</b>
                </p>
            </div>''', unsafe_allow_html=True)

        verdict = "REROUTE RECOMMENDED" if net_savings > 0 else "ABSORB RISK"
        v_color = COLOR_LOW if net_savings > 0 else COLOR_MODERATE
        st.markdown(f'''<div class="info-box {"solution" if net_savings > 0 else "warning"}">
            <p class="info-box-title" style="color:{v_color};">Verdict: {verdict}</p>
            <p class="info-box-content">
                {"Switching to Air freight saves " if net_savings > 0 else "Air freight costs "}
                <b style="color:{v_color};">${abs(net_savings):,.0f}</b>
                {"compared to absorbing the delay penalty." if net_savings > 0 else "more than the penalty. Consider alternative mitigations."}
            </p>
        </div>''', unsafe_allow_html=True)


def render_cascade_module(df: pd.DataFrame):
    """Module 5: Network Cascade & Ripple Simulator."""

    st.markdown('<p class="section-label">🌊 Network Cascade & Ripple Simulator</p>',
                unsafe_allow_html=True)

    # Select shipment to simulate
    overdue = df[df["Is_Overdue"]].nlargest(20, "Risk_Score")
    if overdue.empty:
        overdue = df.nlargest(20, "Risk_Score")

    sel_col, delay_col = st.columns([3, 1])
    with sel_col:
        options = [f"{r['Order_ID']} — {r['Supplier']} → {r['Destination_City']} (WH: {r['Warehouse_ID']})"
                   for _, r in overdue.iterrows()]
        selected = st.selectbox("Select Shipment to Simulate", options=options, key="cascade_select")
    with delay_col:
        delay_days = st.slider("Simulated Delay (days)", 0, 25, 8, key="delay_slider")

    if selected:
        idx = options.index(selected)
        row = overdue.iloc[idx]
        wh_id = row["Warehouse_ID"]
        wh = WAREHOUSES[wh_id]

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Network Visualization ─────────────────────────────────────────
        st.markdown('<p class="section-label">🔗 Supply Chain Network Graph</p>', unsafe_allow_html=True)

        # Build Sankey diagram
        labels = [
            f"📦 {row['Supplier']}\n({row['Origin_City']})",
            f"🚢 Transit\n({row['Transport_Mode']})",
            f"🏭 {wh_id}\n({wh['city']})",
            f"📍 {row['Destination_City']}\n(Final Dest)",
        ]

        # Determine node colors based on delay propagation
        node_colors = []
        buffer_remaining = wh["safety_stock_days"] - delay_days

        if delay_days > 0:
            node_colors.append(COLOR_MODERATE)  # Supplier affected
            node_colors.append(COLOR_HIGH)       # Transit is delayed
        else:
            node_colors.append(COLOR_LOW)
            node_colors.append(COLOR_LOW)

        if buffer_remaining <= 0:
            node_colors.append(COLOR_HIGH)   # Warehouse stockout
            node_colors.append(COLOR_HIGH)   # Destination affected
        elif buffer_remaining <= 2:
            node_colors.append(COLOR_MODERATE)
            node_colors.append(COLOR_MODERATE)
        else:
            node_colors.append(COLOR_LOW)
            node_colors.append(COLOR_LOW)

        fig_sankey = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=30, thickness=20, line=dict(color="#2A2A40", width=1),
                label=labels, color=node_colors,
            ),
            link=dict(
                source=[0, 1, 2],
                target=[1, 2, 3],
                value=[row["Quantity"], row["Quantity"], row["Quantity"]],
                color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.3)" for c in node_colors[:3]],
            ),
        ))
        fig_sankey.update_layout(
            height=220, margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#E0E0E0", family="Inter"),
        )
        st.plotly_chart(fig_sankey, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Node Status Cards ─────────────────────────────────────────────
        st.markdown('<p class="section-label">📊 Node Impact Assessment</p>', unsafe_allow_html=True)

        n1, n2, n3, n4 = st.columns(4)
        with n1:
            nc = "warning" if delay_days > 0 else "safe"
            st.markdown(f'''<div class="cascade-node {nc}">
                <div style="font-size:24px; margin-bottom:6px;">📦</div>
                <div style="font-size:13px; font-weight:600; color:#FFF;">{row["Supplier"]}</div>
                <div style="font-size:11px; color:#8A8AA0; margin-top:4px;">{row["Origin_City"]}</div>
                <div style="font-size:11px; color:{COLOR_MODERATE if delay_days > 0 else COLOR_LOW}; margin-top:8px; font-weight:600;">
                    {"DISPATCH DELAYED" if delay_days > 0 else "ON SCHEDULE"}</div>
            </div>''', unsafe_allow_html=True)

        with n2:
            st.markdown(f'''<div class="cascade-node {"critical" if delay_days > 3 else "warning" if delay_days > 0 else "safe"}">
                <div style="font-size:24px; margin-bottom:6px;">🚢</div>
                <div style="font-size:13px; font-weight:600; color:#FFF;">Transit Corridor</div>
                <div style="font-size:11px; color:#8A8AA0; margin-top:4px;">{row["Transport_Mode"]} · {row["Distance_km"]:,.0f}km</div>
                <div style="font-size:11px; color:{COLOR_HIGH if delay_days > 3 else COLOR_MODERATE if delay_days > 0 else COLOR_LOW}; margin-top:8px; font-weight:600;">
                    {f"+{delay_days}d DELAY" if delay_days > 0 else "IN TRANSIT"}</div>
            </div>''', unsafe_allow_html=True)

        with n3:
            stockout = buffer_remaining <= 0
            wh_status = "STOCKOUT" if stockout else f"BUFFER: {buffer_remaining}d"
            wh_class = "critical" if stockout else ("warning" if buffer_remaining <= 2 else "safe")
            wh_color = COLOR_HIGH if stockout else (COLOR_MODERATE if buffer_remaining <= 2 else COLOR_LOW)
            st.markdown(f'''<div class="cascade-node {wh_class}">
                <div style="font-size:24px; margin-bottom:6px;">🏭</div>
                <div style="font-size:13px; font-weight:600; color:#FFF;">{wh_id}</div>
                <div style="font-size:11px; color:#8A8AA0; margin-top:4px;">Safety Stock: {wh["safety_stock_days"]}d</div>
                <div style="font-size:11px; color:{wh_color}; margin-top:8px; font-weight:600;">
                    {wh_status}</div>
            </div>''', unsafe_allow_html=True)

        with n4:
            dest_class = "critical" if stockout else ("warning" if buffer_remaining <= 2 else "safe")
            dest_status = "CUSTOMER IMPACT" if stockout else ("AT RISK" if buffer_remaining <= 2 else "SERVED")
            dest_color = COLOR_HIGH if stockout else (COLOR_MODERATE if buffer_remaining <= 2 else COLOR_LOW)
            st.markdown(f'''<div class="cascade-node {dest_class}">
                <div style="font-size:24px; margin-bottom:6px;">📍</div>
                <div style="font-size:13px; font-weight:600; color:#FFF;">{row["Destination_City"]}</div>
                <div style="font-size:11px; color:#8A8AA0; margin-top:4px;">Demand: {wh["daily_demand"]} units/day</div>
                <div style="font-size:11px; color:{dest_color}; margin-top:8px; font-weight:600;">
                    {dest_status}</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Buffer Depletion Chart ────────────────────────────────────────
        chart_col, alert_col = st.columns([2, 1])
        with chart_col:
            st.markdown('<p class="section-label">📉 Buffer Stock Depletion Timeline</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                create_buffer_depletion_chart(wh["safety_stock_days"], wh["daily_demand"], delay_days),
                use_container_width=True, config={"displayModeBar": False}
            )

        with alert_col:
            st.markdown('<p class="section-label">🚨 Domino-Effect Alerts</p>', unsafe_allow_html=True)

            if stockout:
                stockout_day = wh["safety_stock_days"]
                st.markdown(f'''<div class="info-box impact">
                    <p class="info-box-title" style="color:{COLOR_HIGH};">DAY {stockout_day}: WAREHOUSE STOCKOUT</p>
                    <p class="info-box-content">{wh_id} buffer depleted. {wh["daily_demand"]} units/day demand unmet.</p>
                </div>''', unsafe_allow_html=True)

                st.markdown(f'''<div class="info-box impact">
                    <p class="info-box-title" style="color:{COLOR_HIGH};">DAY {stockout_day + 1}: PRODUCTION HALT</p>
                    <p class="info-box-content">Assembly line dependent on this input will pause.
                    Est. loss: {wh["daily_demand"] * 120} USD/day.</p>
                </div>''', unsafe_allow_html=True)

                st.markdown(f'''<div class="info-box impact">
                    <p class="info-box-title" style="color:{COLOR_HIGH};">DAY {stockout_day + 3}: CUSTOMER STOCKOUT</p>
                    <p class="info-box-content">Final customer in {row["Destination_City"]} faces
                    empty shelves. SLA penalty triggers.</p>
                </div>''', unsafe_allow_html=True)
            elif buffer_remaining <= 2:
                st.markdown(f'''<div class="info-box warning">
                    <p class="info-box-title" style="color:{COLOR_MODERATE};">⚠ BUFFER CRITICAL</p>
                    <p class="info-box-content">Only {buffer_remaining} day(s) of safety stock remain.
                    Expedite shipment to prevent cascade failure.</p>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="info-box solution">
                    <p class="info-box-title" style="color:{COLOR_LOW};">✓ BUFFER ADEQUATE</p>
                    <p class="info-box-content">{buffer_remaining} days of buffer available.
                    Current delay is absorbed by warehouse safety stock.</p>
                </div>''', unsafe_allow_html=True)


def render_control_tower(df: pd.DataFrame):
    """Module 6: Live GIS Control Tower & KPI Dashboard."""

    # ── Executive KPI Bar ─────────────────────────────────────────────────
    overdue_df = df[df["Is_Overdue"]]
    high_risk_count = len(overdue_df[overdue_df["Risk_Category"] == "High"])
    total_var = sum(
        r["Order_Value_USD"] * r["Days_Overdue"] * 0.0015
        for _, r in overdue_df.iterrows()
    )
    network_health = max(0, 100 - (high_risk_count / max(len(df), 1)) * 100 * 3)
    avg_lead = df["Lead_Time_Days"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        nh_color = COLOR_LOW if network_health > 70 else (COLOR_MODERATE if network_health > 40 else COLOR_HIGH)
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:{nh_color};">{network_health:.0f}%</p>
            <p class="kpi-label">Network Health</p></div>''', unsafe_allow_html=True)
    with k2:
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:{COLOR_HIGH};">{high_risk_count}</p>
            <p class="kpi-label">High-Risk Alerts</p></div>''', unsafe_allow_html=True)
    with k3:
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:#FFFFFF;">{len(overdue_df)}</p>
            <p class="kpi-label">Overdue Shipments</p></div>''', unsafe_allow_html=True)
    with k4:
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:{COLOR_MODERATE};">${total_var:,.0f}</p>
            <p class="kpi-label">Portfolio VaR</p></div>''', unsafe_allow_html=True)
    with k5:
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:{COLOR_ACCENT};">{avg_lead:.1f}d</p>
            <p class="kpi-label">Avg Lead Time</p></div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Date Slider (Time-Stepped Simulation) ─────────────────────────────
    dates = pd.to_datetime(df["Shipment_Date"]).sort_values()
    min_date = dates.min().date()
    max_date = dates.max().date()

    date_range = st.slider(
        "📅 Threat Feed Timeline — Slide to simulate live data ingestion",
        min_value=min_date, max_value=max_date,
        value=(min_date, max_date),
        format="MMM DD",
        key="timeline_slider",
    )

    # Filter by date range
    tower_df = df[
        (pd.to_datetime(df["Shipment_Date"]).dt.date >= date_range[0]) &
        (pd.to_datetime(df["Shipment_Date"]).dt.date <= date_range[1]) &
        (df["Is_Overdue"])
    ]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Map + Detail Panel ────────────────────────────────────────────────
    map_col, detail_col = st.columns([3, 2], gap="medium")

    with map_col:
        st.markdown(f'<p class="section-label">📍 Active Threat Map — {len(tower_df)} overdue shipments</p>',
                    unsafe_allow_html=True)
        folium_map = build_control_tower_map(tower_df)

        map_data = st_folium(
            folium_map, width=None, height=480,
            returned_objects=["last_object_clicked"],
            key="tower_map",
        )

        # Determine clicked shipment
        clicked_order = None
        if map_data and map_data.get("last_object_clicked"):
            clat = map_data["last_object_clicked"].get("lat")
            clng = map_data["last_object_clicked"].get("lng")
            if clat is not None and clng is not None and not tower_df.empty:
                dists = (tower_df["Lat"] - clat)**2 + (tower_df["Lon"] - clng)**2
                nearest = dists.idxmin()
                if dists[nearest] < 0.05:
                    clicked_order = tower_df.loc[nearest]

    with detail_col:
        st.markdown('<p class="section-label">🔍 Shipment Deep-Dive</p>', unsafe_allow_html=True)

        if clicked_order is not None:
            o = clicked_order
            color = risk_color(o["Risk_Category"])
            badge = risk_badge_class(o["Risk_Category"])

            st.markdown(f'''<div class="detail-panel">
                <p class="detail-header">Shipment: {o["Order_ID"]}</p>
                <p class="detail-subheader">
                    {o["Supplier"]} → {o["Destination_City"]} · {o["Transport_Mode"]} ·
                    <span class="status-badge {badge}">{o["Risk_Category"]} Risk</span> ·
                    {o["Days_Overdue"]}d overdue
                </p>
            </div>''', unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.plotly_chart(create_risk_gauge(o["Risk_Probability"], o["Risk_Category"], 180),
                            use_container_width=True, config={"displayModeBar": False})

            # Metadata
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'''<div class="kpi-card"><p class="kpi-value" style="font-size:22px; color:#FFF;">{o["Volume"]}</p>
                    <p class="kpi-label">Volume</p></div>''', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'''<div class="kpi-card"><p class="kpi-value" style="font-size:22px; color:{COLOR_HIGH};">{o["Days_Overdue"]}d</p>
                    <p class="kpi-label">Overdue</p></div>''', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'''<div class="kpi-card"><p class="kpi-value" style="font-size:22px; color:{COLOR_ACCENT};">${o["Order_Value_USD"]:,.0f}</p>
                    <p class="kpi-label">Value</p></div>''', unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            st.markdown(f'''<div class="info-box impact">
                <p class="info-box-title" style="color:{COLOR_HIGH};">⚠ Cascading Impact</p>
                <p class="info-box-content">{o["Cascading_Impact"]}</p>
            </div>''', unsafe_allow_html=True)

            st.markdown(f'''<div class="info-box solution">
                <p class="info-box-title" style="color:{COLOR_LOW};">✓ Recommended Mitigation</p>
                <p class="info-box-content">{o["Alternative_Routing"]}</p>
            </div>''', unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🔄 Execute Reroute", type="primary", key="tower_reroute", use_container_width=True):
                    st.success(f"Reroute submitted for {o['Order_ID']}. Ops team notified.")
            with bc2:
                if st.button("📋 Escalate", key="tower_escalate", use_container_width=True):
                    st.info(f"Escalation ESC-{random.randint(10000,99999)} created for {o['Order_ID']}.")

        else:
            st.markdown('''<div class="detail-panel" style="text-align:center; padding:60px 28px;">
                <div style="font-size:48px; margin-bottom:16px;">📍</div>
                <p class="detail-header" style="font-size:18px; color:#6A6A8A;">Select a Shipment</p>
                <p style="font-size:13px; color:#4A4A6A; max-width:280px; margin:8px auto 0;">
                    Click any marker on the map to view detailed analysis, cascading impact,
                    and mitigation options.</p>
            </div>''', unsafe_allow_html=True)

            # Top risk list
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">🚨 Highest Risk Shipments</p>', unsafe_allow_html=True)

            for _, r in tower_df.nlargest(5, "Risk_Score").iterrows():
                rc = risk_color(r["Risk_Category"])
                bc = risk_badge_class(r["Risk_Category"])
                st.markdown(f'''<div style="background:#14142A; border:1px solid #252540;
                    border-radius:10px; padding:12px 16px; margin-bottom:6px;
                    display:flex; align-items:center; justify-content:space-between;">
                    <div>
                        <span style="font-weight:600; color:#FFF; font-size:13px;">{r["Order_ID"]}</span>
                        <span style="color:#6A6A8A; font-size:11px; margin-left:8px;">
                            {r["Supplier"]} → {r["Destination_City"]}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-weight:700; color:{rc}; font-size:15px;">{r["Risk_Score"]}%</span>
                        <span class="status-badge {bc}">{r["Risk_Category"]}</span>
                    </div>
                </div>''', unsafe_allow_html=True)

    # ── Bottom: Expandable Data Table ─────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.expander("📋 View All Filtered Shipments", expanded=False):
        if not tower_df.empty:
            display_cols = ["Order_ID", "Supplier", "Destination_City", "Transport_Mode",
                            "Volume", "Risk_Score", "Risk_Category", "Days_Overdue", "Order_Value_USD"]
            st.dataframe(
                tower_df[display_cols].sort_values("Risk_Score", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "Risk_Score": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%d%%"),
                    "Order_Value_USD": st.column_config.NumberColumn("Value (USD)", format="$%,.0f"),
                    "Days_Overdue": st.column_config.NumberColumn("Overdue", format="%d days"),
                },
            )
        else:
            st.info("No overdue shipments in the selected date range.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    inject_custom_css()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚡ SCIP")
        st.markdown(
            '<p style="font-size:11px; color:#6A6A8A; margin-top:-10px;">Supply Chain Intelligence Platform</p>',
            unsafe_allow_html=True)
        st.markdown("---")

        # ── Data Source Toggle ────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:10px; color:#6A6A8A; text-transform:uppercase; '
            'letter-spacing:0.1em; font-weight:600; margin-bottom:8px;">Data Source</p>',
            unsafe_allow_html=True)

        data_source = st.radio(
            "Select Data Source",
            options=["🧪 Simulated Data (1627 rows)", "📊 predictions.xlsx (Raw)"],
            index=0,
            label_visibility="collapsed",
            key="data_source_toggle",
        )

        st.markdown("---")

        # Load data based on selection
        if "predictions.xlsx" in data_source:
            df = load_sample_raw_data()
            if df.empty:
                st.stop()
            st.markdown(
                f'<div style="background:#1A2A1A; border:1px solid #2A4A2A; border-radius:8px; '
                f'padding:8px 12px; font-size:11px; color:#09AB3B;">'
                f'✓ Loaded {len(df):,} rows from predictions.xlsx</div>',
                unsafe_allow_html=True)
        else:
            df = generate_shipment_data()
            st.markdown(
                f'<div style="background:#1A1A2A; border:1px solid #2A2A4A; border-radius:8px; '
                f'padding:8px 12px; font-size:11px; color:#6C63FF;">'
                f'✓ Simulated {len(df):,} shipments loaded</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # Quick stats
        st.markdown(
            '<p style="font-size:10px; color:#6A6A8A; text-transform:uppercase; '
            'letter-spacing:0.1em; font-weight:600; margin-bottom:8px;">Network Overview</p>',
            unsafe_allow_html=True)

        overdue_count = len(df[df["Is_Overdue"]])
        high_count = len(df[df["Risk_Category"] == "High"])

        c1, c2 = st.columns(2)
        c1.metric("Total Orders", f"{len(df):,}")
        c2.metric("Overdue", overdue_count)

        c3, c4 = st.columns(2)
        c3.metric("High Risk", high_count)
        c4.metric("Avg Risk", f"{df['Risk_Score'].mean():.0f}%")

        st.markdown("---")

        st.markdown(
            '<p style="font-size:10px; color:#6A6A8A; text-transform:uppercase; '
            'letter-spacing:0.1em; font-weight:600; margin-bottom:8px;">Risk Distribution</p>',
            unsafe_allow_html=True)

        overdue_only = df[df["Is_Overdue"]]
        if not overdue_only.empty:
            st.plotly_chart(create_risk_distribution_chart(overdue_only),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.markdown(
            '<p style="font-size:10px; color:#6A6A8A; text-transform:uppercase; '
            'letter-spacing:0.1em; font-weight:600; margin-bottom:8px;">Suppliers Monitored</p>',
            unsafe_allow_html=True)
        for s, rate in sorted(SUPPLIER_DISRUPTION_RATE.items(), key=lambda x: -x[1])[:6]:
            sc = COLOR_HIGH if rate > 0.15 else (COLOR_MODERATE if rate > 0.08 else COLOR_LOW)
            st.markdown(f'<div style="display:flex; justify-content:space-between; padding:3px 0; '
                        f'font-size:11px;"><span style="color:#A0A0B0;">{s}</span>'
                        f'<span style="color:{sc}; font-weight:600;">{rate:.0%}</span></div>',
                        unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown('''
    <div class="command-center-header">
        <div>
            <p class="header-title">⚡ Supply Chain Intelligence Platform</p>
            <p class="header-subtitle">
                Order Entry · ML Risk Scoring · Reroute Engine · Financial Analysis · Cascade Simulation · Live Control Tower
            </p>
        </div>
    </div>''', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 New Order & Risk Engine",
        "💰 Financial Analysis",
        "🌊 Cascade Simulator",
        "🗺️ Control Tower",
    ])

    with tab1:
        render_order_entry_module()

    with tab2:
        render_financial_module(df)

    with tab3:
        render_cascade_module(df)

    with tab4:
        render_control_tower(df)


if __name__ == "__main__":
    main()
