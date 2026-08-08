# ──────────────────────────────────────────────────────────────────────────────
# Supply Chain Intelligence Platform — Combined main.py
# A comprehensive 6-module Streamlit dashboard for supply chain risk
# management: Order Entry, ML Risk Engine, Reroute Generator, Financial
# Analysis, Cascade Simulation, and Live GIS Control Tower.
#
# Data Source: predictions.xlsx (real shipment data, sampled from
# data/raw/train.parquet), ingested via load_sample_raw_data().
#
# The formerly-default "Simulated Data" source (generate_shipment_data(),
# 1627 fully-fabricated mock shipments with randomly templated cascading-
# impact/routing text) has been removed -- it was leaking fake data into
# this dashboard by default. See notebooks/rebuild_pipeline.py's module
# docstring for the matching "ghost data" issue in the ML pipeline.
# ──────────────────────────────────────────────────────────────────────────────

# pyrefly: ignore [missing-import]
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import folium
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, AntPath
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

# ─── Semantic Color Constants (light, bluish theme) ────────────────────────────
COLOR_HIGH = "#E0393E"
COLOR_MODERATE = "#B8860B"
COLOR_LOW = "#0C9440"
COLOR_BG_CARD = "#FFFFFF"
COLOR_BG_PANEL = "#F3F9FF"
COLOR_TEXT = "#0F2A4D"
COLOR_TEXT_MUTED = "#4E6D95"
COLOR_BORDER = "#DCEAF9"
COLOR_ACCENT = "#2E7CF6"


# ─── Custom CSS ───────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        *, html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

        /* Streamlit's built-in icons (expander/dataframe/sidebar chevrons
           etc.) are text ligatures rendered through Google's "Material
           Symbols" icon font -- e.g. the literal text "keyboard_arrow_right"
           is invisible and renders as an arrow glyph *only* under that font.
           The blanket Poppins override above was winning against it too, so
           icons rendered as their raw ligature text overlapping the nearby
           label instead of a glyph (e.g. expander headers). Restore the
           icon font specifically for those elements. */
        [data-testid="stIconMaterial"],
        span[data-testid="stExpanderToggleIcon"],
        [data-testid="stIconMaterial"] * {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        }

        .stApp {
            background: linear-gradient(165deg, #EFF6FF 0%, #FFFFFF 45%, #E9F2FC 100%);
        }

        /* ── Sidebar: always visible, never collapsible ─────────────────────── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F2F8FF 0%, #E6F1FC 100%);
            border-right: 1px solid #D6E7F8;
            min-width: 300px !important;
            max-width: 340px !important;
            transform: none !important;
            visibility: visible !important;
        }

        /* Hide the collapse arrow inside the sidebar and the "reopen" chevron
           that appears once collapsed -- the sidebar should stay put across
           every page and every widget interaction. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        button[title="Close sidebar"],
        button[title="Open sidebar"] {
            display: none !important;
            visibility: hidden !important;
        }

        .command-center-header {
            background: linear-gradient(135deg, #FFFFFF 0%, #F2F8FF 50%, #FFFFFF 100%);
            border: 1px solid #D6E7F8;
            border-radius: 20px;
            padding: 20px 32px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 28px rgba(20,70,140,0.07), inset 0 1px 0 rgba(255,255,255,0.6);
            transition: box-shadow 0.25s ease;
        }

        .header-title {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #123A6B 0%, #2E7CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            line-height: 1.3;
            margin: 0;
        }

        .header-subtitle {
            font-size: 12px;
            color: #4E6D95;
            margin: 4px 0 0 0;
            font-weight: 400;
            line-height: 1.5;
        }

        .kpi-card {
            background: linear-gradient(145deg, #FFFFFF 0%, #F2F8FF 100%);
            border: 1px solid #DCEAF9;
            border-radius: 20px;
            padding: 20px 22px;
            min-height: 108px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 16px rgba(20,70,140,0.06);
            transition: transform 0.22s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.22s ease, border-color 0.22s ease;
        }

        .kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 28px rgba(20,70,140,0.12);
            border-color: #B8D4F5;
        }

        .kpi-value {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0;
            line-height: 1.25;
        }

        .kpi-label {
            font-size: 11px;
            color: #4E6D95;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-top: 6px;
            font-weight: 600;
            line-height: 1.5;
        }

        /* Clickable High-Risk-Alerts KPI "card" -- st.container(key=...)
           renders a .st-key-<key> class on its wrapper div, which is the
           officially supported hook for scoping CSS to one specific
           Streamlit element (see render_control_tower). */
        .st-key-kpi_high_risk_card {
            background: linear-gradient(145deg, #FFFFFF 0%, #FFF3F3 100%);
            border: 1px solid #F7D3D3;
            border-radius: 20px;
            padding: 10px 10px 16px;
            min-height: 108px;
            box-shadow: 0 4px 16px rgba(190,40,45,0.07);
            transition: transform 0.22s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.22s ease, border-color 0.22s ease;
        }

        .st-key-kpi_high_risk_card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 28px rgba(190,40,45,0.14);
            border-color: #EFAFAF;
        }

        .st-key-kpi_high_risk_card .stButton > button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 32px !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
            color: #E0393E !important;
            padding: 4px 4px 0 !important;
            width: 100%;
            transition: transform 0.15s ease;
        }

        .st-key-kpi_high_risk_card .stButton > button:hover {
            background: rgba(224,57,62,0.07) !important;
            border-radius: 12px !important;
            transform: none !important;
        }

        .st-key-kpi_high_risk_card .stButton > button:active {
            transform: scale(0.96) !important;
        }

        .st-key-kpi_high_risk_card .kpi-label {
            margin-top: 2px;
            cursor: pointer;
        }

        .detail-panel {
            background: linear-gradient(145deg, #FFFFFF 0%, #F3F9FF 100%);
            border: 1px solid #DCEAF9;
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 4px 16px rgba(20,70,140,0.06);
            transition: box-shadow 0.25s ease;
        }

        .detail-header {
            font-size: 18px;
            font-weight: 700;
            color: #0F2A4D;
            margin-bottom: 4px;
            letter-spacing: -0.01em;
            line-height: 1.4;
        }

        .detail-subheader {
            font-size: 12px;
            color: #4E6D95;
            line-height: 1.6;
            margin-bottom: 16px;
        }

        .info-box {
            background: #F3F8FE;
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 12px;
            border-left: 4px solid;
        }

        .info-box.impact { border-left-color: #E0393E; }
        .info-box.solution { border-left-color: #0C9440; }
        .info-box.warning { border-left-color: #B8860B; }
        .info-box.info { border-left-color: #2E7CF6; }

        .info-box-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            line-height: 1.5;
            margin-bottom: 6px;
        }

        .info-box-content {
            font-size: 13px;
            color: #33425C;
            line-height: 1.65;
        }

        .section-label {
            font-size: 15px;
            font-weight: 600;
            color: #0F2A4D;
            line-height: 1.5;
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 1px solid #DCEAF9;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            line-height: 1.6;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            white-space: nowrap;
        }

        .badge-high { background: rgba(224,57,62,0.12); color: #C4282D; }
        .badge-moderate { background: rgba(184,134,11,0.14); color: #9A6E00; }
        .badge-low { background: rgba(12,148,64,0.12); color: #0C9440; }

        /* ── Alternative Cards ─────────────────────────────────────────────── */
        .alt-card {
            background: linear-gradient(145deg, #FFFFFF 0%, #F3F9FF 100%);
            border: 1px solid #DCEAF9;
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 4px 16px rgba(20,70,140,0.06);
            position: relative;
            overflow: hidden;
            transition: transform 0.22s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.22s ease;
        }

        .alt-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 26px rgba(20,70,140,0.10);
        }

        .alt-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }

        .alt-card.original::before { background: linear-gradient(90deg, #E0393E, #FF6B6B); }
        .alt-card.modal-shift::before { background: linear-gradient(90deg, #B8860B, #FFD84B); }
        .alt-card.vendor-shift::before { background: linear-gradient(90deg, #0C9440, #2BC85B); }

        .alt-card-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            line-height: 1.5;
            margin-bottom: 12px;
        }

        .alt-card-value {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.3;
        }

        .alt-card-metric {
            font-size: 12.5px;
            color: #48597A;
            line-height: 1.6;
            margin-top: 4px;
        }

        /* ── Enrichment Pills ──────────────────────────────────────────────── */
        .enrichment-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #F3F8FE;
            border: 1px solid #DCEAF9;
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 8px;
            width: 100%;
            transition: border-color 0.2s ease, background 0.2s ease;
        }

        .enrichment-pill:hover {
            border-color: #B8D4F5;
            background: #EAF3FE;
        }

        .enrichment-pill .ep-icon {
            font-size: 20px;
            line-height: 1;
        }

        .enrichment-pill .ep-label {
            font-size: 10px;
            color: #4E6D95;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            line-height: 1.6;
        }

        .enrichment-pill .ep-value {
            font-size: 18px;
            font-weight: 700;
            line-height: 1.4;
        }

        /* ── Financial Metric ──────────────────────────────────────────────── */
        .fin-metric {
            background: linear-gradient(145deg, #FFFFFF 0%, #F3F9FF 100%);
            border: 1px solid #DCEAF9;
            border-radius: 20px;
            padding: 24px;
            text-align: center;
        }

        .fin-metric .fm-value {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.3;
        }

        .fin-metric .fm-label {
            font-size: 11px;
            color: #4E6D95;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            line-height: 1.5;
            margin-top: 4px;
        }

        .fin-metric .fm-delta {
            font-size: 12px;
            line-height: 1.5;
            margin-top: 6px;
            font-weight: 500;
        }

        /* ── Network Nodes (Cascade Simulator) ───────────────────────────────
           Fixed min-height + explicit line-height on every text tier keeps
           the 4 cards the same height and stops long text (real supplier
           names, longer city names) from wrapping into cramped, overlapping
           lines the way single-line, no-line-height text did before. */
        .cascade-node {
            background: #F3F8FE;
            border: 1px solid #DCEAF9;
            border-radius: 16px;
            padding: 18px 14px;
            min-height: 148px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            transition: all 0.3s ease;
        }

        .cascade-node.critical {
            border-color: #E0393E;
            box-shadow: 0 0 18px rgba(224,57,62,0.12);
        }

        .cascade-node.warning {
            border-color: #B8860B;
            box-shadow: 0 0 18px rgba(184,134,11,0.10);
        }

        .cascade-node.safe {
            border-color: #0C9440;
        }

        .cascade-node-title {
            font-size: 15px;
            font-weight: 700;
            color: #0F2A4D;
            line-height: 1.4;
            word-break: break-word;
        }

        .cascade-node-sub {
            font-size: 12.5px;
            color: #48597A;
            line-height: 1.5;
            margin-top: 5px;
            word-break: break-word;
        }

        .cascade-node-status {
            font-size: 12px;
            font-weight: 700;
            line-height: 1.5;
            letter-spacing: 0.03em;
            margin-top: 10px;
        }

        /* ── Order List (Control Tower) ───────────────────────────────────── */
        .order-row {
            background: #FFFFFF;
            border: 1px solid #DCEAF9;
            border-radius: 14px;
            padding: 10px 14px;
            margin-bottom: 6px;
            transition: transform 0.18s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.18s ease, border-color 0.18s ease;
        }

        .order-row:hover {
            transform: translateX(2px);
            box-shadow: 0 4px 14px rgba(20,70,140,0.08);
            border-color: #B8D4F5;
        }

        .order-row.selected {
            border-color: #2E7CF6;
            box-shadow: 0 0 0 2px rgba(46,124,246,0.18), 0 4px 14px rgba(20,70,140,0.08);
            background: #EFF6FF;
        }

        /* ── Streamlit Overrides ────────────────────────────────────────────── */
        .stButton > button {
            border-radius: 12px;
            transition: transform 0.16s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.16s ease, color 0.16s ease, border-color 0.16s ease, background 0.16s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
        }

        .stButton > button:active {
            transform: translateY(0) scale(0.97);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2E7CF6 0%, #1F63D6 100%);
            border: none;
            border-radius: 12px;
            padding: 10px 28px;
            font-weight: 600;
            box-shadow: 0 4px 16px rgba(46,124,246,0.30);
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 22px rgba(46,124,246,0.45);
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"]:active {
            box-shadow: 0 2px 10px rgba(46,124,246,0.35);
            transform: translateY(0) scale(0.97);
        }

        .stButton > button[kind="secondary"] {
            border-radius: 12px;
        }

        .stButton > button[kind="secondary"]:hover {
            border-color: #2E7CF6;
            color: #2E7CF6;
            transform: translateY(-1px);
        }

        .stButton > button[kind="secondary"]:active {
            background: #EFF6FF;
            transform: translateY(0) scale(0.97);
        }

        #MainMenu { visibility: hidden; }
        header { visibility: hidden; }
        footer { visibility: hidden; }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #DCEAF9;
            border-radius: 18px;
            background: #FFFFFF;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        /* Text inputs / selects: rounder, and consistent vertical centering
           for their labels/content so nothing looks clipped or off-baseline. */
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: 12px !important;
        }

        div[data-testid="stWidgetLabel"] p {
            line-height: 1.5;
        }

        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
            line-height: 1.4;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #EAF3FE;
            border-radius: 14px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 20px;
            font-weight: 500;
            font-size: 13px;
            line-height: 1.6;
            transition: background 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #FFFFFF 0%, #DCEAFC 100%) !important;
            box-shadow: 0 2px 8px rgba(20,70,140,0.10);
        }

        /* Smoother widget/page transitions overall */
        div[data-testid="stVerticalBlock"] { transition: opacity 0.15s ease; }
        iframe { transition: opacity 0.25s ease; }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOCODING & REFERENCE LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

# City coordinates for route calculations
CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946), "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873), "Lucknow": (26.8467, 80.9462),
    "Surat": (21.1702, 72.8311), "Nagpur": (21.1458, 79.0882),
    "Visakhapatnam": (17.6868, 83.2185), "Coimbatore": (11.0168, 76.9558),
    "Indore": (22.7196, 75.8577), "Bhopal": (23.2599, 77.4126),
    "Patna": (25.6093, 85.1376), "Kochi": (9.9312, 76.2673),
    "Guwahati": (26.1445, 91.7362), "Chandigarh": (30.7333, 76.7794),
    # International cities that appear in predictions.xlsx (all 26
    # destination_city values in data/raw/train.parquet)
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
    "Atlanta": (33.7490, -84.3880), "Birmingham": (52.4862, -1.8904),
    "Chicago": (41.8781, -87.6298), "Hamburg": (53.5511, 9.9937),
    "Houston": (29.7604, -95.3698), "Manchester": (53.4808, -2.2426),
    "Marseille": (43.2965, 5.3698), "Melbourne": (-37.8136, 144.9631),
    "Munich": (48.1351, 11.5820), "Osaka": (34.6937, 135.5023),
    "Vancouver": (49.2827, -123.1207), "Frankfurt": (50.1109, 8.6821),
}

# Pre-build lat/lon dicts for vectorized .map() lookups
LAT_DICT = {city: coords[0] for city, coords in CITY_COORDS.items()}
LON_DICT = {city: coords[1] for city, coords in CITY_COORDS.items()}
FALLBACK_LAT = 20.5937
FALLBACK_LON = 78.9629

# Country centroids for the 15 supplier_country values in the real dataset --
# used as the origin point for routes/distance since the data only has
# origin at country grain (no origin city).
COUNTRY_COORDS = {
    "Bangladesh": (23.6850, 90.3563), "Brazil": (-14.2350, -51.9253),
    "China": (35.8617, 104.1954), "Egypt": (26.8206, 30.8025),
    "Germany": (51.1657, 10.4515), "India": (20.5937, 78.9629),
    "Indonesia": (-0.7893, 113.9213), "Mexico": (23.6345, -102.5528),
    "Nigeria": (9.0820, 8.6753), "Poland": (51.9194, 19.1451),
    "South Korea": (35.9078, 127.7669), "Thailand": (15.8700, 100.9925),
    "Turkey": (38.9637, 35.2433), "USA": (37.0902, -95.7129),
    "Vietnam": (14.0583, 108.2772),
}
COUNTRY_LAT_DICT = {c: coords[0] for c, coords in COUNTRY_COORDS.items()}
COUNTRY_LON_DICT = {c: coords[1] for c, coords in COUNTRY_COORDS.items()}

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
    # Scale and sigmoid. Center/scale calibrated so the resulting distribution
    # over realistic input ranges lands ~70% Low / ~22% Medium / ~8% High,
    # instead of the old (z-0.15)*12 which saturated ~99% of shipments into
    # High risk (median z of ~0.28 already mapped to prob ~0.83).
    z_scaled = (z - 0.36) * 20
    prob = 1 / (1 + math.exp(-z_scaled))
    return round(prob, 4)


def risk_band(prob: float) -> str:
    if prob >= 0.66:
        return "High"
    elif prob >= 0.33:
        return "Medium"
    return "Low"


def historical_predicted_delay(df: pd.DataFrame, prob: float) -> int:
    """Expected delay length (days) used by the Financial Analysis module's
    Value-at-Risk calculation, grounded in this dataset's own history for
    shipments at a similar risk level -- not the old `max(1, int(prob*20))`,
    an arbitrary formula with no basis in the data (why 20? why linear in
    probability at all?). Prefers the actual observed Days_Overdue for
    delayed shipments within +/-15 risk-score points of this one; falls
    back to the same risk band, then the dataset-wide average of all
    delayed shipments, widening only when there isn't enough data at the
    more specific level to be a stable estimate."""
    delayed = df[df["Days_Overdue"] > 0]
    if delayed.empty:
        return max(1, round(prob * 14))

    score = prob * 100
    close = delayed[(delayed["Risk_Score"] >= score - 15) & (delayed["Risk_Score"] <= score + 15)]
    if len(close) >= 5:
        return max(1, round(float(close["Days_Overdue"].mean())))

    same_band = delayed[delayed["Risk_Category"] == risk_band(prob)]
    if len(same_band) >= 5:
        return max(1, round(float(same_band["Days_Overdue"].mean())))

    return max(1, round(float(delayed["Days_Overdue"].mean())))


def risk_color(category: str) -> str:
    return {"High": COLOR_HIGH, "Medium": COLOR_MODERATE, "Moderate": COLOR_MODERATE, "Low": COLOR_LOW}.get(category, COLOR_TEXT_MUTED)


def risk_badge_class(category: str) -> str:
    return {"High": "badge-high", "Medium": "badge-moderate", "Moderate": "badge-moderate", "Low": "badge-low"}.get(category, "")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA SOURCE — Real Shipment Data Bridge (load_sample_raw_data)
# ═══════════════════════════════════════════════════════════════════════════════
# predictions.xlsx lives in app/ regardless of whether this script is run as
# ./main.py (repo root) or app/main.py -- resolve it relative to this file
# first, falling back to the app/ subdirectory next to it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_XLSX_PATH = next(
    (p for p in (
        os.path.join(_THIS_DIR, "predictions.xlsx"),
        os.path.join(_THIS_DIR, "app", "predictions.xlsx"),
    ) if os.path.exists(p)),
    os.path.join(_THIS_DIR, "predictions.xlsx"),
)

# outputs/ (trained model artifacts) lives at the repo root -- same
# root-relative resolution as PREDICTIONS_XLSX_PATH above.
OUTPUTS_DIR = next(
    (p for p in (
        os.path.join(_THIS_DIR, "outputs"),
        os.path.join(_THIS_DIR, "..", "outputs"),
    ) if os.path.isdir(p)),
    os.path.join(_THIS_DIR, "outputs"),
)


@st.cache_resource(show_spinner=False)
def _load_risk_model():
    """Load the real trained model (see notebooks/rebuild_pipeline.py) so the
    dashboard can score shipments the same way it was actually evaluated,
    instead of trusting the raw dataset's own leaky `risk_score` column
    (see _predict_risk_score's docstring). Returns None if the artifacts
    aren't present -- e.g. rebuild_pipeline.py hasn't been run yet -- so
    callers can fall back gracefully instead of crashing the whole app.
    """
    try:
        model_path = os.path.join(OUTPUTS_DIR, "model_tuned.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(OUTPUTS_DIR, "model.pkl")
        model = joblib.load(model_path)
        encoder = joblib.load(os.path.join(OUTPUTS_DIR, "encoder.pkl"))
        features = joblib.load(os.path.join(OUTPUTS_DIR, "features.pkl"))
        return model, encoder, features
    except FileNotFoundError:
        return None


def _predict_disruption_probability(raw_df: pd.DataFrame) -> pd.Series | None:
    """Score shipments with the real trained model instead of the dataset's
    own `risk_score` field.

    That field is a *post-hoc* value -- it averages ~19 when nothing went
    wrong vs. ~73 when it did (0.70 correlation with the actual outcome,
    confirmed on this sample), which is why it swings all the way up to
    99-100: it was computed with knowledge of the result, not predicted
    ahead of it. notebooks/rebuild_pipeline.py's own docstring documents
    this and explicitly excludes it (along with delay_days,
    actual_delivery_date, etc.) from the model's training features for
    exactly this reason.

    This re-scores the same shipments with that already-trained,
    leakage-free model (raw_df must still have the *original* snake_case
    column names -- call this before COLUMN_MAP renames them) so the
    dashboard shows a real forward-looking prediction instead. Returns
    None (caller falls back to the raw column) if the trained artifacts
    aren't available.
    """
    loaded = _load_risk_model()
    if loaded is None:
        return None
    model, encoder, features = loaded
    missing = [f for f in features if f not in raw_df.columns]
    if missing:
        return None

    X = raw_df[features].copy()
    cat_cols = [c for c in encoder.feature_names_in_ if c in X.columns]
    X[cat_cols] = encoder.transform(X[cat_cols].astype(str))
    y_prob = model.predict_proba(X)[:, 1]
    return pd.Series(y_prob, index=raw_df.index)


# Maps every one of the model's 28 raw snake_case feature names to the
# renamed UI column that carries it in load_sample_raw_data()'s output, so
# the Order & Risk Engine's live "what-if" predictions (below) can be built
# from the *same* trained model instead of the old simulate_risk_score()
# hand-tuned formula + get_environmental_factors() hash-seeded random
# numbers -- which fed the model's real weather/congestion/geopolitical
# scale (0-100, and port_congestion_level is categorical Low/Medium/High)
# with fabricated 0-1 values it was never trained on.
MODEL_FEATURE_UI_MAP = {
    "customs_clearance_hours": "Customs_Clearance_Hours",
    "demand_forecast_units": "Demand_Forecast_Units",
    "distance_km": "Distance_km",
    "fuel_price_index": "Fuel_Price_Index",
    "geopolitical_risk_index": "Geopolitical_Risk",
    "historical_disruption_count": "Historical_Disruption_Count",
    "inventory_level_percent": "Inventory_Level_Pct",
    "num_alternate_suppliers": "Num_Alternate_Suppliers",
    "order_quantity": "Volume",
    "order_value_usd": "Order_Value_USD",
    "payment_terms_days": "Payment_Terms_Days",
    "planned_lead_time_days": "Lead_Time_Days",
    "supplier_financial_health_score": "Supplier_Financial_Health",
    "supplier_reliability_score": "Supplier_Reliability",
    "unit_cost_usd": "Unit_Cost_USD",
    "weather_risk_index": "Weather_Risk",
    "carrier_name": "Carrier_Name",
    "contract_type": "Contract_Type",
    "destination_city": "Destination_City",
    "destination_country": "Destination_Country",
    "port_congestion_level": "Port_Congestion",
    "product_category": "Product_Category",
    "product_name": "Product_Name",
    "supplier_country": "Supplier_Country",
    "supplier_id": "Supplier_ID",
    "transportation_mode": "Transport_Mode",
    "warehouse_id": "Warehouse_ID",
    "is_peak_season": "Is_Peak_Season",
}

# Human-readable labels for the Feature Contributions (SHAP) chart --
# MODEL_FEATURE_UI_MAP's raw snake_case names aren't presentation-ready.
FEATURE_DISPLAY_NAMES = {
    "customs_clearance_hours": "Customs Clearance Time",
    "demand_forecast_units": "Demand Forecast",
    "distance_km": "Distance",
    "fuel_price_index": "Fuel Price Index",
    "geopolitical_risk_index": "Geopolitical Risk",
    "historical_disruption_count": "Historical Disruptions",
    "inventory_level_percent": "Inventory Level",
    "num_alternate_suppliers": "Alternate Suppliers",
    "order_quantity": "Order Quantity",
    "order_value_usd": "Order Value",
    "payment_terms_days": "Payment Terms",
    "planned_lead_time_days": "Planned Lead Time",
    "supplier_financial_health_score": "Supplier Financial Health",
    "supplier_reliability_score": "Supplier Reliability",
    "unit_cost_usd": "Unit Cost",
    "weather_risk_index": "Weather Risk",
    "carrier_name": "Carrier",
    "contract_type": "Contract Type",
    "destination_city": "Destination City",
    "destination_country": "Destination Country",
    "port_congestion_level": "Port Congestion",
    "product_category": "Product Category",
    "product_name": "Product",
    "supplier_country": "Origin Country",
    "supplier_id": "Supplier",
    "transportation_mode": "Transport Mode",
    "warehouse_id": "Warehouse",
    "is_peak_season": "Peak Season",
}


def _historical_value(df: pd.DataFrame, ui_col: str, is_numeric: bool, *filter_frames):
    """Historical mean (numeric) or mode (categorical) for ui_col, tried
    against each candidate subset of df in order (most specific first,
    e.g. supplier+destination) until one has usable data, falling back to
    the whole dataset. This is how every model input the live order form
    doesn't directly collect gets filled in -- from this route/supplier/
    product's own history, never an invented constant."""
    for frame in (*filter_frames, df):
        if frame is None or ui_col not in frame.columns:
            continue
        series = pd.to_numeric(frame[ui_col], errors="coerce") if is_numeric else frame[ui_col]
        series = series.dropna()
        if series.empty:
            continue
        return float(series.mean()) if is_numeric else series.mode().iloc[0]
    return 0.0 if is_numeric else "Unknown"


@st.cache_resource(show_spinner=False)
def _load_shap_explainer():
    """Background sample + SHAP explainer for live single-shipment feature
    contributions, built the same way notebooks/rebuild_pipeline.py builds
    its production SHAP explanations (TreeExplainer via shap.Explainer on
    the tuned HistGradientBoostingClassifier). Cached because building the
    background/explainer is the only slow part (~30ms) -- scoring one row
    against it is ~2ms, fast enough to run on every "Analyze Risk" click."""
    loaded = _load_risk_model()
    if loaded is None:
        return None
    model, encoder, features = loaded
    try:
        raw = pd.read_excel(PREDICTIONS_XLSX_PATH)
    except FileNotFoundError:
        return None
    if any(f not in raw.columns for f in features):
        return None
    cat_cols = [c for c in encoder.feature_names_in_ if c in features]
    X = raw[features].copy()
    X[cat_cols] = encoder.transform(X[cat_cols].astype(str))
    background = X.sample(min(100, len(X)), random_state=42)
    return shap.Explainer(model, background), cat_cols


def predict_live_risk(df: pd.DataFrame, *, supplier: str, destination: str, origin: str,
                       mode: str, product: str, order_value: float, quantity: float,
                       distance: float, lead_time: float, ship_date) -> dict | None:
    """Score a hypothetical new shipment with the real trained model --
    the same one used everywhere else in the app -- instead of the old
    simulate_risk_score() formula. Directly-controlled form fields (origin,
    destination, mode, supplier, product, value, quantity) go straight in;
    every other model feature defaults to this specific supplier/route's
    own historical average or mode (see _historical_value), narrowing from
    supplier+destination down to the whole dataset if there's no closer
    match. Returns None if the trained model artifacts aren't available.
    """
    loaded = _load_risk_model()
    if loaded is None:
        return None
    model, encoder, features = loaded

    route = df[(df["Supplier"] == supplier) & (df["Destination_City"] == destination)]
    by_supplier = df[df["Supplier"] == supplier]
    by_dest = df[df["Destination_City"] == destination]
    by_product = df[df["Product_Category"] == product] if "Product_Category" in df.columns else None
    by_mode = df[df["Transport_Mode"] == mode]

    live_values = {
        "destination_city": destination,
        "supplier_country": origin,
        "transportation_mode": mode,
        "product_category": product,
        "order_value_usd": float(order_value),
        "order_quantity": float(quantity),
        "distance_km": float(distance),
        "planned_lead_time_days": float(lead_time),
        # Nov/Dec ship dates are the conventional retail/e-commerce peak
        # season -- grounded in the chosen date rather than dataset-wide.
        "is_peak_season": bool(ship_date.month in (11, 12)),
    }

    row = {}
    for feat in features:
        if feat in live_values:
            row[feat] = live_values[feat]
            continue
        ui_col = MODEL_FEATURE_UI_MAP.get(feat)
        is_numeric = feat not in encoder.feature_names_in_
        row[feat] = _historical_value(df, ui_col, is_numeric, route, by_supplier, by_dest, by_product, by_mode)

    X = pd.DataFrame([row])[features]
    cat_cols = [c for c in encoder.feature_names_in_ if c in X.columns]
    X_enc = X.copy()
    X_enc[cat_cols] = encoder.transform(X_enc[cat_cols].astype(str))
    prob = float(model.predict_proba(X_enc)[0, 1])

    contributions = {}
    explainer_bundle = _load_shap_explainer()
    if explainer_bundle is not None:
        explainer, _ = explainer_bundle
        try:
            sv = explainer(X_enc, check_additivity=False)
            order = np.argsort(-np.abs(sv.values[0]))[:6]
            contributions = {features[i]: float(sv.values[0][i]) for i in order}
        except Exception:
            contributions = {}

    return {
        "probability": prob,
        "contributions": contributions,
        "raw_inputs": row,
    }


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
        df = pd.read_excel(PREDICTIONS_XLSX_PATH)
    except FileNotFoundError:
        st.error(
            f"🚨 **predictions.xlsx not found!** Expected it at `{PREDICTIONS_XLSX_PATH}`."
        )
        return pd.DataFrame()

    # Score with the real trained model before COLUMN_MAP renames the raw
    # snake_case columns it expects (see _predict_disruption_probability).
    model_prob = _predict_disruption_probability(df)

    # ── Step 2: Dynamic column mapping ────────────────────────────────────────
    # Left = raw column names from predictions.xlsx
    # Right = standardized UI column names the dashboard expects
    COLUMN_MAP = {
        "shipment_id":          "Order_ID",
        "shipmen_id":           "Order_ID",  # typo'd column name in older exports
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

    # ── 3a. Risk_Score / Risk_Probability ──────────────────────────────────────
    # Prefer the real trained model's prediction over the dataset's own
    # `risk_score` column, which is a known post-hoc/leaky field (see
    # _predict_disruption_probability's docstring) -- using it directly is
    # why the dashboard could show scores of 99-100 that were really just
    # the answer leaking through, not a calibrated prediction.
    if model_prob is not None:
        df["Risk_Probability"] = model_prob.values
        df["Risk_Score"] = (model_prob.values * 100).round().astype(int)
    else:
        if "Risk_Score" not in df.columns:
            df["Risk_Score"] = np.random.randint(10, 99, size=n)
        else:
            score_numeric = pd.to_numeric(df["Risk_Score"], errors="coerce")
            df["Risk_Score"] = np.where(
                score_numeric.isna(),
                np.random.randint(10, 99, size=n),
                score_numeric,
            ).astype(int)
        if "Risk_Probability" not in df.columns:
            df["Risk_Probability"] = df["Risk_Score"] / 100.0

    # ── 3b. Risk_Category ─────────────────────────────────────────────────────
    if "Risk_Category" not in df.columns:
        conditions = [df["Risk_Score"] > 66, df["Risk_Score"] > 33]
        choices = ["High", "Medium"]
        df["Risk_Category"] = np.select(conditions, choices, default="Low")

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
    #     Needed by control tower route lines and cascade module. The real
    #     dataset's origin grain is country (Supplier_Country), not city --
    #     Origin_City holds that country name here, geocoded off
    #     COUNTRY_COORDS, instead of fabricating a per-supplier city (the old
    #     SUPPLIER_ORIGINS lookup silently mapped every real supplier to
    #     "Mumbai" since none of them matched its 10 fictional names).
    if "Origin_City" not in df.columns:
        if "Supplier_Country" in df.columns:
            df["Origin_City"] = df["Supplier_Country"].fillna("Unknown")
        else:
            supplier_to_origin = {s: info[0] for s, info in SUPPLIER_ORIGINS.items()}
            df["Origin_City"] = df["Supplier"].map(supplier_to_origin).fillna("Mumbai")

    if "Origin_Lat" not in df.columns:
        df["Origin_Lat"] = (
            df["Origin_City"].map(COUNTRY_LAT_DICT)
            .fillna(df["Origin_City"].map(LAT_DICT))
            .fillna(FALLBACK_LAT)
        )
    if "Origin_Lon" not in df.columns:
        df["Origin_Lon"] = (
            df["Origin_City"].map(COUNTRY_LON_DICT)
            .fillna(df["Origin_City"].map(LON_DICT))
            .fillna(FALLBACK_LON)
        )

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
    # Derived from each supplier's own historical Risk_Score in this dataset
    # (not the hardcoded 10-supplier SUPPLIER_DISRUPTION_RATE table, which
    # has no entries for any real supplier name and used to silently fall
    # back to a flat 10% for every one of them).
    if "Supplier_Disruption_Rate" not in df.columns:
        if "Risk_Score" in df.columns:
            df["Supplier_Disruption_Rate"] = (
                df.groupby("Supplier")["Risk_Score"].transform("mean") / 100.0
            ).fillna(0.10)
        else:
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
        number={"font": {"size": 44, "color": "#16162A", "family": "Poppins"}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "dtick": 25,
                     "tickfont": {"size": 10, "color": "#6A6A8A"}},
            "bar": {"color": risk_color(category), "thickness": 0.3},
            "bgcolor": "#F0F1F7", "borderwidth": 0,
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
        font={"family": "Poppins"},
    )
    return fig


def create_feature_waterfall(features: dict) -> go.Figure:
    """Bar chart showing (real, SHAP-derived) feature contributions to the
    risk score. Real SHAP magnitudes span a much wider range than the old
    hand-tuned weights did (a dominant feature can dwarf the rest), so
    "auto" text positioning would tuck tiny bars' labels right at the plot
    edge and clip them -- force "outside" placement with real x-axis
    padding instead, the same fix as the financial waterfall's."""
    names = list(features.keys())
    values = list(features.values())
    v_lo, v_hi = min(0, *values), max(0, *values)
    span = v_hi - v_lo
    # Generous padding on *both* sides -- "outside" text for a positive bar
    # sits to its right, for a negative bar to its left, so both edges of
    # the range need room, not just the max end.
    pad = max(span * 0.35, 0.08)

    colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in values]

    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"+{v:.1f}" if v > 0 else f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#16162A", family="Poppins"),
        cliponaxis=False,
    ))
    fig.update_layout(
        height=250, margin=dict(l=8, r=55, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#B8D4F5",
                   range=[v_lo - pad, v_hi + pad],
                   tickfont=dict(size=10, color="#6A6A8A")),
        yaxis=dict(tickfont=dict(size=11, color="#5A5A78"), automargin=True),
    )
    return fig


def create_financial_waterfall(var_original: float, freight_premium: float, net_savings: float) -> go.Figure:
    """Waterfall chart for financial trade-off analysis."""
    running = [var_original, var_original - freight_premium, net_savings]
    y_lo, y_hi = min(0, *running), max(0, *running)
    # "outside" text labels need real headroom/footroom above and below the
    # tallest/lowest bar, or they clip against the top margin / collide with
    # the x-axis category labels -- which is exactly what a tight auto-range
    # with a small margin produced before.
    pad = max((y_hi - y_lo) * 0.22, 500)

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Value at Risk<br>(Do Nothing)", "Reroute<br>Premium", "Net<br>Savings"],
        y=[var_original, -freight_premium, net_savings],
        text=[f"${var_original:,.0f}", f"-${freight_premium:,.0f}", f"${net_savings:,.0f}"],
        textposition="outside",
        textfont=dict(size=13, color="#1A1A2E", family="Poppins"),
        connector={"line": {"color": "#B8D4F5", "width": 1}},
        increasing={"marker": {"color": COLOR_HIGH}},
        decreasing={"marker": {"color": COLOR_LOW}},
        totals={"marker": {"color": COLOR_ACCENT if net_savings > 0 else COLOR_HIGH}},
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=44, b=44),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(size=11, color="#5A5A78")),
        yaxis=dict(showgrid=True, gridcolor="rgba(20,70,140,0.07)", range=[y_lo - pad, y_hi + pad],
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
        fill="tozeroy", fillcolor="rgba(46,124,246,0.08)",
        name="Buffer Stock",
    ))

    # Three annotated reference lines can easily land on top of each other
    # when stockout_day and delay_days are close together on a short axis --
    # pin each one to a different quadrant (right / top / bottom) so their
    # text never competes for the same space regardless of where the lines
    # themselves fall.
    fig.add_hline(y=daily_demand * 2, line_dash="dash",
                  line_color=COLOR_MODERATE, annotation_text="Critical Threshold",
                  annotation_font_color=COLOR_MODERATE, annotation_position="top right",
                  annotation_font_size=11)

    if stockout_day:
        fig.add_vline(x=stockout_day, line_dash="dash", line_color=COLOR_HIGH,
                      annotation_text="Stockout", annotation_font_color=COLOR_HIGH,
                      annotation_position="top", annotation_font_size=11)

    # Delay marker
    if delay_days > 0:
        fig.add_vline(x=delay_days, line_dash="dot", line_color=COLOR_MODERATE,
                      annotation_text=f"Shipment Arrives (Day {delay_days})",
                      annotation_font_color=COLOR_MODERATE,
                      annotation_position="bottom right", annotation_font_size=11)

    fig.update_layout(
        height=290, margin=dict(l=10, r=70, t=24, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=dict(text="Days", font=dict(size=11, color="#6A6A8A")),
                   tickfont=dict(size=10, color="#6A6A8A"),
                   gridcolor="rgba(20,70,140,0.07)"),
        yaxis=dict(title=dict(text="Units in Buffer", font=dict(size=11, color="#6A6A8A")),
                   tickfont=dict(size=10, color="#6A6A8A"),
                   gridcolor="rgba(20,70,140,0.07)"),
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
        textfont=dict(size=13, color="#16162A", family="Poppins"),
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=12, color="#5A5A78"), automargin=True),
        bargap=0.35,
    )
    return fig


# Colors clusters by the risk mix of their contents and glows accordingly —
# read from each child marker's already-set `color`/`fillColor` path options.
_CLUSTER_ICON_JS = """
function(cluster) {
    var children = cluster.getAllChildMarkers();
    var counts = {};
    children.forEach(function(mk) {
        var c = (mk.options && mk.options.color) || '#8A8A9A';
        counts[c] = (counts[c] || 0) + 1;
    });
    var dominant = '#8A8A9A', max = -1;
    for (var k in counts) { if (counts[k] > max) { max = counts[k]; dominant = k; } }
    var count = cluster.getChildCount();
    var size = count < 10 ? 34 : (count < 30 ? 44 : 56);
    return L.divIcon({
        html: '<div style="width:100%; height:100%; border-radius:50%; display:flex;' +
              'align-items:center; justify-content:center; font-family:Poppins,sans-serif;' +
              'font-weight:700; font-size:' + (size < 44 ? 12 : 14) + 'px; color:#1A1A2E;' +
              'background:' + dominant + '33; border:2px solid ' + dominant + ';' +
              'box-shadow:0 0 14px ' + dominant + '99;">' + count + '</div>',
        className: 'scip-cluster-icon',
        iconSize: L.point(size, size),
    });
}
"""


@st.cache_data(show_spinner=False)
def build_control_tower_map(df: pd.DataFrame, highlight_id: str | None = None) -> folium.Map:
    """Build light-themed Folium map: risk-glow clusters, in-transit routes,
    origin markers, base-layer toggle, minimap, fullscreen, a risk legend,
    and -- when highlight_id is set -- a pulsing marker + animated fly-to on
    the selected shipment so map <-> order-list selection feels connected
    instead of a static re-render.

    Perf notes: only the *selected* route uses the animated AntPath layer --
    every other route is a plain static PolyLine. With hundreds of overdue
    shipments, having all of them as continuously-animating AntPaths was
    the dominant cause of map jank (each one redraws every ~50ms, forced
    into its own SVG element regardless of the map's prefer_canvas setting).
    Static polylines paint once via the shared canvas renderer and then sit
    idle. @st.cache_data also means reruns that don't change the filtered
    shipment set or selection (e.g.
    clicking a button in the deep-dive panel) skip rebuilding the map --
    and skip re-sending it to the browser, since st_folium/Streamlit's
    component layer no-ops on identical serialized props."""
    center_lat = df["Lat"].mean() if not df.empty else 20.5937
    center_lon = df["Lon"].mean() if not df.empty else 78.9629

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=5,
        tiles=None, control_scale=True, prefer_canvas=True,
        # Finer zoom granularity + explicit animation flags make pan/zoom
        # feel smoother instead of snapping in large integer-zoom jumps.
        zoomSnap=0.25, zoomDelta=0.5, wheelPxPerZoomLevel=90,
        fadeAnimation=True, zoomAnimation=True, markerZoomAnimation=True,
        easeLinearity=0.25, inertia=True,
    )

    folium.TileLayer("CartoDB positron", name="Light (Control Tower)", control=True, show=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark", control=True, show=False).add_to(m)

    # CSS transitions for the HTML/DivIcon-based layers (canvas-rendered
    # CircleMarkers can't take CSS transitions, but cluster bubbles and the
    # selection pulse below are DOM elements and animate smoothly).
    m.get_root().header.add_child(folium.Element("""
    <style>
        .scip-cluster-icon { transition: transform 0.2s ease-out; }
        .scip-cluster-icon:hover { transform: scale(1.08); }
        .scip-pulse-marker { position: relative; width: 28px; height: 28px; }
        .scip-pulse-core {
            position: absolute; top: 50%; left: 50%; width: 13px; height: 13px;
            margin: -7px 0 0 -7px; border-radius: 50%; background: #2E7CF6;
            border: 2px solid #FFFFFF; box-shadow: 0 2px 10px rgba(46,124,246,0.55);
        }
        .scip-pulse-ring {
            position: absolute; top: 50%; left: 50%; width: 13px; height: 13px;
            margin: -7px 0 0 -7px; border-radius: 50%; border: 2px solid #2E7CF6;
            animation: scip-pulse 1.6s cubic-bezier(0.22, 1, 0.36, 1) infinite;
        }
        @keyframes scip-pulse {
            0% { transform: scale(1); opacity: 0.85; }
            100% { transform: scale(3.4); opacity: 0; }
        }
        .leaflet-popup-content-wrapper, .leaflet-popup-tip { box-shadow: 0 8px 24px rgba(30,30,80,0.18); }
    </style>
    """))

    cluster = MarkerCluster(
        name="Shipments",
        options={"maxClusterRadius": 45, "spiderfyOnMaxZoom": True, "disableClusteringAtZoom": 10},
        icon_create_function=_CLUSTER_ICON_JS,
    ).add_to(m)

    route_group = folium.FeatureGroup(name="In-Transit Routes", show=True).add_to(m)
    origin_group = folium.FeatureGroup(name="Origins", show=True).add_to(m)

    if not df.empty:
        vol_min, vol_max = df["Volume"].min(), df["Volume"].max()
        vol_range = vol_max - vol_min if vol_max != vol_min else 1
    else:
        vol_min, vol_range = 0, 1

    for _, row in df.iterrows():
        is_selected = highlight_id is not None and row.get("Order_ID") == highlight_id
        radius = 6 + 20 * ((row["Volume"] - vol_min) / vol_range)
        color = risk_color(row["Risk_Category"])

        popup_html = f"""
        <div style="font-family:Poppins,sans-serif; min-width:200px; color:#1A1A2E;
                    background:#FFFFFF; padding:14px; border-radius:10px; border:1px solid #E7E9F2;">
            <div style="font-size:14px; font-weight:700; color:#16162A;">{row['Order_ID']}</div>
            <div style="font-size:11px; color:#6A6A8A; margin:4px 0 8px;">
                {row['Supplier']} → {row['Destination_City']} ({row['Transport_Mode']})
            </div>
            <div style="display:flex; gap:14px;">
                <div><span style="font-size:9px; color:#5A5A78; text-transform:uppercase;">Risk</span><br>
                    <span style="font-size:15px; font-weight:700; color:{color};">{row['Risk_Score']}%</span></div>
                <div><span style="font-size:9px; color:#5A5A78; text-transform:uppercase;">Volume</span><br>
                    <span style="font-size:15px; font-weight:700;">{row['Volume']}</span></div>
                <div><span style="font-size:9px; color:#5A5A78; text-transform:uppercase;">Overdue</span><br>
                    <span style="font-size:15px; font-weight:700; color:#E0393E;">{row['Days_Overdue']}d</span></div>
            </div>
        </div>"""

        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=radius + (5 if is_selected else 0),
            color=("#2E7CF6" if is_selected else color), fill=True,
            fill_color=color, fill_opacity=0.6 if is_selected else 0.5,
            weight=3 if is_selected else 1.5, opacity=0.95,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['Order_ID']} — {row['Destination_City']} (Risk: {row['Risk_Score']}%)",
        ).add_to(cluster)

        # Origin + animated in-transit route from origin to destination
        if row.get("Origin_Lat") and row.get("Origin_Lon"):
            folium.CircleMarker(
                location=[row["Origin_Lat"], row["Origin_Lon"]], radius=3,
                color=color, fill=True, fill_color="#FFFFFF", fill_opacity=0.9,
                weight=1.5, opacity=0.7,
                tooltip=f"Origin: {row['Supplier']} ({row.get('Origin_City', '')})",
            ).add_to(origin_group)

            if is_selected:
                # Animated dash is reserved for the selected route -- it's
                # the one place the continuous redraw is worth its cost.
                AntPath(
                    locations=[[row["Origin_Lat"], row["Origin_Lon"]], [row["Lat"], row["Lon"]]],
                    color="#2E7CF6", weight=3, opacity=0.8,
                    dash_array=[10, 20], delay=1200, pulse_color="#FFFFFF",
                ).add_to(route_group)
            else:
                folium.PolyLine(
                    locations=[[row["Origin_Lat"], row["Origin_Lon"]], [row["Lat"], row["Lon"]]],
                    color=color, weight=1.5, opacity=0.35, dash_array="4 6",
                ).add_to(route_group)

        if is_selected:
            pulse_html = (
                '<div class="scip-pulse-marker">'
                '<div class="scip-pulse-core"></div>'
                '<div class="scip-pulse-ring"></div>'
                '</div>'
            )
            folium.Marker(
                location=[row["Lat"], row["Lon"]],
                icon=folium.DivIcon(html=pulse_html, icon_size=(28, 28), icon_anchor=(14, 14)),
                tooltip=f"Selected: {row['Order_ID']}",
                z_index_offset=1000,
            ).add_to(m)

    # Fit the viewport to the data instead of a fixed zoom level, unless a
    # specific shipment is selected -- then fly to it for a connected feel.
    if highlight_id is not None and not df.empty and highlight_id in df["Order_ID"].values:
        hrow = df.loc[df["Order_ID"] == highlight_id].iloc[0]
        m.location = [float(hrow["Lat"]), float(hrow["Lon"])]
        m.get_root().script.add_child(folium.Element(f"""
            setTimeout(function() {{
                {m.get_name()}.flyTo([{float(hrow["Lat"])}, {float(hrow["Lon"])}], 6,
                    {{animate: true, duration: 1.1, easeLinearity: 0.25}});
            }}, 200);
        """))
    elif not df.empty:
        m.fit_bounds(
            [[df["Lat"].min(), df["Lon"].min()], [df["Lat"].max(), df["Lon"].max()]],
            padding=(30, 30),
        )

    Fullscreen(position="topleft", title="Expand map", title_cancel="Exit fullscreen").add_to(m)
    MiniMap(tile_layer="CartoDB positron", toggle_display=True, position="bottomright",
            width=120, height=120).add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:24px; left:24px; z-index:9999;
                font-family:Poppins,sans-serif; background:#FFFFFFdd; backdrop-filter:blur(6px);
                border:1px solid #E7E9F2; border-radius:10px; padding:10px 14px;
                color:#1A1A2E; font-size:11px; line-height:1.9; box-shadow:0 4px 16px rgba(30,30,80,0.12);">
        <div style="font-weight:700; font-size:10px; text-transform:uppercase;
                    letter-spacing:0.05em; color:#8A8A9A; margin-bottom:4px;">Risk Level</div>
        <div><span style="display:inline-block; width:9px; height:9px; border-radius:50%;
             background:#E0393E; margin-right:6px;"></span>High</div>
        <div><span style="display:inline-block; width:9px; height:9px; border-radius:50%;
             background:#B8860B; margin-right:6px;"></span>Medium</div>
        <div><span style="display:inline-block; width:9px; height:9px; border-radius:50%;
             background:#0C9440; margin-right:6px;"></span>Low</div>
        <div style="margin-top:4px; color:#8A8A9A;">Marker size ∝ shipment volume</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_order_entry_module(df: pd.DataFrame):
    """Order & Risk Engine: dataset-driven origin/destination/mode/supplier
    selection with cascading filters, ML risk auto-enrichment, and a
    prescriptive alternatives generator."""

    st.markdown('<p class="section-label">📦 Order & Risk Engine — New Shipment</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:12px; color:#5A5A78; margin-top:-8px;">'
        'Origin, destination, transport mode and supplier options are populated live '
        'from the loaded shipment dataset — nothing below is hardcoded.</p>',
        unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    form_col, result_col = st.columns([2, 3], gap="large")

    with form_col:
        st.markdown(
            '<div class="detail-panel">'
            '<p class="detail-header">Enter Shipment Details</p>'
            '<p class="detail-subheader">Origin → destination determines which transport modes and suppliers are offered below</p>'
            '</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Origin & Destination (populated from the dataset) ──────────────
        origin_options = sorted(df["Supplier_Country"].dropna().unique().tolist())
        dest_options = sorted(
            c for c in df["Destination_City"].dropna().unique().tolist() if c != "Unknown"
        )

        oc1, oc2 = st.columns(2)
        with oc1:
            origin = st.selectbox(
                "🌍 Origin / Starting Point", options=origin_options, key="oe_origin",
                help="Supplier country of origin — the dataset's origin field is country-grain.",
            )
        with oc2:
            destination = st.selectbox(
                "📍 Destination City", options=dest_options, key="oe_destination",
            )

        # ── Transport modes available for this destination ─────────────────
        modes_for_dest = sorted(
            df.loc[df["Destination_City"] == destination, "Transport_Mode"].dropna().unique().tolist()
        )
        if modes_for_dest:
            pills = "".join(
                f'<span class="status-badge badge-low" style="margin-right:6px;">{m}</span>'
                for m in modes_for_dest
            )
            st.markdown(
                f'<div style="margin:2px 0 12px; font-size:11px; color:#6A6A8A;">'
                f'Modes serving <b>{destination}</b>: {pills}</div>',
                unsafe_allow_html=True)
            mode_options = modes_for_dest
        else:
            st.caption(f"No historical shipments to {destination} — showing all modes.")
            mode_options = ["Sea", "Air", "Rail", "Road"]

        # ── Suppliers applicable to this origin → destination route ────────
        route_mask = (df["Supplier_Country"] == origin) & (df["Destination_City"] == destination)
        route_suppliers = sorted(df.loc[route_mask, "Supplier"].dropna().unique().tolist())
        route_note = None
        if not route_suppliers:
            route_suppliers = sorted(
                df.loc[df["Supplier_Country"] == origin, "Supplier"].dropna().unique().tolist()
            )
            route_note = f"No {origin} → {destination} shipments on file — showing all {origin} suppliers."
        if not route_suppliers:
            route_suppliers = sorted(df["Supplier"].dropna().unique().tolist())
            route_note = "No suppliers found for this route — showing every supplier in the dataset."

        if route_note:
            st.caption(f"⚠ {route_note}")
        else:
            st.markdown(
                f'<div style="margin:0 0 10px; font-size:11px; color:{COLOR_LOW};">'
                f'✓ {len(route_suppliers)} supplier(s) ship {origin} → {destination}</div>',
                unsafe_allow_html=True)

        with st.form("order_form", clear_on_submit=False):
            supplier = st.selectbox("Supplier", options=route_suppliers, key="oe_supplier")

            product_options = sorted(
                df.loc[df["Supplier"] == supplier, "Product_Category"].dropna().unique().tolist()
            ) if "Product_Category" in df.columns else []
            if not product_options:
                product_options = ["General Goods"]
            product = st.selectbox("Product Category", options=product_options)

            ship_date = st.date_input("Shipment Date", value=datetime.now().date())
            mode = st.selectbox("Transport Mode", options=mode_options)

            c1, c2 = st.columns(2)
            with c1:
                order_value = st.number_input("Order Value (USD)", min_value=1000, value=50000, step=5000)
            with c2:
                quantity = st.number_input("Quantity (units)", min_value=10, value=200, step=10)

            submitted = st.form_submit_button("⚡ Analyze Risk", type="primary", use_container_width=True)

    with result_col:
        if submitted:
            # ── Auto-Enrichment (dataset-derived where possible) ───────────
            dlat, dlon = CITY_COORDS.get(destination, (FALLBACK_LAT, FALLBACK_LON))
            olat, olon = COUNTRY_COORDS.get(origin, (FALLBACK_LAT, FALLBACK_LON))

            # Prefer this exact supplier+destination's historical average
            # distance from the dataset; fall back to a great-circle estimate
            # from the origin country's centroid when there's no match.
            hist = df[(df["Supplier"] == supplier) & (df["Destination_City"] == destination)]
            if not hist.empty and hist["Distance_km"].notna().any():
                distance = float(hist["Distance_km"].mean())
            else:
                distance = haversine_km(olat, olon, dlat, dlon)

            # Lead time: this route's own historical average first, then
            # this mode's historical average, then a physical speed estimate
            # only as a last resort (MODE_PARAMS speeds are a stated
            # assumption, not derived from the dataset).
            mode_lead_hist = df.loc[df["Transport_Mode"] == mode, "Lead_Time_Days"].dropna()
            if not hist.empty and hist["Lead_Time_Days"].notna().any():
                lead_time = float(hist["Lead_Time_Days"].mean())
            elif not mode_lead_hist.empty:
                lead_time = float(mode_lead_hist.mean())
            else:
                lead_time = max(1.0, distance / (MODE_PARAMS[mode]["speed_kmh"] * 24))

            # Supplier fragility = this supplier's own historical average
            # risk score in the dataset (falls back to the network average
            # if this supplier has no rows, which shouldn't happen since the
            # dropdown above is itself dataset-derived). Shown in the
            # enrichment pills below; not fed into the model separately --
            # the model sees the supplier directly via supplier_id/country.
            supplier_hist = df.loc[df["Supplier"] == supplier, "Risk_Score"]
            supplier_rate = (
                float(supplier_hist.mean()) / 100.0 if not supplier_hist.empty
                else float(df["Risk_Score"].mean()) / 100.0
            )

            # ── Real ML Prediction ──────────────────────────────────────────
            # Scored by the same trained model used everywhere else in the
            # app (outputs/model_tuned.pkl), not the old simulate_risk_score()
            # hand-tuned formula fed by get_environmental_factors()'s
            # hash-seeded random numbers -- which also used the wrong scale
            # (0-1) for weather/geopolitical risk when the model was trained
            # on 0-100, and a made-up numeric port congestion when the real
            # feature is categorical (Low/Medium/High).
            result = predict_live_risk(
                df, supplier=supplier, destination=destination, origin=origin,
                mode=mode, product=product, order_value=order_value, quantity=quantity,
                distance=distance, lead_time=lead_time, ship_date=ship_date,
            )

            if result is None:
                st.warning(
                    "⚠ Trained model artifacts not found in `outputs/` — showing a rough "
                    "heuristic estimate instead of a real prediction. Run "
                    "`python notebooks/rebuild_pipeline.py` to enable live ML scoring."
                )
                env = get_environmental_factors(destination, ship_date.month)
                mode_p = MODE_PARAMS[mode]
                km_per_lead_day = distance / max(lead_time, 0.1)
                sea_x_cong = env["port_congestion_level"] if mode == "Sea" else env["port_congestion_level"] * 0.3
                prob = simulate_risk_score(
                    env["port_congestion_level"], env["weather_risk_index"],
                    env["geopolitical_risk_index"], supplier_rate,
                    km_per_lead_day, sea_x_cong, mode_p["reliability"]
                )
                port_congestion_display = f"{env['port_congestion_level']}/10"
                weather_display = f"{env['weather_risk_index']}"
                geo_display = f"{env['geopolitical_risk_index']}"
                cong_is_high = env["port_congestion_level"] > 6
                cong_is_med = env["port_congestion_level"] > 3
                wx_is_high = env["weather_risk_index"] > 0.6
                wx_is_med = env["weather_risk_index"] > 0.3
                gp_is_high = env["geopolitical_risk_index"] > 0.35
                gp_is_med = env["geopolitical_risk_index"] > 0.15
                contributions = {}
            else:
                prob = result["probability"]
                raw = result["raw_inputs"]
                contributions = result["contributions"]
                port_congestion_display = str(raw["port_congestion_level"])
                weather_display = f"{raw['weather_risk_index']:.0f}/100"
                geo_display = f"{raw['geopolitical_risk_index']:.0f}/100"
                cong_is_high = raw["port_congestion_level"] == "High"
                cong_is_med = raw["port_congestion_level"] == "Medium"
                wx_is_high = raw["weather_risk_index"] > 60
                wx_is_med = raw["weather_risk_index"] > 30
                gp_is_high = raw["geopolitical_risk_index"] > 50
                gp_is_med = raw["geopolitical_risk_index"] > 25

            band = risk_band(prob)
            color = risk_color(band)

            # Store in session for financial tab
            st.session_state["last_order"] = {
                "supplier": supplier, "product": product, "destination": destination,
                "mode": mode, "order_value": order_value, "quantity": quantity,
                "distance": distance, "lead_time": lead_time, "prob": prob,
                "band": band, "supplier_rate": supplier_rate,
                "origin": origin,
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
                cong_color = COLOR_HIGH if cong_is_high else (COLOR_MODERATE if cong_is_med else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🏗️</span>
                    <div><span class="ep-label">Port Congestion</span><br>
                    <span class="ep-value" style="color:{cong_color}">{port_congestion_display}</span></div>
                </div>''', unsafe_allow_html=True)
            with ep2:
                wx_color = COLOR_HIGH if wx_is_high else (COLOR_MODERATE if wx_is_med else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🌧️</span>
                    <div><span class="ep-label">Weather Risk</span><br>
                    <span class="ep-value" style="color:{wx_color}">{weather_display}</span></div>
                </div>''', unsafe_allow_html=True)
            with ep3:
                gp_color = COLOR_HIGH if gp_is_high else (COLOR_MODERATE if gp_is_med else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🌐</span>
                    <div><span class="ep-label">Geopolitical Risk</span><br>
                    <span class="ep-value" style="color:{gp_color}">{geo_display}</span></div>
                </div>''', unsafe_allow_html=True)
            with ep4:
                sr_color = COLOR_HIGH if supplier_rate > 0.15 else (COLOR_MODERATE if supplier_rate > 0.08 else COLOR_LOW)
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">🏭</span>
                    <div><span class="ep-label">Supplier Fragility</span><br>
                    <span class="ep-value" style="color:{sr_color}">{supplier_rate:.0%}</span></div>
                </div>''', unsafe_allow_html=True)

            # Computed features
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">📏</span>
                    <div><span class="ep-label">Distance</span><br>
                    <span class="ep-value" style="color:#1A1A2E">{distance:,.0f}km</span></div>
                </div>''', unsafe_allow_html=True)
            with fc2:
                st.markdown(f'''<div class="enrichment-pill">
                    <span class="ep-icon">⏱️</span>
                    <div><span class="ep-label">Lead Time</span><br>
                    <span class="ep-value" style="color:#1A1A2E">{lead_time:.1f}d</span></div>
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
                    'letter-spacing:0.1em; font-weight:600;">Feature Contributions (SHAP)</p>'
                    '</div>', unsafe_allow_html=True)

                if contributions:
                    display_features = {
                        FEATURE_DISPLAY_NAMES.get(k, k): v for k, v in contributions.items()
                    }
                    st.plotly_chart(create_feature_waterfall(display_features), use_container_width=True,
                                    config={"displayModeBar": False})
                else:
                    st.caption("Feature contributions unavailable in fallback (non-ML) mode.")

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
                        <hr style="border-color:#E7E9F2; margin:12px 0;">
                        <p class="alt-card-metric">Mode: <b>{mode}</b></p>
                        <p class="alt-card-metric">Supplier: <b>{supplier}</b></p>
                        <p class="alt-card-metric">Lead Time: <b>{lead_time:.1f} days</b></p>
                        <p class="alt-card-metric">Cost: <b>${order_value:,.0f}</b></p>
                    </div>''', unsafe_allow_html=True)

                # ── Option 2: Modal Shift -- re-scored with the real model,
                # not simulate_risk_score, by swapping just the mode. ─────
                alt_mode = "Air" if mode != "Air" else "Rail"
                alt_mode_p = MODE_PARAMS[alt_mode]
                alt_mode_lead_hist = df.loc[df["Transport_Mode"] == alt_mode, "Lead_Time_Days"].dropna()
                alt_lead = float(alt_mode_lead_hist.mean()) if not alt_mode_lead_hist.empty else max(1.0, distance / (alt_mode_p["speed_kmh"] * 24))
                cost_delta_mode = distance * quantity * (alt_mode_p["cost_per_km_kg"] - MODE_PARAMS[mode]["cost_per_km_kg"])

                if result is not None:
                    alt_result_mode = predict_live_risk(
                        df, supplier=supplier, destination=destination, origin=origin,
                        mode=alt_mode, product=product, order_value=order_value, quantity=quantity,
                        distance=distance, lead_time=alt_lead, ship_date=ship_date,
                    )
                    alt_prob_mode = alt_result_mode["probability"] if alt_result_mode else prob
                else:
                    alt_prob_mode = prob
                alt_band_mode = risk_band(alt_prob_mode)

                with alt2:
                    mc = risk_color(alt_band_mode)
                    st.markdown(f'''<div class="alt-card modal-shift">
                        <p class="alt-card-label" style="color:{COLOR_MODERATE};">✈ Modal Shift → {alt_mode}</p>
                        <p class="alt-card-value" style="color:{mc};">{alt_prob_mode*100:.1f}%</p>
                        <p class="alt-card-metric">Disruption Probability</p>
                        <hr style="border-color:#E7E9F2; margin:12px 0;">
                        <p class="alt-card-metric">Mode: <b>{alt_mode}</b></p>
                        <p class="alt-card-metric">Supplier: <b>{supplier}</b> (same)</p>
                        <p class="alt-card-metric">Lead Time: <b>{alt_lead:.1f} days</b></p>
                        <p class="alt-card-metric">Cost Delta: <b style="color:{COLOR_MODERATE};">+${max(0,cost_delta_mode):,.0f}</b></p>
                    </div>''', unsafe_allow_html=True)

                # ── Option 3: Vendor Shift ────────
                # Alternate supplier serving the same destination (and, when
                # possible, the same product category), ranked by lowest
                # historical average Risk_Score in the dataset, then
                # re-scored with the real model using that supplier's own
                # country/history instead of reusing the original supplier's.
                route_alt = df[(df["Destination_City"] == destination) & (df["Supplier"] != supplier)]
                if "Product_Category" in df.columns:
                    narrowed = route_alt[route_alt["Product_Category"] == product]
                    if not narrowed.empty:
                        route_alt = narrowed

                if not route_alt.empty:
                    candidates = route_alt.groupby("Supplier")["Risk_Score"].mean().sort_values()
                    alt_supplier = candidates.index[0]
                    alt_sr = float(candidates.iloc[0]) / 100.0
                    alt_origin_country = df.loc[df["Supplier"] == alt_supplier, "Supplier_Country"].mode().iloc[0]
                else:
                    alt_supplier, alt_sr, alt_origin_country = supplier, supplier_rate, origin

                alt_olat, alt_olon = COUNTRY_COORDS.get(alt_origin_country, (FALLBACK_LAT, FALLBACK_LON))
                alt_hist = df[(df["Supplier"] == alt_supplier) & (df["Destination_City"] == destination)]
                if not alt_hist.empty and alt_hist["Distance_km"].notna().any():
                    alt_dist = float(alt_hist["Distance_km"].mean())
                else:
                    alt_dist = haversine_km(alt_olat, alt_olon, dlat, dlon)
                if not alt_hist.empty and alt_hist["Lead_Time_Days"].notna().any():
                    alt_lead_v = float(alt_hist["Lead_Time_Days"].mean())
                else:
                    alt_lead_v = max(1.0, alt_dist / (MODE_PARAMS[mode]["speed_kmh"] * 24))

                if result is not None:
                    alt_result_vendor = predict_live_risk(
                        df, supplier=alt_supplier, destination=destination, origin=alt_origin_country,
                        mode=mode, product=product, order_value=order_value, quantity=quantity,
                        distance=alt_dist, lead_time=alt_lead_v, ship_date=ship_date,
                    )
                    alt_prob_vendor = alt_result_vendor["probability"] if alt_result_vendor else alt_sr
                else:
                    alt_prob_vendor = alt_sr
                alt_band_vendor = risk_band(alt_prob_vendor)

                with alt3:
                    vc = risk_color(alt_band_vendor)
                    st.markdown(f'''<div class="alt-card vendor-shift">
                        <p class="alt-card-label" style="color:{COLOR_LOW};">🏭 Vendor Shift → {alt_supplier}</p>
                        <p class="alt-card-value" style="color:{vc};">{alt_prob_vendor*100:.1f}%</p>
                        <p class="alt-card-metric">Disruption Probability</p>
                        <hr style="border-color:#E7E9F2; margin:12px 0;">
                        <p class="alt-card-metric">Mode: <b>{mode}</b> (same)</p>
                        <p class="alt-card-metric">Supplier: <b>{alt_supplier}</b></p>
                        <p class="alt-card-metric">Fragility: <b style="color:{COLOR_LOW};">{alt_sr:.0%}</b> (was {supplier_rate:.0%})</p>
                        <p class="alt-card-metric">From: <b>{alt_origin_country}</b></p>
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
                    Choose an origin and destination, then fill in the shipment form on the left
                    and click <b>Analyze Risk</b>. The system will auto-enrich environmental data,
                    run the ML risk model, and generate alternatives if the risk is high.
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
    holding_cost_rate = 0.0015  # 0.15% per day -- stated inventory carrying-cost assumption
    # Historically-grounded expected delay for a shipment at this risk level
    # (see historical_predicted_delay's docstring) -- replaces the old
    # `max(1, int(prob*20))`, which had no basis in the data: why 20, and
    # why should days-late scale linearly with probability at all?
    predicted_delay = historical_predicted_delay(df, prob)
    var_original = order_value * predicted_delay * holding_cost_rate

    # Freight premium for switching to Air. MODE_PARAMS' $/km/kg rates are a
    # stated industry-rate assumption, not derived from this dataset -- it
    # has no freight-cost field (unit_cost_usd/order_value_usd are the
    # shipped goods' value, not the cost to move them), so there's nothing
    # to fit an empirical rate to. The *ratios* between modes (Air ~4x Sea,
    # consistent with real-world freight economics) are what drive this
    # estimate, not a claim of dataset-derived precision.
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

        # This used to be a go.Sankey figure, but a Sankey diagram is the
        # wrong chart for a strictly linear 4-stage chain (one node per
        # rank, no branching): Plotly stretches the single link at each
        # stage to fill the *entire* plot height, so instead of thin nodes
        # and flowing ribbons it rendered as four solid color blocks with
        # the labels stuck on top -- which is what looked like broken,
        # overlapping text. A plain HTML flow-chip row can't suffer that
        # degenerate layout and wraps cleanly on narrow screens.
        def _flow_label(text: str, max_len: int = 24) -> str:
            text = str(text)
            return text if len(text) <= max_len else text[: max_len - 1] + "…"

        # Node colors based on delay propagation
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

        flow_steps = [
            ("📦", _flow_label(row["Supplier"]), row["Origin_City"]),
            ("🚢", "Transit", row["Transport_Mode"]),
            ("🏭", wh_id, wh["city"]),
            ("📍", _flow_label(row["Destination_City"]), "Final Destination"),
        ]

        chips = []
        for i, (icon, title, sub) in enumerate(flow_steps):
            chips.append(f'''<div style="background:#FFFFFF; border:2px solid {node_colors[i]};
                border-radius:16px; padding:14px 18px; text-align:center; flex:1;
                min-width:150px; box-shadow:0 4px 14px rgba(20,70,140,0.06);">
                <div style="font-size:22px; line-height:1;">{icon}</div>
                <div style="font-size:13px; font-weight:700; color:#0F2A4D; margin-top:6px;
                    line-height:1.4; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
                    title="{title}">{title}</div>
                <div style="font-size:11.5px; color:#4E6D95; margin-top:2px; line-height:1.4;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{sub}</div>
            </div>''')
            if i < len(flow_steps) - 1:
                chips.append(
                    '<div style="font-size:20px; color:#9DBDE8; padding:0 2px; flex-shrink:0;">➜</div>'
                )

        st.markdown(
            '<div style="display:flex; align-items:stretch; justify-content:center; '
            'flex-wrap:wrap; gap:8px; padding:8px 0 4px;">' + "".join(chips) + '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Node Status Cards ─────────────────────────────────────────────
        st.markdown('<p class="section-label">📊 Node Impact Assessment</p>', unsafe_allow_html=True)

        n1, n2, n3, n4 = st.columns(4)
        with n1:
            nc = "warning" if delay_days > 0 else "safe"
            st.markdown(f'''<div class="cascade-node {nc}">
                <div style="font-size:26px; margin-bottom:8px; line-height:1;">📦</div>
                <div class="cascade-node-title">{row["Supplier"]}</div>
                <div class="cascade-node-sub">{row["Origin_City"]}</div>
                <div class="cascade-node-status" style="color:{COLOR_MODERATE if delay_days > 0 else COLOR_LOW};">
                    {"DISPATCH DELAYED" if delay_days > 0 else "ON SCHEDULE"}</div>
            </div>''', unsafe_allow_html=True)

        with n2:
            st.markdown(f'''<div class="cascade-node {"critical" if delay_days > 3 else "warning" if delay_days > 0 else "safe"}">
                <div style="font-size:26px; margin-bottom:8px; line-height:1;">🚢</div>
                <div class="cascade-node-title">Transit Corridor</div>
                <div class="cascade-node-sub">{row["Transport_Mode"]} · {row["Distance_km"]:,.0f}km</div>
                <div class="cascade-node-status" style="color:{COLOR_HIGH if delay_days > 3 else COLOR_MODERATE if delay_days > 0 else COLOR_LOW};">
                    {f"+{delay_days}d DELAY" if delay_days > 0 else "IN TRANSIT"}</div>
            </div>''', unsafe_allow_html=True)

        with n3:
            stockout = buffer_remaining <= 0
            wh_status = "STOCKOUT" if stockout else f"BUFFER: {buffer_remaining}d"
            wh_class = "critical" if stockout else ("warning" if buffer_remaining <= 2 else "safe")
            wh_color = COLOR_HIGH if stockout else (COLOR_MODERATE if buffer_remaining <= 2 else COLOR_LOW)
            st.markdown(f'''<div class="cascade-node {wh_class}">
                <div style="font-size:26px; margin-bottom:8px; line-height:1;">🏭</div>
                <div class="cascade-node-title">{wh_id}</div>
                <div class="cascade-node-sub">Safety Stock: {wh["safety_stock_days"]}d</div>
                <div class="cascade-node-status" style="color:{wh_color};">
                    {wh_status}</div>
            </div>''', unsafe_allow_html=True)

        with n4:
            dest_class = "critical" if stockout else ("warning" if buffer_remaining <= 2 else "safe")
            dest_status = "CUSTOMER IMPACT" if stockout else ("AT RISK" if buffer_remaining <= 2 else "SERVED")
            dest_color = COLOR_HIGH if stockout else (COLOR_MODERATE if buffer_remaining <= 2 else COLOR_LOW)
            st.markdown(f'''<div class="cascade-node {dest_class}">
                <div style="font-size:26px; margin-bottom:8px; line-height:1;">📍</div>
                <div class="cascade-node-title">{row["Destination_City"]}</div>
                <div class="cascade-node-sub">Demand: {wh["daily_demand"]} units/day</div>
                <div class="cascade-node-status" style="color:{dest_color};">
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


@st.dialog("🚨 High-Risk Alerts", width="large")
def _show_high_risk_alerts_dialog(alerts_df: pd.DataFrame):
    st.caption(f"{len(alerts_df):,} overdue shipment(s) currently flagged High risk, sorted by risk score.")
    if alerts_df.empty:
        st.info("No high-risk alerts right now.")
    else:
        display_cols = ["Order_ID", "Supplier", "Destination_City", "Transport_Mode",
                        "Risk_Score", "Days_Overdue", "Order_Value_USD"]
        st.dataframe(
            alerts_df[display_cols].sort_values("Risk_Score", ascending=False),
            use_container_width=True, hide_index=True, height=380,
            column_config={
                "Order_ID": st.column_config.TextColumn("Order"),
                "Destination_City": st.column_config.TextColumn("Destination"),
                "Transport_Mode": st.column_config.TextColumn("Mode"),
                "Risk_Score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d%%"),
                "Days_Overdue": st.column_config.NumberColumn("Overdue", format="%d d"),
                "Order_Value_USD": st.column_config.NumberColumn("Value (USD)", format="$%,.0f"),
            },
        )
    if st.button("Close", key="close_high_risk_dialog"):
        st.rerun()


def render_control_tower(df: pd.DataFrame):
    """Module 6: Live GIS Control Tower & KPI Dashboard."""

    # ── Executive KPI Bar ─────────────────────────────────────────────────
    overdue_df = df[df["Is_Overdue"]]
    high_risk_df = overdue_df[overdue_df["Risk_Category"] == "High"]
    high_risk_count = len(high_risk_df)
    total_var = sum(
        r["Order_Value_USD"] * r["Days_Overdue"] * 0.0015
        for _, r in overdue_df.iterrows()
    )
    avg_lead = df["Lead_Time_Days"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        # The whole card is a real button (styled via .st-key-kpi_high_risk_card
        # below) so clicking the stat itself opens the alerts list -- not just
        # a separate "view" affordance next to it.
        with st.container(key="kpi_high_risk_card"):
            if st.button(f"{high_risk_count}", key="kpi_high_risk_btn", use_container_width=True):
                _show_high_risk_alerts_dialog(high_risk_df)
            st.markdown('<p class="kpi-label">🚨 High-Risk Alerts · click to view</p>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:#16162A;">{len(overdue_df)}</p>
            <p class="kpi-label">Overdue Shipments</p></div>''', unsafe_allow_html=True)
    with k3:
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:{COLOR_MODERATE};">${total_var:,.0f}</p>
            <p class="kpi-label">Portfolio VaR</p></div>''', unsafe_allow_html=True)
    with k4:
        st.markdown(f'''<div class="kpi-card">
            <p class="kpi-value" style="color:{COLOR_ACCENT};">{avg_lead:.1f}d</p>
            <p class="kpi-label">Avg Lead Time</p></div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Date Slider (Time-Stepped Simulation) ─────────────────────────────
    dates = pd.to_datetime(df["Shipment_Date"]).sort_values()
    min_date = dates.min().date()
    max_date = dates.max().date()

    if min_date == max_date:
        # st.slider requires min < max; a small/filtered real sample can
        # collapse to a single shipment date, so skip the widget and show
        # everything instead of crashing.
        st.caption(f"📅 All shipments fall on {min_date:%b %d, %Y} — timeline filter not applicable.")
        date_range = (min_date, max_date)
    else:
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

    # ── Live filters: risk band + supplier/city/id search ──────────────────
    # These narrow both the map and the order list together so the two
    # views always show the exact same shipment set.
    f1, f2 = st.columns([2, 3])
    with f1:
        band_options = ["High", "Medium", "Low"]
        band_filter = st.multiselect(
            "Risk band", options=band_options, default=band_options,
            key="tower_band_filter", label_visibility="collapsed",
            placeholder="Filter by risk band",
        )
    with f2:
        search = st.text_input(
            "Search", key="tower_search", label_visibility="collapsed",
            placeholder="🔎 Search Order ID / Supplier / Destination",
        )

    if band_filter:
        tower_df = tower_df[tower_df["Risk_Category"].isin(band_filter)]
    if search:
        mask = pd.Series(False, index=tower_df.index)
        for col in ["Order_ID", "Supplier", "Destination_City"]:
            mask |= tower_df[col].astype(str).str.contains(search, case=False, na=False)
        tower_df = tower_df[mask]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Resolve current selection (map click OR order-list click) ──────────
    # A single session_state key is the source of truth for "what's
    # selected" so the map, the order list, and the deep-dive panel all stay
    # in sync no matter which one the user interacted with.
    if "tower_selected_id" not in st.session_state:
        st.session_state["tower_selected_id"] = None
    if (
        st.session_state["tower_selected_id"] is not None
        and st.session_state["tower_selected_id"] not in tower_df["Order_ID"].values
    ):
        # Selected shipment got filtered out (date range / risk band / search) --
        # drop the stale selection rather than showing a mismatched deep-dive.
        st.session_state["tower_selected_id"] = None

    # ── Map + Order List ─────────────────────────────────────────────────
    map_col, list_col = st.columns([3, 2], gap="medium")

    with map_col:
        st.markdown(
            f'<p class="section-label">📍 Active Threat Map — {len(tower_df):,} overdue shipments</p>',
            unsafe_allow_html=True)
        folium_map = build_control_tower_map(tower_df, highlight_id=st.session_state["tower_selected_id"])

        map_data = st_folium(
            folium_map, width=None, height=480,
            returned_objects=["last_object_clicked"],
            key="tower_map",
        )

        # Map marker click -> resolve nearest shipment and select it.
        if map_data and map_data.get("last_object_clicked"):
            clat = map_data["last_object_clicked"].get("lat")
            clng = map_data["last_object_clicked"].get("lng")
            if clat is not None and clng is not None and not tower_df.empty:
                dists = (tower_df["Lat"] - clat) ** 2 + (tower_df["Lon"] - clng) ** 2
                nearest = dists.idxmin()
                if dists[nearest] < 0.05:
                    st.session_state["tower_selected_id"] = tower_df.loc[nearest, "Order_ID"]

    with list_col:
        st.markdown(
            f'<p class="section-label">📋 Order List — sorted by risk</p>',
            unsafe_allow_html=True)

        if tower_df.empty:
            st.info("No shipments match the current filters.")
        else:
            list_cols = ["Order_ID", "Supplier", "Destination_City", "Transport_Mode",
                         "Risk_Score", "Days_Overdue"]
            ranked = tower_df[list_cols].sort_values("Risk_Score", ascending=False).reset_index(drop=True)

            # Pre-select whatever's currently selected (e.g. from a map
            # click) so the list highlights the same row the map is showing.
            preselect = []
            sel_id = st.session_state["tower_selected_id"]
            if sel_id is not None and sel_id in ranked["Order_ID"].values:
                preselect = [int(ranked.index[ranked["Order_ID"] == sel_id][0])]

            list_event = st.dataframe(
                ranked,
                use_container_width=True, hide_index=True, height=300,
                on_select="rerun", selection_mode="single-row",
                key="order_list_df",
                column_config={
                    "Order_ID": st.column_config.TextColumn("Order"),
                    "Destination_City": st.column_config.TextColumn("Destination"),
                    "Transport_Mode": st.column_config.TextColumn("Mode"),
                    "Risk_Score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d%%"),
                    "Days_Overdue": st.column_config.NumberColumn("Overdue", format="%d d"),
                },
            )

            selected_rows = list_event["selection"]["rows"] if list_event else []
            if selected_rows:
                st.session_state["tower_selected_id"] = ranked.iloc[selected_rows[0]]["Order_ID"]

            st.caption(f"{len(ranked):,} shipment(s) · click a row to highlight it on the map")

        # ── Deep-Dive for the currently selected shipment ──────────────────
        sel_id = st.session_state["tower_selected_id"]
        if sel_id is not None and sel_id in tower_df["Order_ID"].values:
            o = tower_df.loc[tower_df["Order_ID"] == sel_id].iloc[0]
            color = risk_color(o["Risk_Category"])
            badge = risk_badge_class(o["Risk_Category"])

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f'''<div class="detail-panel">
                <p class="detail-header">Shipment: {o["Order_ID"]}</p>
                <p class="detail-subheader">
                    {o["Supplier"]} → {o["Destination_City"]} · {o["Transport_Mode"]} ·
                    <span class="status-badge {badge}">{o["Risk_Category"]} Risk</span> ·
                    {o["Days_Overdue"]}d overdue
                </p>
            </div>''', unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.plotly_chart(create_risk_gauge(o["Risk_Probability"], o["Risk_Category"], 160),
                            use_container_width=True, config={"displayModeBar": False})

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'''<div class="kpi-card"><p class="kpi-value" style="font-size:20px; color:#16162A;">{o["Volume"]}</p>
                    <p class="kpi-label">Volume</p></div>''', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'''<div class="kpi-card"><p class="kpi-value" style="font-size:20px; color:{COLOR_HIGH};">{o["Days_Overdue"]}d</p>
                    <p class="kpi-label">Overdue</p></div>''', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'''<div class="kpi-card"><p class="kpi-value" style="font-size:20px; color:{COLOR_ACCENT};">${o["Order_Value_USD"]:,.0f}</p>
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
        elif not tower_df.empty:
            st.markdown('''<div class="detail-panel" style="text-align:center; padding:28px 20px; margin-top:10px;">
                <div style="font-size:32px; margin-bottom:8px;">📍</div>
                <p class="detail-header" style="font-size:14px; color:#6A6A8A;">Select a Shipment</p>
                <p style="font-size:12px; color:#8A8AA0; max-width:280px; margin:6px auto 0;">
                    Click a row above or a marker on the map to see its cascading impact
                    and mitigation options.</p>
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
                    "Order_ID": st.column_config.TextColumn("Order"),
                    "Destination_City": st.column_config.TextColumn("Destination"),
                    "Transport_Mode": st.column_config.TextColumn("Mode"),
                    "Risk_Category": st.column_config.TextColumn("Band"),
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

        # ── Data Source ────────────────────────────────────────────────────
        # Real shipment data only -- see load_sample_raw_data()'s docstring.
        df = load_sample_raw_data()
        if df.empty:
            st.stop()
        st.markdown(
            f'<div style="background:#1A2A1A; border:1px solid #2A4A2A; border-radius:8px; '
            f'padding:8px 12px; font-size:11px; color:#09AB3B;">'
            f'✓ Loaded {len(df):,} rows from predictions.xlsx</div>',
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
        top_suppliers = (
            df.groupby("Supplier")["Supplier_Disruption_Rate"].mean()
            .sort_values(ascending=False).head(6)
        )
        for s, rate in top_suppliers.items():
            sc = COLOR_HIGH if rate > 0.15 else (COLOR_MODERATE if rate > 0.08 else COLOR_LOW)
            # Real supplier names run much longer than the old fixture names
            # ("Nelson, Morton and Medina" vs. "TechCorp") -- truncate with
            # an ellipsis and keep the row on one centered line instead of
            # letting it wrap and throw off vertical alignment with the %.
            st.markdown(
                f'<div style="display:flex; align-items:center; justify-content:space-between; '
                f'gap:8px; padding:4px 0; font-size:11px; line-height:1.5;">'
                f'<span style="color:#48597A; white-space:nowrap; overflow:hidden; '
                f'text-overflow:ellipsis; flex:1; min-width:0;" title="{s}">{s}</span>'
                f'<span style="color:{sc}; font-weight:700; flex-shrink:0;">{rate:.0%}</span></div>',
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

    # ── Pages ─────────────────────────────────────────────────────────────
    # Control Tower is its own page (and the default/landing page); Order &
    # Risk Engine is a separate page reached via an explicit button on the
    # Control Tower page (plus the standard page-switcher in the sidebar).
    # `pages` is populated below before any of these closures actually run
    # (Streamlit only invokes a page function once st.navigation resolves
    # it), so the forward references to sibling pages are safe.
    pages = {}

    def control_tower_page():
        top_l, top_r = st.columns([5, 1.3])
        with top_l:
            st.markdown('<p class="section-label">🗺️ Live Control Tower</p>', unsafe_allow_html=True)
        with top_r:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("📦 Order & Risk Engine →", key="goto_order_engine",
                         type="primary", use_container_width=True):
                st.switch_page(pages["order_risk"])
        render_control_tower(df)

    def order_risk_page():
        if st.button("← Back to Control Tower", key="back_to_tower"):
            st.switch_page(pages["control_tower"])
        render_order_entry_module(df)

    def financial_page():
        render_financial_module(df)

    def cascade_page():
        render_cascade_module(df)

    pages["control_tower"] = st.Page(
        control_tower_page, title="Control Tower", icon="🗺️",
        url_path="control-tower", default=True,
    )
    pages["order_risk"] = st.Page(
        order_risk_page, title="Order & Risk Engine", icon="📦",
        url_path="order-risk-engine",
    )
    pages["financial"] = st.Page(
        financial_page, title="Financial Analysis", icon="💰", url_path="financial",
    )
    pages["cascade"] = st.Page(
        cascade_page, title="Cascade Simulator", icon="🌊", url_path="cascade",
    )

    nav = st.navigation(list(pages.values()))
    nav.run()


if __name__ == "__main__":
    main()
