"""
STEP 4: STREAMLIT INTERFACE — "TerraScan" Edition
-----------------------------------------------------
Run:  streamlit run app.py
"""

import streamlit as st
import base64
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pydeck as pdk
from datetime import datetime

from model import MultimodalFusionModel
from data_loader import FEATURE_COLS, CLASS_NAMES, generate_features_for_class
from grad_cam import GradCAM, overlay_heatmap
import shap_utils as shx
import report_utils as rpt
import extras_utils as ex
import llm_utils as llm

st.set_page_config(
    page_title="TerraScan · Multi-Hazard Risk Console",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
APP_VERSION = "v1.0"

HAZARD_META = {
    "Flood":      {"icon": "🌊", "color": "#38BDF8", "glow": "56,189,248"},
    "Wildfire":   {"icon": "🔥", "color": "#FB7A3C", "glow": "251,122,60"},
    "Cyclone":    {"icon": "🌀", "color": "#B892F0", "glow": "184,146,240"},
    "Earthquake": {"icon": "🟤", "color": "#F0567A", "glow": "240,86,122"},
}

FEATURE_META = {
    "rainfall":         {"label": "Rainfall",          "unit": "mm",   "min": 0.0,   "max": 400.0, "default": 100.0},
    "temperature":      {"label": "Temperature",       "unit": "°C",   "min": -10.0, "max": 55.0,  "default": 28.0},
    "humidity":         {"label": "Humidity",          "unit": "%",    "min": 0.0,   "max": 100.0, "default": 60.0},
    "wind_speed":       {"label": "Wind Speed",        "unit": "km/h", "min": 0.0,   "max": 220.0, "default": 30.0},
    "soil_moisture":    {"label": "Soil Moisture",     "unit": "",     "min": 0.0,   "max": 1.0,   "default": 0.4},
    "ndvi":             {"label": "NDVI",               "unit": "",     "min": -1.0,  "max": 1.0,   "default": 0.2},
    "seismic_activity": {"label": "Seismic Activity",  "unit": "Mw",   "min": 0.0,   "max": 8.0,   "default": 1.5},
    "elevation":        {"label": "Elevation",          "unit": "m",    "min": 0.0,   "max": 3000.0,"default": 200.0},
}

# =========================================================
# CSS — "ORBIT": black + blue satellite/space console theme
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        /* Earth-like glowing sphere, top-right */
        radial-gradient(circle 340px at 82% 6%, rgba(226,246,255,0.55) 0%, rgba(147,220,255,0.45) 8%, rgba(56,189,248,0.38) 22%, rgba(34,139,230,0.28) 38%, rgba(15,60,120,0.14) 55%, transparent 68%),
        /* terminator shadow so the sphere reads as a lit globe, not a flat glow */
        radial-gradient(circle 340px at 91% 2%, rgba(2,4,10,0.55) 0%, rgba(2,4,10,0.35) 30%, transparent 55%),
        /* soft cyan atmosphere rim */
        radial-gradient(circle 380px at 82% 6%, transparent 60%, rgba(94,234,212,0.10) 66%, transparent 74%),
        /* ambient glows lower on the page */
        radial-gradient(ellipse 1400px 700px at 50% 115%, rgba(56,189,248,0.22), rgba(37,99,235,0.08) 45%, transparent 65%),
        radial-gradient(circle at 8% 15%, rgba(59,130,246,0.08), transparent 40%),
        /* starfield */
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='500' height='500'%3E%3Cg fill='%23FFFFFF'%3E%3Ccircle cx='40' cy='60' r='1.2' opacity='0.9'/%3E%3Ccircle cx='120' cy='20' r='0.9' opacity='0.6'/%3E%3Ccircle cx='210' cy='90' r='1.4' opacity='0.8'/%3E%3Ccircle cx='300' cy='40' r='0.8' opacity='0.5'/%3E%3Ccircle cx='370' cy='120' r='1.1' opacity='0.7'/%3E%3Ccircle cx='450' cy='60' r='1.3' opacity='0.9'/%3E%3Ccircle cx='60' cy='180' r='0.9' opacity='0.6'/%3E%3Ccircle cx='150' cy='220' r='1.2' opacity='0.8'/%3E%3Ccircle cx='250' cy='170' r='0.8' opacity='0.5'/%3E%3Ccircle cx='330' cy='240' r='1.1' opacity='0.7'/%3E%3Ccircle cx='420' cy='200' r='1.4' opacity='0.9'/%3E%3Ccircle cx='480' cy='280' r='0.9' opacity='0.6'/%3E%3Ccircle cx='20' cy='320' r='1.2' opacity='0.8'/%3E%3Ccircle cx='100' cy='380' r='0.8' opacity='0.5'/%3E%3Ccircle cx='190' cy='340' r='1.3' opacity='0.9'/%3E%3Ccircle cx='270' cy='410' r='0.9' opacity='0.6'/%3E%3Ccircle cx='360' cy='360' r='1.1' opacity='0.7'/%3E%3Ccircle cx='440' cy='430' r='1.2' opacity='0.8'/%3E%3Ccircle cx='70' cy='460' r='0.8' opacity='0.5'/%3E%3Ccircle cx='160' cy='480' r='1.4' opacity='0.9'/%3E%3C/g%3E%3C/svg%3E"),
        /* flowing satellite-telemetry wave lines */
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='500' viewBox='0 0 900 500'%3E%3Cg fill='none' stroke='%2338BDF8' stroke-opacity='0.07' stroke-width='1.4'%3E%3Cpath d='M0,120 C150,80 250,180 400,130 C550,80 650,180 900,110'/%3E%3Cpath d='M0,200 C150,160 250,260 400,210 C550,160 650,260 900,190'/%3E%3Cpath d='M0,280 C150,240 250,340 400,290 C550,240 650,340 900,270'/%3E%3Cpath d='M0,360 C150,320 250,420 400,370 C550,320 650,420 900,350'/%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(180deg, #02040A 0%, #050B18 55%, #02060F 100%);
    background-attachment: fixed, fixed, fixed, fixed, fixed, fixed, fixed, fixed;
    background-repeat: no-repeat, no-repeat, no-repeat, no-repeat, no-repeat, repeat, repeat, no-repeat;
    background-size: cover, cover, cover, cover, cover, 500px 500px, 900px 500px, cover;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

/* ---- Ticker / system notice bar ---- */
.ticker {
    display: flex; align-items: center; gap: 10px;
    background: rgba(56,189,248,0.06);
    backdrop-filter: blur(14px) saturate(180%);
    -webkit-backdrop-filter: blur(14px) saturate(180%);
    border: 1px solid rgba(56,189,248,0.30);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 20px;
    font-size: 12.5px;
    color: #BAE6FD;
    box-shadow: 0 8px 24px rgba(37,99,235,0.18);
}
.ticker b { color: #38BDF8; }

/* ---- Hero (glass + earth-glow shadow) ---- */
.hero {
    padding: 32px 36px;
    border-radius: 22px;
    border: 1px solid rgba(56,189,248,0.22);
    background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(34,211,238,0.08));
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 25px 60px rgba(37,99,235,0.30), inset 0 1px 0 rgba(255,255,255,0.10);
}
.hero::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #2563EB, #38BDF8, #22D3EE, #60A5FA);
    background-size: 200% 100%;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { 0% {background-position: 0% 0;} 100% {background-position: 200% 0;} }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    color: #22D3EE;
    font-size: 12px;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg, #F8FAFC, #BAE6FD 45%, #93C5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 0.5px;
}
.hero-sub {
    color: #94C5E8;
    font-size: 14.5px;
    margin-top: 10px;
    max-width: 760px;
}
.hero-chips { margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }

/* ---- Stat strip ---- */
.stat-box {
    border-radius: 16px;
    padding: 16px 12px;
    text-align: center;
    background: linear-gradient(145deg, rgba(37,99,235,0.14), rgba(34,211,238,0.06));
    border: 1px solid rgba(56,189,248,0.18);
    box-shadow: 8px 8px 20px rgba(0,2,8,0.55), -6px -6px 16px rgba(37,99,235,0.08), inset 0 1px 0 rgba(255,255,255,0.06);
}
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(90deg, #38BDF8, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 10.5px;
    color: #7DA9CC;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ---- Glass panels ---- */
.panel {
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 18px;
    background: rgba(56,189,248,0.045);
    backdrop-filter: blur(18px) saturate(180%);
    -webkit-backdrop-filter: blur(18px) saturate(180%);
    border: 1px solid rgba(56,189,248,0.16);
    box-shadow: 0 16px 40px rgba(2,6,16,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.panel:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 50px rgba(37,99,235,0.32), inset 0 1px 0 rgba(255,255,255,0.10);
}
.panel-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 15.5px;
    font-weight: 700;
    color: #F1F8FF;
    margin-bottom: 4px;
}
.panel-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: #38BDF8;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.panel label, .panel [data-testid="stWidgetLabel"] p { color: #CFE7FA !important; font-weight: 500; }
.panel p, .panel [data-testid="stMarkdownContainer"] p, .panel .stCaption { color: #8FB9DA !important; }

/* ---- Result card ---- */
.result-card {
    border-radius: 20px;
    padding: 28px 32px;
    border: 1px solid rgba(var(--glow), 0.5);
    background: linear-gradient(135deg, rgba(var(--glow),0.22), rgba(56,189,248,0.05));
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    box-shadow: 0 0 60px rgba(var(--glow), 0.35), 0 25px 50px rgba(0,2,8,0.55), inset 0 1px 0 rgba(255,255,255,0.12);
}
.result-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    letter-spacing: 2.5px;
    color: #BAE6FD;
    text-transform: uppercase;
}
.result-hazard {
    font-family: 'Orbitron', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #FFFFFF;
    margin: 8px 0 4px 0;
    text-shadow: 0 0 30px rgba(56,189,248,0.6);
}
.result-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    color: #BAE6FD;
}
.risk-chip {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 4px 12px;
    border-radius: 20px;
    color: #FFFFFF;
    margin-top: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4);
}

/* ---- Probability bars ---- */
.prob-row { display: flex; align-items: center; gap: 10px; margin: 9px 0; }
.prob-label { width: 108px; font-size: 13px; color: #CFE7FA; font-family: 'JetBrains Mono', monospace; }
.prob-track { flex: 1; background: rgba(56,189,248,0.08); border-radius: 8px; height: 11px; overflow: hidden; box-shadow: inset 2px 2px 6px rgba(0,0,0,0.5); }
.prob-fill { height: 100%; border-radius: 8px; box-shadow: 0 0 12px rgba(56,189,248,0.5); }
.prob-val { width: 50px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: #7DA9CC; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #02040A 0%, #071227 100%);
    border-right: 1px solid rgba(56,189,248,0.18);
}
.sb-title { font-family: 'Orbitron', sans-serif; font-weight: 800; color: #38BDF8; font-size: 19px; }
.sb-sub { color: #6D93B8; font-size: 12px; margin-bottom: 14px; }
.sb-item { font-size: 13px; color: #CFE7FA; margin: 4px 0; }
.sb-badge {
    display:inline-block; font-family:'JetBrains Mono', monospace; font-size: 11px;
    background: rgba(56,189,248,0.14); color:#38BDF8; padding: 3px 10px; border-radius: 20px;
    border: 1px solid rgba(56,189,248,0.4);
}

/* ---- Buttons ---- */
.stButton>button {
    background: linear-gradient(145deg, #2563EB, #38BDF8);
    color: #FFFFFF;
    font-weight: 700;
    border: none;
    border-radius: 14px;
    padding: 12px 0;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.5px;
    box-shadow: 6px 6px 16px rgba(0,2,8,0.55), -4px -4px 12px rgba(37,99,235,0.18), inset 0 1px 1px rgba(255,255,255,0.25);
    transition: all 0.15s ease;
}
.stButton>button:hover {
    filter: brightness(1.12);
    color: #FFFFFF;
    transform: translateY(-2px);
    box-shadow: 8px 8px 22px rgba(0,2,8,0.6), -5px -5px 14px rgba(37,99,235,0.22), inset 0 1px 1px rgba(255,255,255,0.3);
}
.stButton>button:active {
    transform: translateY(1px);
    box-shadow: inset 4px 4px 10px rgba(0,2,8,0.5), inset -2px -2px 8px rgba(37,99,235,0.15);
}
.stDownloadButton>button {
    background: linear-gradient(145deg, rgba(37,99,235,0.25), rgba(34,211,238,0.2));
    color: #F1F8FF !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 12px;
    box-shadow: 4px 4px 12px rgba(0,2,8,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(56,189,248,0.06);
    backdrop-filter: blur(10px);
    border-radius: 10px 10px 0 0;
    color: #7DA9CC;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    padding: 8px 14px;
    border: 1px solid rgba(56,189,248,0.10);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(145deg, #2563EB, #38BDF8) !important;
    color: #FFFFFF !important;
    box-shadow: 0 -4px 16px rgba(56,189,248,0.4);
}

/* ---- Alert / severity panel ---- */
.alert-panel {
    border-radius: 16px;
    padding: 16px 20px;
    margin: 16px 0;
    border-left: 5px solid var(--sev-color, #DC2626);
    background: rgba(56,189,248,0.04);
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 28px rgba(0,2,8,0.45);
}
.alert-panel div, .alert-panel li { color: #E6F4FF !important; }
.alert-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 20px;
    color: #FFFFFF;
    background: var(--sev-color, #DC2626);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.uncertainty-banner {
    border: 1px solid rgba(244,63,94,0.5);
    background: rgba(244,63,94,0.12);
    backdrop-filter: blur(10px);
    color: #FECDD3;
    border-radius: 12px;
    padding: 10px 16px;
    margin: 12px 0;
    font-size: 13px;
    font-weight: 600;
}

/* ---- Sidebar text + expander ---- */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] summary,
section[data-testid="stSidebar"] .stMarkdown {
    color: #CFE7FA !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(56,189,248,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 12px;
    margin-bottom: 10px;
    box-shadow: 4px 4px 14px rgba(0,0,0,0.4);
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-weight: 700 !important;
}

/* ---- Misc widgets ---- */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stAlert { border-radius: 12px !important; backdrop-filter: blur(10px); }

hr { border-color: rgba(56,189,248,0.22); }
[data-testid="stImage"] img { border-radius: 14px; box-shadow: 0 12px 30px rgba(0,2,8,0.55); }

/* ---- Footer ---- */
.footer-disclaimer {
    text-align: center;
    max-width: 760px;
    margin: 4px auto 0 auto;
    font-size: 11.5px;
    color: #64748B;
    line-height: 1.6;
}
.footer-credits {
    text-align: center;
    font-size: 11px;
    color: #4B6B85;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# OPTIONAL: use YOUR OWN local photo as the page background instead of the
# CSS-generated space scene above. Just drop an image file named
# "background.jpg" (or .png) into this same folder — if found, it overrides
# the built-in background automatically. If not found, the CSS space theme
# above is used as-is.
# =========================================================
def _load_local_background():
    for fname in ["background.jpg", "background.jpeg", "background.png"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = fname.split(".")[-1]
            return data, ext
    return None, None


_bg_data, _bg_ext = _load_local_background()
if _bg_data:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(180deg, rgba(2,4,10,0.55) 0%, rgba(2,4,10,0.35) 40%, rgba(2,4,10,0.65) 100%),
            url("data:image/{_bg_ext};base64,{_bg_data}") !important;
        background-size: cover !important;
        background-position: center top !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model():
    model = MultimodalFusionModel(
        tabular_in_dim=len(FEATURE_COLS),
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )
    state = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")


@st.cache_resource
def get_gradcam(_model):
    target_layer = _model.image_encoder.features[-2]
    return GradCAM(_model, target_layer)


transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_gps_from_image(image):
    try:
        exif_data = image._getexif()
    except Exception:
        exif_data = None
    if not exif_data:
        return None

    gps_info = {}
    for tag, value in exif_data.items():
        if TAGS.get(tag, tag) == "GPSInfo":
            for gps_tag in value:
                gps_info[GPSTAGS.get(gps_tag, gps_tag)] = value[gps_tag]

    if not gps_info or "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
        return None

    def _to_degrees(dms, ref):
        degrees = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
        if ref in ("S", "W"):
            degrees = -degrees
        return degrees

    try:
        lat = _to_degrees(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
        lon = _to_degrees(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))
        return (lat, lon)
    except (KeyError, TypeError, ZeroDivisionError, IndexError):
        return None


def predict(model, scaler, image, feature_values):
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    raw = np.array([[feature_values[c] for c in FEATURE_COLS]], dtype=np.float32)
    scaled = scaler.transform(raw)
    tab_tensor = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor, tab_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, img_tensor


def get_gradcam_overlay(gradcam, img_tensor, scaler, feature_values, image, class_idx):
    raw = np.array([[feature_values[c] for c in FEATURE_COLS]], dtype=np.float32)
    scaled = scaler.transform(raw)
    tab_tensor = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
    cam = gradcam.generate(img_tensor, tab_tensor, class_idx)
    return overlay_heatmap(image, cam, alpha=0.45)


def make_probability_chart(probs, class_names, hazard_meta, pred_idx):
    order = np.argsort(probs)
    classes_sorted = [class_names[i] for i in order]
    probs_sorted = probs[order] * 100
    colors = [hazard_meta[class_names[i]]["color"] for i in order]
    edge_colors = ["#F1F8FF" if i == pred_idx else "none" for i in order]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_facecolor("#050B18")
    ax.set_facecolor("#050B18")
    bars = ax.barh(classes_sorted, probs_sorted, color=colors, edgecolor=edge_colors, linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", color="#C4B5FD", fontsize=9)
    ax.tick_params(colors="#CFE7FA", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#123A5C")
    ax.grid(axis="x", color="#0E2E4A", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, probs_sorted):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", color="#F1F8FF", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


def risk_level_from_severity(severity, confidence=100.0):
    """Map prediction confidence to a traffic-light risk chip.

    NOTE: this is driven by CONFIDENCE, not by the hazard's fixed severity
    label. Severity (e.g. Flood = HIGH) is a property of the hazard TYPE and
    is shown separately in the "SEVERITY: ..." badge below. If risk were
    based on severity alone, hazards whose severity is always HIGH/CRITICAL
    in HAZARD_ACTIONS would show "High Risk" for every single prediction,
    even a shaky 40% one — and Moderate/Low would never appear. Using
    confidence instead means the risk chip actually reflects how sure THIS
    prediction is:
      - confidence >= 80%  -> 🔴 High Risk       (model is very sure)
      - confidence 60-80%  -> 🟡 Moderate Risk    (reasonably sure)
      - confidence 50-60%  -> 🟢 Low Risk         (weak signal)
      - confidence < 50%   -> ⚪ Uncertain Risk   (not reliable — verify manually)
    """
    if confidence < 50:
        return "⚪", "Uncertain Risk", "#64748B"
    if confidence < 60:
        return "🟢", "Low Risk", "#22C55E"
    if confidence < 80:
        return "🟡", "Moderate Risk", "#F59E0B"
    return "🔴", "High Risk", "#DC2626"


def analyze_image_plausibility(image):
    """Lightweight sanity check to flag inputs that don't look like typical
    satellite/aerial disaster imagery — e.g. ID cards, documents, screenshots,
    certificates, or studio-style photos with plain backgrounds.

    WHY THIS EXISTS: the classifier only knows 4 classes (Flood, Wildfire,
    Cyclone, Earthquake). It has no "not a disaster photo" option, so ANY
    image — including an ID card — gets force-fit into one of those 4
    classes, sometimes with high confidence, because softmax always outputs
    numbers that sum to 100% regardless of whether the input makes sense.
    This function does NOT fix that (only retraining with a 5th "unknown"
    class truly would) — it's a fast, approximate pre-check based on colour
    statistics that catches the most common non-disaster inputs (documents,
    ID cards, screenshots) so the user gets a visible caution instead of
    silently trusting a confident-looking but meaningless prediction.

    Returns (is_suspect: bool, reason: str).
    """
    try:
        small = image.convert("RGB").resize((160, 160))
        arr = np.asarray(small).astype(np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        brightness = (r + g + b) / 3.0

        # Fraction of near-white/near-blank pixels — common in scanned
        # documents, ID cards, and screenshots with plain backgrounds.
        white_frac = float(np.mean(brightness > 225))

        # Hasler & Süsstrunk "colorfulness" metric — natural aerial/satellite
        # scenes (water, vegetation, soil, smoke, cloud) tend to score
        # moderate-to-high; printed documents/ID cards (mostly white + black
        # text + a small logo/photo) score low.
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorfulness = float(
            np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
            + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        )

        if white_frac > 0.40 and colorfulness < 28:
            return True, "large plain/white background with low colour variety — looks more like a scanned document, ID card, or screenshot than aerial imagery"
        if colorfulness < 12:
            return True, "very low colour variety (mostly flat/grayscale) — unlike a natural satellite or aerial scene"
        return False, ""
    except Exception:
        return False, ""


model_ready, load_error = True, None
try:
    model = load_model()
    scaler = load_scaler()
    gradcam = get_gradcam(model)
except Exception as e:
    model_ready, load_error = False, str(e)

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sb-title">🧭 TerraScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Environmental Risk Intelligence Console</div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-badge">● MODEL ONLINE</span>' if model_ready else '<span class="sb-badge">● MODEL OFFLINE</span>', unsafe_allow_html=True)

    st.write("")

    tcol1, tcol2 = st.columns(2)
    light_mode = tcol1.toggle("☀️ Light mode", value=False)
    lang = "hi" if tcol2.toggle("🗣️ हिंदी", value=False) else "en"

    with st.expander("❓ How TerraScan Works", expanded=False):
        st.markdown('<div class="sb-item">1️⃣ Upload a satellite/aerial image or use your camera</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">2️⃣ Set or fetch live environmental telemetry for the location</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">3️⃣ Run Analysis → view hazard, confidence, Grad-CAM/SHAP explanations, map & AI report</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About TerraScan", expanded=False):
        st.markdown(
            '<div class="sb-item">A multimodal deep-learning system that fuses satellite imagery with '
            'environmental sensor data to classify Flood, Wildfire, Cyclone &amp; Earthquake risk — with '
            'built-in explainability (Grad-CAM + SHAP) and an AI Copilot for reports, Q&amp;A and advisories.</div>',
            unsafe_allow_html=True,
        )

    # ---- AI Copilot API key: auto-load from .streamlit/secrets.toml if present,
    # so you don't have to paste it every time. Falls back to manual entry. ----
    try:
        saved_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        saved_key = ""
    if saved_key:
        st.session_state.claude_api_key = saved_key
        st.session_state.claude_model = st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        st.markdown('<span class="sb-badge">✅ AI COPILOT AUTO-CONNECTED</span>', unsafe_allow_html=True)
        st.write("")
        with st.expander("🤖 AI Copilot Settings (Gemini API)", expanded=False):
            llm_model = st.selectbox("Model", ["gemini-3.5-flash-lite", "gemini-3.6-flash"], index=0)
            st.session_state.claude_model = llm_model
    else:
        with st.expander("🤖 AI Copilot Settings (Gemini API — free)", expanded=False):
            st.caption("Tip: save your key in `.streamlit/secrets.toml` as `GEMINI_API_KEY = \"...\"` "
                       "to skip pasting it here every time.")
            api_key_input = st.text_input("Google Gemini API key", type="password",
                                           help="Get a free key at aistudio.google.com/apikey")
            llm_model = st.selectbox("Model", ["gemini-3.5-flash-lite", "gemini-3.6-flash"], index=0)
            if api_key_input:
                st.session_state.claude_api_key = api_key_input
                st.session_state.claude_model = llm_model
                st.success("✅ Key set — AI Copilot tab is active.")
            else:
                st.caption("Without a key, the AI Copilot tab will be disabled.")

    with st.expander("🏗️ Architecture", expanded=False):
        st.markdown('<div class="sb-item">🖼️ Image branch — ResNet CNN</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">📊 Tabular branch — Residual MLP</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">🔗 Late fusion → Dense classifier</div>', unsafe_allow_html=True)

    with st.expander("🔬 Explainability Suite", expanded=False):
        st.markdown('<div class="sb-item">🛰️ Grad-CAM — image attention</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">🧬 SHAP — bar / waterfall / beeswarm / decision</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">📄 PDF report + per-plot downloads</div>', unsafe_allow_html=True)

    with st.expander("📈 Training Snapshot", expanded=False):
        st.markdown('<div class="sb-item">Dataset: 2,000 real disaster images</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">Test Accuracy: <b style="color:#38BDF8">96.25%</b></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-item">Classes: Flood · Wildfire · Cyclone · Earthquake</div>', unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center; padding: 8px 0 2px 0;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#7DA9CC; letter-spacing:1.5px; text-transform:uppercase;">Made By</div>
            <div style="font-family:'Orbitron', sans-serif; font-weight:700; color:#F1F8FF; font-size:13.5px; margin-top:6px;">Saloni</div>
            <div style="font-family:'Orbitron', sans-serif; font-weight:700; color:#F1F8FF; font-size:13.5px; margin-top:6px;">Anjali Hazra</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        text-align: center;
        margin-top: 0px;
        margin-bottom: 0px;
        font-size: 14px;
        color: #B9E6FF;
        font-family: sans-serif;
    ">
        Built with PyTorch + Streamlit + SHAP
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div style="text-align:center;"><span class="sb-badge">TerraScan {APP_VERSION}</span></div>', unsafe_allow_html=True)

if light_mode:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(160deg, #F0F9FF 0%, #F5FAFF 50%, #E0F2FE 100%) !important; }
    .panel { background: rgba(255,255,255,0.65) !important; border-color: rgba(56,189,248,0.25) !important;
             box-shadow: 0 10px 30px rgba(56,189,248,0.15) !important; }
    .panel-title { color: #0C4A6E !important; }
    .panel-eyebrow { color: #0284C7 !important; }
    .panel label, .panel [data-testid="stWidgetLabel"] p { color: #0C4A6E !important; }
    .panel p, .panel [data-testid="stMarkdownContainer"] p, .panel .stCaption { color: #0369A1 !important; }
    .stat-box { background: rgba(255,255,255,0.65) !important; border-color: rgba(56,189,248,0.25) !important; }
    .stat-label { color: #0369A1 !important; }
    .ticker { background: rgba(255,255,255,0.6) !important; color: #0C4A6E !important; border-color: rgba(56,189,248,0.3) !important; }
    .hero-sub { color: #075985 !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] div { color: #0C4A6E; }
    [data-testid="stCaptionContainer"] { color: #0369A1 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="ticker">
    📡 <b>SYSTEM NOTICE</b> — Tabular telemetry values default to <b>domain-informed demo values</b>.
    Attach a location below and click "Fetch Live Weather" to pull real data (NASA POWER API).
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">{ex.t('hero_eyebrow', lang)}</div>
    <div class="hero-title">{ex.t('hero_title', lang)}</div>
    <div class="hero-sub">{ex.t('hero_sub', lang)}</div>
    <div class="hero-chips">
        <span class="sb-badge">🧭 {APP_VERSION}</span>
        <span class="sb-badge">🛰️ 4 Hazard Classes</span>
        <span class="sb-badge">🧠 Trained on 2,000 real disaster images</span>
    </div>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
for col, (num, label) in zip(
    [s1, s2, s3, s4],
    [("96.25%", "TEST ACCURACY"), ("4", "HAZARD CLASSES"), ("2,000", "TRAINING IMAGES"), ("8", "ENV. FEATURES")]
):
    col.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.write("")

if not model_ready:
    st.error(
        f"⚠️ Model artefacts not found (`best_model.pt` / `scaler.pkl`). "
        f"Run `python train.py` first, then relaunch the app.\n\nDetails: {load_error}"
    )

# =========================================================
# MAIN LAYOUT
# =========================================================
col_left, col_right = st.columns([1, 1.35], gap="large")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-eyebrow">INPUT · 01</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{ex.t("input1_title", lang)}</div>', unsafe_allow_html=True)

    source_mode = st.radio("Source", ["📁 Upload", "📷 Camera"], horizontal=True, label_visibility="collapsed")
    image = None
    detected_gps = None
    if source_mode == "📁 Upload":
        uploaded_file = st.file_uploader("Drop an image (.jpg / .png)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            detected_gps = extract_gps_from_image(image)
    else:
        camera_file = st.camera_input("Capture", label_visibility="collapsed")
        if camera_file:
            image = Image.open(camera_file)

    if image is not None:
        st.image(image, use_container_width=True, caption="Feed preview")
        image_suspect, image_suspect_reason = analyze_image_plausibility(image)
        if image_suspect:
            st.warning(
                f"⚠️ This doesn't look like typical satellite/aerial imagery — {image_suspect_reason}. "
                f"The model has only 4 hazard classes and will still force a prediction, so treat the "
                f"result below with extra caution and verify manually."
            )
    else:
        image_suspect, image_suspect_reason = False, ""
        st.markdown(f'<div style="text-align:center; color:#7A6F5F; padding: 30px 0; font-family:\'IBM Plex Mono\',monospace; font-size:12.5px;">{ex.t("upload_prompt", lang)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-eyebrow">INPUT · 02</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{ex.t("input2_title", lang)}</div>', unsafe_allow_html=True)

    af1, af2 = st.columns([2, 1])
    demo_hazard = af1.selectbox(
        "🎲 Auto-fill realistic demo values for", ["— none —"] + list(CLASS_NAMES),
        label_visibility="visible",
    )
    if af2.button("Apply", use_container_width=True) and demo_hazard != "— none —":
        rng = np.random.default_rng()
        st.session_state.fetched_weather = generate_features_for_class(demo_hazard, rng)
        st.session_state._values_source = "demo"
        st.success(f"✅ Sliders auto-filled with realistic '{demo_hazard}' demo values.")
        st.rerun()

    _src = st.session_state.get("_values_source")
    if _src == "demo":
        st.caption("🎲 Current values are physically realistic **demo** values for the selected hazard type — not derived from the uploaded image or live data.")
    elif _src == "live":
        st.caption("🌐 Current values were fetched from **live weather data** (NASA POWER / Open-Elevation) for the attached location.")
        _w = st.session_state.get("fetched_weather", {})
        _city_label = st.session_state.get("_weather_city_label", "Selected location")
        st.markdown(f"""
        <div style="display:flex; flex-wrap:wrap; gap:10px; margin:10px 0;">
            <div style="flex:1; min-width:110px; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:10px 8px; text-align:center;">
                <div style="font-size:11px; color:#7DA9CC;">🌡️ TEMP</div>
                <div style="font-size:17px; font-weight:700; color:#F1F8FF; white-space:nowrap;">{_w.get('temperature', 0):.1f}°C</div>
            </div>
            <div style="flex:1; min-width:110px; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:10px 8px; text-align:center;">
                <div style="font-size:11px; color:#7DA9CC;">💧 HUMIDITY</div>
                <div style="font-size:17px; font-weight:700; color:#F1F8FF; white-space:nowrap;">{_w.get('humidity', 0):.0f}%</div>
            </div>
            <div style="flex:1; min-width:110px; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:10px 8px; text-align:center;">
                <div style="font-size:11px; color:#7DA9CC;">🌧️ RAINFALL</div>
                <div style="font-size:17px; font-weight:700; color:#F1F8FF; white-space:nowrap;">{_w.get('rainfall', 0):.0f} mm</div>
            </div>
            <div style="flex:1; min-width:110px; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:10px 8px; text-align:center;">
                <div style="font-size:11px; color:#7DA9CC;">💨 WIND</div>
                <div style="font-size:17px; font-weight:700; color:#F1F8FF; white-space:nowrap;">{_w.get('wind_speed', 0):.1f} km/h</div>
            </div>
            <div style="flex:1; min-width:110px; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:10px 8px; text-align:center;">
                <div style="font-size:11px; color:#7DA9CC;">⛰️ ELEVATION</div>
                <div style="font-size:17px; font-weight:700; color:#F1F8FF; white-space:nowrap;">{_w.get('elevation', 0):.0f} m</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"📍 {_city_label}")
    else:
        st.caption("Set values manually below, use live weather, or auto-fill demo values for a hazard type.")

    fetched = st.session_state.get("fetched_weather", {})
    feature_values = {}
    c1, c2 = st.columns(2)
    for i, (key, meta) in enumerate(FEATURE_META.items()):
        target_col = c1 if i % 2 == 0 else c2
        unit = f" {meta['unit']}" if meta["unit"] else ""
        default_val = fetched.get(key, meta["default"])
        default_val = float(min(max(default_val, meta["min"]), meta["max"]))
        feature_values[key] = target_col.slider(
            f"{meta['label']}{unit}",
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=default_val,
            # NOTE: no 'step' argument here on purpose — Streamlit infers a
            # float step automatically. If you add one back, it MUST be a
            # float (e.g. step=0.1), never a plain int, or it will crash
            # with "Slider value arguments must be of matching types."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-eyebrow">INPUT · 03</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{ex.t("input3_title", lang)}</div>', unsafe_allow_html=True)

    location = None
    location_label = None
    if detected_gps is not None:
        st.success(f"📍 GPS auto-detected from image EXIF: Lat {detected_gps[0]:.4f}, Lon {detected_gps[1]:.4f}")
        location = detected_gps
        location_label = f"GPS location — Lat {detected_gps[0]:.4f}, Lon {detected_gps[1]:.4f}"
    elif image is not None:
        st.caption("ℹ️ No GPS metadata found in this image. You can attach a location manually below.")
        use_location = st.checkbox("Attach coordinates manually")
        if use_location:
            PRESET_CITIES = {
                "New Delhi":  (28.6139, 77.2090),
                "Mumbai":     (19.0760, 72.8777),
                "Chennai":    (13.0827, 80.2707),
                "Kolkata":    (22.5726, 88.3639),
                "Bengaluru":  (12.9716, 77.5946),
                "Guwahati":   (26.1445, 91.7362),
                "Chandigarh": (30.7333, 76.7794),
                "Uttar Pradesh (Lucknow)": (26.8467, 80.9462),
                "Custom":     None,
            }
            city_choice = st.selectbox("Quick-pick a city (or choose Custom)", list(PRESET_CITIES.keys()))
            if city_choice == "Custom":
                lc1, lc2 = st.columns(2)
                lat = lc1.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.5937, format="%.4f")
                lon = lc2.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.9629, format="%.4f")
                location = (lat, lon)
                location_label = f"Custom location — Lat {lat:.4f}, Lon {lon:.4f}"
            else:
                location = PRESET_CITIES[city_choice]
                location_label = f"{city_choice} — Lat {location[0]:.4f}, Lon {location[1]:.4f}"
                st.caption(f"📍 {city_choice} — Lat {location[0]:.4f}, Lon {location[1]:.4f}")
    else:
        st.caption("Upload/capture an image to auto-detect its GPS location (if available).")

    if location is not None:
        if st.button("🌐 Fetch Live Weather for this location", use_container_width=True):
            try:
                with st.spinner("Contacting NASA POWER + Open-Elevation APIs... (up to 15s)"):
                    weather = ex.fetch_live_weather(location[0], location[1])
                    elevation = ex.fetch_elevation(location[0], location[1])
                if weather:
                    if elevation is not None:
                        weather["elevation"] = elevation
                    st.session_state.fetched_weather = weather
                    st.session_state._values_source = "live"
                    st.session_state._weather_city_label = location_label
                    st.session_state._weather_just_fetched = True
                    st.rerun()
                else:
                    st.warning("⚠️ Could not reach the weather API (no data returned). Using demo values instead.")
            except Exception as e:
                st.error(f"❌ Weather fetch failed: {e}")

        if st.session_state.pop("_weather_just_fetched", False):
            st.success("✅ Live weather fetched — sliders above updated with real data.")
    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button(ex.t("run_button", lang), use_container_width=True,
                                 disabled=(not model_ready or image is None))

with col_right:
    st.markdown('<div class="panel-eyebrow">OUTPUT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title" style="margin-bottom:12px;">{ex.t("output_title", lang)}</div>', unsafe_allow_html=True)

    if predict_clicked and image is not None:
        with st.spinner("Analyzing fused signal..."):
            probs, img_tensor = predict(model, scaler, image, feature_values)
        # CRITICAL FIX: persist across reruns. Without this, clicking ANY other
        # button (Fetch Weather, AI Copilot buttons, feedback, etc.) resets
        # predict_clicked to False on the next script run, and this whole
        # result section would disappear -> the "blank page" bug.
        st.session_state.last_result = {
            "probs": probs,
            "img_tensor": img_tensor,
            "image": image,
            "feature_values": feature_values,
            "location": location,
            "image_suspect": image_suspect,
            "image_suspect_reason": image_suspect_reason,
        }

    _stored = st.session_state.get("last_result")

    if _stored is None and image is None and model_ready:
        st.markdown(
            f'<div class="panel" style="text-align:center; color:#7A6F5F; padding: 50px 0;">{ex.t("awaiting", lang)}</div>',
            unsafe_allow_html=True,
        )

    if _stored is not None:
        probs = _stored["probs"]
        img_tensor = _stored["img_tensor"]
        image = _stored["image"]
        feature_values = _stored["feature_values"]
        location = _stored["location"]
        _img_suspect = _stored.get("image_suspect", False)
        _img_suspect_reason = _stored.get("image_suspect_reason", "")

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100
        meta = HAZARD_META[pred_class]

        # Compute severity / recommended actions BEFORE the result card so
        # we can show the traffic-light risk chip inline with the hazard result.
        hazard_info = rpt.HAZARD_ACTIONS.get(pred_class, {})
        sev = hazard_info.get("severity", "—")
        sev_color = hazard_info.get("severity_color", "#DC2626")
        actions = hazard_info.get("actions", [])
        risk_icon, risk_label, risk_color = risk_level_from_severity(sev, confidence)

        st.markdown(f"""
        <div class="result-card" style="--glow:{meta['glow']}">
            <div class="result-eyebrow">{ex.t('predicted_hazard', lang)}</div>
            <div class="result-hazard">{meta['icon']} {pred_class}</div>
            <div class="result-conf">{ex.t('confidence', lang)}&nbsp;&nbsp;<b style="color:{meta['color']}">{confidence:.1f}%</b></div>
            <div class="risk-chip" style="background:{risk_color};">{risk_icon} {risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

        _why_text = (
            "🔍 Curious why? Open the Grad-CAM and SHAP tabs below to see exactly which pixels and "
            "environmental features drove this decision."
            if lang == "en" else
            "🔍 जानना चाहते हैं क्यों? नीचे Grad-CAM और SHAP टैब खोलें और देखें कि किन पिक्सेल्स और "
            "फीचर्स ने इस निर्णय को प्रभावित किया।"
        )
        st.caption(_why_text)

        if _img_suspect:
            st.markdown(
                f'<div class="uncertainty-banner">🧐 UNUSUAL INPUT — the uploaded image {_img_suspect_reason}. '
                f'This model only recognizes 4 hazard classes and has no "not a disaster" option, so it always '
                f'forces a prediction. Treat this result as <b>not reliable</b> unless the image genuinely shows '
                f'a disaster scene.</div>',
                unsafe_allow_html=True,
            )

        if confidence < 70:
            st.markdown(f'<div class="uncertainty-banner">⚠️ LOW CONFIDENCE ({confidence:.1f}%) — this prediction is uncertain, please verify manually before acting on it.</div>', unsafe_allow_html=True)

        tier_label, tier_color = rpt.confidence_tier(confidence)
        actions_html = "".join(f"<li>{a}</li>" for a in actions)
        st.markdown(f"""
        <div class="alert-panel" style="--sev-color:{sev_color}">
            <span class="alert-badge">SEVERITY: {sev}</span>
            &nbsp;&nbsp;<span style="font-size:12px; color:#64748B; font-weight:600;">CERTAINTY: <span style="color:{tier_color}">{tier_label}</span></span>
            <div style="margin-top:10px; font-weight:700; color:#0F172A;">{ex.t('recommended_actions', lang)}</div>
            <ul style="margin:6px 0 0 18px; padding:0; color:#334155; font-size:13.5px; line-height:1.7;">
                {actions_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if sev == "CRITICAL" and st.session_state.get("_alert_played_for") != pred_idx:
            ex.play_critical_alert()
            st.session_state._alert_played_for = pred_idx

        fb1, fb2, fb3 = st.columns([2, 1, 1])
        fb1.caption(ex.t("feedback_prompt", lang))
        if "feedback" not in st.session_state:
            st.session_state.feedback = []
        if fb2.button("👍", key="fb_up", use_container_width=True):
            st.session_state.feedback.append({"time": datetime.now().strftime("%H:%M:%S"), "hazard": pred_class, "correct": True})
            st.toast(ex.t("feedback_thanks", lang))
        if fb3.button("👎", key="fb_down", use_container_width=True):
            st.session_state.feedback.append({"time": datetime.now().strftime("%H:%M:%S"), "hazard": pred_class, "correct": False})
            st.toast(ex.t("feedback_thanks", lang))

        if st.session_state.get("_logged_for") != id(_stored):
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Hazard": f"{meta['icon']} {pred_class}",
                "Confidence": f"{confidence:.1f}%",
                "Certainty": tier_label,
                "Risk": f"{risk_icon} {risk_label}",
                "Location": f"{location[0]:.3f}, {location[1]:.3f}" if location else "—",
            })
            st.session_state._logged_for = id(_stored)

        st.write("")
        tab_gc, tab_shap, tab_conf, tab_map, tab_ai = st.tabs([
            ex.t("tab_gradcam", lang), ex.t("tab_shap", lang), ex.t("tab_confidence", lang),
            ex.t("tab_map", lang), "🤖 AI COPILOT"
        ])

        # ---- Tab 1: Grad-CAM ----
        with tab_gc:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">WHERE THE MODEL IS LOOKING</div>', unsafe_allow_html=True)
            with st.spinner("Computing Grad-CAM attention map..."):
                overlay_img = get_gradcam_overlay(gradcam, img_tensor, scaler, feature_values, image, pred_idx)
            g1, g2 = st.columns(2)
            g1.image(image, caption="Original", use_container_width=True)
            g2.image(overlay_img, caption=f"Grad-CAM · '{pred_class}'", use_container_width=True)
            st.caption("🔴🟡 Red/yellow regions influenced the prediction most · 🔵 Blue regions had little effect.")
            st.download_button("⬇ Download Grad-CAM image", data=rpt.pil_to_png_bytes(overlay_img),
                                file_name=f"gradcam_{pred_class}.png", mime="image/png")
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Tab 2: SHAP Explainability suite ----
        with tab_shap:
            with st.spinner("Running SHAP KernelExplainer on the tabular branch..."):
                user_raw = np.array([feature_values[c] for c in FEATURE_COLS], dtype=np.float32)
                shap_result = shx.compute_shap(model, img_tensor, scaler, pred_idx, user_raw, DEVICE,
                                                n_per_class=3, nsamples=60)

            bar_fig = shx.plot_bar(shap_result, pred_class)
            waterfall_fig = shx.plot_waterfall(shap_result, pred_class)
            beeswarm_fig = shx.plot_beeswarm(shap_result, pred_class)
            decision_fig = shx.plot_decision(shap_result, pred_class)

            p1, p2 = st.columns(2)
            with p1:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-eyebrow">BAR · GLOBAL FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
                st.pyplot(bar_fig, use_container_width=True)
                st.download_button("⬇ Download", data=rpt.fig_to_png_bytes(bar_fig), file_name="shap_bar.png",
                                    mime="image/png", key="dl_bar")
                st.markdown('</div>', unsafe_allow_html=True)
            with p2:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-eyebrow">WATERFALL · THIS PREDICTION</div>', unsafe_allow_html=True)
                st.pyplot(waterfall_fig, use_container_width=True)
                st.download_button("⬇ Download", data=rpt.fig_to_png_bytes(waterfall_fig), file_name="shap_waterfall.png",
                                    mime="image/png", key="dl_waterfall")
                st.markdown('</div>', unsafe_allow_html=True)

            p3, p4 = st.columns(2)
            with p3:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-eyebrow">BEESWARM · IMPACT DISTRIBUTION</div>', unsafe_allow_html=True)
                st.pyplot(beeswarm_fig, use_container_width=True)
                st.download_button("⬇ Download", data=rpt.fig_to_png_bytes(beeswarm_fig), file_name="shap_beeswarm.png",
                                    mime="image/png", key="dl_beeswarm")
                st.markdown('</div>', unsafe_allow_html=True)
            with p4:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-eyebrow">DECISION · CUMULATIVE PATH</div>', unsafe_allow_html=True)
                st.pyplot(decision_fig, use_container_width=True)
                st.download_button("⬇ Download", data=rpt.fig_to_png_bytes(decision_fig), file_name="shap_decision.png",
                                    mime="image/png", key="dl_decision")
                st.markdown('</div>', unsafe_allow_html=True)

            st.caption("SHAP quantifies each environmental feature's contribution to the tabular branch's output, with the uploaded image held fixed.")

        # ---- Tab 3: Confidence chart + quick view ----
        with tab_conf:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">CLASS PROBABILITY BREAKDOWN</div>', unsafe_allow_html=True)
            prob_fig = make_probability_chart(probs, CLASS_NAMES, HAZARD_META, pred_idx)
            st.pyplot(prob_fig, use_container_width=True)
            st.download_button("⬇ Download chart", data=rpt.fig_to_png_bytes(prob_fig), file_name="confidence_chart.png",
                                mime="image/png", key="dl_conf")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">QUICK VIEW</div>', unsafe_allow_html=True)
            st.write("")
            for idx in np.argsort(-probs):
                cls = CLASS_NAMES[idx]
                pct = probs[idx] * 100
                color = HAZARD_META[cls]["color"]
                icon = HAZARD_META[cls]["icon"]
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">{icon} {cls}</div>
                    <div class="prob-track"><div class="prob-fill" style="width:{pct}%; background:{color};"></div></div>
                    <div class="prob-val">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Tab 4: Location map ----
        with tab_map:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">INCIDENT LOCATION</div>', unsafe_allow_html=True)
            if location:
                rgb = tuple(int(meta["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                df_loc = pd.DataFrame({"lat": [location[0]], "lon": [location[1]]})
                layer = pdk.Layer(
                    "ScatterplotLayer", data=df_loc, get_position="[lon, lat]",
                    get_color=list(rgb) + [220], get_radius=15000, pickable=True,
                )
                view_state = pdk.ViewState(latitude=location[0], longitude=location[1], zoom=5)
                # FIX: map_style="road" needs a Mapbox token (none configured) -> blank map.
                # Carto is free/token-free and works out of the box in st.pydeck_chart.
                st.pydeck_chart(pdk.Deck(
                    layers=[layer], initial_view_state=view_state,
                    map_provider="carto", map_style="light",
                    tooltip={"text": f"{pred_class} · {confidence:.1f}%"},
                ))
                st.caption(f"📍 Lat {location[0]:.4f}, Lon {location[1]:.4f} — marker colored by predicted hazard.")
            else:
                st.info("No location for this analysis. Attach coordinates manually in the input panel and re-run.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Tab 5: AI Copilot (LLM + RAG + Agentic + Generative) ----
        with tab_ai:
            api_key = st.session_state.get("claude_api_key")
            model_name = st.session_state.get("claude_model", "gemini-3.5-flash-lite")

            if not api_key:
                st.warning("🔑 Add your free Gemini API key in the sidebar (**AI Copilot Settings**) to unlock this tab.")
            else:
                client = llm.get_client(api_key)

                try:
                    _mean_abs = np.abs(shap_result["shap_values"]).mean(axis=0)
                    _order = np.argsort(-_mean_abs)[:3]
                    shap_top_features = [shap_result["feature_names"][i] for i in _order]
                except Exception:
                    shap_top_features = list(feature_values.keys())[:3]

                sub_report, sub_rag, sub_agent, sub_gen = st.tabs(
                    ["📝 Situation Report", "💬 Ask the Assistant (RAG)", "🤖 Autonomous Agent", "📢 Generate Advisory"]
                )

                with sub_report:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-eyebrow">LLM · NATURAL-LANGUAGE SITREP</div>', unsafe_allow_html=True)
                    if st.button("📝 Generate Situation Report", key="btn_sitrep", use_container_width=True):
                        try:
                            with st.spinner("Gemini is drafting the situation report... (up to 30s)"):
                                sitrep = llm.generate_situation_report(
                                    client, model_name, pred_class, confidence, location,
                                    feature_values, shap_top_features, lang=lang,
                                )
                            st.session_state.sitrep_text = sitrep
                        except Exception as e:
                            st.error(f"❌ Request failed: {e}")
                    if st.session_state.get("sitrep_text"):
                        st.markdown(f'<div style="background:rgba(255,255,255,0.08); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:14px 18px; font-size:14px; line-height:1.7; color:#F1F8FF; backdrop-filter:blur(10px);">{st.session_state.sitrep_text}</div>', unsafe_allow_html=True)
                        st.download_button("⬇ Download report (.txt)", data=st.session_state.sitrep_text,
                                            file_name="situation_report.txt", mime="text/plain")
                    st.markdown('</div>', unsafe_allow_html=True)

                with sub_rag:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-eyebrow">RAG · GROUNDED Q&A OVER DISASTER GUIDANCE</div>', unsafe_allow_html=True)
                    st.caption("Retrieves relevant passages from a curated safety-guidance knowledge base (TF-IDF search), then asks Gemini to answer using only that context.")
                    user_q = st.text_input("Ask a question", placeholder=f"e.g. What should I do during a {pred_class} if I have young children at home?", key="rag_q")
                    if st.button("💬 Ask", key="btn_rag", use_container_width=True) and user_q:
                        try:
                            with st.spinner("Retrieving guidance + asking Gemini... (up to 30s)"):
                                answer, sources = llm.rag_answer(client, model_name, user_q, pred_class, lang=lang)
                            st.session_state.rag_answer = (answer, sources)
                        except Exception as e:
                            st.error(f"❌ Request failed: {e}")
                    if st.session_state.get("rag_answer"):
                        ans, srcs = st.session_state.rag_answer
                        st.markdown(f'<div style="background:rgba(255,255,255,0.08); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:14px 18px; font-size:14px; line-height:1.7; color:#F1F8FF; backdrop-filter:blur(10px);">{ans}</div>', unsafe_allow_html=True)
                        st.caption("📚 Sources: " + ", ".join(srcs))
                    st.markdown('</div>', unsafe_allow_html=True)

                with sub_agent:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-eyebrow">AGENTIC AI · AUTONOMOUS RESPONSE AGENT</div>', unsafe_allow_html=True)
                    st.caption("Gemini autonomously decides which tools to call before drafting a final alert.")
                    if st.button("🤖 Run Autonomous Response Agent", key="btn_agent", use_container_width=True):
                        try:
                            with st.spinner("Agent investigating (may call multiple tools, up to 60s)..."):
                                trace, final_alert = llm.run_response_agent(
                                    client, model_name, pred_class, confidence, location, lang=lang,
                                )
                            st.session_state.agent_trace = trace
                            st.session_state.agent_alert = final_alert
                        except Exception as e:
                            st.error(f"❌ Request failed: {e}")
                    if st.session_state.get("agent_trace") is not None:
                        st.markdown("**Agent reasoning trace:**")
                        for i, step in enumerate(st.session_state.agent_trace, 1):
                            with st.expander(f"Step {i}: called `{step['tool']}`"):
                                st.json(step["input"])
                                st.text(step["output"])
                        st.write("")
                        st.markdown("**Final drafted alert:**")
                        st.markdown(f'<div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:10px; padding:14px 18px; font-size:14px; line-height:1.7; color:#0F172A; font-weight:600;">{st.session_state.get("agent_alert","")}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with sub_gen:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-eyebrow">GENERATIVE AI · PUBLIC ADVISORY</div>', unsafe_allow_html=True)
                    if st.button("📢 Generate Public Advisory", key="btn_advisory", use_container_width=True):
                        try:
                            with st.spinner("Generating advisory text... (up to 30s)"):
                                advisory = llm.generate_public_advisory(client, model_name, pred_class, confidence, location, lang=lang)
                            st.session_state.advisory_text = advisory
                        except Exception as e:
                            st.error(f"❌ Request failed: {e}")
                    if st.session_state.get("advisory_text"):
                        st.markdown(f'<div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:16px 20px; font-size:15px; line-height:1.7; color:#0F172A; font-weight:600;">{st.session_state.advisory_text}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        # ---- PDF report ----
        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-eyebrow">EXPORT</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title" style="margin-bottom:10px;">Full Report</div>', unsafe_allow_html=True)
        with st.spinner("Assembling PDF report..."):
            pdf_bytes = rpt.generate_pdf_report(
                pred_class, confidence, probs, CLASS_NAMES, feature_values,
                image, overlay_img, prob_fig,
                {"bar": bar_fig, "waterfall": waterfall_fig, "beeswarm": beeswarm_fig, "decision": decision_fig},
                location=location,
            )
        st.download_button("📄 Download Full PDF Report", data=pdf_bytes,
                            file_name=f"terrascan_report_{pred_class.lower()}.pdf",
                            mime="application/pdf", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# =========================================================
# BATCH ANALYSIS — multiple images at once
# =========================================================
with st.expander("🗂️ BATCH ANALYSIS — analyze multiple images at once", expanded=False):
    st.caption("Uses the current Environmental Telemetry values (left panel) for every image in the batch.")
    batch_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True, key="batch_uploader")
    if batch_files and st.button("▶ Run Batch Analysis", use_container_width=True, disabled=not model_ready):
        batch_results = []
        cols = st.columns(4)
        with st.spinner(f"Analyzing {len(batch_files)} images..."):
            for i, bf in enumerate(batch_files):
                b_image = Image.open(bf)
                b_probs, _ = predict(model, scaler, b_image, feature_values)
                b_idx = int(np.argmax(b_probs))
                b_class = CLASS_NAMES[b_idx]
                b_conf = b_probs[b_idx] * 100
                b_meta = HAZARD_META[b_class]
                batch_results.append({"File": bf.name, "Hazard": f"{b_meta['icon']} {b_class}", "Confidence": f"{b_conf:.1f}%"})
                with cols[i % 4]:
                    st.image(b_image, use_container_width=True)
                    st.markdown(f"**{b_meta['icon']} {b_class}**  \n{b_conf:.1f}%")
        st.write("")
        st.dataframe(pd.DataFrame(batch_results), use_container_width=True, hide_index=True)

# =========================================================
# SESSION HISTORY
# =========================================================
with st.expander("📜 YOUR PAST ASSESSMENTS — analyses run in this session", expanded=False):
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        hc1, hc2 = st.columns(2)
        hc1.download_button("⬇ Export as CSV", data=hist_df.to_csv(index=False).encode("utf-8"),
                             file_name="terrascan_session_history.csv", mime="text/csv", use_container_width=True)
        if hc2.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No analyses yet — run one above and it will show up here.")

    if st.session_state.get("feedback"):
        st.write("")
        st.caption("Feedback log:")
        st.dataframe(pd.DataFrame(st.session_state.feedback), use_container_width=True, hide_index=True)

# =========================================================
# MODEL PERFORMANCE DASHBOARD
# =========================================================
with st.expander("📉 MODEL PERFORMANCE DASHBOARD — classification report & confusion matrix", expanded=False):
    st.caption("Results from the held-out test set (400 images, 100 per class) after training.")
    st.dataframe(ex.PERFORMANCE_REPORT, use_container_width=True, hide_index=True)
    st.pyplot(ex.plot_confusion_matrix(), use_container_width=True)
    st.caption("Overall test accuracy: **96.25%**")

with st.expander("🧪 MODEL COMPARISON — ResNet-18 vs ResNet-50", expanded=False):
    st.caption("Empirical comparison from `train_compare.py`, trained on the same 2,000-image dataset.")
    comparison_df = pd.DataFrame([
        {"Backbone": "ResNet-18", "Parameters (M)": 11.38, "Train Time (min)": 5.2, "Inference (ms)": 14.89, "Test Accuracy (%)": 99.25},
        {"Backbone": "ResNet-50", "Parameters (M)": 23.91, "Train Time (min)": 5.09, "Inference (ms)": 17.99, "Test Accuracy (%)": 98.75},
    ])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption("✅ ResNet-18 was chosen as the primary backbone — fewer parameters, faster inference, and slightly higher accuracy on this dataset size, with lower overfitting risk.")

st.markdown("---")
st.markdown(f"""
<div style="text-align:center; padding: 10px 0 2px 0;">
    <div style="font-family:'Orbitron', sans-serif; font-weight:700; background: linear-gradient(90deg, #38BDF8, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size:14px;">
        👩‍💻 Made by Saloni &amp; Anjali Hazra
    </div>
    <div style="margin-top:6px;"><span class="sb-badge">TerraScan {APP_VERSION}</span></div>
</div>
<div class="footer-disclaimer">
    ⚠️ TerraScan assists in hazard <b>classification</b>, not real-time forecasting.
    Always follow official disaster-management advisories in an actual emergency.
</div>
<div class="footer-credits">
    Powered by NASA POWER API · Open-Elevation · Google Gemini
</div>
""", unsafe_allow_html=True)
st.caption("TerraScan · Multimodal Environmental Risk Assessment · CNN + Residual-MLP Fusion · Grad-CAM + SHAP Explainability · Built for academic demonstration")