"""
Extra utilities: live weather (NASA POWER), elevation lookup, critical-alert
sound, EN/HI translations, and the static model-performance dashboard.
"""
import io
import wave
import struct
import math
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =========================================================
# 1. Live weather — NASA POWER API (free, no key required)
# =========================================================
def fetch_live_weather(lat, lon, timeout=8):
    """
    Fetches long-term CLIMATOLOGY averages (annual) for a point from NASA POWER.
    Returns dict with rainfall(mm), temperature(C), humidity(%), wind_speed(km/h)
    or None if the request fails (e.g. no internet).
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,RH2M,PRECTOTCORR,WS2M&community=RE"
        f"&longitude={lon}&latitude={lat}&format=JSON"
    )
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()["properties"]["parameter"]
        return {
            "temperature": float(data["T2M"]["ANN"]),
            "humidity": float(data["RH2M"]["ANN"]),
            "rainfall": float(data["PRECTOTCORR"]["ANN"]) * 30.4,   # mm/day -> approx mm/month
            "wind_speed": float(data["WS2M"]["ANN"]) * 3.6,          # m/s -> km/h
        }
    except Exception:
        return None


def fetch_elevation(lat, lon, timeout=8):
    """Open-Elevation API (free, no key). Returns meters or None."""
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return float(resp.json()["results"][0]["elevation"])
    except Exception:
        return None


# =========================================================
# 2. Critical-alert beep (generated on the fly, no audio file needed)
# =========================================================
def _generate_beep_b64(freq=880, duration=0.30, volume=0.5, sample_rate=44100, repeats=2, gap=0.12):
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for r in range(repeats):
            n_samples = int(sample_rate * duration)
            for i in range(n_samples):
                t = i / sample_rate
                value = int(volume * 32767 * math.sin(2 * math.pi * freq * t))
                wf.writeframesraw(struct.pack("<h", value))
            if r < repeats - 1:
                silence = int(sample_rate * gap)
                for _ in range(silence):
                    wf.writeframesraw(struct.pack("<h", 0))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def play_critical_alert():
    b64 = _generate_beep_b64()
    st.markdown(
        f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


# =========================================================
# 3. EN / HI translations (core UI strings)
# =========================================================
TRANSLATIONS = {
    "hero_eyebrow": {"en": "MULTIMODAL FUSION · SATELLITE + TABULAR · EXPLAINABLE AI",
                      "hi": "मल्टीमॉडल फ्यूज़न · सैटेलाइट + तालिका डेटा · व्याख्या-योग्य AI"},
    "hero_title": {"en": "Multi-Hazard Environmental Risk Console",
                    "hi": "बहु-आपदा पर्यावरणीय जोखिम कंसोल"},
    "hero_sub": {"en": "Fusing aerial/satellite imagery with environmental telemetry to classify Flood, Wildfire, Cyclone & Earthquake risk — with full Grad-CAM + SHAP explainability.",
                  "hi": "बाढ़, दावानल, चक्रवात और भूकंप जोखिम को वर्गीकृत करने के लिए सैटेलाइट छवियों को पर्यावरणीय डेटा के साथ जोड़ना — पूर्ण Grad-CAM + SHAP व्याख्या सहित।"},
    "input1_title": {"en": "Satellite / Aerial Feed", "hi": "सैटेलाइट / हवाई फ़ीड"},
    "input2_title": {"en": "Environmental Telemetry", "hi": "पर्यावरणीय डेटा"},
    "input3_title": {"en": "Location", "hi": "स्थान"},
    "run_button": {"en": "▶  RUN RISK ANALYSIS", "hi": "▶  जोखिम विश्लेषण चलाएँ"},
    "output_title": {"en": "Risk Assessment", "hi": "जोखिम आकलन"},
    "predicted_hazard": {"en": "PREDICTED HAZARD", "hi": "अनुमानित आपदा"},
    "confidence": {"en": "CONFIDENCE", "hi": "विश्वास स्तर"},
    "recommended_actions": {"en": "🚨 Recommended Actions", "hi": "🚨 अनुशंसित कार्रवाई"},
    "tab_gradcam": {"en": "🛰️ GRAD-CAM", "hi": "🛰️ ग्रैड-कैम"},
    "tab_shap": {"en": "🧬 SHAP EXPLAINABILITY", "hi": "🧬 SHAP व्याख्या"},
    "tab_confidence": {"en": "📈 CONFIDENCE", "hi": "📈 विश्वास स्तर"},
    "tab_map": {"en": "🗺️ LOCATION MAP", "hi": "🗺️ स्थान मानचित्र"},
    "upload_prompt": {"en": "◇ NO FEED LOADED ◇", "hi": "◇ कोई फ़ीड लोड नहीं ◇"},
    "awaiting": {"en": "📡 Awaiting image feed — upload a satellite/aerial image on the left, set telemetry values, then run analysis.",
                  "hi": "📡 छवि की प्रतीक्षा है — बाईं ओर एक सैटेलाइट/हवाई छवि अपलोड करें, मान सेट करें, फिर विश्लेषण चलाएँ।"},
    "feedback_prompt": {"en": "Was this prediction correct?", "hi": "क्या यह अनुमान सही था?"},
    "feedback_thanks": {"en": "✅ Thanks — logged for future model retraining.", "hi": "✅ धन्यवाद — भविष्य के मॉडल प्रशिक्षण हेतु दर्ज किया गया।"},
}


def t(key, lang="en"):
    entry = TRANSLATIONS.get(key)
    if not entry:
        return key
    return entry.get(lang, entry.get("en", key))


# =========================================================
# 4. Static model-performance dashboard (hardcoded real training results)
# =========================================================
PERFORMANCE_REPORT = pd.DataFrame([
    {"Class": "Flood", "Precision": 0.91, "Recall": 0.97, "F1-score": 0.94, "Support": 100},
    {"Class": "Wildfire", "Precision": 1.00, "Recall": 0.96, "F1-score": 0.98, "Support": 100},
    {"Class": "Cyclone", "Precision": 1.00, "Recall": 1.00, "F1-score": 1.00, "Support": 100},
    {"Class": "Earthquake", "Precision": 0.95, "Recall": 0.92, "F1-score": 0.93, "Support": 100},
])

_CONFUSION_MATRIX = np.array([
    [97, 0, 0, 3],
    [2, 96, 0, 2],
    [0, 0, 100, 0],
    [8, 0, 0, 92],
])
_CM_CLASSES = ["Flood", "Wildfire", "Cyclone", "Earthquake"]


def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    fig.patch.set_facecolor("#170A2E")
    ax.set_facecolor("#170A2E")
    im = ax.imshow(_CONFUSION_MATRIX, cmap="RdPu")
    ax.set_xticks(range(len(_CM_CLASSES)))
    ax.set_yticks(range(len(_CM_CLASSES)))
    ax.set_xticklabels(_CM_CLASSES, rotation=30, ha="right", fontsize=9, color="#E9D5FF")
    ax.set_yticklabels(_CM_CLASSES, fontsize=9, color="#E9D5FF")
    ax.set_xlabel("Predicted", fontsize=9, color="#C4B5FD")
    ax.set_ylabel("True", fontsize=9, color="#C4B5FD")
    for i in range(len(_CM_CLASSES)):
        for j in range(len(_CM_CLASSES)):
            val = _CONFUSION_MATRIX[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > 50 else "#2E1065", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#E9D5FF")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#E9D5FF")
    plt.tight_layout()
    return fig