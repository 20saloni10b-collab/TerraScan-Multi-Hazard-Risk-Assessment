"""
Report utilities — PDF export + hazard severity/action metadata + map helper.
"""
import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HAZARD_ACTIONS = {
    "Flood": {
        "severity": "HIGH",
        "severity_color": "#D97706",
        "actions": [
            "Move to higher ground immediately; avoid low-lying areas.",
            "Avoid walking/driving through flood water — 6 inches can knock an adult down.",
            "Turn off electricity/gas at the mains if water is entering the building.",
            "Keep an emergency kit (water, food, torch, first aid) ready.",
            "Monitor official flood warnings and evacuation orders.",
        ],
    },
    "Wildfire": {
        "severity": "CRITICAL",
        "severity_color": "#DC2626",
        "actions": [
            "Evacuate immediately if in the fire's path — don't wait if visibly unsafe.",
            "Close all windows/doors; clear flammable material from around structures.",
            "Wear an N95 mask if smoke is present; limit outdoor exposure.",
            "Keep a 'go-bag' with documents, medication, and water ready.",
            "Follow local fire department / forest department advisories.",
        ],
    },
    "Cyclone": {
        "severity": "HIGH",
        "severity_color": "#D97706",
        "actions": [
            "Secure loose outdoor objects and board up windows.",
            "Stock at least 3 days of water, food, and medical supplies.",
            "Stay indoors, away from windows, during the storm.",
            "Charge phones/power banks before landfall — expect outages.",
            "Follow coast guard / meteorological department evacuation advisories.",
        ],
    },
    "Earthquake": {
        "severity": "CRITICAL",
        "severity_color": "#DC2626",
        "actions": [
            "Drop, Cover, and Hold On — get under sturdy furniture immediately.",
            "Stay away from windows, glass, and heavy hanging objects.",
            "If outdoors, move to open ground, away from buildings and power lines.",
            "After shaking stops, check for gas leaks before re-entering buildings.",
            "Be prepared for aftershocks; keep an emergency kit accessible.",
        ],
    },
}


def confidence_tier(confidence_pct):
    """Returns (label, color) describing how certain the prediction is."""
    if confidence_pct >= 85:
        return "CONFIRMED", "#16A34A"
    elif confidence_pct >= 60:
        return "LIKELY", "#D97706"
    else:
        return "UNCERTAIN — VERIFY MANUALLY", "#DC2626"


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


def pil_to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_pdf_report(pred_class, confidence, probs, class_names, feature_values,
                         image, overlay_img, prob_fig, shap_figs, location=None):
    """
    Builds a multi-page in-memory PDF summarizing one prediction.
    shap_figs: dict with keys 'bar','waterfall','beeswarm','decision' -> matplotlib figures
    Returns: BytesIO (ready for st.download_button)
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---- Page 1: Summary ----
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.patch.set_facecolor("white")
        fig.suptitle("TerraScan — Environmental Risk Assessment Report", fontsize=15, fontweight="bold", y=0.975)
        fig.text(0.5, 0.945, datetime.now().strftime("Generated on %d %b %Y, %H:%M"),
                  ha="center", fontsize=9, color="#555555")

        fig.text(0.08, 0.905, "Predicted Hazard:", fontsize=11, fontweight="bold")
        fig.text(0.36, 0.905, f"{pred_class}   (Confidence: {confidence:.1f}%)", fontsize=11, color="#C2410C")

        tier, _ = confidence_tier(confidence)
        fig.text(0.08, 0.880, "Certainty Level:", fontsize=11, fontweight="bold")
        fig.text(0.36, 0.880, tier, fontsize=11)

        row_y = 0.855
        if location:
            fig.text(0.08, row_y, "Location:", fontsize=11, fontweight="bold")
            fig.text(0.36, row_y, f"Lat {location[0]:.4f}, Lon {location[1]:.4f}", fontsize=11)
            row_y -= 0.025

        ax1 = fig.add_axes([0.08, 0.55, 0.40, 0.27])
        ax1.imshow(image)
        ax1.set_title("Original Image", fontsize=10)
        ax1.axis("off")

        ax2 = fig.add_axes([0.52, 0.55, 0.40, 0.27])
        ax2.imshow(overlay_img)
        ax2.set_title("Grad-CAM Attention", fontsize=10)
        ax2.axis("off")

        fig.text(0.08, 0.50, "Class Probabilities:", fontsize=11, fontweight="bold")
        y = 0.47
        for cls, p in zip(class_names, probs):
            fig.text(0.10, y, f"•  {cls}: {p * 100:.1f}%", fontsize=10)
            y -= 0.025

        fig.text(0.08, 0.335, "Environmental Telemetry (input values):", fontsize=11, fontweight="bold")
        y = 0.305
        items = list(feature_values.items())
        for i in range(0, len(items), 2):
            line = "     |     ".join(f"{k}: {v:.2f}" for k, v in items[i:i + 2])
            fig.text(0.10, y, line, fontsize=9.5)
            y -= 0.022

        actions = HAZARD_ACTIONS.get(pred_class, {}).get("actions", [])
        fig.text(0.08, 0.175, "Recommended Actions:", fontsize=11, fontweight="bold", color="#B91C1C")
        y = 0.148
        for a in actions:
            fig.text(0.10, y, f"–  {a}", fontsize=9)
            y -= 0.024

        fig.text(0.5, 0.02, "TerraScan · Multimodal Environmental Risk Assessment · Academic Demonstration",
                  ha="center", fontsize=7.5, color="#888888")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: probability chart ----
        if prob_fig is not None:
            pdf.savefig(prob_fig)

        # ---- Page 3+: SHAP figures ----
        for key in ["bar", "waterfall", "beeswarm", "decision"]:
            f = shap_figs.get(key) if shap_figs else None
            if f is not None:
                pdf.savefig(f)

    buf.seek(0)
    return buf
