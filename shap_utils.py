"""
SHAP Explainability for the TABULAR branch of the fusion model
------------------------------------------------------------------
Idea: image embedding ko FIX kar dete hain (jo bhi uploaded image hai uska),
aur sirf tabular features ko vary karke dekhte hain ki fused prediction
kaise badalta hai. Isse SHAP sirf tabular branch ka marginal contribution
explain karta hai — ye fast bhi hai (CNN dubara-dubara chalane ki zaroorat
nahi) aur conceptually sahi bhi hai fusion model ke liye.

Plots:
  - Bar         -> global mean |SHAP value| per feature (importance ranking)
  - Beeswarm    -> feature value vs SHAP impact, sabhi reference samples ke liye
  - Waterfall   -> ek single prediction (uploaded image + user ke sliders) ka breakdown
  - Decision    -> multiple samples ka cumulative decision path
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_loader import generate_features_for_class, CLASS_NAMES, FEATURE_COLS

# NOTE: `shap` library ka import bahut heavy hai (sklearn, numba, scipy jaise
# dependencies load karta hai) — isliye ise TOP-LEVEL pe import nahi karte.
# Agar top pe import karte to Streamlit app OPEN hote hi (bina kisi prediction
# ke) ye sab load ho jaata, jisse startup bahut slow lagta. Ab ye sirf tab
# import hota hai jab user "RUN RISK ANALYSIS" dabata hai (compute_shap call).


# -----------------------------
# 1. Reference/background dataset (domain-informed, same generator as training data)
# -----------------------------
def generate_reference_tabular(n_per_class=12, seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    for cls in CLASS_NAMES:
        for _ in range(n_per_class):
            feats = generate_features_for_class(cls, rng)
            rows.append(feats)
    df = pd.DataFrame(rows)[FEATURE_COLS]
    return df


# -----------------------------
# 2. Fast predict function: image embedding fixed, only tabular varies
# -----------------------------
def build_fast_predict_fn(model, image_tensor, class_idx, device):
    with torch.no_grad():
        img_feat = model.image_encoder(image_tensor)  # (1, D) — computed ONCE

    def predict_fn(tabular_scaled_batch):
        tab_tensor = torch.tensor(tabular_scaled_batch, dtype=torch.float32).to(device)
        with torch.no_grad():
            tab_feat = model.tabular_encoder(tab_tensor)
            img_feat_rep = img_feat.repeat(tab_feat.shape[0], 1)
            fused = torch.cat([img_feat_rep, tab_feat], dim=1)
            logits = model.classifier(fused)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs[:, class_idx]

    return predict_fn


# -----------------------------
# 3. Compute SHAP values (speed-optimized)
# -----------------------------
def compute_shap(model, image_tensor, scaler, class_idx, user_raw_row, device,
                  n_per_class=3, nsamples=64, seed=123):
    """
    Speed notes:
      - n_per_class chhota rakha hai (background/eval set jitna bada, utna slow)
      - background ko kmeans se 5 points tak summarize karte hain (KernelExplainer
        ki cost seedha background size pe depend karti hai)
      - l1_reg="num_features(n)" use karte hain — default "aic" wala Lasso
        regularization-path search bahut slow hota hai, ye seedha fixed feature
        count use karta hai (bahut fast, minor accuracy trade-off jo demo ke liye fine hai)
    """
    import shap  # lazy import — sirf yahan chahiye, app startup fast rakhta hai
    ref_df = generate_reference_tabular(n_per_class=n_per_class, seed=seed)
    ref_scaled = scaler.transform(ref_df.values)

    user_scaled = scaler.transform(user_raw_row.reshape(1, -1))

    eval_raw = np.vstack([ref_df.values, user_raw_row.reshape(1, -1)])
    eval_scaled = np.vstack([ref_scaled, user_scaled])

    predict_fn = build_fast_predict_fn(model, image_tensor, class_idx, device)

    background = shap.kmeans(ref_scaled, min(5, len(ref_scaled)))
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(
        eval_scaled, nsamples=nsamples, silent=True,
        l1_reg=f"num_features({len(FEATURE_COLS)})",
    )

    return {
        "shap_values": shap_values,              # (n_eval, n_features)
        "eval_raw": eval_raw,                     # unscaled, human-readable feature values
        "expected_value": explainer.expected_value,
        "feature_names": FEATURE_COLS,
        "user_idx": eval_raw.shape[0] - 1,         # last row = the user's instance
    }


# -----------------------------
# 4. Light-themed plot helpers (white bg + sky-blue accents, matches app.py panels)
# -----------------------------
def _dark_fig(figsize):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#170A2E")
    return fig


def _style_axes(ax):
    ax.set_facecolor("#170A2E")
    ax.tick_params(colors="#E9D5FF", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#3B1466")
    ax.xaxis.label.set_color("#C4B5FD")
    ax.yaxis.label.set_color("#C4B5FD")
    ax.title.set_color("#F5F3FF")


def plot_bar(result, class_name):
    plt.close("all")   # koi bhi stale/leftover figure clear kar do
    shap_values = result["shap_values"]
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    labels = [result["feature_names"][i] for i in order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    fig.patch.set_facecolor("#170A2E")
    _style_axes(ax)
    ax.barh(labels, vals, color="#EC4899")
    ax.set_xlabel(f"Mean |SHAP value|  →  impact on P({class_name})")
    ax.grid(axis="x", color="#3B1466", linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_beeswarm(result, class_name):
    import shap
    plt.close("all")   # CRITICAL: bina iske ye pichle plot (waterfall) ke figure pe hi draw hota tha
    plt.figure(figsize=(5.6, 3.6))
    shap.summary_plot(
        result["shap_values"], features=result["eval_raw"],
        feature_names=result["feature_names"], plot_type="dot",
        show=False, plot_size=None, cmap="cool",
    )
    fig = plt.gcf()
    fig.patch.set_facecolor("#170A2E")
    for ax in fig.axes:
        ax.set_facecolor("#170A2E")
        ax.tick_params(colors="#E9D5FF", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#3B1466")
    plt.title(f"Feature impact on P({class_name})", color="#F5F3FF", fontsize=11)
    plt.tight_layout()
    return fig


def plot_waterfall(result, class_name):
    import shap
    plt.close("all")
    idx = result["user_idx"]
    explanation = shap.Explanation(
        values=result["shap_values"][idx],
        base_values=result["expected_value"],
        data=result["eval_raw"][idx],
        feature_names=result["feature_names"],
    )
    plt.figure(figsize=(5.6, 3.8))
    shap.plots.waterfall(explanation, show=False, max_display=8)
    fig = plt.gcf()
    fig.patch.set_facecolor("#170A2E")
    for ax in fig.axes:
        ax.set_facecolor("#170A2E")
        ax.tick_params(colors="#E9D5FF", labelsize=9)
    plt.title(f"Your instance → P({class_name})", color="#F5F3FF", fontsize=11)
    plt.tight_layout()
    return fig


def plot_decision(result, class_name):
    import shap
    plt.close("all")
    plt.figure(figsize=(5.6, 3.8))
    shap.decision_plot(
        result["expected_value"], result["shap_values"], result["eval_raw"],
        feature_names=result["feature_names"], show=False,
        highlight=[result["user_idx"]],
    )
    fig = plt.gcf()
    fig.patch.set_facecolor("#170A2E")
    for ax in fig.axes:
        ax.set_facecolor("#170A2E")
        ax.tick_params(colors="#E9D5FF", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#3B1466")
    plt.title(f"Decision path → P({class_name})", color="#F5F3FF", fontsize=11)
    plt.tight_layout()
    return fig