# 🌍 TerraScan: Multimodal Environmental Risk Assessment

<p align="center">
  <strong>AI-Powered Environmental Hazard Detection & Risk Analysis</strong>
  <br>
  <em>Internship Project | C-DAC Mohali | June – July 2026</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Generative%20AI-Gemini-8E75B2?style=for-the-badge" alt="Generative AI"/>
</p>

---

## 🌱 Overview

**TerraScan** is a multimodal environmental risk assessment system that combines satellite/aerial imagery with structured environmental telemetry to classify four hazard categories: **Flood, Wildfire, Cyclone, and Earthquake**.

The system integrates Deep Learning, Explainable AI (XAI), live environmental data, and Generative AI to deliver interpretable risk assessments through an interactive **Streamlit web application**.

## 🚨 Hazard Categories

| 🌊 Flood | 🔥 Wildfire | 🌀 Cyclone | 🌎 Earthquake |
|:---:|:---:|:---:|:---:|
| Flood risk analysis | Wildfire risk analysis | Cyclone risk analysis | Earthquake risk analysis |

## ✨ Key Features

- 🧠 **Multimodal Deep Learning** — ResNet-18 CNN for image feature extraction and a Residual MLP for environmental telemetry.
- 🔗 **Late Fusion Architecture** — Combines image and tabular feature embeddings for hazard classification.
- 🔍 **Explainable AI (XAI)** — Grad-CAM visual explanations and SHAP feature attribution.
- 🤖 **AI Copilot** — LLM-generated situation reports, RAG-based question answering, Agentic AI tools, and public safety advisories.
- 🌦️ **Environmental Data Integration** — NASA POWER and Open-Elevation APIs.
- 🛡️ **Input Robustness Check** — Flags potentially implausible image inputs.
- 🗺️ **Interactive Dashboard** — Image upload and camera input, mapping, risk assessment, and session history.
- 📊 **Batch Analysis** — Supports analysis of multiple inputs.
- 📄 **PDF Report Export** — Generates downloadable assessment reports.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Deep Learning | PyTorch, Torchvision |
| Computer Vision | ResNet-18, CNN |
| Tabular Learning | Residual MLP |
| Explainable AI | SHAP, Grad-CAM |
| Web Application | Streamlit |
| Generative AI | Google Gemini API |
| Retrieval-Augmented Generation | RAG |
| Environmental APIs | NASA POWER, Open-Elevation |
| Visualization | PyDeck |

## 🏗️ Project Architecture

```text
          🛰️ Satellite / Aerial Image
                       |
                       v
                ResNet-18 CNN
                       |
                Image Embeddings
                       |
                       +----------+
                                  |
                                  v
                           Late Fusion
                                  |
                                  v
                         Hazard Classifier
                                  ^
                                  |
                       Tabular Embeddings
                                  |
                         Residual MLP
                                  ^
                                  |
          🌦️ Environmental Telemetry
                                  |
                                  v
                   🌍 Risk Assessment Output
                                  |
                +-----------------+----------------+
                |                 |                |
                v                 v                v
           Grad-CAM / SHAP    AI Copilot      Streamlit UI
                               RAG + LLM      Reports + Maps
```

### 🔬 How It Works

1. **Image Processing:** ResNet-18 extracts meaningful visual features from satellite/aerial imagery.
2. **Environmental Analysis:** A Residual MLP processes eight telemetry features: rainfall, temperature, humidity, wind speed, soil moisture, NDVI, seismic activity, and elevation.
3. **Feature Fusion:** The image and environmental representations are combined to classify the hazard.
4. **Explainability:** Grad-CAM highlights relevant image regions, while SHAP helps interpret tabular feature contributions.
5. **AI Copilot:** Generative AI, retrieval-based question answering, and tools support situation reports and safety advisories.
6. **Interactive Results:** Streamlit presents predictions, visualizations, maps, history, and PDF reports.

## 📈 Model Performance

| Evaluation | Reported Accuracy |
|---|---:|
| Primary held-out test evaluation | **96.25%** |
| Separate preferred ResNet-18 backbone comparison | **99.25%** |

> 📌 These figures represent separate reported experiments and should not be interpreted as the same evaluation. Results may vary depending on the dataset, configuration, and evaluation procedure.

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create a Python 3.11 Environment

**Windows:**

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file or configure Streamlit secrets according to your application's code.

Example `.env` entry:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. Keep your API key private and **never upload `.env` or secret credentials to GitHub**.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the TerraScan dashboard.

> 💡 Ensure that the repository includes `app.py`, `requirements.txt`, the required model files, and any supporting modules. Exact installation steps may vary with the project structure.

## 📁 Project Components

The project includes the following logical components:

- 🖼️ Image feature extraction
- 🌦️ Environmental telemetry processing
- 🔗 Multimodal feature fusion
- 🔍 Grad-CAM and SHAP explainability
- 🤖 Gemini-powered AI Copilot and RAG
- 🗺️ Interactive risk assessment dashboard
- 📄 Assessment and PDF report generation

*The actual repository directory structure may differ.*

## 🎓 Internship Details

**Organization:** Centre for Development of Advanced Computing (C-DAC), Mohali

**Internship:** Machine Learning to Generative AI & LLMs

**Duration:** 1 June 2026 – 10 July 2026

**Project:** TerraScan — Multimodal Environmental Risk Assessment

This project was developed as part of the internship to explore the practical integration of Deep Learning, Computer Vision, Explainable AI, environmental data, and Generative AI into an interactive decision-support application.

## 🔮 Future Scope

- 🌐 Integration of additional real-time environmental and satellite data sources.
- 📍 More detailed location-based risk analysis.
- 📉 Expanded model validation across independent datasets.
- 🧠 Improved explainability and uncertainty estimation.
- 🚀 Deployment for accessible web-based demonstrations.

## ⚠️ Disclaimer

TerraScan is an **academic project and decision-support prototype**. Its outputs are experimental and must not be treated as official disaster warnings, emergency instructions, or a substitute for information from authorized disaster management agencies.

## 🙌 Acknowledgements

Special thanks to **C-DAC Mohali** for the internship opportunity and learning experience in Machine Learning, Generative AI, and Large Language Models.

---

<p align="center">
  🌍 <strong>TerraScan</strong>
  <br>
  <em>Combining AI and environmental data for interpretable risk assessment.</em>
</p>
