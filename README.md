# 🎯 Customer Segmentation Pro

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A professional AI-powered data science dashboard that clusters customer data using K-Means and provides automated marketing strategies via a 100% offline local LLM.**

[🚀 Live Demo](#) · [📖 Documentation](#architecture) · [🐛 Report Bug](https://github.com/Prasunnandi/customer-segmentation-data-mining-/issues)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **K-Means Clustering** | Unsupervised machine learning to segment customers based on Annual Income and Spending Score |
| 📊 **Interactive Dashboard** | Beautiful visualizations using Seaborn and Matplotlib directly within Streamlit |
| 📈 **Cluster Analytics** | Instantly compute aggregate statistics (mean, min, max) and Silhouette Scores for your segments |
| 💬 **AI Marketing Assistant** | A built-in chat interface that reads your segment data and suggests targeted marketing strategies |
| 🌐 **100% Free & Offline LLM** | Uses a local `distilgpt2` model via Hugging Face pipelines — absolutely no API keys or paid subscriptions required |

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│         Dark Theme · Tabs · Sidebar · Chat Interface        │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────────┐  ┌───────────────────┐
│  Data Upload │   │   Scikit-Learn   │  │   Cluster Stats   │
│ (CSV format) │   │     (K-Means)    │  │   Aggregation     │
└──────┬───────┘   └────────┬─────────┘  └────────┬──────────┘
       │                    │                     │
       └────────────────────┴─────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Local LLM Engine    │
                │   distilgpt2 (82MB)   │
                │   Runs 100% offline   │
                └───────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A dataset with `Annual_Income_k$` and `Spending_Score_1_100` columns (e.g., standard Mall Customers dataset)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Prasunnandi/customer-segmentation-data-mining-.git
cd customer-segmentation-data-mining-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> ⚡ **First run:** The app will automatically download the `distilgpt2` model (~82MB) locally. Subsequent runs start instantly and require no internet access.

---

## 📦 Project Structure

```text
customer-segmentation-data-mining-/
│
├── app.py                  # Main Streamlit application & UI
├── hf_llm.py               # Custom local offline LLM pipeline wrapper
├── generate_data.py        # Helper script to generate mock CSV data
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Machine Learning** | Scikit-learn (K-Means, StandardScaler, Metrics) |
| **Data Processing** | Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **LLM Engine** | Hugging Face Transformers (`distilgpt2`) |
| **Orchestration** | LangChain |

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your fork → set main file to `app.py`
4. Click **Deploy** 

*(No secrets or API keys are required for this deployment!)*

---

## 📸 Screenshots

| Tab | Feature |
|---|---|
| 📊 Dashboard & Visualization | View the 2D scatter plot of your customer segments and overall metrics |
| 📈 Cluster Analytics | Drill down into the specific mathematical boundaries of each generated cluster |
| 🤖 AI Marketing Assistant | Ask business questions like *"How should I target Cluster 1?"* and get AI strategies |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/Prasunnandi/customer-segmentation-data-mining-/issues).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
Made with ❤️ using Streamlit, Scikit-Learn & Hugging Face
</div>
