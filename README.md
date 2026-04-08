# 📊 Invoice Intelligence ML Project

An end-to-end Machine Learning application to **predict freight cost** and **detect suspicious invoices** using real-world business logic.

🚀 Built with **Python, Scikit-learn, and Streamlit**

---

## 🔗 Live App

👉 [https://your-streamlit-app-link.streamlit.app](https://invoiceintelligencemlproject-at2qmcsogge9elknbynx8h.streamlit.app/)

---


## 🎯 Project Overview

This project solves two critical business problems:

### 1️⃣ Freight Cost Prediction

Predicts expected freight cost using:

* Invoice Quantity
* Invoice Dollars

📌 Helps in:

* Budget forecasting
* Cost optimization
* Vendor benchmarking

---

### 2️⃣ Invoice Flag Detection 🚨

Classifies invoices as:

* ✅ Normal
* 🚨 Suspicious

Based on:

* Invoice vs Item Value mismatch
* Freight anomalies
* Quantity inconsistencies

---

## 🧠 How It Works

### 🔹 Data Processing

* Cleaned and structured invoice data
* Handled missing values and inconsistencies

---

### 🔹 Feature Engineering

Key features used:

* `invoice_quantity`
* `invoice_dollars`
* `Freight`
* `total_item_quantity`
* `total_item_dollars`

---

### 🔹 Label Creation (Invoice Flag)

Invoices are flagged if:

* 💰 Invoice amount differs from item totals
* ⏱️ Receiving delay is unusually high

---

### 🔹 Model Training

Models trained and evaluated:

* Linear Regression
* Decision Tree
* Random Forest ✅ (Best performing)

---

### 🔹 Deployment

* Interactive UI built with **Streamlit**
* Deployed for real-time predictions

---

## ⚙️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## 📊 Features

✔ Freight cost prediction
✔ Invoice anomaly detection
✔ Explainable AI ("Why flagged?")
✔ Clean and interactive dashboard
✔ Real-time inference

---

## 📁 Project Structure

```
invoice_intelligence_ml_project/
│
├── app.py
├── models/
│   ├── freight_cost_model.pkl
│   └── predict_flag_invoice.pkl
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── notebooks/
│   └── Invoice Flagging.ipynb
|   └── Predicting Freight Cost.ipynb
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

```bash
## ▶️ How to Run This Project

### Clone the repository:

```bash
git clone https://github.com/your-username/invoice_intelligence_ml_project.git
cd invoice_intelligence_ml_project
```

---

### Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Ensure model files are present:

```bash
models/
├── freight_cost_model.pkl
└── predict_flag_invoice.pkl
```

---

### Run the Streamlit app:

```bash
streamlit run app.py
```

---

### Open in browser:

```bash
http://localhost:8501
```

---

### (Optional) Run Notebook:

```bash
notebooks/Invoice Flagging.ipynb
```


---

## 💡 Business Impact

* 📉 Reduced cost leakages
* 🚨 Early anomaly detection
* ⚡ Faster invoice validation
* 📊 Better financial decision-making

---

## 🧠 Key Learnings

* End-to-end ML pipeline development
* Feature engineering for real-world problems
* Model evaluation and selection
* Deployment using Streamlit
* Handling feature mismatches in production

---

## 👤 Author

**Sudiksha Aslesha**
Data Analyst | Machine Learning Enthusiast


---
