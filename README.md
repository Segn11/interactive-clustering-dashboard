# 📊 Interactive Clustering Dashboard with Unsupervised Learning

This project is an interactive dashboard built with [Dash (Plotly)](https://dash.plotly.com/) that visualizes consumer finance survey data using **unsupervised learning (KMeans clustering)**.  
Users can explore high-variance financial features and adjust clustering parameters in real-time, gaining insight into consumer behavior patterns.

---

## 🚀 Features

- 📈 **Dynamic Variance Analysis**  
  Visualize the top 5 features with the highest variance in the dataset (with an option to use trimmed or untrimmed variance).

- 🔍 **Unsupervised Clustering with KMeans**  
  Perform KMeans clustering on the selected features to discover hidden groupings in the data.

- 🎚️ **Interactive Controls**  
  - Toggle between trimmed vs untrimmed variance.
  - Adjust the number of clusters `k` with a slider.
  - Instantly view **inertia** and **silhouette score** as evaluation metrics.

- 🧠 **No Labels Required**  
  This project uses **unsupervised machine learning** — it finds patterns without any pre-existing labels or targets.

---

## 📁 Dataset

The dataset used is from the **2019 U.S. Survey of Consumer Finances (SCF)**.

The data is filtered to include only:
- Households fearful of credit use (`TURNFEAR == 1`)
- Households with net worth under $2 million

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **Dash** | Web app framework for interactive dashboards |
| **Plotly Express** | Easy and beautiful plotting |
| **Scikit-learn** | KMeans, Scaling, Silhouette Score |
| **Pandas** | Data manipulation |
| **SciPy** | Trimmed variance calculation |

---

## 🧪 Unsupervised Learning Workflow

1. **Feature Selection**: Use trimmed or untrimmed variance to choose top 5 features.
2. **Scaling**: Standardize features to have mean 0 and variance 1.
3. **Clustering**: Apply `KMeans` to group data into `k` clusters.
4. **Evaluation**: Display `Inertia` and `Silhouette Score` to help assess cluster quality.

---


