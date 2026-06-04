# 🛒 SmartCart — Customer Segmentation System

> An unsupervised machine learning system that segments e-commerce customers into meaningful clusters based on demographics, purchase behaviour, spending patterns, and engagement — enabling personalised marketing and data-driven customer retention strategies.

---

## 📌 Problem Statement

**SmartCart** is a growing e-commerce platform serving customers across multiple countries with **2,240 customer records** and **22 attributes** covering demographics, purchase behaviour, website activity, and campaign response.

Currently, SmartCart applies generic marketing strategies for all customers without understanding distinct behaviour patterns — resulting in inefficient marketing, missed retention opportunities, and delayed identification of churn-prone users.

This project solves that by building an **intelligent customer segmentation system** using **unsupervised machine learning** to group customers into meaningful clusters based on purchasing behaviour, engagement levels, and loyalty indicators — supporting **data-driven decision-making** for personalised marketing.

---

## 🗂️ Dataset

**File:** `smartcart_customers.csv` — 2,240 rows × 22 columns

### Feature Categories

| Category | Features |
|---|---|
| **Customer Demographics** | `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer` |
| **Purchase Behaviour (Amount Spent)** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| **Purchase Behaviour (Frequency)** | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| **Customer Feedback & Activity** | `Recency`, `Complain`, `Response` |

---

## ⚙️ ML Pipeline

```
Raw Data (2240 records, 22 features)
   │
   ├─ 1. Data Cleaning
   │     └─ Missing Value Imputation → Income filled with median
   │
   ├─ 2. Feature Engineering
   │     ├─ Age               ← derived from Year_Birth
   │     ├─ Customer_Tenure   ← days since Dt_Customer enrollment
   │     ├─ Total_Spending    ← sum of all 6 MntXxx columns
   │     ├─ Total_Children    ← Kidhome + Teenhome
   │     ├─ Education         ← grouped into 3 tiers
   │     │     (Basic/2n Cycle → Undergraduate, Graduation → Graduate,
   │     │      Master/PhD → Postgraduate)
   │     └─ Living_With       ← Marital_Status → Partner / Alone
   │
   ├─ 3. Outlier Removal
   │     ├─ Age < 90
   │     └─ Income < 600,000
   │
   ├─ 4. Encoding
   │     └─ One-Hot Encoding → Education, Living_With
   │
   ├─ 5. Scaling
   │     └─ StandardScaler on all features
   │
   ├─ 6. Dimensionality Reduction
   │     └─ PCA → 3 components (for 3D visualization & clustering)
   │
   ├─ 7. Optimal K Selection
   │     ├─ Elbow Method (KneeLocator on WCSS)
   │     └─ Silhouette Score (k = 2 to 10)
   │
   ├─ 8. Clustering (k = 4)
   │     ├─ K-Means Clustering
   │     └─ Agglomerative Clustering (Ward linkage)
   │
   └─ 9. Cluster Characterization
         ├─ Cluster size distribution
         ├─ Income vs. Total Spending scatter
         └─ Group mean summary across all features
```

---

## 📊 Visualizations

| Plot | Purpose |
|---|---|
| Pair Plot | Feature relationships across Income, Spending, Age, etc. |
| Correlation Heatmap | Numeric feature correlations |
| 3D PCA Scatter | Raw data projected into 3 principal components |
| Elbow Curve | WCSS vs K for optimal cluster count |
| Silhouette Score Plot | Cluster quality vs K |
| Combined Elbow + Silhouette | Dual-axis comparison for K selection |
| 3D Cluster Scatter (K-Means) | Final K-Means segments in PCA space |
| 3D Cluster Scatter (Agglomerative) | Final Agglomerative segments in PCA space |
| Cluster Count Plot | Distribution of customers per cluster |
| Income vs Spending Scatter | Cluster-wise spending behaviour |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, feature engineering |
| `matplotlib` | Plotting and 3D visualization |
| `seaborn` | Statistical plots (heatmap, pairplot, scatter) |
| `scikit-learn` | PCA, KMeans, AgglomerativeClustering, StandardScaler, OneHotEncoder, silhouette_score |
| `kneed` | Automated elbow detection (KneeLocator) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/smartcart.git
cd smartcart
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the Dataset

Ensure `smartcart_customers.csv` is in the root directory.

### 4. Run the Notebook

```bash
jupyter notebook smartcart.ipynb
```

---

## 📁 Project Structure

```
smartcart/
│
├── smartcart.ipynb                  # Main Jupyter Notebook
├── smartcart_customers.csv          # Dataset (2240 records, 22 features)
├── SmartCart_Clustering_System.pdf  # Problem statement & dataset description
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

**`requirements.txt`**
```
pandas
matplotlib
seaborn
scikit-learn
kneed
jupyter
```

---

## 🔍 Cluster Insights

After segmentation, each of the **4 clusters** is profiled using group means. Typical segments include:

| Segment | Profile |
|---|---|
| 🟢 Premium Customers | High income, high total spending, low children |
| 🔵 Young Families | Mid income, moderate spending, more children at home |
| 🟡 Budget-Conscious | Lower income, low spending, high discount purchases |
| 🔴 Loyal Seniors | Older, long-tenured, consistent moderate spenders |

*(Exact cluster profiles depend on final model output.)*

---

## 🎓 About

This project was built as a **Minor Project for Machine Learning** coursework. It demonstrates a complete end-to-end unsupervised ML workflow — from raw data preprocessing to actionable business insights through customer segmentation.

---

## 📄 License

For academic use. Feel free to reference or build upon this project.
