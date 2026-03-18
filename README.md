# Mall Customer Segmentation — K-Means Clustering Case Study

![Python](https://img.shields.io/badge/Python-3.x-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-K--Means%20Clustering-purple)
![Type](https://img.shields.io/badge/Type-Unsupervised%20Learning-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete unsupervised machine learning project that segments mall customers into distinct groups based on their Annual Income and Spending Score. The project uses the **K-Means Clustering** algorithm and applies the **Elbow Method** to find the optimal number of clusters.

---

## Problem Statement

A mall wants to better understand its customers. By grouping customers based on their income and spending behaviour, the business can design targeted marketing strategies for each customer segment.

Unlike the previous case studies, this is an **unsupervised learning** problem — there are no predefined labels or correct answers. The model discovers hidden patterns and groups in the data on its own.

---

## Supervised vs Unsupervised Learning

| Type | Label Required | Goal | Example |
|------|---------------|------|---------|
| Supervised Learning | Yes | Predict a known output | Titanic Survival, Iris Classification |
| Unsupervised Learning | No | Discover hidden patterns | Customer Segmentation |

---

## Project Workflow

| Step | Description |
|------|-------------|
| 1 | Load the dataset, preview records, check shape and missing values |
| 2 | Select relevant features — Annual Income and Spending Score |
| 3 | Scale the features using Standard Scaler |
| 4 | Apply the Elbow Method to find the optimal number of clusters |
| 5 | Train the final K-Means model using the best K and assign cluster labels |

---

## Dataset

**File:** `Mall_Customers.csv`

**Features Selected (X):**
- `AnnualIncome` — customer's annual income (in thousands)
- `SpendingScore` — score assigned by the mall based on spending behaviour (1 to 100)

**No Target Variable (Y)** — this is unsupervised learning. There is no label column to predict.

---

## Why Feature Scaling?

K-Means is a **distance-based algorithm**, just like KNN. Without scaling, a feature with larger values (like Annual Income in thousands) will dominate the distance calculation over a feature with smaller values (like Spending Score out of 100). `StandardScaler` brings both features to the same scale so the clustering is fair and accurate.

---

## Elbow Method — Finding the Best K

Instead of guessing the number of clusters, the **Elbow Method** is used:

- K-Means is run for K values from 1 to 10
- For each K, the **WCSS (Within-Cluster Sum of Squares)** is recorded
- WCSS measures how tightly packed the data points are within each cluster
- A line graph of K vs WCSS is plotted
- The point where the curve bends like an elbow is the optimal K

**Lower WCSS = tighter, better-formed clusters**

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | K-Means Clustering |
| Number of Clusters (K) | 4 |
| K Selection Method | Elbow Method |
| Feature Scaling | StandardScaler |
| n_init | 10 |
| Random State | 42 |
| Library | scikit-learn |

---

## Output

After training, each customer in the dataset is assigned a **Cluster label (0, 1, 2, or 3)**. This label is added as a new column in the dataset, allowing the business to identify which segment each customer belongs to.

Example output:

| CustomerID | AnnualIncome | SpendingScore | Cluster |
|------------|-------------|---------------|---------|
| 1 | 15 | 39 | 2 |
| 2 | 70 | 82 | 0 |
| 3 | 90 | 14 | 3 |

---

## Evaluation

K-Means clustering does not use accuracy, confusion matrix, or classification report because there are no true labels to compare against. Instead, model quality is judged by:

- **WCSS (Within-Cluster Sum of Squares)** — lower is better
- **Elbow Plot** — visual confirmation of the best K
- **Business Interpretability** — do the clusters make real-world sense?

---

## Tech Stack

- Python 3
- pandas — data loading and analysis
- matplotlib — Elbow Method graph
- scikit-learn — feature scaling and K-Means clustering

---

## How to Run

1. Clone this repository
2. Place `Mall_Customers.csv` in the same folder as the script
3. Install the required libraries:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
4. Run the script:
   ```bash
   python MallCustomerKMeanElbow_Final.py
   ```

---

## Key Concepts Covered

- Unsupervised Machine Learning
- K-Means Clustering
- Feature Scaling (StandardScaler)
- Elbow Method (finding optimal K)
- WCSS (Within-Cluster Sum of Squares)
- Customer Segmentation
- Cluster Label Assignment



---

## Author

**Raviraj Aade**

Built as part of a Machine Learning Case Study series to understand unsupervised learning, clustering algorithms, and how to discover hidden patterns in data without labeled outputs.
