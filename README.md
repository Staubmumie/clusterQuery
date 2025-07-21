# Cluster Queries

This project focuses on clustering a set of queries over event streams. Three different vectorization approaches are implemented, followed by clustering using the K-Means algorithm:

1. **Structural Distance**  
   Based on a hierarchical structure that measures the distance between queries.

2. **Syntactic Similarity**  
   Inspired by Information Retrieval techniques, clustering is performed based on the syntactic form of the queries.

3. **Semantic Similarity**  
   Uses an extended version of the Word2Vec model to represent the semantic meaning of the queries.

For a detailed explanation of the methodology, implementation, and evaluation, please refer to the included PDF:  
**`Bachelor's Thesis`**

---

## 📁 Project Structure

- **`/data`**  
  Contains the dataset of queries that are clustered in this project.

- **`main.py`**  
  The main script where vector representations are created, clustered, and evaluated.

- **`/clustering`**  
  Includes the clustering methods used in `main.py`.

- **`/preprocessing`**  
  Contains the implementations of all three vectorization approaches.

- **`/results`**  
  Stores all visualizations and tables used for analyzing the clustering performance.

- **`/variable_tuning`**  
  Includes results from parameter experiments conducted during development.

---

> ⚠️ Note: This project does not include a graphical user interface (GUI).
