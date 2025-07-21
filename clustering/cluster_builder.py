# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:07:45 2024

@author: mobau

Stellt Funktionen bereit, welche zum Vergleich und Evaluierung von Clustern genutzt werden können
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

def cluster_data(data_points, num_clusters):
    """
    Führt die k-Means-Clusterung durch.

    Parameters:
        data_points (array): Die Datenpunkte, die geclustert werden sollen.
        num_clusters (int): Die Anzahl der Cluster.

    Returns:
        tuple: (kmeans_model, cluster_labels), wobei
            - kmeans_model: Das trainierte KMeans-Modell.
            - cluster_labels: Die Cluster-Zuordnungen der Datenpunkte.
    """
    data_points = np.array(data_points)  # Sicherstellen, dass es ein NumPy-Array ist

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")

        kmeans_model = KMeans(n_clusters=num_clusters, random_state=5)
        cluster_labels = kmeans_model.fit_predict(data_points)

    return kmeans_model, cluster_labels

def determine_k(data_points, max_k=10):
    """
    Bestimmt die optimale Anzahl der Cluster k basierend auf der Elbow-Methode
    und dem Silhouetten-Koeffizienten.

    Parameters:
        X (array): Die Datenpunkte, die geclustert werden sollen.
        max_k (int): Maximale Anzahl von Clustern.

    Returns:
        int: Optimale Anzahl von Clustern k.
    """
    silhouette_scores = []
    K_range = range(3, min(max_k, len(data_points)))

    for k in K_range:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")

            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_points)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(data_points, labels))

    

    # Silhouetten-Koeffizient Plot
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouette_scores, marker='o', color='orange', label='Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.legend()
    plt.show()

    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Automatisch gewähltes optimales k: {optimal_k}")

    return optimal_k