# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:58:48 2024
Main-File Kombiniert ui und Server
@author: mobau
"""
from preprocessing.DictionaryVectorizer import DictionaryVectorizer 
from preprocessing.TreeVectorizer import TreeVectorizer
from preprocessing.EmbeddingVectorizer import EmbeddingVectorizer

import clustering.cluster_analysis as ca
import clustering.cluster_builder as cb

EMBEDDING = "embedding"
DICTIONARY = "dictionary"
TREE = "tree"

def get_vectorized_points(method: str,  csv_file_path):
    if method == EMBEDDING:
        vectorizer = EmbeddingVectorizer(csv_file_path)
    elif method == DICTIONARY:
        vectorizer = DictionaryVectorizer(csv_file_path)
    elif method == TREE:
        vectorizer = TreeVectorizer(csv_file_path)
    else:
        raise ValueError(f"Unbekannte Vektorisierungsmethode: {method}")

    return vectorizer.get_points(), vectorizer.queries

def get_cluster(points, max_k = 100):
    """Findet den optimalen K-Wert und gibt das Cluster zurück"""
    optimal_k = cb.determine_k(points, max_k)
    kmeans_model, cluster_labels = cb.cluster_data(points, optimal_k)
    
    return  cluster_labels

def analyze_cluster(points, queries, cluster_labels, out_dir):
    """Wertet ein Cluster in Mikro- und Makroauswertung aus"""
    ca.macro_cluster_clustering(points, cluster_labels, out_dir)
    ca.micro_cluster_analysis(queries, cluster_labels, points, out_dir)

if __name__ == "__main__":
    csv_file_path = r'data/dataset.csv' 
    
    # Methode auswählen: EMBEDDING, DICTIONARY oder TREE
    method = TREE
    points, queries = get_vectorized_points(method, csv_file_path)
    
    cluster_labels = get_cluster(points)
    
    out_dir = r'C:\Users\mobau\OneDrive\Dokumente\__Uni\_Semester6\Bachelorarbeit\NeuerOrdner'
    analyze_cluster(points, queries, cluster_labels, out_dir)

    
    
