import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import umap.umap_ as umap



def bfs_depth(root):
    """Berechnet die Tiefe eines Baumes mit Breitensuche (BFS)."""
    depth = 0
    queue = [(root, 0)]  # (Knoten, Tiefe)
    visited = set()

    while queue:
        node, current_depth = queue.pop(0)

        if node in visited:
            continue
        visited.add(node)

        # Alle Elternknoten als Knoten für die nächste Ebene hinzufügen
        for parent, _ in node.parents:
            if parent not in visited:
                queue.append((parent, current_depth + 1))

        depth = max(depth, current_depth)

    return depth

def micro_cluster_analysis(queries, kmeans_labels, data_points, output_dir="micro_results"):
    os.makedirs(output_dir, exist_ok=True)

    # Prüfen, ob TreeNode-Objekte übergeben wurden
    is_tree = hasattr(queries[0], "name") and hasattr(queries[0], "parents")

    # Falls TreeNodes, extrahiere Namen für Verarbeitung
    if is_tree:
        names = [node.name for node in queries]
    else:
        names = queries

    # Cluster-Abgleich
    clustered_queries = get_clusters(queries if not is_tree else names, kmeans_labels)
    results = []

    for i, (original_label, cluster_names) in enumerate(clustered_queries.items(), start=1):
        cluster_id = i
        
        data_points = np.array(data_points)#rausnehmen falls es zum Typeerror ferhler kommt
        
        # Indizes im Cluster
        indices = [idx for idx, lbl in enumerate(kmeans_labels) if lbl == original_label]
        vectors = data_points[indices]

        centroid = np.mean(vectors, axis=0)
        dists_to_centroid = euclidean_distances(vectors, [centroid])
        rep_idx = np.argmin(dists_to_centroid)
        representative_sentence = cluster_names[rep_idx]
        representative_sentence_safe = f'"{representative_sentence}"'

        # Wortfrequenzen
        token_lists = [s.lower().split() for s in cluster_names]
        all_tokens = [token for tokens in token_lists for token in tokens]
        total_word_freq = Counter(all_tokens)
        word_in_sentence_count = Counter()
        for tokens in token_lists:
            word_in_sentence_count.update(set(tokens))

        def plot_word_freq(counter, title, filename, color):
            if len(counter) == 0:
                return
            top_words = counter.most_common(15)
            words, freqs = zip(*top_words)
            words = [w.replace('$', r'\$') for w in words]
            plt.figure(figsize=(10, 5))
            sns.barplot(x=list(freqs), y=list(words), orient="h", color=color)
            plt.title(title)
            plt.xlabel("Anzahl")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        plot_word_freq(total_word_freq, f"Cluster {cluster_id} – Gesamtereignisfrequenz",
                       f"{output_dir}/cluster_{cluster_id}_total_freq.png", "skyblue")
        plot_word_freq(word_in_sentence_count, f"Cluster {cluster_id} – Ereignisverteilung über Anfragen",
                       f"{output_dir}/cluster_{cluster_id}_sentence_freq.png", "lightgreen")

        avg_similarity = dists_to_centroid.mean()
        sentence_lengths = [len(tokens) for tokens in token_lists]
        avg_len = np.mean(sentence_lengths)
        std_len = np.std(sentence_lengths)

        result_row = {
            "id": cluster_id,
            "original_label": original_label,
            "anzahl_anfragen": len(cluster_names),
            "repräsentant": representative_sentence_safe,
            "intra_kohärenz": avg_similarity,
            "anfragenlänge_avg": avg_len,
            "anfragenlänge_std": std_len,
            "anfragen": cluster_names 
        }

        # Falls TreeNodes, Tiefenanalyse hinzufügen
        if is_tree:
            cluster_nodes = [queries[idx] for idx in indices]
            depths = [bfs_depth(node) for node in cluster_nodes]
            avg_depth = np.mean(depths)
            std_depth = np.std(depths)
        
            result_row["tiefe_avg"] = avg_depth
            result_row["tiefe_std"] = std_depth
            # Tiefenverteilung als Counter
            depth_counter = Counter(depths)
            
            # In CSV-taugliches Format umwandeln: z.B. "2:3, 3:5, 4:2"
            depth_dist_str = ", ".join(f"{depth}:{count}" for depth, count in sorted(depth_counter.items()))
            result_row["tiefe_verteilung"] = depth_dist_str
            # Tiefendiagramm
            plt.figure(figsize=(10, 5))
            sns.histplot(depths, kde=True, color="orchid", bins=15)
            plt.title(f"Tiefenverteilung für Cluster {cluster_id}")
            plt.xlabel("Tiefe")
            plt.ylabel("Anzahl der Knoten")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/cluster_{cluster_id}_depth_distribution.png")
            plt.close()


        # Satzlängen-Diagramm
        plt.figure(figsize=(10, 5))
        sns.histplot(sentence_lengths, kde=True, color="skyblue", bins=20)
        plt.title(f"Verteilung der Anfragenlängen für Cluster {cluster_id}")
        plt.xlabel("Anfragelänge")
        plt.ylabel("Anzahl der Anfragen")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cluster_{cluster_id}_sentence_length_distribution.png")
        plt.close()

        results.append(result_row)

    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/mikroauswertung.csv"
    df.to_csv(csv_path, index=False)
    plot_cluster_quality(df, output_dir)
    print(f"[✔] Mikro-Auswertung abgeschlossen: {len(results)} Cluster analysiert.")
    print(f"[📄] CSV gespeichert unter: {csv_path}")

def plot_cluster_quality(df, output_dir="micro_results"):
    """
    Erstellt ein Diagramm, das verschiedene Cluster-Qualitätsmetriken darstellt.
    
    Parameters:
        df (DataFrame): Die DataFrame, die alle Cluster-Metriken enthält.
        output_dir (str): Das Verzeichnis, in dem die Diagramme gespeichert werden.
    """
    # Clustergröße (Anzahl der Sätze)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['id'], y=df['anzahl_anfragen'], color='lightblue')
    plt.title("Clustergröße (Anzahl der Anffragen)")
    plt.xlabel("Cluster ID")
    plt.ylabel("Anzahl der Anfragen")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_size_distribution.png")
    plt.close()

    # Intra-Cluster-Kohärenz
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['id'], y=df['intra_kohärenz'], color='skyblue')
    plt.title("Intra-Cluster Kohärenz")
    plt.xlabel("Cluster ID")
    plt.ylabel("Intra-Cluster Kohärenz")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/intra_cluster_coherence.png")
    plt.close()

    # Durchschnittliche Satzlänge
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['id'], y=df['anfragenlänge_avg'], color='lightgreen')
    plt.title("Durchschnittliche Anfragelänge")
    plt.xlabel("Cluster ID")
    plt.ylabel("Durchschnittliche Anfragelänge")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_sentence_length.png")
    plt.close()

    # Standardabweichung der Satzlängen
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['id'], y=df['anfragenlänge_std'], color='salmon')
    plt.title("Standardabweichung der Anfragenlänge")
    plt.xlabel("Cluster ID")
    plt.ylabel("Standardabweichung der Anfragelänge")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentence_length_std.png")
    plt.close()

    print("[✔] Cluster-Qualitäts-Diagramme gespeichert.")


def macro_cluster_clustering(data_points, cluster_labels, output_dir):
    """
    Führt die Auswertung der KMeans-Clusterung durch, einschließlich Berechnung von Silhouetten-Score,
    Davies-Bouldin-Index, Clustergrößenverteilung und UMAP-Visualisierung.
    
    Parameters:
        data_points (array): Die Datenpunkte, die geclustert wurden.
        cluster_labels (array): Die Cluster-Zuordnungen der Datenpunkte.
        save_metrics_path (str): Der Pfad, unter dem die Metriken gespeichert werden.
        output_dir (str): Der Ordner, in dem die Diagramme und Ergebnisse gespeichert werden.
    
    Returns:
        tuple: (cluster_labels, sil_score, db_index)
    """
    # Metriken berechnen
    sil_score = silhouette_score(data_points, cluster_labels)
    db_index = davies_bouldin_score(data_points, cluster_labels)
    
    
    
    # Clustergrößen zählen
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # ✅ Metriken speichern
    save_metrics_path = f"{output_dir}/makroauswertung.txt"
    with open(save_metrics_path, "w") as f:
        f.write("Makro-Auswertung der Cluster:\n")
        f.write(f"Anzahl Cluster: {len(unique_labels)}\n")
        f.write(f"Silhouetten-Score: {sil_score:.4f}\n")
        f.write(f"Davies-Bouldin Index: {db_index:.4f}\n")
        f.write("Clustergrößen:\n")
        for label, count in zip(unique_labels, counts):
            f.write(f"  Cluster {label}: {count} Punkte\n")

    print(f"[✔] Metriken gespeichert in '{save_metrics_path}'")

    # 📊 Clustergrößenverteilung plotten
    plt.figure(figsize=(10, 5))
    if len(unique_labels) > 60:  # Wenn es viele Cluster sind, dann als Histogramm
        plt.hist(counts, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel("Clustergröße (Anzahl Punkte)")
        plt.ylabel("Häufigkeit")
        plt.title("Clustergrößenverteilung (Histogramm)")
    else:  # Wenn es weniger Cluster gibt, dann als Balkendiagramm
        plt.bar(unique_labels, counts, color='skyblue')
        plt.xlabel("Cluster-Label")
        plt.ylabel("Anzahl der Punkte")
        plt.title("Clustergrößenverteilung")
        plt.xticks(unique_labels)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_size_distribution.png")
    plt.close()

    # 📍 UMAP-Visualisierung
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(data_points)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1],
                    hue=cluster_labels, palette='tab20', s=50, alpha=0.8)
    plt.title("UMAP-Darstellung der Cluster")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    # Legende unter den Plot
    plt.legend(
        title="Cluster", bbox_to_anchor=(0.5, -0.3), loc='upper center',
        ncol=6, frameon=False
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_cluster_visualization.png")
    plt.close()

    return cluster_labels, sil_score, db_index

def get_clusters(queries, kmeans_labels):
    """
    Gibt die Anfragen zurück, die zu den jeweiligen Clustern gehören.

    Parameters:
        queries (list): Liste der ursprünglichen Anfragen.
        kmeans_labels (array): Die Labels der KMeans-Cluster (die Clusterzuordnung für jede Anfrage).

    Returns:
        dict: Ein Dictionary, das jedem Cluster die zugehörigen Anfragen zuordnet.
    """
    clustered_queries = {}

    # Durchlaufe die KMeans-Labels und ordne die Anfragen ihren Clustern zu
    for i, label in enumerate(kmeans_labels):
        if label not in clustered_queries:
            clustered_queries[label] = []
        clustered_queries[label].append(queries[i])

    return clustered_queries