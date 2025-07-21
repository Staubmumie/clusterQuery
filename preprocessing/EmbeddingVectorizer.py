# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:48:58 2024

@author: mobau
"""
import fasttext
import numpy as np
import os
from preprocessing.VectorizerTemplate import VectorizerTemplate

        
class EmbeddingVectorizer(VectorizerTemplate):
    """
    Diese Klasse erstellt aus den entgegen genommenen Anfragen Vektoren, indem es die Semantik vergleicht
    """
    
    def __init__(self, csv_path, n_min = 3, n_max = 6, import_model = None) -> None:
        """
            Initialisierung des EmbeddingVectorizer durch Angabe des CSV-Pfades

            Args:
                csv_path: Pfad der zur CSV-Datei führt 
        """
        self.queries = self.get_queries_from_csv(csv_path)
        self.query_embeddings=[]
        
        # Workaround: Erstellung einer Textdatei, weil das Training über Textdateien funktioniert
        query_data = "\n".join(self.queries)
        temp_file_name = "temp_queries.txt"
        with open(temp_file_name, "w", encoding="utf-8") as f:
            f.write(query_data)
        
        if import_model == None:
            self.model = fasttext.train_unsupervised('temp_queries.txt', model='skipgram', minCount=1, minn=n_min,maxn=n_max)
            self.model.save_model('result\semantik\fasttext_model1.bin')
        else:
            self.model= import_model
        
        os.remove(temp_file_name)    
        
    def get_points(self):
        for query in self.queries:
            events = query.split()
            
            event_embeddings = [self.model.get_word_vector(event) for event in events]
            
            if event_embeddings: 
                query_embedding = np.mean(event_embeddings, axis=0)
            else:
                query_embedding = np.zeros(self.model.get_dimension())  # Null-Vektor, falls der Satz leer ist
            
            self.query_embeddings.append(query_embedding)
            
        return self.query_embeddings
            
            
    
        






