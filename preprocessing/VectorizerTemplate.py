# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:26:06 2024

@author: mobau
"""
import csv
from abc import ABC, abstractmethod


class VectorizerTemplate(ABC):
    queries=[]
    @abstractmethod
    def get_points(self):
        """
        Eine abstrakte Methode, die in den abgeleiteten Klassen überschrieben werden muss.
        """
        pass
    
    @staticmethod
    def get_queries_from_csv(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')
            if 'Vertex' not in reader.fieldnames or 'Parents' not in reader.fieldnames:
                raise ValueError("Die CSV-Datei benötigt die Spalten 'Vertex' und 'Parents'.")
            
            vertices = set()
            parents = set()
            
            for row in reader:
                vertex = row['Vertex'].strip()
                if vertex:
                    vertices.add(vertex)
                
                if row['Parents'].strip():
                    parents.update(p.strip() for p in row['Parents'].split('||') if p.strip())
            
        return [vertex for vertex in vertices if vertex not in parents]



