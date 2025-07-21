# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:05:38 2024

@author: mobau

Methoden um die Bäume zu einem Vektoren umzuwandeln
"""
import numpy as np
from sklearn.manifold import MDS
import csv
from collections import deque
import heapq

from preprocessing.VectorizerTemplate import VectorizerTemplate

class TreeVectorizer(VectorizerTemplate):
    """
    Diese Klasse erstellt aus den entgegen genommenen Anfragen Vektoren, indem 
    die Vektoren mithilfe einer Baumstruktur erstellt werden.
    """
    def __init__(self, csv_path, term_weight= 1, variablen_weight=1, length_weigth=1.0):
        """
        Initialisiert die Klasse indem die Baumstruktur mithilfe der CSV-Datei gebaut wird.
        
        Args:
            csv_path: Pfad der zur CSV-Datei führt
        """
        treestructure = TreeStructure(csv_path, term_weight, variablen_weight, length_weigth)
        self.queries = treestructure.get_queries()
        self.root = treestructure.get_root()

    def get_distance_matrix(self) -> np.ndarray:
        """
        Berechnet die Distanzmatrix für eine Liste von Abfragen (queries) und gibt sie als numpy-Array zurück.
        
        Args:
            root: Der Wurzelknoten des Baums, von dem aus die Distanzen berechnet werden.
            queries: Eine Liste von Knoten oder Elementen, für die Distanzen berechnet werden.
        
        Returns:
            np.ndarray: Eine nxn-Distanzmatrix als numpy-Array.
        """
        n = len(self.queries)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    # Eingabe der verschiedenen Distanzmetriken
                    matrix[i, j] = self.wu_palmer_ansatz(self.queries[i], self.queries[j])
        
        return matrix

    
    def wu_palmer_ansatz(self, node1, node2):
        """
        Berechnet die Wu-Palmer-Ähnlichkeit zwischen zwei Knoten in einem Baum.
    
        Args:
            node1 (TreeNode): Der erste Knoten.
            node2 (TreeNode): Der zweite Knoten.
    
        Returns:
            float: Der Wu-Palmer-Ähnlichkeitswert zwischen node1 und node2.
        """
        nca = TreeVectorizer.find_nearest_common_ancestor(node1, node2)
        
        root_to_nca = TreeVectorizer.distance_from_ancestor(nca, self.root )        
        nca_to_node1 = TreeVectorizer.distance_from_ancestor(node1, nca)
        nca_to_node2 = TreeVectorizer.distance_from_ancestor(node2, nca)
    
        top = 2 * root_to_nca
        bottom = nca_to_node1 + nca_to_node2 + 2 * root_to_nca
       
        if bottom == 0:
            return 0.0
    
        return top / bottom
    
    def get_points(self, n_components=2):
        distance_matrix = self.get_distance_matrix()
        mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42)
        points = mds.fit_transform(distance_matrix)
        return points

    
    
    @staticmethod
    def distance_from_ancestor(start, ancestor):
        """
        Berechnet die minimalen Kosten für den Abstand von einem Knoten zu einem Vorfahren.
        """
        queue = [(0, id(start), start)]  # (Kosten, id(Knoten), Knoten)
        costs = {start: 0}

        while queue:
            cost, _, current = heapq.heappop(queue)

            if current == ancestor:
                return cost
            if current != None:
                for parent, weight in current.parents:
                    new_cost = cost + weight
                    if parent not in costs or new_cost < costs[parent]:
                        costs[parent] = new_cost
                        heapq.heappush(queue, (new_cost, id(parent), parent))  # id(parent) als eindeutiger Vergleichswert

        return float('inf')
    
    @staticmethod
    def find_nearest_common_ancestor(target1, target2):
       """
       Findet den kleinsten gemeinsamen Vorfahren mit der kürzesten Distanz zwischen den beiden Knoten.
       Falls mehrere Knoten denselben Abstand haben, wird der mit den geringsten kombinierten Kosten gewählt.
       """
       if not target1 or not target2:
           return None

       def bfs_levels(start):
           queue = deque([(start, 0, 0)])  # (Knoten, Level, Kosten)
           visited = {}
           
           while queue:
               node, level, cost = queue.popleft()
               if node not in visited or level < visited[node][0] or (level == visited[node][0] and cost < visited[node][1]):
                   visited[node] = (level, cost)
                   for parent, weight in node.parents:
                       queue.append((parent, level + 1, cost + weight))
           
           return visited

       levels1 = bfs_levels(target1)
       levels2 = bfs_levels(target2)
       
       common_ancestors = set(levels1.keys()) & set(levels2.keys())
       if not common_ancestors:
           return None
       
       return min(common_ancestors, key=lambda node: (max(levels1[node][0], levels2[node][0]), levels1[node][1] + levels2[node][1]))

    

class TreeStructure:
    def __init__(self,csv_file_path, term_weight, variablen_weight, length_weigth):
        self.root = None
        self.querie_list = []  # Liste der Blattknoten (Knoten, die niemals als Elternknoten genutzt wurden)
        self.term_weight= term_weight
        self.variablen_weight= variablen_weight
        self.length_weigth = length_weigth
        data = self.read_csv(csv_file_path)
        self.build_tree(data)
            

    class TreeNode:
        """
        Knoten der Struktur, der ausschließlich die Elternbeziehungen verwaltet.
        """
        def __init__(self, name, weights):
            self.name = name
            self.parents = []  # Liste der Elternknoten als Tupel (parent_node, weight)
            self.used_as_parent = False  # Kennzeichen, ob dieser Knoten als Elternknoten für einen anderen genutzt wird
            self.weights = weights

        def add_parent(self, parent_node, variablenmerkmal='$'):
            """
            Fügt diesem Knoten einen Elternknoten hinzu und berechnet dabei eine Gewichtung.
            
            Args:
                parent_node (TreeNode): Der hinzuzufügende Elternknoten.
                variablenmerkmal (str): Zeichenfolge zur Kennzeichnung von Variabilität. Standard: '$'.
            """
            weight = self.get_weight(parent_node, variablenmerkmal)
            self.parents.append((parent_node, weight))
            # Markiere den übergebenen Knoten als verwendet als Elternknoten
            parent_node.used_as_parent = True

        def get_weight(self, parent_node, variablenmerkmal):
            """
            Berechnet eine Gewichtung zwischen diesem Knoten und einem potenziellen Elternknoten.
            
            Hierbei wird beispielhaft der Unterschied der Knotennamen ausgewertet.
            """
            len_child= len(self.name.split(' '))
            len_parent= len(parent_node.name.split(' '))
            term_child, var_child = TreeStructure.count_attributes(self.name)
            term_parent, var_parent = TreeStructure.count_attributes(parent_node.name)
            
            dif_length = len_child - len_parent
            dif_term = term_child - term_parent
            dif_var = var_child - var_parent
            
            weight = self.weights.term_weight * dif_term + self.weights.variablen_weight * dif_var + self.weights.length_weigth * dif_length
            
            return weight

        def __repr__(self, level=0):
            indent = "  " * level
            result = f"{indent}- {self.name}\n"
            #for parent, weight in self.parents:
             #   result += f"{indent}  <- {parent.name} (Weight: {weight})\n"
            return result
    
    @staticmethod
    def count_attributes(s1):
        term_counter = 0
        var_counter = 0

        s1 = s1.replace(' ', ';')
        s1= s1.split(';')
        
        for item in s1:
            if item != '':
                if '$' in item:  # Zähle Elemente, die '$' enthalten
                    var_counter += 1
                else:
                    term_counter += 1
        
        return term_counter, var_counter


    def build_tree(self, data):
        """
        Baut die Struktur aus einer Liste von Dictionaries auf.
        Erwartet wird, dass jedes Dictionary die Schlüssel 'Vertex' und 'Parents' enthält.
        
        Für jeden Datensatz wird:
          - der Knoten (Vertex) erstellt (falls noch nicht vorhanden),
          - die angegebenen Elternknoten werden ermittelt und dem Knoten hinzugefügt,
          
        Zum Schluss wird in der Klassenvariablen 'querie_list' die Liste aller Knoten gespeichert,
        die **niemals** als Elternknoten für einen anderen Knoten genutzt wurden (also die Blätter).
        
        Args:
            data (list): Liste von Dictionaries mit den Schlüsseln 'Vertex' und 'Parents'.
        
        Returns:
            TreeNode: Ein exemplarischer Wurzelknoten (erster Knoten, der als Elternknoten nicht verwendet wurde)
                      oder None, falls keiner existiert.
        """
        nodes = {}

        for row in data:
            vertex = row['Vertex'].strip()
            #if vertex == '':
             #   print('war hier')
              #  vertex = '__'
            parents = []
            if not row['Parents'].strip():
                parents = ['']
            else:
                parents = [p.strip() for p in row['Parents'].split('||') if p.strip()]
            

            # Erstelle den Knoten, falls er noch nicht existiert
            if vertex not in nodes:
                nodes[vertex] = TreeStructure.TreeNode(vertex,self)

            # Für jeden angegebenen Elternknoten:
            for parent in parents:
                if parent not in nodes:
                    nodes[parent] = TreeStructure.TreeNode(parent,self)
                # Füge den Elternknoten zum aktuellen Knoten hinzu
                nodes[vertex].add_parent(nodes[parent])
        nodes[''].parents = []
        
        self.root = nodes['']
        # Sammle alle Knoten, die nicht als Elternknoten verwendet wurden (Blattknoten)
        self.querie_list = [node for node in nodes.values() if not node.used_as_parent]


        

    def read_csv(self, file_path):
        """
        Liest eine CSV-Datei ein und erstellt eine Liste von Dictionaries.
        Erwartet wird eine CSV-Datei mit den Spalten 'Vertex' und 'Parents', getrennt durch ';'.
        
        Args:
            file_path (str): Pfad zur CSV-Datei.
        
        Returns:
            list: Liste von Dictionaries mit den Schlüsseln 'Vertex' und 'Parents'.
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            header_line = file.readline().strip()
            fieldnames = header_line.split(';')
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter=';')
            if 'Vertex' not in reader.fieldnames or 'Parents' not in reader.fieldnames:
                raise ValueError("Die CSV-Datei benötigt die Spalten 'Vertex' und 'Parents'.")
            for row in reader:
                data.append({'Vertex': row['Vertex'], 'Parents': row['Parents']})
        return data

    def get_root(self):
        return self.root

    def get_queries(self):
        return sorted(self.querie_list, key=lambda x: x.name)

