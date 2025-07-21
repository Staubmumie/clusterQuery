# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:55:43 2024

@author: mobau

Methoden zum Erstellen von Vektoren
"""
import re
import numpy as np
from typing import List, Tuple, Dict, Set, Any, Union
import matplotlib.pyplot as plt
from collections import Counter

from preprocessing.VectorizerTemplate import VectorizerTemplate


class DictionaryVectorizer(VectorizerTemplate):
    def __init__(
        self, 
        csv_path: str, 
        n_ev: int = 1, 
        n_att: int = 1,
        weight_event: float = 1.0, 
        weight_event_ngram: float = 1.0,
        weight_att_individual: float = 1.0, 
        weight_att_ngram: float = 1.0
    ) -> None:
        self.queries: List[str] = self.get_queries_from_csv(csv_path) 
        self.n_ev = n_ev
        self.n_att = n_att
        self.weight_event = weight_event
        self.weight_event_ngram = weight_event_ngram
        self.weight_att_individual = weight_att_individual
        self.weight_att_ngram = weight_att_ngram
        
        self.event_dictionary: Set[Tuple[str, ...]] = set()  
        self.att_dictionary: Set[Tuple[int, Tuple[str, ...]]] = set()
    
    def initialize_meta_vectors(self):
        self.meta_event_vector = np.zeros(len(self.event_dictionary), dtype=int)
        self.meta_attribute_vector = np.zeros(len(self.att_dictionary), dtype=int)
        
    def get_points(self) -> np.ndarray:
        self.build_event_dictionary()
        tokenized_queries = [self.tokenize_query_attributes(query) for query in self.queries]
        self.att_dictionary = self.generate_attribute_dict(tokenized_queries)
        
        self.initialize_meta_vectors()
        return np.array([self._vectorize_query(query) for query in self.queries])
    
    def build_event_dictionary(self) -> None:
        for query in self.queries:
            self.event_dictionary.update(self._generate_event_ngrams(query))
             
    def _generate_attribute_dict(
        self, tokenized_sentences: List[List[List[str]]]
    ) -> Set[Tuple[int, Tuple[str, ...]]]:
        att_set: Set[Tuple[int, Tuple[str, ...]]] = set()
        for sentence in tokenized_sentences:
            max_len = max(len(word) for word in sentence)
            for pos in range(max_len):
                tokens_at_pos = [word[pos] if pos < len(word) else "_" for word in sentence]
                max_n = min(self.n_att, len(tokens_at_pos))
                for gram_size in range(1, max_n + 1):
                    for start in range(len(tokens_at_pos) - gram_size + 1):
                        ngram = tokens_at_pos[start:start + gram_size]
                        var_map: Dict[str, str] = {}
                        normalized_ngram = tuple(self._normalize_variable(tok, var_map) for tok in ngram)
                        att_set.add((pos, normalized_ngram))
        return att_set

    def generate_attribute_dict(self, tokenized_sentences: List[Any]) -> Set[Tuple[int, Tuple[str, ...]]]:
        if tokenized_sentences and isinstance(tokenized_sentences[0][0], str):
            tokenized_sentences = [tokenized_sentences]
        return self._generate_attribute_dict(tokenized_sentences)
    
    def _generate_and_count_attribute_dict(
            self, tokenized_sentences: List[List[List[str]]]
        ) -> Dict[Tuple[int, Tuple[str, ...]], int]:
        att_dict: Dict[Tuple[int, Tuple[str, ...]], int] = {}
        for sentence in tokenized_sentences:
            max_len = max(len(word) for word in sentence)
            for pos in range(max_len):
                tokens_at_pos = [word[pos] if pos < len(word) else "_" for word in sentence]
                max_n = min(self.n_att, len(tokens_at_pos))
                for gram_size in range(1, max_n + 1):
                    for start in range(len(tokens_at_pos) - gram_size + 1):
                        ngram = tokens_at_pos[start:start + gram_size]
                        var_map: Dict[str, str] = {}
                        normalized_ngram = tuple(self._normalize_variable(tok, var_map) for tok in ngram)
                        if (pos, normalized_ngram) in att_dict:
                            att_dict[(pos, normalized_ngram)] += 1
                        else:
                            att_dict[(pos, normalized_ngram)] = 1
        return att_dict
    
    def sentence_to_attribute_vector(self, sentence: str) -> List[float]:
        sorted_keys = sorted(self.att_dictionary, key=lambda x: (x[0], x[1]))
        sentence_tokens = self.tokenize_query_attributes(sentence)
        sentence_att_dict = self._generate_and_count_attribute_dict([sentence_tokens])
        
        vector = np.zeros(len(sorted_keys), dtype=float)
        for i, key in enumerate(sorted_keys):
            if key in sentence_att_dict:
                # Gewichtung unter Berücksichtigung der Häufigkeit des Attributs
                weight = self.weight_att_individual * sentence_att_dict[key] if len(key[1]) == 1 else self.weight_att_ngram * sentence_att_dict[key] #* len(key[1])
                vector[i] = weight
                self.meta_attribute_vector[i] += 1
        return vector
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        event_ngrams = self._generate_and_count_event_ngrams(query)
        sorted_event_keys = sorted(self.event_dictionary, key=lambda x: (len(x), x))
        event_vector = np.zeros(len(sorted_event_keys), dtype=float)
        
        for i, key in enumerate(sorted_event_keys):
            if key in event_ngrams:
                event_vector[i] = self.weight_event * event_ngrams[key] if len(key) == 1 else self.weight_event_ngram * event_ngrams[key] #* len(key)
        
        self.meta_event_vector += (event_vector > 0).astype(int)
        attribute_vector = np.array(self.sentence_to_attribute_vector(query), dtype=float)
        
        return np.concatenate((event_vector, attribute_vector))
    
    def _generate_and_count_event_ngrams(self, query: str) -> Dict[Tuple[str, ...], int]:
        events = query.split()
        ngrams: Dict[Tuple[str, ...], int] = {}
        for n in range(1, self.n_ev + 1):
            for i in range(len(events) - n + 1):
                var_map: Dict[str, str] = {}
                normalized_ngram = tuple(self._normalize_event_token(event, var_map) for event in events[i:i+n])
                # Zähle, wie oft jedes N-Gramm vorkommt
                if normalized_ngram in ngrams:
                    ngrams[normalized_ngram] += 1
                else:
                    ngrams[normalized_ngram] = 1
        return ngrams
    
    def _generate_event_ngrams(self, query: str) -> Set[Tuple[str, ...]]:
        events = query.split()
        ngrams: Set[Tuple[str, ...]] = set()
        for n in range(1, self.n_ev + 1):
            for i in range(len(events) - n + 1):
                var_map: Dict[str, str] = {}
                normalized_ngram = tuple(self._normalize_event_token(event, var_map) for event in events[i:i+n])
                ngrams.add(normalized_ngram)
        return ngrams

    @staticmethod
    def tokenize_query_attributes(query: str) -> List[List[str]]:
        tokenized = [event.split(";") for event in query.split()]
        for attributes in tokenized:
            for idx, attr in enumerate(attributes):
                if attr == "":
                    attributes[idx] = "_"
        return tokenized
    
    def _normalize_variable(self, token: str, var_map: Dict[str, str]) -> str:
        if token.startswith("$"):
            if token not in var_map:
                var_map[token] = f"%{len(var_map)}"
            return var_map[token]
        return token

    def _normalize_event_token(self, token: str, var_map: Dict[str, str]) -> str:
        def repl(match):
            var = match.group(0)
            if var not in var_map:
                var_map[var] = f"%{len(var_map)}"
            return var_map[var]
        return re.sub(r'\$[A-Za-z0-9]+', repl, token)
    
    def get_indices_by_length(self, dictionary_type: str, length: int, return_elements: bool = False) -> Union[List[int], List[Tuple[int, Union[Tuple, List]]]]:
        # Wähle das richtige Dictionary basierend auf dem Typ
        dictionary = sorted(self.att_dictionary if dictionary_type == "att" else self.event_dictionary, key=lambda x: (len(x), x))
    
        # Wenn return_elements True ist, gib sowohl Indizes als auch die entsprechenden Elemente zurück
        if return_elements:
            if dictionary_type == "att":
                return [(i, dictionary[i]) for i in range(len(dictionary)) if len(dictionary[i][1]) == length]
            else:  # Event-Dictionary
                return [(i, dictionary[i]) for i in range(len(dictionary)) if len(dictionary[i]) == length]
        else:
            # Andernfalls gib nur die Indizes zurück
            if dictionary_type == "att":
                return [i for i, key in enumerate(dictionary) if len(key[1]) == length]
            else:  # Event-Dictionary
                return [i for i, key in enumerate(dictionary) if len(key) == length]
    
    def get_meta_vector_values(self, dictionary_type: str, length: int, return_elements: bool = False) -> Union[np.ndarray, List[Tuple[int, Union[Tuple, List]]]]:
        # Holen der Indizes (und eventuell der Elemente)
        indices_or_elements = self.get_indices_by_length(dictionary_type, length, return_elements)
    
        # Wenn return_elements auf True gesetzt ist, gib sowohl den Meta-Vektor-Wert als auch das Dictionary-Element zurück
        if return_elements:
            if dictionary_type == "att":
                return [(self.meta_attribute_vector[i], element) for i, element in indices_or_elements]
            else:  # Event-Dictionary
                return [(self.meta_event_vector[i], element) for i, element in indices_or_elements]
    
        # Ansonsten nur die Meta-Vektor-Werte
        if dictionary_type == "att":
            return self.meta_attribute_vector[indices_or_elements]
        else:  # Event-Dictionary
            return self.meta_event_vector[indices_or_elements]

    def test_function_and_plot(self, num_iterations):
        # Initialisiere eine Liste für alle Vektoren
        all_vectors = []

        # Schleife über die Anzahl der Iterationen
        for i in range(1, num_iterations + 1):
            # Hole den Vektor für den aktuellen Wert von i
            vector = self.get_meta_vector_values('att', i, False)
            all_vectors.append(vector)

            # Ausgabe der Vektorgröße
            print(f"Größe des Vektors für i={i}: {len(vector)}")

            # Zähle die Häufigkeit der einzelnen Zahlen im Vektor
            counter = Counter(vector)
            print(f"Zählung der Vektorelemente für i={i}: {dict(counter)}")

            # Visualisierung des Vektors als Histogramm
            plt.figure(figsize=(10, 6))
            plt.hist(vector, bins=np.arange(min(vector), max(vector) + 2) - 0.5, edgecolor='black', alpha=0.7)
            plt.title(f"Histogramm der Vektorelemente für i={i}")
            plt.xlabel('Werte')
            plt.ylabel('Häufigkeit')
            plt.grid(True)
            plt.show()