import pandas as pd
import torch
import numpy as np
from collections import defaultdict

class MMKGProcessor:
    def __init__(self, mmkg_path, image_map_path):
        self.mmkg_df = pd.read_csv(mmkg_path, header=None, 
            names=['image_id', 'relation', 'concept', 'image_name', 'description'])
        self.image_map_df = pd.read_csv(image_map_path, sep=' ', 
            names=['IID', 'Path'])
        
        # Create relation mappings
        self.relation_map = {
            'R66': 'uncertain',
            'R53': 'negative',
            'R62': 'positive'
        }
        
        # Build knowledge graph structure
        self.build_graph()
        
    def build_graph(self):
        self.image_to_concepts = defaultdict(list)
        self.concept_to_images = defaultdict(list)
        self.image_relations = defaultdict(dict)
        
        for _, row in self.mmkg_df.iterrows():
            img_id = row['image_id']
            concept = row['concept']
            relation = self.relation_map[row['relation']]
            
            self.image_to_concepts[img_id].append(concept)
            self.concept_to_images[concept].append(img_id)
            self.image_relations[img_id][concept] = relation
            
    def get_image_concepts(self, image_id):
        return self.image_to_concepts[image_id]
    
    def get_concept_images(self, concept):
        return self.concept_to_images[concept]
    
    def get_relation(self, image_id, concept):
        return self.image_relations[image_id].get(concept, None)
    
    def get_image_path(self, image_id):
        match = self.image_map_df[self.image_map_df['IID'] == image_id]
        return match['Path'].values[0] if not match.empty else None

    def build_adjacency_matrix(self, image_ids):
        """Build adjacency matrix for a batch of images"""
        n = len(image_ids)
        adj_matrix = torch.zeros((n, n))
        
        for i, img1 in enumerate(image_ids):
            for j, img2 in enumerate(image_ids):
                if i == j:
                    adj_matrix[i,j] = 1
                    continue
                    
                # Calculate similarity based on shared concepts
                concepts1 = set(self.get_image_concepts(img1))
                concepts2 = set(self.get_image_concepts(img2))
                shared = len(concepts1.intersection(concepts2))
                if shared > 0:
                    adj_matrix[i,j] = shared / len(concepts1.union(concepts2))
                    
        return adj_matrix