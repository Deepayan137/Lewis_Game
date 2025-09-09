import os
import faiss
import json
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

class SimpleClipRetriever:
    def __init__(self, 
                 embed_dim=768,
                 create_index=False,
                 batch_size=6,
                 dataset='PerVA',
                 json_path=None,
                 category='decoration',
                 device="cuda",
                 clip_model="openai/clip-vit-large-patch14-336",
                 seed=42):
        
        self.device = device
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.category = category
        self.seed = seed
        random.seed(seed)
        self.target_dir = f'outputs/{dataset}/{category}'
        self.json_path = json_path
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(clip_model)
        
        # Single FAISS index for all images
        self.index = faiss.IndexFlatIP(embed_dim)
        
        # Mapping from index ID to image path
        self.id2path = {}
        if create_index:
            self._create_index()
        else:
            self._load_index()
        
    def _create_index(self):
        json_path = self.json_path
        with open(json_path, 'r') as f:
            data = json.load(f)
        category = self.category
        dir_names = data[category].keys()
        image_paths = []
        for dir_name in tqdm(dir_names, desc="Processing classes"):
            filepaths = data[category][dir_name]['train']
            file_path = random.choice(filepaths)
            image_paths.append(file_path)
        print(f"Creating index from {len(image_paths)} images")
        
        all_features = []
        valid_paths = []
        
        # Process images in batches
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_features = self._extract_batch_features(batch_paths)
            
            if batch_features is not None:
                all_features.append(batch_features)
                valid_paths.extend(batch_paths)
        
        if not all_features:
            raise ValueError("No valid images found to create index")
        
        # Concatenate all features
        all_features = np.vstack(all_features)
        
        # Add to FAISS index
        self.index.add(all_features)
        
        # Create ID to path mapping
        self.id2path = {i: path for i, path in enumerate(valid_paths)}
        
        # Save index if path provided
        self.save_index()
        
        print(f"Index created with {self.index.ntotal} images")
    
    def _extract_batch_features(self, image_paths):
        """Extract features for a batch of images"""
        batch_images = []
        valid_paths = []
        
        # Load and validate images
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                batch_images.append(image)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if not batch_images:
            return None
        
        try:
            # Process batch
            inputs = self.feature_extractor(
                images=batch_images, 
                return_tensors="pt", 
                padding=True
            )
            
            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(
                    inputs["pixel_values"].to(self.device)
                )
                # Normalize features
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.detach().cpu().numpy()
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None
    
    def get_image_features(self, image_path):
        """Extract CLIP features for a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(
                    inputs["pixel_values"].to(self.device)
                )
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.detach().cpu().numpy()
        
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def image_search(self, query_image_path, k=5):
        """
        Search for top-k most similar images
        
        Args:
            query_image_path: Path to query image
            k: Number of results to return
            
        Returns:
            List of dictionaries with 'path' and 'similarity' keys
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Please create index first.")
        
        # Extract query features
        query_features = self.get_image_features(query_image_path)
        if query_features is None:
            return []
        
        # Search index
        similarities, indices = self.index.search(query_features, k)
        
        # Format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append({
                    'path': self.id2path[idx],
                    'name': self.id2path[idx].split('/')[-2],
                    'similarity': float(sim)
                })
        
        return results

    def save_index(self, ):
        """Save the index and mappings to disk"""
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.target_dir, "simple_index.faiss"))
        
        # Save path mappings
        with open(os.path.join(self.target_dir, "id2path.json"), 'w') as f:
            json.dump(self.id2path, f)
        
        print(f"Index saved to {self.target_dir}")
    
    def load_index(self,):
        """Load index and mappings from disk"""
        # Load FAISS index
        index_path = os.path.join(self.target_dir, "simple_index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load path mappings
        mapping_path = os.path.join(self.target_dir, "id2path.json")
        with open(mapping_path, 'r') as f:
            loaded_mappings = json.load(f)
            # Convert string keys back to int
            self.id2path = {int(k): v for k, v in loaded_mappings.items()}
        
        print(f"Index loaded from {target_dir} with {self.index.ntotal} images")