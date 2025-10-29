import os
import faiss
import json
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from transformers import CLIPModel, CLIPTextModel, CLIPProcessor

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
                 seed=42,
                 db_type='original'):
        
        self.device = device
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.category = category
        self.seed = seed
        self.db_type = db_type
        random.seed(seed)
        self.target_dir = f'outputs/{dataset}/{category}/seed_{seed}'
        self.json_path = json_path
        # FIX: 'args' was undefined - use 'seed' parameter instead
        self.description_json = Path("outputs") / dataset / category / f"seed_{seed}" / f"descriptions_{db_type}.json"
        
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(clip_model)
        
        # Single FAISS index for all images
        self.index = faiss.IndexFlatIP(embed_dim)
        self.text_index = faiss.IndexFlatIP(embed_dim)
        
        # Mapping from index ID to image path
        self.id2path = {}
        self.id2concept = {}  # FIX: Add mapping for text concepts
        
        if create_index:
            self._create_image_index()  # FIX: Method name typo
            self._create_text_index()
        else:
            self._load_index()

    def _create_text_index(self):
        # FIX: Use self.description_json instead of self.json_path
        json_path = self.description_json
        
        if not os.path.exists(json_path):
            print(f"Warning: Description file not found at {json_path}")
            return
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        concepts = list(data.keys())
        descriptions = []
        lengths = []
        for concept in concepts:
            desc = [data[concept]['general'][0] + ' ' + data[concept]['distinguishing features'][0].split('.')[0]]
            descriptions.append(desc)
            lengths.append(len(desc))
        
        print(f"Creating text index from {len(descriptions)} descriptions")
        
        all_features = []
        # FIX: Use descriptions instead of image_paths
        for i in tqdm(range(0, len(descriptions), self.batch_size), desc="Processing descriptions"):
            batch_descs = descriptions[i:i + self.batch_size]
            batch_lengths = lengths[i:i + self.batch_size]
            batch_features = self._extract_batch_text_features(batch_descs, batch_lengths)
            if batch_features is not None:
                all_features.append(batch_features)

        if not all_features:
            print("Warning: No valid text features extracted")
            return
        all_features = np.vstack(all_features)
        self.text_index.add(all_features)
        
        # Create ID to concept mapping
        self.id2concept = {i: concept for i, concept in enumerate(concepts)}
        
        self.save_index(index_type='text')

    def _create_image_index(self):
        json_path = self.json_path
        
        if json_path is None or not os.path.exists(json_path):
            raise ValueError(f"JSON path not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        category = self.category
        
        if category not in data:
            raise ValueError(f"Category '{category}' not found in JSON data")
            
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
            batch_features = self._extract_batch_image_features(batch_paths)
            
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
        self.save_index(index_type='image')
        
        print(f"Index created with {self.index.ntotal} images")
    
    def _extract_batch_text_features(self, texts, lengths):
        if not texts:
            return None
        # Flatten texts if nested
        if isinstance(texts[0], list):
            texts = [item for sublist in texts for item in sublist]
        try:
            inputs_text = self.feature_extractor(
                text=texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            with torch.no_grad():
                text_embeddings = self.clip_model.get_text_features(
                    input_ids=inputs_text['input_ids']
                )
                text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True) 

            # Now, compute the mean of every k items, where k is from lengths
            embeddings_list = []
            idx = 0
            for k in lengths:
                if k == 0:
                    continue
                chunk = text_embeddings[idx:idx+k]
                if chunk.size(0) == 0:
                    continue
                mean_emb = chunk.mean(dim=0, keepdim=True)
                embeddings_list.append(mean_emb)
                idx += k
            if len(embeddings_list) == 0:
                return None
            # Stack to shape (len(lengths), embedding_dim)
            final_embeddings = torch.cat(embeddings_list, dim=0)
            # Ensure the size(0) matches len(lengths)
            assert final_embeddings.size(0) == len(lengths), f"Expected {len(lengths)} embeddings, got {final_embeddings.size(0)}"
            return final_embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return None
        
    def _extract_batch_image_features(self, image_paths):
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
    
    def image_search_on_text(self, query_image_path, k=5, return_paths=False):
        """
        Search text index using image query
        
        Args:
            query_image_path: Path to query image
            k: Number of results to return
            return_paths: If True, return image paths instead of concepts
            
        Returns:
            List of dicts with 'concept'/'path', 'name', 'similarity'
        """
        if self.text_index.ntotal == 0:
            print("Warning: Text index is empty")
            return []
            
        query_features = self.get_image_features(query_image_path)
        if query_features is None:
            return []
        
        similarities, indices = self.text_index.search(query_features, k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx < len(self.id2concept):
                concept = self.id2concept[idx]
                img_path = self.id2path[idx]
                # if return_paths:
                #     # Convert concept to image paths
                #     image_paths = self._get_images_for_concept(concept)
                #     for img_path in image_paths:
                results.append({
                    'path': img_path,
                    'name': img_path.split('/')[-2],
                    'concept': concept,
                    'similarity': float(sim)
                })
                # else:
                #     results.append({
                #         'concept': concept,
                #         'similarity': float(sim)
                #     })
        return results
    
    def _get_images_for_concept(self, concept):
        """Get all image paths for a given concept"""
        image_paths = []
        
        if self.json_path is None:
            return image_paths
        
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            if self.category in data and concept in data[self.category]:
                # Get train images
                image_paths.extend(data[self.category][concept].get('train', []))
                # Optionally get test images too
                # image_paths.extend(data[self.category][concept].get('test', []))
        except Exception as e:
            print(f"Error getting images for concept {concept}: {e}")
        
        return image_paths

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
    
    def hybrid_search(self, 
                     query_image_path, 
                     k=10,
                     alpha=0.5,
                     normalization='minmax',
                     aggregation='weighted_sum',
                     image_k=None,
                     text_k=None):
        """
        Combine image and text search results to produce a unified ranked list.
        
        This method calls image_search() and image_search_on_text() internally,
        then combines their similarities to create a final ranking.
        
        Args:
            query_image_path: Path to query image
            k: Number of final results to return
            alpha: Weight for image search (0-1). text_weight = 1-alpha
            normalization: How to normalize scores before combining
                - 'minmax': Scale to [0,1] range
                - 'zscore': Standardize using mean and std
                - 'none': Use raw scores
            aggregation: How to combine scores
                - 'weighted_sum': alpha * img_score + (1-alpha) * text_score
                - 'max': Take maximum of the two scores
                - 'min': Take minimum of the two scores
                - 'product': img_score * text_score
                - 'harmonic_mean': 2*img*text/(img+text)
            image_k: Number of results to fetch from image_search (default: 2*k)
            text_k: Number of results to fetch from text_search (default: 2*k)
        
        Returns:
            List of dicts with:
                - 'path': Image path
                - 'name': Image name (directory name)
                - 'image_similarity': Similarity from image search
                - 'text_similarity': Similarity from text search
                - 'combined_score': Final combined score
        """
        # Default to fetching more results than needed
        if image_k is None:
            image_k = min(k * 2, self.index.ntotal) if self.index.ntotal > 0 else k
        if text_k is None:
            text_k = min(k * 2, self.text_index.ntotal) if self.text_index.ntotal > 0 else k
        
        # Get results from both search methods
        image_results = []
        text_results = []
        
        if self.index.ntotal > 0:
            image_results = self.image_search(query_image_path, k=image_k)
        
        if self.text_index.ntotal > 0:
            text_results = self.image_search_on_text(query_image_path, k=text_k, return_paths=True)
        # Handle edge cases
        if not image_results and not text_results:
            return []
        if not image_results:
            return text_results[:k]
        if not text_results:
            return image_results[:k]
        
        # Build a unified dictionary: path -> {similarities}
        path_scores = {}
        
        # Add image search results
        for res in image_results:
            path = res['path']
            path_scores[path] = {
                'path': path,
                'name': res['name'],
                'image_similarity': res['similarity'],
                'text_similarity': 0.0  # Default if not found in text search
            }
        
        # Add/update with text search results
        for res in text_results:
            path = res['path']
            if path in path_scores:
                # Path found in both searches
                path_scores[path]['text_similarity'] = res['similarity']
            else:
                # Path only in text search
                path_scores[path] = {
                    'path': path,
                    'name': res['name'],
                    'image_similarity': 0.0,
                    'text_similarity': res['similarity']
                }
        
        # Convert to list
        combined_results = list(path_scores.values())
        
        # Normalize scores if requested
        if normalization != 'none' and len(combined_results) > 1:
            combined_results = self._normalize_scores(combined_results, method=normalization)
        
        # Compute combined scores
        for res in combined_results:
            # Use normalized scores if available, otherwise raw scores
            img_sim = res.get('image_similarity_norm', res['image_similarity'])
            txt_sim = res.get('text_similarity_norm', res['text_similarity'])
            
            if aggregation == 'weighted_sum':
                res['combined_score'] = alpha * img_sim + (1 - alpha) * txt_sim
            
            elif aggregation == 'max':
                res['combined_score'] = max(img_sim, txt_sim)
            
            elif aggregation == 'min':
                res['combined_score'] = min(img_sim, txt_sim)
            
            elif aggregation == 'product':
                res['combined_score'] = img_sim * txt_sim
            
            elif aggregation == 'harmonic_mean':
                if img_sim + txt_sim > 0:
                    res['combined_score'] = 2 * img_sim * txt_sim / (img_sim + txt_sim)
                else:
                    res['combined_score'] = 0.0
            
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top k
        return combined_results[:k]
    
    def _normalize_scores(self, results_list, method='minmax'):
        """
        Normalize similarity scores across all results.
        
        Args:
            results_list: List of result dicts with 'image_similarity' and 'text_similarity'
            method: 'minmax' or 'zscore'
        
        Returns:
            Updated results_list with '_norm' fields added
        """
        if not results_list:
            return results_list
        
        img_sims = [r['image_similarity'] for r in results_list]
        txt_sims = [r['text_similarity'] for r in results_list]
        
        if method == 'minmax':
            # Scale to [0, 1]
            img_min, img_max = min(img_sims), max(img_sims)
            txt_min, txt_max = min(txt_sims), max(txt_sims)
            
            for res in results_list:
                if img_max - img_min > 0:
                    res['image_similarity_norm'] = (res['image_similarity'] - img_min) / (img_max - img_min)
                else:
                    res['image_similarity_norm'] = res['image_similarity']
                
                if txt_max - txt_min > 0:
                    res['text_similarity_norm'] = (res['text_similarity'] - txt_min) / (txt_max - txt_min)
                else:
                    res['text_similarity_norm'] = res['text_similarity']
        
        elif method == 'zscore':
            # Z-score normalization: (x - mean) / std
            img_mean = np.mean(img_sims)
            img_std = np.std(img_sims)
            txt_mean = np.mean(txt_sims)
            txt_std = np.std(txt_sims)
            
            for res in results_list:
                if img_std > 0:
                    res['image_similarity_norm'] = (res['image_similarity'] - img_mean) / img_std
                else:
                    res['image_similarity_norm'] = 0.0
                
                if txt_std > 0:
                    res['text_similarity_norm'] = (res['text_similarity'] - txt_mean) / txt_std
                else:
                    res['text_similarity_norm'] = 0.0
        
        return results_list

    def save_index(self, index_type='image'):
        """Save the index and mappings to disk"""
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Save FAISS index
        if index_type == 'image':
            faiss.write_index(self.index, os.path.join(self.target_dir, "simple_image_index.faiss"))
            # Save path mappings
            with open(os.path.join(self.target_dir, "id2path.json"), 'w') as f:
                json.dump(self.id2path, f)
            
            print(f"Image index saved to {self.target_dir}")
        elif index_type == 'text':
            faiss.write_index(self.text_index, os.path.join(self.target_dir, "simple_text_index.faiss"))
            # FIX: Save concept mappings for text index
            with open(os.path.join(self.target_dir, "id2concept.json"), 'w') as f:
                json.dump(self.id2concept, f)
            print(f"Text index saved to {self.target_dir}")
    
    def _load_index(self):  # FIX: Method name had wrong prefix
        """Load index and mappings from disk"""
        # Load image FAISS index
        image_index_path = os.path.join(self.target_dir, "simple_image_index.faiss")
        if os.path.exists(image_index_path):
            self.index = faiss.read_index(image_index_path)
            
            # Load path mappings
            mapping_path = os.path.join(self.target_dir, "id2path.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    loaded_mappings = json.load(f)
                    # Convert string keys back to int
                    self.id2path = {int(k): v for k, v in loaded_mappings.items()}
                print(f"Image index loaded from {self.target_dir} with {self.index.ntotal} images")
            else:
                print(f"Warning: id2path.json not found at {mapping_path}")
        else:
            print(f"Warning: Image index not found at {image_index_path}")
        
        # Load text FAISS index
        text_index_path = os.path.join(self.target_dir, "simple_text_index.faiss")
        if os.path.exists(text_index_path):
            self.text_index = faiss.read_index(text_index_path)
            
            # Load concept mappings
            concept_mapping_path = os.path.join(self.target_dir, "id2concept.json")
            if os.path.exists(concept_mapping_path):
                with open(concept_mapping_path, 'r') as f:
                    loaded_mappings = json.load(f)
                    self.id2concept = {int(k): v for k, v in loaded_mappings.items()}
                print(f"Text index loaded from {self.target_dir} with {self.text_index.ntotal} entries")
            else:
                print(f"Warning: id2concept.json not found at {concept_mapping_path}")
        else:
            print(f"Warning: Text index not found at {text_index_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reasoning-based personalization evaluation")
    parser.add_argument("--data_name", type=str, default="YoLLaVA")
    parser.add_argument("--catalog_file", type=str, default="main_catalog_seed_23.json")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--concept_name", type=str, default="bo")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--db_type", type=str, default="original", choices=["original", "original_7b", "finetuned", 'finetuned_7b', 'lora_finetuned_7b'])
    # parser.add_argument("--model_type", type=str, default="base_qwen", choices=["base_qwen", "ft_qwen"])
    # parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--k_retrieval", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    def set_seed(seed: int) -> None:
        """Set seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    manifests_dir = Path("manifests") / args.data_name
    description_json = Path("outputs") / args.data_name / args.category / f"seed_{args.seed}" / f"descriptions_{args.db_type}.json"
    catalog_json = str(manifests_dir / args.catalog_file)
    if not description_json.exists():
        LOG.error("Descriptions file not found at %s", description_json)
        raise FileNotFoundError(f"Descriptions file not found: {description_json}")

    # initialize retriever
    retriever = SimpleClipRetriever(
        dataset=args.data_name,
        category=args.category,
        json_path=catalog_json,
        create_index=True,
        seed=args.seed,
        db_type=args.db_type
    )

    with open(catalog_json) as f:
        data = json.load(f)
    data = data[args.category]
    concepts = data.keys()
    all_paths = []
    top3_acc = 0
    count=0
    # concepts = ['chicken_bean_bag']
    for concept in tqdm(concepts):
        image_paths = data[concept]['test']
        # all_paths.extend(image_paths)
        for im_path in image_paths:
            results = retriever.hybrid_search(
                    query_image_path=im_path,
                    k=3,
                    alpha=0.5,
                    normalization='minmax',
                    aggregation='weighted_sum',
                    image_k=40,
                    text_k=40
                )
            top_3 = [r["name"] for r in results]
            if concept in top_3:
                top3_acc +=1
            count+=1
    acc = top3_acc/count
    print(f"Top 3:{top3_acc}")
    print(f"Total: {count}")
    print(f'Accuracy:{acc}')