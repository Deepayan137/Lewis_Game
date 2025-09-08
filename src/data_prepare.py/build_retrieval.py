import faiss
import requests, json
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
import argparse
import torch
import torchvision.transforms as T
import numpy as np
from scipy.spatial.distance import cdist
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel, CLIPProcessor
import random
from collections import defaultdict

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

DEFAULT_CLIP_MODEL = 'openai/clip-vit-large-patch14-336'

class HierarchicalClipRetriever():
    def __init__(self, 
        data_dir, 
        embed_dim = 768, 
        create_index = False, 
        batch_size = 6, 
        device = "cuda", 
        vis_feat_extractor='clip',
        catalogue_file="train_seed_42.json",
        dir_names=None,
        clip_model_name: str = DEFAULT_CLIP_MODEL):
        
        self.device = device
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.vis_feat_extractor = vis_feat_extractor
        self.clip_model_name = clip_model_name
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.root = os.path.dirname(data_dir)
        self.catalogue_path = os.path.join(data_dir, catalogue_file)
        self.class_index = faiss.IndexFlatIP(embed_dim)  # Class mean embeddings
        self.image_index = faiss.IndexFlatIP(embed_dim)  # Individual image embeddings
        
        if create_index:
            self._create_hierarchical_index()
        else:
            self._load_hierarchical_index()
    
    def _create_hierarchical_index(self):
        """Create both class-level and image-level indices"""
        with open(self.catalogue_path, 'r') as f:
            data = json.load(f)
        category = os.path.basename(self.data_dir)
        dir_names = data[category].keys()
        print(f"Creating hierarchical database from {self.data_dir}")
        self.class_id2name = {}  # class_id -> class_name
        self.image_id2info = {}  # image_id -> {'path': path, 'class_name': name, 'class_id': id}
        self.class_name2images = defaultdict(list)  # class_name -> list of image_ids
        
        class_id = 0
        image_id = 0
        for dir_name in tqdm(dir_names, desc="Processing classes"):
            filepaths = data[category][dir_name]['test']
            self.class_id2name[class_id] = dir_name
            image_features_list = []
            
            for file_path in filepaths:
                try:
                    image = Image.open(file_path)
                    inputs = self.feature_extractor(images=image, return_tensors="pt", padding=True)
                    image_features = self.clip_model.get_image_features(inputs["pixel_values"].to(self.device))
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    image_features = image_features.detach().cpu()

                    self.image_index.add(image_features.numpy())
                    self.image_id2info[image_id] = {
                        'path': file_path,
                        'class_name': dir_name,
                        'class_id': class_id
                    }
                    self.class_name2images[dir_name].append(image_id)
                    
                    image_features_list.append(image_features)
                    image_id += 1
                    image.close()
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

            if image_features_list:
                class_mean_features = torch.cat(image_features_list).mean(0, keepdim=True)
                self.class_index.add(class_mean_features.numpy())
            
            class_id += 1
        
        faiss.write_index(self.class_index, f"{self.data_dir}/class_means.faiss")
        faiss.write_index(self.image_index, f"{self.data_dir}/individual_images.faiss")
        
        with open(f'{self.data_dir}/class_mappings.json', 'w') as f:
            json.dump({
                'class_id2name': self.class_id2name,
                'image_id2info': self.image_id2info,
                'class_name2images': dict(self.class_name2images)
            }, f)
    
    def _load_hierarchical_index(self):
        """Load pre-built indices and mappings"""
        print(f"Loading hierarchical database from {self.data_dir}")
        
        self.class_index = faiss.read_index(f"{self.data_dir}/class_means.faiss")
        self.image_index = faiss.read_index(f"{self.data_dir}/individual_images.faiss")
        
        with open(f'{self.data_dir}/class_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.class_id2name = {int(k): v for k, v in mappings['class_id2name'].items()}
            self.image_id2info = {int(k): v for k, v in mappings['image_id2info'].items()}
            self.class_name2images = {k: v for k, v in mappings['class_name2images'].items()}
    
    def get_image_features(self, image_path):
        """Extract CLIP features for a single image"""
        image = load_image(image_path)
        inputs = self.feature_extractor(images=image, return_tensors="pt", padding=True)
        image_features = self.clip_model.get_image_features(inputs["pixel_values"].to(self.device))
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.detach().cpu().numpy()

    def get_alternate_query(self, query_image_path, remaining_paths, alpha=0.6, beta=0.8):
        # Extract query features
        query_features = self.get_image_features(query_image_path)  # (1, d)
        if query_features.ndim == 1:
            query_features = query_features[np.newaxis, :]  # ensure (1, d)
        remaining_features = [self.get_image_features(item) for item in remaining_paths]
        remaining_features = np.vstack(remaining_features)  # (N, d)
        similarity = 1 - cdist(query_features, remaining_features, 'cosine')  # (1, N)
        similarity = similarity.flatten()
        sorted_idx = np.argsort(similarity)
        median_idx = sorted_idx[0]
        return remaining_paths[median_idx]
    
    def hierarchical_distractor_sampling(self, query_image_path, num_distractors=4, 
                                       candidate_classes=8, selection_strategy='most_similar'):
        query_features = self.get_image_features(query_image_path)
        class_distances, class_indices = self.class_index.search(query_features, candidate_classes)
        class_distances = class_distances[0]  # Remove batch dimension
        class_indices = class_indices[0]
        query_class_name = self._get_class_from_path(query_image_path)
        distractor_candidates = []
        
        for class_idx, class_distance in zip(class_indices, class_distances):
            if class_idx < 0:  # Invalid index
                continue
                
            class_name = self.class_id2name[class_idx]
            if class_name == query_class_name:
                continue
            image_ids_in_class = self.class_name2images[class_name]
            if not image_ids_in_class:
                continue
            
            if selection_strategy == 'most_similar':
                best_image_id, best_similarity = self._find_most_similar_in_class(
                    query_features, image_ids_in_class
                )
            elif selection_strategy == 'random':
                best_image_id = random.choice(image_ids_in_class)
                best_similarity = class_distance
            else:
                if random.random() < 0.5:
                    best_image_id, best_similarity = self._find_most_similar_in_class(
                        query_features, image_ids_in_class
                    )
                else:
                    best_image_id = random.choice(image_ids_in_class)
                    best_similarity = class_distance
            
            distractor_candidates.append({
                'image_id': best_image_id,
                'image_path': self.image_id2info[best_image_id]['path'],
                'class_name': class_name,
                'similarity': best_similarity
            })
        distractor_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        selected_distractors = distractor_candidates[:num_distractors]
        return selected_distractors
    
    def _find_most_similar_in_class(self, query_features, image_ids_in_class):
        """Find the most similar image within a specific class"""
        best_similarity = -1
        best_image_id = None
        for image_id in image_ids_in_class:
            image_features = self.image_index.reconstruct(image_id).reshape(1, -1)
            similarity = np.dot(query_features, image_features.T)[0, 0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_image_id = image_id
        
        return best_image_id, best_similarity
    
    def _get_class_from_path(self, image_path):
        """Extract class name from image path - you might need to adapt this"""
        path_parts = image_path.split(os.sep)
        for part in path_parts:
            if part in self.class_id2name.values():
                return part
        return None
    
    def create_training_batch(self, query_image_path, remaining_paths, category, num_distractors=4):
        """
        Create a complete training batch for Lewis game
        Returns query path, distractor paths, and target index (always 0)
        """
        distractors = self.hierarchical_distractor_sampling(query_image_path, num_distractors)
        try:
            new_query_image_path = self.get_alternate_query(query_image_path, remaining_paths)
        except Exception as e:
            print(f"Problem at query: {query_image_path}")
            new_query_image_path = query_image_path
        all_image_paths = [new_query_image_path]
        all_image_paths.extend([d['image_path'] for d in distractors])
        random.shuffle(all_image_paths)
        target_index = all_image_paths.index(new_query_image_path)
        
        return {
            'image_paths': all_image_paths,
            'target_index': target_index,
            'query_path': query_image_path,
            'distractor_info': distractors,
            'category':category
        }

def main():
    parser = argparse.ArgumentParser(description="Create training batches for Lewis game")
    parser.add_argument('--root', type=str, default="/gpfs/projects/ehpc171/ddas/projects/YoLLaVA/yollava-data/train_")
    parser.add_argument('--catalogue_file', type=str, default="PerVA_catalogue_train_seed_23.json")
    parser.add_argument('--distractors', type=int, default=4)
    args = parser.parse_args()
    catalogue_path = os.path.join(args.root, args.catalogue_file)
    with open(catalogue_path, 'r') as f:
        data = json.load(f)
    categories = data.keys()
    all_data = []
    for category in categories:
        if category in ['clothe']:
            data_dir = os.path.join(args.root, category)
            concepts = data[category].keys()
            retriever = HierarchicalClipRetriever(
                data_dir=data_dir,
                create_index=True,  # Set to True for first time
                dir_names=concepts
            )
            concept_list = []
            for concept in tqdm(concepts):
                for image_path in data[category][concept]['test']:
                    remaining_paths = [item for item in data[category][concept]['test'] if item != image_path]
                    batch = retriever.create_training_batch(image_path, remaining_paths, category, num_distractors=args.distractors)
                    concept_list.append({
                        "query_path":batch['query_path'],
                        "ret_paths":batch['image_paths'],
                        'label':batch['target_index'],
                        'category':batch['category'],
                    })
            category_json_path = os.path.join(root, f'{category}_retrieval_top{args.distractors}_subset.json')
            with open(category_json_path, 'w') as f:
                json.dump(concept_list, f, indent=2)
            print(f"{category} data saved at {category_json_path}")
            all_data.extend(concept_list)


if __name__ == "__main__":
    main()