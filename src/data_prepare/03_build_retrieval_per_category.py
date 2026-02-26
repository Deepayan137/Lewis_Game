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
import re
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
        out_dir,
        embed_dim = 768,
        # create_index = False, 
        batch_size = 6, 
        device = "cuda", 
        vis_feat_extractor='clip',
        catalog_file="train_catalog_seed_42.json",
        split="train",
        dir_names=None,
        clip_model_name: str = DEFAULT_CLIP_MODEL, with_negative=False):
        
        self.device = device
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.split = split
        self.vis_feat_extractor = vis_feat_extractor
        self.clip_model_name = clip_model_name
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.catalog_path = os.path.join(catalog_file)
        self.class_index = faiss.IndexFlatIP(embed_dim)  # Class mean embeddings
        self.image_index = faiss.IndexFlatIP(embed_dim)  # Individual image embeddings
        self.with_negative = with_negative

    def _create_hierarchical_index(self, category, seed):
        """Create both class-level and image-level indices"""
        with open(self.catalog_path, 'r') as f:
            data = json.load(f)
        dir_names = data[category].keys()
        print(f"Creating hierarchical database")
        os.makedirs(os.path.join(self.out_dir, category), exist_ok=True)
        self.class_id2name = {}  # class_id -> class_name
        self.image_id2info = {}  # image_id -> {'path': path, 'class_name': name, 'class_id': id}
        self.class_name2images = defaultdict(list)  # class_name -> list of image_ids
        
        class_id = 0
        image_id = 0
        for dir_name in tqdm(dir_names, desc="Processing classes"):
            if self.with_negative:
                filepaths = data[category][dir_name]['negative']
            else:
                filepaths = data[category][dir_name][self.split]
            # filepaths = random.sample(filepaths, 10)
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
        class_means_vdb = "class_means_with_neg.faiss" if self.with_negative else "class_means.faiss"
        individual_images_vdb = "individual_images_with_neg.faiss" if self.with_negative else "individual_images.faiss"
        class_mappings_json = "class_mappings_with_neg.json" if self.with_negative else "class_mappings.json" 
        faiss.write_index(self.class_index, f"{self.out_dir}/{category}/seed_{seed}/{class_means_vdb}")
        faiss.write_index(self.image_index, f"{self.out_dir}/{category}/seed_{seed}/{individual_images_vdb}")
        
        with open(f'{self.out_dir}/{category}/seed_{seed}/{class_mappings_json}', 'w') as f:
            json.dump({
                'class_id2name': self.class_id2name,
                'image_id2info': self.image_id2info,
                'class_name2images': dict(self.class_name2images)
            }, f)
    
    def _load_hierarchical_index(self, category, seed):
        """Load pre-built indices and mappings"""
        print(f"Loading hierarchical database from {self.out_dir}/seed_{seed}/{category}")
        class_means_vdb = "class_means_with_neg.faiss" if self.with_negative else "class_means.faiss"
        individual_images_vdb = "individual_images_with_neg.faiss" if self.with_negative else "individual_images.faiss"
        class_mappings_json = "class_mappings_with_neg.json" if self.with_negative else "class_mappings.json"
        self.class_index = faiss.read_index(f"{self.out_dir}/{category}/seed_{seed}/{class_means_vdb}")
        self.image_index = faiss.read_index(f"{self.out_dir}/{category}/seed_{seed}/{individual_images_vdb}")
        
        with open(f'{self.out_dir}/{category}//seed_{seed}/{class_mappings_json}', 'r') as f:
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

    def get_alternate_query(self, query_image_path, remaining_paths, alpha=0.6, beta=0.8,
                            random_negative=False, easy_pos_prob: float = 0.0):
        if not remaining_paths:
            return None

        if random_negative:
            return random.choice(remaining_paths)

        # Extract query features
        query_features = self.get_image_features(query_image_path)  # (d,) or (1, d)
        if query_features is None:
            return None
        if query_features.ndim == 1:
            query_features = query_features[np.newaxis, :]  # ensure (1, d)

        # Extract remaining features and stack
        remaining_features = [self.get_image_features(item) for item in remaining_paths]
        # Filter out any None features just in case
        valid = [(p, f) for p, f in zip(remaining_paths, remaining_features) if f is not None]
        if not valid:
            return None
        paths_valid, feats_valid = zip(*valid)
        remaining_features = np.vstack(feats_valid)  # (N, d)

        # Cosine similarity: 1 - cosine_distance
        similarity = 1 - cdist(query_features, remaining_features, 'cosine')  # (1, N)
        similarity = similarity.flatten()  # (N,)

        # indices for most and least similar
        most_sim_idx = int(np.argmax(similarity))
        least_sim_idx = int(np.argmin(similarity))

        # Decide based on easy_pos_prob
        if easy_pos_prob <= 0.0:
            chosen_idx = least_sim_idx
        elif easy_pos_prob >= 1.0:
            chosen_idx = most_sim_idx
        else:
            if random.random() < float(easy_pos_prob):
                chosen_idx = most_sim_idx
            else:
                chosen_idx = least_sim_idx

        return paths_valid[chosen_idx]

    
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
    
    def _find_most_dissimilar_in_class(self, query_features, image_ids_in_class, k=4, reverse=False):
        """
        Find the k most dissimilar (least similar) images within a specific class.

        Args:
            query_features: numpy array, (1, dim)
            image_ids_in_class: list of image ids
            k: number of most dissimilar images to return

        Returns:
            List of tuples: [(image_id, similarity), ...] sorted by increasing similarity (most dissimilar first)
        """
        similarities = []
        for image_id in image_ids_in_class:
            image_features = self.image_index.reconstruct(image_id).reshape(1, -1)
            similarity = np.dot(query_features, image_features.T)[0, 0]
            similarities.append((image_id, similarity))
        # Sort by similarity (ascending, so least similar are first)
        similarities.sort(key=lambda x: x[1], reverse=reverse)
        worst_image_id = [item for item, _ in similarities]
        return worst_image_id[:k]
    
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
    
    def create_training_batch(self, query_image_path, remaining_paths, category, num_distractors=4,
                              random_negative=False, selection_strategy='most_similar',
                              easy_pos_prob=0.0):
        """
        Create a complete training batch for Lewis game.

        Standard mode:
            - Speaker sees: query_image_path (single view)
            - Listener pool: [hard_positive (alt view), neg1, neg2]
        """
        if self.with_negative:
            distractors = self.hierarchical_distractor_sampling_with_negatives(
                query_image_path, num_distractors, selection_strategy=selection_strategy)
        else:
            distractors = self.hierarchical_distractor_sampling(
                query_image_path, num_distractors, selection_strategy=selection_strategy)

        
        try:
            new_query_image_path = self.get_alternate_query(
                query_image_path, remaining_paths,
                random_negative=random_negative, easy_pos_prob=easy_pos_prob)
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
            'query_path': query_image_path,  # single path in standard mode
            'distractor_info': distractors,
            'category': category,
        }

def main():
    parser = argparse.ArgumentParser(description="Create training batches for Lewis game")
    parser.add_argument('--category', type=str, default="clothe")
    parser.add_argument('--model_type', type=str, default="original_7b")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--catalog_file', type=str, default="manifests/PerVA/train_combined_concepts_seed_23.json")
    parser.add_argument('--distractors', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='outputs/PerVA', help='Optional output directory')
    parser.add_argument('--random_negative', action='store_true', default=False, help='Use random negative sampling (default: False)')
    parser.add_argument('--with_negative', action='store_true', default=False, help='use negatives sourced from LAION')
    parser.add_argument('--easy_pos_prob', type=float, default=0.0)
    parser.add_argument('--for_recognition', action='store_true', default=False, help='If set, create data for recognition task')
    args = parser.parse_args()
    args.dataset = os.path.basename(args.out_dir)
    seed = args.seed
    catalog_path =args.catalog_file
    with open(catalog_path, 'r') as f:
        data = json.load(f)
    category = args.category
    os.makedirs(os.path.join(args.out_dir, category, f'seed_{seed}'), exist_ok=True)
    all_data = []
    concepts = data[category].keys()
    with_negative = args.with_negative
    retriever = HierarchicalClipRetriever(
        out_dir=args.out_dir,
        catalog_file=args.catalog_file,
        split=args.split,
        dir_names=concepts,
        with_negative=with_negative)
    out_path = f'{args.out_dir}/{category}/seed_{seed}/'
    class_mappings_json = "class_mappings_with_neg.json" if with_negative else "class_mappings.json"
    print("Creating Index")
    retriever._create_hierarchical_index(category, seed)
    concept_list = []
    for concept in tqdm(concepts):
        selection_strategy = 'most_similar'
        image_paths = data[category][concept][args.split]
        for image_path in image_paths:
            remaining_paths = [item for item in data[category][concept][args.split] if item != image_path]
            batch = retriever.create_training_batch(
                image_path, remaining_paths, category,
                num_distractors=args.distractors,
                random_negative=args.random_negative,
                selection_strategy=selection_strategy,
                easy_pos_prob=args.easy_pos_prob)
            concept_list.append({
                "query_path": batch['query_path'],
                "ret_paths": batch['image_paths'],
                'label': batch['target_index'],
                'category': batch['category'],
            })
    if args.for_recognition:
        import sys;sys.path.append('./src')
        from inference_utils.prompts import get_description_prompt
        from inference_utils.common  import PERVA_CATEGORY_MAP, set_seed, get_model_config
        from inference_utils.model import setup_model, speaker_describes_batch
        from inference_utils.cleanup import parse_descriptions
        from copy import deepcopy
        from generate_descriptions import process_batch_efficiently
        from PIL import Image
        set_seed(args.seed)
        model_config = get_model_config(
            args.model_type,
            dataset='PerVA',
            seed=args.seed,
        )
        model_name_or_path = model_config['path']
        use_peft = model_config['use_peft']
        print(f"Loading model from {model_name_or_path} (use_peft={use_peft})")
        speaker_model, processor = setup_model(model_name_or_path, use_peft=use_peft)
        # concept_list_copy = concept_list.deepcopy()
        batch_items = []
        # concept_list = concept_list[:4]
        num_samples = len(concept_list)
        num_refs = []
        for item in concept_list:
            CATEGORY = PERVA_CATEGORY_MAP.get(item['category'], item['category'])
            problem = get_description_prompt(CATEGORY)
            num_refs.append(len(item['ret_paths']))
            for ref_path in item['ret_paths']:
            # ref_path = item['ret_paths'][0]
                name = ref_path.split('/')[-2]
                batch_items.append({
                    'image': Image.open(ref_path).convert('RGB'),
                    'name': name,
                    'problem': problem,
                    'path': ref_path
                })
        raw_results = process_batch_efficiently(speaker_model, processor, batch_items=batch_items, max_new_tokens=100)
        idx = 0  # Running counter for indexing raw_results
        for i in range(num_samples):
            ret_descs = []
            for j in range(num_refs[i]):
                category, ref_path, desc = raw_results[idx]
                ref_concept_name = ref_path.split('/')[-2]
                parsed = parse_descriptions(desc[0])
                coarse = parsed["coarse"] or f"A photo of a {category}"
                detailed = parsed["detailed"]
                ref_concept_description = f"Name: {ref_concept_name}\n" + "Description: " + coarse + " " + detailed
                ret_descs.append(ref_concept_description)
                idx += 1  # Increment after processing each reference
            concept_list[i]['ret_descs'] = ret_descs
    K = args.distractors + 1
    base_catalog = os.path.basename(args.catalog_file)
    subset_match = re.search(r'subset_(\d+)', base_catalog)
    split_tag = f"subset_{subset_match.group(1)}" if subset_match else ""

    if base_catalog.startswith("validation") and split_tag:
        save_file = f'val_retrieval_top{K}_{split_tag}.json'
    elif split_tag:
        save_file = f'retrieval_top{K}_{split_tag}_{args.model_type}.json'
    else:
        save_file = f'retrieval_top{K}.json'

    if args.easy_pos_prob != 0.0:
        save_file = save_file.replace('.json', f'_easy_{args.easy_pos_prob}.json')
    category_json_path = os.path.join(args.out_dir, args.category, f'seed_{seed}', save_file)
    with open(category_json_path, 'w') as f:
        json.dump(concept_list, f, indent=2)
    print(f"{category} data saved at {category_json_path}")
    all_data.extend(concept_list)


if __name__ == "__main__":
    main()