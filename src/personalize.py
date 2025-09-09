import faiss
import json
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import os
import json
import random
import argparse
import sys
import re
sys.path.insert(0, 'src')
from gen_des import SimpleImageDataset, create_data_loader, setup_model, speaker_describes_batch, extract_answer_content

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
# INSERT_YOUR_CODE
from torch.utils.data import Dataset, DataLoader
# test comment
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
        if not self.json_path:
            self.json_path = "/gpfs/projects/ehpc171/ddas/projects/YoLLaVA/yollava-data/train_/train_test_val_seed_42_num_train_1.json"
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

def get_prompt(descriptions, category):
    answer_format = {
            "Reasoning": "<Your reasoning in 1-2 sentences.>",
            "Answer": f"<name of the object>",
        }
    prompt = f"""
            You are provided with a query image containing a {category} object along with the name and detailed distinguishing features of several {category} objects.
            Below are the name and their descriptions:
            {json.dumps(descriptions, indent=2)}
            Your Task:
            - Carefully compare the {category} object in the query image with the provided description(s):
            - Identify the name of object in the query image by matching the features and other characteristics mentioned in the descriptions.
            - Your response should contain the name of the identified object and nothing else.
            - **Ignore superficial details** such as background, pose and lighting.
            - Please provide a reasoning for your answer generate your response with JSON format {json.dumps(answer_format)}.
            Any deviation from this format will be considered incorrect. Output only the JSON response, without any additional text.
            """
    # prompt += ("\n\nOutput the answer in <answer> </answer> tags.\nThe output answer format should be as follows:\n"
    #                   "<answer> ... </answer>\nPlease strictly follow the format.")
    return prompt

def prepare_test_retrieval_items(args, desc_path, retriever):
    test_ret_path = f'outputs/{args.data_name}/{args.category}/test_ret_{args.model_type}.json'
    # if not os.path.exists(test_ret_path):
    # output_path = f'outputs/{args.data_name}/{args.category}/llm_judge_prompt_{args.model_type}.json'
    # if os.path.exists(output_path):
    #     print(f"Output file {output_path} already exists. Loading and returning existing data.")
    #     with open(output_path, 'r') as f:
    #         return json.load(f)
    print("Preparing ret data")
    dataset = SimpleImageDataset(
        json_path=os.path.join(f'manifests/{args.data_name}', args.catalog_file),
        category=args.category,
        split="test",
        seed=args.seed
    )
    with open(desc_path, 'r') as f:
        capt_dict = json.load(f)
    new_items = []
    # import pdb;pdb.set_trace()
    for item in tqdm(dataset, desc="Generating prompts"):
        path = item['path']
        # Only load image if needed elsewhere; not used here
        results = retriever.image_search(path, k=5)
        # Collect unique names and their descriptions
        descriptions, ret_paths = [], []
        for result in results:
            ret_paths.append(result['path'])
            descriptions.append(capt_dict[result['name']])
        # Defensive: fallback if no descriptions found
        if not descriptions:
            descriptions = ["No description available."]
        prompt = get_prompt(descriptions, args.category)
        new_items.append(
            {'problem':prompt, 
            'path':item['path'], 
            'solution':item["name"],
            'solution_desc':capt_dict[item['name']],
            'ret_path':ret_paths})
    # Write once at the end
    # with open(test_ret_path, 'w') as f:
    #     json.dump(new_items, f, indent=2)
    # else:
    #     print('Loading ret data')
    #     with open(test_ret_path, 'r') as f:
    #         new_items = json.load(f)
    return new_items


class DictListDataset(Dataset):
    def __init__(self, dict_list):
        self.data = dict_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def dict_collate_fn(batch):
    # Batch is a list of dictionaries; return as-is or stack fields if needed
    return batch

def extract_answer_term(text: str, term: str) -> str:
    """
    Extracts the value for a given term from the text.
    It first tries to match a quoted value, then an unquoted word.
    """
    patterns = {
        'Answer': r'"Answer":\s*(?:"([^"]+)"|([\w-]+))',
        'Confidence': r'"Confidence":\s*(?:"([^"]+)"|([\w.]+))',
        'Choice': r'"Choice":\s*(?:"([^"]+)"|([\w-]+))',
        'A': r'"A":\s*(?:"([^"]+)"|([\w-]+))',
        'B': r'"B":\s*(?:"([^"]+)"|([\w-]+))',
        'C': r'"C":\s*(?:"([^"]+)"|([\w-]+))',
        'D': r'"D":\s*(?:"([^"]+)"|([\w-]+))',
        'E': r'"E":\s*(?:"([^"]+)"|([\w-]+))',
        'F': r'"F":\s*(?:"([^"]+)"|([\w-]+))',
        # 'Caption': r'"Caption":\s*(?:"([^"]+)"|([\w-]+))'
    }
    pattern = patterns.get(term)
    if not pattern:
        return None
    match = re.search(pattern, text)
    if match:
        return (match.group(1) or match.group(2)).strip()
    else:
        # Fallback if regex doesn't match.
        parts = text.split(term)
        if parts:
            return re.sub(r'[^a-zA-Z0-9\s]', '', parts[-1]).strip()
        return None

# Example usage
if __name__ == "__main__":
    # Initialize retriever
    parser = argparse.ArgumentParser(description='Optimized Lewis Game Evaluation')
    parser.add_argument("--data_name", type=str, default='PerVA',
                       help='name of the dataset')
    parser.add_argument("--catalog_file", type=str, default="test_catalog_seed_23.json",
                       help="Path to the catalog JSON file")
    parser.add_argument("--category", type=str, default='clothe',
                       help='Model type: original or finetuned')
    parser.add_argument("--seed", type=int, default=23,
                       help='random seed')
    parser.add_argument("--model_type", type=str, default='original',
                       help='Model type: original or finetuned')                   
    args = parser.parse_args()
    
    retriever = SimpleClipRetriever(category=args.category, 
        json_path=os.path.join(f'manifests/{args.data_name}', args.catalog_file),
        create_index=True,
        seed=args.seed)
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    if args.model_type == 'original':
        desc_path = f'outputs/PerVA/{args.category}/descriptions_original.json'
    else:
        # model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2.5-VL-2B-Instruct_GRPO_lewis_{args.category}"
        desc_path = f'outputs/PerVA/{args.category}/descriptions_finetuned.json'
    
    with open(desc_path) as f:
        capt_dict = json.load(f)    
    # test_ret_path = f'outputs/{args.data_name}/{args.category}/test_ret_{args.model_type}.json'
    ret_items = prepare_test_retrieval_items(args, desc_path, retriever)
    print(f"Loading model from {model_path}")
    model, processor = setup_model(model_path)
    dataset = DictListDataset(ret_items)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=dict_collate_fn, num_workers=4)
    # INSERT_YOUR_CODE
    correct = []
    all_responses = []
    results = []
    for batch in tqdm(data_loader, desc="Generating model responses"):
        images = [item['image'] if 'image' in item else Image.open(item['path']).convert('RGB') for item in batch]
        problems = [item['problem'] for item in batch]
        names = [item['solution'] for item in batch]
        sol_descs = [item['solution_desc'] for item in batch]
        paths = [item['path'] if 'path' in item else None for item in batch]
        ret_paths = [item['ret_path'] for item in batch]
        responses = speaker_describes_batch(model, processor, images, problems, max_new_tokens=128)
        # pred_names = [extract_answer_content(item) for item in responses]
        pred_names = [extract_answer_term(item, "Answer").lower().strip() for item in responses]
        # Ensure responses is a list
        if isinstance(pred_names, str):
            pred_names = [pred_names]
        for item in zip(paths, names, problems, responses, pred_names, ret_paths, sol_descs):
            path, name, problem, response, pred, ret_path, sol_desc = item
            correct.append(int(pred == name))
            results.append({
                "image_path": path,
                "problem": problem,
                "solution": name,
                "solution_desc":sol_desc,
                "ret_paths":ret_path,
                "response": response,
                "pred_name": pred
            })
        torch.cuda.empty_cache()
    savedir = f'results/{args.data_name}/{args.category}'
    os.makedirs(savedir, exist_ok=True)
    results_path = f'{savedir}/results_{args.model_type}.json'
    accuracy = sum(correct) / len(correct)
    output = {
        "accuracy": accuracy,
        "results": results
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Accuracy:{accuracy}")