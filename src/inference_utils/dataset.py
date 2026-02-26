from PIL import Image
import json
import random
from tqdm import tqdm
try:
    from torch.utils.data import Dataset, DataLoader
except Exception:
    Dataset = object  # fallback if torch not installed
    DataLoader = None
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import sys
sys.path.insert(0, 'src/')
from inference_utils.common import YOLLAVA_CATEGORY_MAP, MYVLM_CATEGORY_MAP

class DictListDataset(Dataset):
    """Wrap a list of dicts into a torch Dataset."""

    def __init__(self, dict_list: Sequence[Dict[str, Any]]):
        self.data = list(dict_list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def dict_collate_fn(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identity collate that returns the list of dicts as-is."""
    return list(batch)

class SimpleImageDataset(Dataset):
    """
    Simple dataset that returns PIL images based on index
    """
    def __init__(self, category, json_path=None, split='train', seed=42, data_name="PerVA"):
        """
        Initialize the dataset
        
        Args:
            category: Category name (e.g., 'animals', 'vehicles', etc.)
            json_path: Path to the JSON file containing image paths
            split: Data split to use ('train', 'val', 'test')
        """
        self.category = category
        self.split = split
        self.json_path = json_path
        self.data_name = data_name
        random.seed(seed)
        self.image_paths = self._load_image_paths()
        logging.info(f"Dataset initialized with {len(self.image_paths)} images from category '{category}' ({split} split)")
    
    def _load_image_paths(self):
        """Load image paths from JSON file"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        if self.category not in data:
            raise ValueError(f"Category '{self.category}' not found in JSON data. Available categories: {list(data.keys())}")
        
        dir_names = data[self.category].keys()
        image_paths = []
        
        for dir_name in tqdm(dir_names, desc=f"Loading {self.split} image paths"):
            if self.split not in data[self.category][dir_name]:
                logging.warning(f"Warning: Split '{self.split}' not found for directory '{dir_name}'. Skipping.")
                continue
                
            filepaths = data[self.category][dir_name][self.split]
            if filepaths and self.split == 'train': #sampling one image per concept when split == train
                file_path = random.choice(filepaths)
                image_paths.append(file_path)
            else:
                image_paths.extend(filepaths)
        
        return image_paths
    
    def __len__(self):
        """Return the total number of images"""
        return len(self.image_paths)
    
    def __repr__(self):
        return f"<SimpleImageDataset category={self.category} split={self.split} n={len(self.image_paths)}>"

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
        image_path = self.image_paths[idx]
        name = image_path.split('/')[-2]
        with Image.open(image_path) as _img:
            image = _img.convert("RGB")
        if self.data_name == "YoLLaVA":
            category = YOLLAVA_CATEGORY_MAP[name]
        elif self.data_name == "MyVLM":
            category = MYVLM_CATEGORY_MAP[name]
        else:
            category = self.category
        # problem = (
        #     f'Provide descriptions of the {category} in the image in four parts:\n'
        #     f'1. Coarse: A 5-6 word description starting with "A photo of a {category}"\n'
        #     f'2. Detailed: Describe ONLY permanent identity features (color, patterns, markings, shape, facial features, eye color, build, etc.). '
        #     f'Write one sentence with "The {category}" highlighting 3-4 permanent attributes.\n'
        #     f'3. State: Describe pose and position of the {category} in the image (eg. lying, open, closed, sitting, standing, running, hanging, folded, etc.).\n'
        #     f'4. Location: Describe positioning and background (outside, inside, on the floor, on the shelf/table, near objects, background elements etc.).\n\n'
        #     'Examples:\n\n'
        #     'Example 1 (cat):\n'
        #     '<thinking>I need to separate permanent features from temporary state. The cat has white and brown fur with green eyes - these are identity features. It is sitting with paws tucked - this is state. It is on a wooden floor - this is location.</thinking>\n'
        #     '<coarse>A photo of a cat</coarse>\n'
        #     '<detailed>The cat has a white chest and face with brown fur on its back and ears, bright green eyes, and a distinctive pink nose.</detailed>\n'
        #     '<state>Sitting upright with front paws tucked under its body</state>\n'
        #     '<location>On a wooden floor near a window</location>\n\n'
        #     'Example 2 figurine:\n'
        #     '<thinking>Permanent features include shape, color and pattern on the figurine. Where and how it is placed are not permanent, so they go in location and state respectively.</thinking>\n'
        #     '<coarse>A photo of a deer shaped ceramic figurine</coarse>\n'
        #     '<detailed>The figurine is shaped like a deer with four legs and two antlers, featuring brown coloring on its upper body with yellow specks and white on its underside.</detailed>\n'
        #     '<state>lying hrizontally</state>\n'
        #     '<location>on the shelf</location>\n\n'
        #     f'Now describe the {category} in the image following this format:\n'
        #     "<thinking>Your reasoning</thinking>\n"
        #     f"<coarse>A photo of a {category}</coarse>\n"
        #     f"<detailed>The {category} ...</detailed>\n"
        #     "<state>...</state>\n"
        #     "<location>...</location>"
        # )
        problem = (
            f'Provide two descriptions of the {category} in the image:\n'
            f'1. A coarse 5-6 word description starting with "A photo of a "\n'
            f'2. A detailed description: Describe the {category} so it can be distinguished from other {category}s. '
            "Do NOT mention background, location or state. "
            "If the image contains a person, avoid mentioning clothing or accessories. "
            f'Write exactly one fluent sentence beginning with "The " and highlighting 3-4 visible distinguishing attributes. '
            "Keep it concise and natural, without lists or brackets.\n\n"
            "Output format:\n"
            "<thinking>Your reasoning</thinking>\n"
            f"<coarse>A photo of a ...</coarse>\n"
            f"<detailed>The ...</detailed>"
        )
        return {
            'image': image,
            'problem': problem,
            'path': image_path,
            'name':name,
            'index': idx
        }

def create_data_loader(dataset, batch_size=4, shuffle=False, num_workers=4):
    def collate_fn(batch):
        return batch  # Return as-is since we handle batching manually
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        num_workers=num_workers
    )

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Optimized Lewis Game Evaluation')
    parser.add_argument("--data_name", type=str, default='PerVA',
                       help='name of the dataset')
    parser.add_argument("--catalog_file", type=str, default="main_catalog.json",
                       help="Path to the catalog JSON file")
    parser.add_argument("--category", type=str, default='clothe',
                       help='Model type: original or finetuned')
    parser.add_argument("--seed", type=int, default=23,
                    help='random seed')
    args = parser.parse_args()

    dataset = SimpleImageDataset(
        json_path=os.path.join(f'manifests/{args.data_name}', args.catalog_file),
        category=args.category,
        split="train",
        seed=args.seed,
        data_name=args.data_name
    )

    data_loader = create_data_loader(dataset, batch_size=4, shuffle=False)
    for batch in data_loader:
        # Example: print batch info
        print(f"Batch size: {len(batch)}")
        for item in batch:
            print(f"Image path: {item['path']}, Name: {item['name']}")