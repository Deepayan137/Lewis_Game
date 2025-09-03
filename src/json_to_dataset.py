import os
import time
import json
import sys
from datasets import DatasetDict, Dataset
from PIL import Image
from tqdm import *
import argparse
from tqdm import *
def save_dataset(dataset_dict, save_path):
    # save DatasetDict to your disk
    dataset_dict.save_to_disk(save_path)

def load_dataset(save_path):
    return DatasetDict.load_from_disk(save_path)

def json_to_dataset(json_file_path):
    """
    Reads a JSON file, processes images by resizing and padding them to a target size,
    and returns a Hugging Face DatasetDict.
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Initialize lists to hold the processed data
    query_images = []
    ret_paths = []
    solutions = []

    # Iterate through each record to process images in pairs
    idx = []
    categories = []
    speaker_problems =[]
    for i, item in enumerate(tqdm(data)):
        # 1. Load the grid_image to get the target dimensions
        query_image = Image.open(item['query_path']).convert('RGB')
        target_size = query_image.size
        
        category = item['category']
        speaker_problem = f"""Describe the {category} in the image so that it can be distinguished from other {category} objects. DO NOT mention the background or location of the object. Output in the format: "{category}: [concise distinguishing qualities]"
        Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags. The output answer format should be as follows: <think> ... </think> <answer> ... </answer>. Please strictly follow the format.
        """
        query_images.append(query_image)
        # ret_image = [Image.open(x).convert('RGB') for x in item['ret_paths'] ]
        ret_paths.append(item['ret_paths'])
        speaker_problems.append(speaker_problem)
        solutions.append(item['label'])
        categories.append(item['category'])
        uid = os.path.basename(item['query_path']).split('.')[0]
        idx.append(uid)
    # Create the dictionary for the dataset
    dataset_dict = {
        'image': query_images,
        'ret_paths': ret_paths,
        'speaker_problem':speaker_problems,
        'solution': solutions,
        'category':categories,
        'example_idx':idx
    }

    # Convert to a Hugging Face Dataset and then a DatasetDict
    dataset = Dataset.from_dict(dataset_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })

    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine MC-LLaVA data and convert to dataset format.")
    parser.add_argument('--category', type=str, default='clothe',
                        help='artifact of the data (e.g., cartoon, person).')
    args = parser.parse_args()
    json_path = f"/gpfs/projects/ehpc171/ddas/projects/YoLLaVA/yollava-data/train_/{args.category}.json"
    dataset_dict = json_to_dataset(json_path)
    save_path = f'/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_data/PerVA_{args.category}'
    os.makedirs(save_path, exist_ok=True)
    save_dataset(dataset_dict, save_path)
    print("Dataset saved...Now loading")
    data = load_dataset(save_path)


