import json
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class PredictionAnalyzer:
    def __init__(self, original_json_path, finetuned_json_path, output_dir="analysis"):
        self.original_json_path = original_json_path
        self.finetuned_json_path = finetuned_json_path
        self.output_dir = output_dir
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary output directories"""
        for case in ['ftr', 'ogr']:
            os.makedirs(f"{self.output_dir}/{case}/images", exist_ok=True)
            os.makedirs(f"{self.output_dir}/{case}/json", exist_ok=True)
    
    def load_json_files(self):
        """Load both JSON files"""
        with open(self.original_json_path, 'r') as f:
            original_data = json.load(f)
        
        with open(self.finetuned_json_path, 'r') as f:
            finetuned_data = json.load(f)
            
        return original_data, finetuned_data
    
    def extract_object_name(self, ret_path):
        """Extract object name from path (basename without extension)"""
        return ret_path.split('/')[-2]
    
    def compare_predictions(self, original_data, finetuned_data):
        """Compare predictions and identify case1 and case2 items"""
        # Create lookup dict for finetuned data by image_path
        finetuned_lookup = {item['image_path']: item for item in finetuned_data}
        
        case1_items = []  # finetuned right, original wrong
        case2_items = []  # original right, finetuned wrong
        
        for orig_item in original_data:
            img_path = orig_item['image_path']
            
            # Find matching finetuned item
            if img_path not in finetuned_lookup:
                continue
                
            ft_item = finetuned_lookup[img_path]
            
            # Check if solution matches predictions
            orig_correct = orig_item['pred_name'] == orig_item['solution']
            ft_correct = ft_item['pred_name'] == ft_item['solution']
            
            # Case 1: finetuned right, original wrong
            if ft_correct and not orig_correct:
                case1_items.append((orig_item, ft_item))
            
            # Case 2: original right, finetuned wrong  
            elif orig_correct and not ft_correct:
                case2_items.append((orig_item, ft_item))
                
        return case1_items, case2_items
    
    def create_grid_image(self, query_img_path, ret_paths, grid_size=(2, 3), img_size=(300, 300)):
        """Create 2x3 grid with query image and retrieved images"""
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
        fig.suptitle('Query Image and Retrieved Images', fontsize=16, y=0.95)
        # Load and display query image (top-left)
        # try:
        query_img = Image.open(query_img_path)
        # query_name = query_image_path.split('/')[-2]
        axes[0, 0].imshow(query_img)
        query_name = query_img_path.split('/')[-2]
        axes[0, 0].set_title(query_name, fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
    # except Exception as e:
        axes[0, 0].text(0.5, 0.5, f'Query Image\nNot Found', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].axis('off')
        
        # Load and display retrieved images
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, (ret_path, pos) in enumerate(zip(ret_paths, positions)):
            row, col = pos
            try:
                ret_img = Image.open(ret_path)
                axes[row, col].imshow(ret_img)
                
                # Extract object name and set as title
                obj_name = self.extract_object_name(ret_path)
                axes[row, col].set_title(obj_name, fontsize=10)
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Image {i+1}\nNot Found', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'Retrieved {i+1}', fontsize=10)
                axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_grid_image(self, fig, image_id, case_dir):
        """Save grid image"""
        output_path = f"{self.output_dir}/{case_dir}/images/{image_id}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    def save_analysis_json(self, data, json_id, case_dir):
        """Save analysis JSON"""
        output_path = f"{self.output_dir}/{case_dir}/json/{json_id}.json"
        with open(output_path, 'w') as f:
            json.dump({str(json_id): data}, f, indent=2)
        return output_path
    
    def analyze_case(self, case_items, case_name, case_description):
        """Analyze a case with given items"""
        print(f"Analyzing {case_name}: {len(case_items)} items ({case_description})")
        
        for i, (orig_item, ft_item) in enumerate(case_items):
            # Create grid image
            fig = self.create_grid_image(
                orig_item['image_path'], 
                orig_item['ret_paths']
            )
            
            # Save grid image
            image_path = self.save_grid_image(fig, i, case_name)
            
            # Prepare JSON data
            json_data = {
                'response_original': orig_item['response'],
                'response_finetuned': ft_item['response'],
                'solution_desc_original': orig_item['solution_desc'],
                'solution_desc_finetuned': ft_item['solution_desc'],
                'pred_original': orig_item['pred_name'],
                'pred_finetuned': ft_item['pred_name'],
                'solution': orig_item['solution'],
                'image_path': orig_item['image_path'],
                'problem': orig_item['problem']
            }
            
            # Save JSON data
            json_path = self.save_analysis_json(json_data, i, case_name)
            
            print(f"  Item {i}: Saved image to {image_path}, JSON to {json_path}")
    
    def run_analysis(self):
        """Main pipeline to run the complete analysis"""
        print("Loading JSON files...")
        original_data, finetuned_data = self.load_json_files()
        
        print(f"Original data: {len(original_data)} items")
        print(f"Finetuned data: {len(finetuned_data)} items")
        original_data = original_data['results']
        finetuned_data = finetuned_data['results']
        print("Comparing predictions...")
        case1_items, case2_items = self.compare_predictions(original_data, finetuned_data)
        
        print(f"Found {len(case1_items)} Case 1 items and {len(case2_items)} Case 2 items")
        
        # Analyze both cases
        if case1_items:
            self.analyze_case(case1_items, 'ftr', 'finetuned correct, original wrong')
        else:
            print("No Case 1 items found.")
            
        if case2_items:
            self.analyze_case(case2_items, 'ogr', 'original correct, finetuned wrong')
        else:
            print("No Case 2 items found.")
        
        print("Analysis complete!")
        print(f"Results saved in '{self.output_dir}' directory")


# Usage example:
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = PredictionAnalyzer(
        original_json_path="results/PerVA/bag/results_original.json",
        finetuned_json_path="results/PerVA/bag/results_finetuned.json",
        output_dir="analysis2"
    )
    
    # Run complete analysis
    analyzer.run_analysis()