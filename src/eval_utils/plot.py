import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import textwrap
import json
import os

def truncate_text(text, max_length=100):
    """Truncate text and add ellipsis if too long"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def extract_key_features(desc_dict):
    """Extract key features from description dictionary"""
    features = []
    if 'distinct features' in desc_dict:
        feat = desc_dict['distinct features']
        if isinstance(feat, list):
            features.extend(feat)
        elif isinstance(feat, str):
            # Parse string like "[feature1, feature2]"
            feat = feat.strip('[]')
            features = [f.strip() for f in feat.split(',')]
    return features[:3]  # Limit to top 3 features

def parse_reasoning(reason_str):
    """Parse reasoning string (may be JSON formatted)"""
    try:
        reason_dict = json.loads(reason_str)
        if 'Reasoning' in reason_dict:
            return reason_dict['Reasoning']
        return str(reason_dict)
    except:
        return reason_str

def get_gt_label_from_path(ret_path):
    """Extract ground truth label from retrieval path"""
    # Extract label from path like "data/PerVA/train_/clothe/fse/1.jpg" -> "fse"
    parts = ret_path.split('/')
    return parts[-2] if len(parts) >= 2 else 'unknown'

def visualize_comparison(data_dict, save_path='cvpr_comparison.png', dpi=300):
    """
    Create a publication-ready comparison visualization
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with query_path as key containing predictions, descriptions, and reasoning
    save_path : str
        Path to save the figure
    dpi : int
        Resolution for the saved figure
    """
    
    # Extract the first (and likely only) query from the dict
    query_path = list(data_dict.keys())[0]
    data = data_dict[query_path]
    
    # Get ground truth label from retrieval path
    gt_label = get_gt_label_from_path(data['ret_path'])
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, 
                          left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # Color scheme
    colors = {
        'correct': '#4CAF50',
        'incorrect': '#F44336',
        'neutral': '#2196F3',
        'gt': '#FF9800'
    }
    
    # Title
    fig.suptitle('Method Comparison: R2P vs RRG (Ours)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # ==================== ROW 1: Images ====================
    
    # Query Image
    ax_query = fig.add_subplot(gs[0, 0])
    try:
        if os.path.exists(query_path):
            query_img = Image.open(query_path)
            ax_query.imshow(query_img)
            ax_query.axis('off')
        else:
            ax_query.text(0.5, 0.5, 'Query\nImage', ha='center', va='center', 
                          fontsize=14, fontweight='bold')
            ax_query.set_xlim(0, 1)
            ax_query.set_ylim(0, 1)
            ax_query.axis('off')
            rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor=colors['neutral'], 
                                  facecolor='lightblue', 
                                  linewidth=3, alpha=0.3)
            ax_query.add_patch(rect)
    except Exception as e:
        ax_query.text(0.5, 0.5, 'Query\nImage', ha='center', va='center', 
                      fontsize=14, fontweight='bold')
        ax_query.set_xlim(0, 1)
        ax_query.set_ylim(0, 1)
        ax_query.axis('off')
    ax_query.set_title('Query', fontsize=12, fontweight='bold', pad=10)
    
    # Ground Truth
    ax_gt = fig.add_subplot(gs[0, 1])
    try:
        if os.path.exists(data['ret_path']):
            gt_img = Image.open(data['ret_path'])
            ax_gt.imshow(gt_img)
            ax_gt.axis('off')
        else:
            ax_gt.text(0.5, 0.5, 'Ground\nTruth', ha='center', va='center', 
                       fontsize=14, fontweight='bold', color=colors['gt'])
            ax_gt.set_xlim(0, 1)
            ax_gt.set_ylim(0, 1)
            ax_gt.axis('off')
            rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor=colors['gt'], 
                                  facecolor='lightyellow', 
                                  linewidth=3, alpha=0.3)
            ax_gt.add_patch(rect)
    except Exception as e:
        ax_gt.text(0.5, 0.5, 'Ground\nTruth', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=colors['gt'])
        ax_gt.set_xlim(0, 1)
        ax_gt.set_ylim(0, 1)
        ax_gt.axis('off')
    ax_gt.set_title(f'GT: {gt_label}', fontsize=12, fontweight='bold', 
                    color=colors['gt'], pad=10)
    
    # R2P Prediction
    ax_r2p = fig.add_subplot(gs[0, 2])
    is_r2p_correct = (data['r2p_pred'] == gt_label)
    try:
        if os.path.exists(data['r2p_pred_path']):
            r2p_img = Image.open(data['r2p_pred_path'])
            ax_r2p.imshow(r2p_img)
            ax_r2p.axis('off')
        else:
            color = colors['correct'] if is_r2p_correct else colors['incorrect']
            ax_r2p.text(0.5, 0.5, 'R2P\nPrediction', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color=color)
            ax_r2p.set_xlim(0, 1)
            ax_r2p.set_ylim(0, 1)
            ax_r2p.axis('off')
            rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor=color, 
                                  facecolor='mistyrose' if not is_r2p_correct else 'honeydew', 
                                  linewidth=3, alpha=0.3)
            ax_r2p.add_patch(rect)
    except Exception as e:
        ax_r2p.text(0.5, 0.5, 'R2P\nPrediction', ha='center', va='center', 
                    fontsize=14, fontweight='bold')
        ax_r2p.set_xlim(0, 1)
        ax_r2p.set_ylim(0, 1)
        ax_r2p.axis('off')
    
    # Add checkmark or X
    if not is_r2p_correct:
        ax_r2p.text(0.9, 0.1, '✗', fontsize=40, color=colors['incorrect'], 
                    ha='center', va='center', fontweight='bold',
                    transform=ax_r2p.transAxes)
        ax_r2p.set_title(f'R2P: {data["r2p_pred"]} ✗', 
                         fontsize=12, fontweight='bold', 
                         color=colors['incorrect'], pad=10)
    else:
        ax_r2p.text(0.9, 0.1, '✓', fontsize=40, color=colors['correct'], 
                    ha='center', va='center', fontweight='bold',
                    transform=ax_r2p.transAxes)
        ax_r2p.set_title(f'R2P: {data["r2p_pred"]} ✓', 
                         fontsize=12, fontweight='bold', 
                         color=colors['correct'], pad=10)
    
    # RRG Prediction (Ours)
    ax_rrg = fig.add_subplot(gs[0, 3])
    is_rrg_correct = (data['rrg_pred_name'] == gt_label)
    
    try:
        if 'rrg_pred_path' in data and os.path.exists(data['rrg_pred_path']):
            rrg_img = Image.open(data['rrg_pred_path'])
            ax_rrg.imshow(rrg_img)
            ax_rrg.axis('off')
        else:
            color = colors['correct'] if is_rrg_correct else colors['incorrect']
            ax_rrg.text(0.5, 0.5, 'RRG\nPrediction\n(Ours)', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color=color)
            ax_rrg.set_xlim(0, 1)
            ax_rrg.set_ylim(0, 1)
            ax_rrg.axis('off')
            rect_color = colors['correct'] if is_rrg_correct else colors['incorrect']
            rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor=rect_color, 
                                  facecolor='honeydew' if is_rrg_correct else 'mistyrose', 
                                  linewidth=3, alpha=0.3)
            ax_rrg.add_patch(rect)
    except Exception as e:
        color = colors['correct'] if is_rrg_correct else colors['incorrect']
        ax_rrg.text(0.5, 0.5, 'RRG\nPrediction\n(Ours)', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color=color)
        ax_rrg.set_xlim(0, 1)
        ax_rrg.set_ylim(0, 1)
        ax_rrg.axis('off')
    
    # Add check mark or X
    if is_rrg_correct:
        ax_rrg.text(0.9, 0.1, '✓', fontsize=40, color=colors['correct'], 
                    ha='center', va='center', fontweight='bold',
                    transform=ax_rrg.transAxes)
        ax_rrg.set_title(f'RRG: {data["rrg_pred_name"]} ✓', 
                         fontsize=12, fontweight='bold', 
                         color=colors['correct'], pad=10)
    else:
        ax_rrg.text(0.9, 0.1, '✗', fontsize=40, color=colors['incorrect'], 
                    ha='center', va='center', fontweight='bold',
                    transform=ax_rrg.transAxes)
        ax_rrg.set_title(f'RRG: {data["rrg_pred_name"]} ✗', 
                         fontsize=12, fontweight='bold', 
                         color=colors['incorrect'], pad=10)
    
    # ==================== ROW 2: Key Features Comparison ====================
    
    ax_features = fig.add_subplot(gs[1, :])
    ax_features.axis('off')
    
    # Extract key features for comparison
    r2p_gt_features = extract_key_features(data['r2p_desc'].get(gt_label, {}))
    r2p_pred_features = extract_key_features(data['r2p_desc'].get(data['r2p_pred'], {}))
    
    rrg_gt_features = extract_key_features(data['rrg_desc'].get(gt_label, {}))
    rrg_pred_features = extract_key_features(data['rrg_desc'].get(data['rrg_pred_name'], {}))
    
    # Create comparison table
    table_data = []
    table_data.append(['Method', 'Ground Truth Features', 'Predicted Features', 'Match'])
    
    # R2P row
    r2p_gt_str = '\n'.join([f"• {truncate_text(f, 40)}" for f in r2p_gt_features[:2]]) if r2p_gt_features else "N/A"
    r2p_pred_str = '\n'.join([f"• {truncate_text(f, 40)}" for f in r2p_pred_features[:2]]) if r2p_pred_features else "N/A"
    table_data.append(['R2P', r2p_gt_str, r2p_pred_str, '✓' if is_r2p_correct else '✗'])
    
    # RRG row
    rrg_gt_str = '\n'.join([f"• {truncate_text(f, 40)}" for f in rrg_gt_features[:2]]) if rrg_gt_features else "N/A"
    rrg_pred_str = '\n'.join([f"• {truncate_text(f, 40)}" for f in rrg_pred_features[:2]]) if rrg_pred_features else "N/A"
    table_data.append(['RRG\n(Ours)', rrg_gt_str, rrg_pred_str, '✓' if is_rrg_correct else '✗'])
    
    # Create table
    table = ax_features.table(cellText=table_data, 
                             cellLoc='left',
                             loc='center',
                             colWidths=[0.12, 0.38, 0.38, 0.12],
                             bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Style the table
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            table_cell = table[(i, j)]
            if i == 0:  # Header
                table_cell.set_facecolor('#E3F2FD')
                table_cell.set_text_props(weight='bold', size=10)
            elif i == 1:  # R2P row
                if j == 3:
                    if is_r2p_correct:
                        table_cell.set_facecolor('#E8F5E9')
                        table_cell.set_text_props(color=colors['correct'], 
                                                 weight='bold', size=16)
                    else:
                        table_cell.set_facecolor('#FFEBEE')
                        table_cell.set_text_props(color=colors['incorrect'], 
                                                 weight='bold', size=16)
                else:
                    table_cell.set_facecolor('#FAFAFA')
            elif i == 2:  # RRG row
                if j == 3:
                    if is_rrg_correct:
                        table_cell.set_facecolor('#E8F5E9')
                        table_cell.set_text_props(color=colors['correct'], 
                                                 weight='bold', size=16)
                    else:
                        table_cell.set_facecolor('#FFEBEE')
                        table_cell.set_text_props(color=colors['incorrect'], 
                                                 weight='bold', size=16)
                else:
                    table_cell.set_facecolor('#F1F8E9')
            
            table_cell.set_edgecolor('#BDBDBD')
            table_cell.set_linewidth(1.5)
    
    # Add title
    ax_features.text(0.5, 1.05, 'Key Feature Comparison', 
                    ha='center', fontsize=14, fontweight='bold', 
                    transform=ax_features.transAxes)
    
    # ==================== ROW 3: Reasoning Summary ====================
    
    ax_r2p_reason = fig.add_subplot(gs[2, :2])
    ax_r2p_reason.axis('off')
    
    # R2P Reasoning (truncated and parsed)
    r2p_reason = parse_reasoning(data['r2p_reason'])
    r2p_reason = truncate_text(r2p_reason, 200)
    wrapped_r2p = textwrap.fill(r2p_reason, width=60)
    
    box_color = colors['correct'] if is_r2p_correct else colors['incorrect']
    box = FancyBboxPatch((0.05, 0.15), 0.9, 0.75, 
                         boxstyle="round,pad=0.02", 
                         edgecolor=box_color, 
                         facecolor='#E8F5E9' if is_r2p_correct else '#FFEBEE', 
                         linewidth=2, alpha=0.5)
    ax_r2p_reason.add_patch(box)
    
    ax_r2p_reason.text(0.5, 0.9, 'R2P Reasoning', 
                      ha='center', fontsize=12, fontweight='bold',
                      color=box_color, transform=ax_r2p_reason.transAxes)
    ax_r2p_reason.text(0.5, 0.45, wrapped_r2p, 
                      ha='center', va='center', fontsize=9,
                      transform=ax_r2p_reason.transAxes)
    
    ax_rrg_reason = fig.add_subplot(gs[2, 2:])
    ax_rrg_reason.axis('off')
    
    # RRG Reasoning (truncated and parsed)
    rrg_reason = parse_reasoning(data['rrg_reason'])
    rrg_reason = truncate_text(rrg_reason, 200)
    wrapped_rrg = textwrap.fill(rrg_reason, width=60)
    
    box_color = colors['correct'] if is_rrg_correct else colors['incorrect']
    box = FancyBboxPatch((0.05, 0.15), 0.9, 0.75, 
                         boxstyle="round,pad=0.02", 
                         edgecolor=box_color, 
                         facecolor='#E8F5E9' if is_rrg_correct else '#FFEBEE', 
                         linewidth=2, alpha=0.5)
    ax_rrg_reason.add_patch(box)
    
    ax_rrg_reason.text(0.5, 0.9, 'RRG Reasoning (Ours)', 
                      ha='center', fontsize=12, fontweight='bold',
                      color=box_color, transform=ax_rrg_reason.transAxes)
    ax_rrg_reason.text(0.5, 0.45, wrapped_rrg, 
                      ha='center', va='center', fontsize=9,
                      transform=ax_rrg_reason.transAxes)
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {save_path}")
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Your data with correct structure
    with open('for_visualize_clothe.json') as f:
        data = json.load(f)
    from tqdm import *
    for i, sample_data in tqdm(enumerate(data)):
        fig = visualize_comparison(sample_data, save_path=f'plots/method_comparison_{i}.png', dpi=300)
        plt.show()
        # sample_data = {
        #     "data/PerVA/test_/clothe/gnb/20.jpg": {
        #         "ret_path": "data/PerVA/train_/clothe/fse/1.jpg",
        #         "r2p_desc": {
        #             "fse": {
        #                 "general": "<fse> is a neatly folded pink cloth, possibly made of cotton or linen.",
        #                 "category": "Clothing",
        #                 "distinct features": "[Pink color, checkered pattern, appears to be a handkerchief or small towel]"
        #             },
        #             "tql": {
        #                 "general": "<tql> is a beige, fabric cover for an umbrella.",
        #                 "category": "Clothing",
        #                 "distinct features": "[Designed to protect the handle and shaft of an umbrella, made from durable material, fits over the entire length of the umbrella shaft, typically used in outdoor settings to shield against weather elements]"
        #             }
        #         },
        #         "r2p_reason": '{\n  "A": "[Black, fabric, umbrella cover]",\n  "B": "[Pink, checkered, handkerchief]",\n  "C": "[Red, velvet, folded]",\n  "Reasoning": "The image shows a black, fabric item that appears to be an umbrella cover, which matches the description of option A.",\n  "Answer": "A"\n}',
        #         "r2p_pred": "tql",
        #         "r2p_pred_path": "data/PerVA/train_/clothe/tql/1.jpg",
        #         "rrg_desc": {
        #             "fse": {
        #                 "category": "clothe",
        #                 "general": [
        #                     "A photo of a pink handkerchief."
        #                 ],
        #                 "distinct features": [
        #                     "The handkerchief is a light pink color with a smooth texture. It has a neat, precise fold with a visible seam along the top edge. There are three thin, horizontal lines running parallel to each other on the surface of the handkerchief."
        #                 ]
        #             },
        #             "tql": {
        #                 "category": "clothe",
        #                 "general": [
        #                     "A photo of a beige, long-sleeved shirt."
        #                 ],
        #                 "distinct features": [
        #                     "The beige, long-sleeved shirt features a high neckline, a relaxed fit, and a slight sheen to the fabric, indicating a possible cotton or cotton blend material. The shirt is designed with a loose, comfortable style, suitable for casual wear."
        #                 ]
        #             }
        #         },
        #         "rrg_reason": '{"Reasoning": "The query image shows a black dress with a high neckline, a flowing, asymmetrical skirt, and a loose, comfortable fit. It has a distinctive, layered look with a sheer, flowing overlay. This matches the description of the \'gnb\' category in the provided gnbs.", "Answer": "C"}',
        #         "rrg_pred_name": "gnb"
        #     }
        # }
        
    # Create visualization
    fig = visualize_comparison(sample_data, save_path='method_comparison.png', dpi=300)
    plt.show()