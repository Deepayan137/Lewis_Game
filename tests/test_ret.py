import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from inference_utils.retriever import SimpleClipRetriever

def print_results(results, title, max_display=10):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    if not results:
        print("No results found!")
        return
    
    for i, res in enumerate(results[:max_display], 1):
        print(f"\n{i}. {res['name']}")
        print(f"   Path: {res['path']}")
        
        if 'image_similarity' in res:
            print(f"   Image Similarity: {res['image_similarity']:.4f}")
        
        if 'text_similarity' in res:
            print(f"   Text Similarity:  {res['text_similarity']:.4f}")
        
        if 'combined_score' in res:
            print(f"   Combined Score:   {res['combined_score']:.4f}")
        
        if 'similarity' in res:
            print(f"   Similarity:       {res['similarity']:.4f}")
    
    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")
    
    print(f"\nTotal results: {len(results)}")


def compare_search_methods(retriever, query_image_path, k=10):
    """Compare all three search methods side by side"""
    print(f"\n{'#'*80}")
    print(f"Comparing Search Methods for: {query_image_path}")
    print(f"{'#'*80}")
    
    # 1. Image-only search
    print("\n[1/3] Running Image Search...")
    img_results = retriever.image_search(query_image_path, k=k)
    print_results(img_results, "IMAGE SEARCH RESULTS", max_display=5)
    
    # 2. Text-only search
    print("\n[2/3] Running Text Search...")
    txt_results = retriever.image_search_on_text(query_image_path, k=k, return_paths=True)
    print_results(txt_results, "TEXT SEARCH RESULTS", max_display=5)
    
    # 3. Hybrid search
    print("\n[3/3] Running Hybrid Search...")
    hybrid_results = retriever.hybrid_search(
        query_image_path=query_image_path,
        k=k,
        alpha=0.5,
        normalization='minmax',
        aggregation='weighted_sum'
    )
    print_results(hybrid_results, "HYBRID SEARCH RESULTS (alpha=0.5)", max_display=10)
    
    return img_results, txt_results, hybrid_results


def test_different_alphas(retriever, query_image_path, k=10):
    """Test hybrid search with different alpha values"""
    print(f"\n{'#'*80}")
    print(f"Testing Different Alpha Values")
    print(f"{'#'*80}")
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for alpha in alphas:
        print(f"\n{'='*80}")
        print(f"Alpha = {alpha} (Image: {alpha*100:.0f}%, Text: {(1-alpha)*100:.0f}%)")
        print(f"{'='*80}")
        
        results = retriever.hybrid_search(
            query_image_path=query_image_path,
            k=k,
            alpha=alpha,
            normalization='minmax',
            aggregation='weighted_sum',
            image_k=40,
            text_k=40
        )
        
        # Show top 5 results
        for i, res in enumerate(results[:5], 1):
            print(f"{i}. {res['name']:<30} | "
                  f"Img: {res['image_similarity']:.3f} | "
                  f"Txt: {res['text_similarity']:.3f} | "
                  f"Combined: {res['combined_score']:.3f}")


def test_different_aggregations(retriever, query_image_path, k=10):
    """Test hybrid search with different aggregation methods"""
    print(f"\n{'#'*80}")
    print(f"Testing Different Aggregation Methods")
    print(f"{'#'*80}")
    
    aggregations = ['weighted_sum', 'max', 'min', 'product', 'harmonic_mean']
    
    for agg in aggregations:
        print(f"\n{'='*80}")
        print(f"Aggregation: {agg.upper()}")
        print(f"{'='*80}")
        
        results = retriever.hybrid_search(
            query_image_path=query_image_path,
            k=k,
            alpha=0.5,
            normalization='minmax',
            aggregation=agg
        )
        
        # Show top 5 results
        for i, res in enumerate(results[:5], 1):
            print(f"{i}. {res['name']:<30} | "
                  f"Img: {res['image_similarity']:.3f} | "
                  f"Txt: {res['text_similarity']:.3f} | "
                  f"Combined: {res['combined_score']:.3f}")


def analyze_overlap(img_results, txt_results, hybrid_results):
    """Analyze overlap between different search methods"""
    print(f"\n{'='*80}")
    print("OVERLAP ANALYSIS")
    print(f"{'='*80}")
    
    img_names = {r['name'] for r in img_results}
    txt_names = {r['name'] for r in txt_results}
    hybrid_names = {r['name'] for r in hybrid_results}
    
    print(f"\nUnique items in Image Search:  {len(img_names)}")
    print(f"Unique items in Text Search:   {len(txt_names)}")
    print(f"Unique items in Hybrid Search: {len(hybrid_names)}")
    
    overlap_img_txt = img_names.intersection(txt_names)
    print(f"\nOverlap (Image ∩ Text):        {len(overlap_img_txt)}")
    
    only_img = img_names - txt_names
    only_txt = txt_names - img_names
    print(f"Only in Image Search:          {len(only_img)}")
    print(f"Only in Text Search:           {len(only_txt)}")
    
    print(f"\nHybrid contains from Image:    {len(hybrid_names.intersection(img_names))}")
    print(f"Hybrid contains from Text:     {len(hybrid_names.intersection(txt_names))}")
    
    # Show some examples
    if only_img:
        print(f"\nExample items only in Image Search: {list(only_img)[:3]}")
    if only_txt:
        print(f"Example items only in Text Search:  {list(only_txt)[:3]}")


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Search")
    parser.add_argument("--query_image", type=str, required=True,
                       help="Path to query image")
    parser.add_argument("--json_path", type=str,
                       default="manifests/YoLLaVA/main_catalog_seed_23.json")
    parser.add_argument("--category", type=str, default="all",
                       help="Category to search in")
    parser.add_argument("--dataset", type=str, default="YoLLaVA",
                       help="Dataset name")
    parser.add_argument("--seed", type=int, default=23,
                       help="Random seed")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of results to return")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--create_index", action="store_true",
                       help="Create new index (otherwise load existing)")
    parser.add_argument("--test_mode", type=str, default="all",
                       choices=["all", "compare", "alphas", "aggregations", "basic"],
                       help="Which tests to run")
    
    args = parser.parse_args()
    
    # Verify query image exists
    if not Path(args.query_image).exists():
        print(f"Error: Query image not found: {args.query_image}")
        return
    
    print("="*80)
    print("HYBRID SEARCH TEST")
    print("="*80)
    print(f"Query Image:    {args.query_image}")
    print(f"JSON Path:      {args.json_path}")
    print(f"Category:       {args.category}")
    print(f"Dataset:        {args.dataset}")
    print(f"K:              {args.k}")
    print(f"Create Index:   {args.create_index}")
    print(f"Test Mode:      {args.test_mode}")
    print("="*80)
    
    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = SimpleClipRetriever(
        embed_dim=768,
        create_index=args.create_index,
        batch_size=6,
        dataset=args.dataset,
        json_path=args.json_path,
        category=args.category,
        device=args.device,
        clip_model="openai/clip-vit-large-patch14-336",
        seed=args.seed,
        db_type='finetuned_7b'
    )
    print("Retriever ready!")
    
    # Run tests based on mode
    if args.test_mode == "basic":
        # Just run hybrid search
        results = retriever.hybrid_search(
            query_image_path=args.query_image,
            k=args.k,
            alpha=0.5,
            normalization='minmax',
            aggregation='weighted_sum'
        )
        print_results(results, "HYBRID SEARCH RESULTS", max_display=args.k)
    
    elif args.test_mode == "compare":
        # Compare all three methods
        img_results, txt_results, hybrid_results = compare_search_methods(
            retriever, args.query_image, k=args.k
        )
        analyze_overlap(img_results, txt_results, hybrid_results)
    
    elif args.test_mode == "alphas":
        # Test different alpha values
        test_different_alphas(retriever, args.query_image, k=args.k)
    
    elif args.test_mode == "aggregations":
        # Test different aggregation methods
        test_different_aggregations(retriever, args.query_image, k=args.k)
    
    elif args.test_mode == "all":
        # Run all tests
        img_results, txt_results, hybrid_results = compare_search_methods(
            retriever, args.query_image, k=args.k
        )
        analyze_overlap(img_results, txt_results, hybrid_results)
        test_different_alphas(retriever, args.query_image, k=args.k)
        test_different_aggregations(retriever, args.query_image, k=args.k)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()