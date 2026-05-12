"""Command-line interface for mango processing pipeline"""
import os
import cv2
import time
from pathlib import Path

# Import from new modular structure
from config.settings import Settings
from fruits.mango import MangoPipeline


def save_frame(image, filename, output_dir):
    """Save image to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


def process_single_frame(frame, metadata, pipeline, output_dir):
    """
    Process a single peak frame using the pipeline
    
    Args:
        frame: Input frame
        metadata: Dictionary with frame metadata
        pipeline: MangoPipeline instance
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with all results and file paths
    """
    # Generate file ID
    frame_id = f"{metadata['frame_idx']:06d}_{metadata['timestamp_ms']}"
    
    # Process through pipeline
    print("  Processing frame...")
    start_time = time.time()
    result = pipeline.process_single_frame_detailed(frame)
    processing_time = time.time() - start_time
    
    print(f"  Predicted: {result['class_name']} "
          f"(confidence: {result['confidence']:.2%})")
    print(f"  Processing time: {processing_time:.2f}s")
    
    # Save original
    original_path = save_frame(
        result['original'],
        f"peak_{frame_id}_original.jpg",
        output_dir
    )
    
    # Save preprocessed
    processed_path = save_frame(
        result['preprocessed'],
        f"peak_{frame_id}_processed.jpg",
        output_dir
    )
    
    # Initialize file paths
    paths = {
        'original': original_path,
        'processed': processed_path
    }
    
    # Save segmentation if available
    if result['segmentation_overlay'] is not None:
        print(f"  Disease coverage: {result['disease_percentage']:.2f}%")
        
        if 'disease_statistics' in result:
            stats = result['disease_statistics']
            print(f"  Number of disease regions: {stats['num_regions']}")
        
        # Save mask
        mask_path = save_frame(
            result['segmentation_mask'] * 255,  # Scale to 0-255
            f"peak_{frame_id}_mask.jpg",
            output_dir
        )
        
        # Save overlay
        overlay_path = save_frame(
            result['segmentation_overlay'],
            f"peak_{frame_id}_segmented.jpg",
            output_dir
        )
        
        # Create annotated result
        annotated = result['segmentation_overlay'].copy()
        cv2.putText(
            annotated,
            f"{result['class_name']} ({result['confidence']:.2%})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated,
            f"Disease Area: {result['disease_percentage']:.1f}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        result_path = save_frame(
            annotated,
            f"peak_{frame_id}_result.jpg",
            output_dir
        )
        
        paths.update({
            'mask': mask_path,
            'overlay': overlay_path,
            'result': result_path
        })
    else:
        # No segmentation - just annotate classification
        annotated = result['preprocessed'].copy()
        cv2.putText(
            annotated,
            f"{result['class_name']} ({result['confidence']:.2%})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        result_path = save_frame(
            annotated,
            f"peak_{frame_id}_result.jpg",
            output_dir
        )
        paths['result'] = result_path
    
    # Return comprehensive result
    return {
        'metadata': metadata,
        'class_name': result['class_name'],
        'confidence': result['confidence'],
        'disease_percentage': result['disease_percentage'],
        'top_predictions': result.get('top_k_predictions', []),
        'paths': paths,
        'processing_time': processing_time
    }


def run_pipeline(video_path, output_dir=None):
    """
    Run complete mango processing pipeline
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save outputs (default: from settings)
    """
    # Load settings
    settings = Settings()
    
    if output_dir is None:
        output_dir = settings.OUTPUT_DIR
    
    print("=" * 60)
    print("MANGO DISEASE DETECTION PIPELINE")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Create pipeline
    print("Initializing mango pipeline...")
    pipeline = MangoPipeline(settings)
    pipeline.load_models()
    print()
    
    # Extract peak frames
    print("Extracting peak frames from video...")
    peak_frames = pipeline.frame_extractor.extract_peak_frames(
        video_path,
        display_debug=False
    )
    print(f"Extracted {len(peak_frames)} peak frames")
    print()
    
    # Process each frame
    all_results = []
    for idx, (frame, metadata) in enumerate(peak_frames):
        print(f"Processing frame {idx + 1}/{len(peak_frames)}...")
        
        result = process_single_frame(frame, metadata, pipeline, output_dir)
        all_results.append(result)
        print()
    
    # Summary
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {len(all_results)}")
    
    # Count by variety
    variety_counts = {}
    for r in all_results:
        variety = r['class_name']
        variety_counts[variety] = variety_counts.get(variety, 0) + 1
    
    print("\nVariety distribution:")
    for variety, count in sorted(variety_counts.items()):
        print(f"  {variety}: {count}")
    
    # Disease statistics
    diseased = [r for r in all_results if r['disease_percentage'] > 0]
    if diseased:
        avg_disease = sum(r['disease_percentage'] for r in diseased) / len(diseased)
        print(f"\nDiseased mangoes: {len(diseased)}")
        print(f"Average disease coverage: {avg_disease:.2f}%")
    
    print(f"\nResults saved to: {output_dir}")
    
    return all_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mango Disease Detection Pipeline")
    parser.add_argument('video', help="Path to input video file")
    parser.add_argument('-o', '--output', help="Output directory")
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(args.video, args.output)


if __name__ == "__main__":
    main()
