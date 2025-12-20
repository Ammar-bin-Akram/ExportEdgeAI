"""Main pipeline for mango disease detection - Backward compatibility wrapper"""
# This file now uses the new modular structure but maintains the old interface
import os
import cv2
import time
from pathlib import Path

# Import from new modular structure
from config.settings import Settings
from fruits.mango import MangoPipeline
from vision import ROIExtractor

# Backward compatibility: keep old imports working
import config
from video_utils import extract_peak_frames, extract_roi
from preprocessing import preprocess_image
from model_utils import load_model_weights, predict_disease
from segmentation_utils import load_segmentation_model, segment_disease, get_disease_statistics


def save_frame(image, filename, output_dir=None):
    """Save image to output directory"""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


def process_single_frame(frame, metadata, model, segmentation_model=None, output_dir=None):
    """
    Process a single peak frame: preprocess, predict, and segment if diseased
    
    Args:
        frame: Input frame
        metadata: Dictionary with frame metadata
        model: Loaded classification model
        segmentation_model: Loaded segmentation model (optional)
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with all results and file paths
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    # Extract ROI
    roi = extract_roi(frame)
    
    # Generate file ID
    frame_id = f"{metadata['frame_idx']:06d}_{metadata['timestamp_ms']}"
    
    # Save original
    original_path = save_frame(roi, f"peak_{frame_id}_original.jpg", output_dir)
    
    # Preprocess
    print("  Preprocessing...")
    preprocess_start = time.time()
    processed_image = preprocess_image(roi)
    preprocess_end = time.time()
    print(f"  Preprocessing time: {preprocess_end - preprocess_start:.2f}s")
    
    # Save preprocessed
    processed_path = save_frame(processed_image, f"peak_{frame_id}_processed.jpg", output_dir)
    
    # Predict
    print("  Running classification...")
    prediction = predict_disease(model, processed_image, return_probabilities=True)
    
    print(f"  Predicted: {prediction['class_name']} "
          f"(confidence: {prediction['confidence']:.2%})")
    print(f"  Inference time: {prediction['inference_time']:.3f}s")
    
    # Initialize result dictionary
    result = {
        'metadata': metadata,
        'prediction': prediction,
        'paths': {
            'original': original_path,
            'processed': processed_path
        }
    }
    
    # Check if segmentation should be performed (now runs for all mangoes)
    should_segment = (config.ENABLE_SEGMENTATION and 
                     segmentation_model is not None)
    
    if should_segment:
        print("  Running segmentation...")
        
        # Segment disease regions
        segmentation_result = segment_disease(segmentation_model, roi)
        
        print(f"  Disease coverage: {segmentation_result['disease_percentage']:.2f}%")
        print(f"  Segmentation time: {segmentation_result['segmentation_time']:.3f}s")
        
        # Get disease statistics
        stats = get_disease_statistics(segmentation_result['mask'])
        print(f"  Number of disease regions: {stats['num_regions']}")
        
        # Save segmentation results
        mask_path = save_frame(
            segmentation_result['mask'] * 255,  # Scale mask to 0-255
            f"peak_{frame_id}_mask.jpg",
            output_dir
        )
        
        overlay_path = save_frame(
            segmentation_result['overlay'],
            f"peak_{frame_id}_segmented.jpg",
            output_dir
        )
        
        # Create final annotated result with segmentation info
        annotated = segmentation_result['overlay'].copy()
        cv2.putText(
            annotated,
            f"{prediction['class_name']} ({prediction['confidence']:.2%})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated,
            f"Disease Area: {segmentation_result['disease_percentage']:.1f}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        result_path = save_frame(annotated, f"peak_{frame_id}_result.jpg", output_dir)
        
        # Add segmentation data to result
        result['segmentation'] = {
            'disease_percentage': segmentation_result['disease_percentage'],
            'statistics': stats,
            'segmentation_time': segmentation_result['segmentation_time']
        }
        result['paths'].update({
            'mask': mask_path,
            'overlay': overlay_path,
            'result': result_path
        })
    else:
        # No segmentation - just save annotated classification result
        annotated = processed_image.copy()
        cv2.putText(
            annotated,
            f"{prediction['class_name']} ({prediction['confidence']:.2%})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        result_path = save_frame(annotated, f"peak_{frame_id}_result.jpg", output_dir)
        result['paths']['result'] = result_path
        result['segmentation'] = None
    
    return result


def run_pipeline(video_path=None, output_dir=None, enable_segmentation=None):
    """
    Run the complete mango disease detection pipeline with segmentation
    
    This function now uses the new modular architecture but maintains
    the same interface for backward compatibility.
    
    Args:
        video_path: Path to video file (if None, uses config)
        output_dir: Output directory (if None, uses config)
        enable_segmentation: Override config segmentation setting (optional)
    
    Returns:
        List of result dictionaries
    """
    # Load settings
    settings = Settings()
    
    if video_path is None:
        video_path = str(settings.VIDEO_SOURCE)
    
    if output_dir is None:
        output_dir = str(settings.OUTPUT_DIR)
    
    if enable_segmentation is not None:
        settings.ENABLE_SEGMENTATION = enable_segmentation
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Mango Disease Detection Pipeline")
    print("=" * 50)
    
    # Use new modular pipeline
    print("\nInitializing pipeline...")
    pipeline = MangoPipeline(settings)
    pipeline.load_models()
    
    # Extract peak frames from video
    print(f"\nExtracting peak frames from: {video_path}")
    peak_frames = pipeline.frame_extractor.extract_peak_frames(
        video_path, 
        display_debug=settings.DISPLAY_DEBUG
    )
    
    if len(peak_frames) == 0:
        print("No peak frames found!")
        return []
    
    print(f"\nProcessing {len(peak_frames)} peak frames...")
    
    # Process each peak frame
    roi_extractor = ROIExtractor(settings)
    results = []
    for idx, (frame, metadata) in enumerate(peak_frames, 1):
        print(f"\n--- Peak Frame {idx}/{len(peak_frames)} ---")
        
        # Extract ROI from full frame first
        roi = roi_extractor.extract_roi(frame)
        
        # Process ROI through pipeline
        result = pipeline.process_single_frame_detailed(roi)
        
        # Save results
        frame_id = f"{metadata['frame_idx']:06d}_{metadata['timestamp_ms']}"
        
        # Save original full frame
        save_frame(frame, f"peak_{frame_id}_fullframe.jpg", output_dir)
        
        # Save ROI (original extracted region)
        save_frame(roi, f"peak_{frame_id}_roi.jpg", output_dir)
        
        # Save preprocessed image
        save_frame(result['preprocessed'], f"peak_{frame_id}_processed.jpg", output_dir)
        
        print(f"  Predicted: {result['class_name']} ({result['confidence']:.2%})")
        
        # Create and save result image with classification text
        result_image = roi.copy()
        class_text = f"{result['class_name']} ({result['confidence']:.1%})"
        
        # Add black background rectangle for better text visibility
        (text_width, text_height), baseline = cv2.getTextSize(
            class_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(result_image, (5, 5), (text_width + 15, text_height + baseline + 15), (0, 0, 0), -1)
        
        # Add classification text
        cv2.putText(
            result_image,
            class_text,
            (10, text_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),  # Green text
            2
        )
        
        # Add disease percentage if segmentation was performed
        if result['segmentation_overlay'] is not None:
            disease_text = f"Disease: {result['disease_percentage']:.1f}%"
            cv2.putText(
                result_image,
                disease_text,
                (10, text_height + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),  # Yellow text
                2
            )
        
        save_frame(result_image, f"peak_{frame_id}_result.jpg", output_dir)
        
        if result['segmentation_overlay'] is not None:
            print(f"  Disease coverage: {result['disease_percentage']:.2f}%")
            # Save colored mask visualization
            colored_mask = pipeline.segmentation_model.create_colored_mask(result['segmentation_mask'])
            save_frame(colored_mask, f"peak_{frame_id}_mask.jpg", output_dir)
            save_frame(result['segmentation_overlay'], f"peak_{frame_id}_segmented.jpg", output_dir)
        
        # Convert to old format for backward compatibility
        old_format_result = {
            'metadata': metadata,
            'prediction': {
                'class_name': result['class_name'],
                'confidence': result['confidence'],
            },
            'segmentation': {
                'disease_percentage': result['disease_percentage']
            } if result['segmentation_overlay'] is not None else None
        }
        
        results.append(old_format_result)
    
    # Summary
    print("\n" + "=" * 50)
    print("Pipeline Completed Successfully!")
    print(f"Processed: {len(results)} frames")
    print(f"Results saved to: {output_dir}")
    print("=" * 50)
    
    # Print summary of predictions
    print("\nPrediction Summary:")
    diseased_count = 0
    segmented_count = 0
    
    for idx, result in enumerate(results, 1):
        pred = result['prediction']
        meta = result['metadata']
        
        status_str = f"  Frame {meta['frame_idx']}: {pred['class_name']} ({pred['confidence']:.2%})"
        
        if pred['class_name'] != 'Healthy':
            diseased_count += 1
        
        # Count segmented frames (now includes all mangoes)
        if result['segmentation'] is not None:
            segmented_count += 1
            disease_pct = result['segmentation']['disease_percentage']
            status_str += f" - Disease area: {disease_pct:.1f}%"
        
        print(status_str)
    
    print(f"\nTotal diseased: {diseased_count}/{len(results)}")
    if settings.ENABLE_SEGMENTATION:
        print(f"Segmented: {segmented_count}/{len(results)}")
    
    return results


if __name__ == "__main__":
    # Run the pipeline
    results = run_pipeline()
    
    # Optional: Access individual results
    # for result in results:
    #     print(f"Frame: {result['metadata']['frame_idx']}")
    #     print(f"Prediction: {result['prediction']['class_name']}")
    #     print(f"Files: {result['paths']}")
