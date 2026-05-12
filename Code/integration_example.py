# Example integration of OpenCV defect detection with existing main.py pipeline
# THIS MAY BE REMOVED AS INTEGRATION WILL BE DONE IN MAIN.PY (AS IT IS THE MIAN FILE)

"""
This file shows how to modify your existing main.py to integrate OpenCV defect detection.
You can copy the relevant parts to enhance your current pipeline.
"""

# Add this import to your main.py imports section:
from vision.defect_detector import MangoDefectDetector
from vision.integrated_analyzer import IntegratedMangoAnalyzer

def enhanced_process_single_frame(frame, metadata, model, segmentation_model=None, output_dir=None):
    """
    Enhanced version of your process_single_frame function with OpenCV defect detection
    
    This is an example of how to modify your existing function to include defect detection.
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    # Extract ROI (your existing code)
    roi = extract_roi(frame)
    
    # Generate file ID (your existing code)
    frame_id = f"{metadata['frame_idx']:06d}_{metadata['timestamp_ms']}"
    
    # Save original (your existing code)
    original_path = save_frame(roi, f"peak_{frame_id}_original.jpg", output_dir)
    
    # Preprocess (your existing code)
    print("  Preprocessing...")
    preprocess_start = time.time()
    processed_image = preprocess_image(roi)
    preprocess_end = time.time()
    print(f"  Preprocessing time: {preprocess_end - preprocess_start:.2f}s")
    
    # Save preprocessed (your existing code)
    processed_path = save_frame(processed_image, f"peak_{frame_id}_processed.jpg", output_dir)
    
    # Predict (your existing code)
    print("  Running classification...")
    prediction = predict_disease(model, processed_image, return_probabilities=True)
    
    print(f"  Predicted: {prediction['class_name']} "
          f"(confidence: {prediction['confidence']:.2%})")
    print(f"  Inference time: {prediction['inference_time']:.3f}s")
    
    # Initialize result dictionary (your existing code)
    result = {
        'metadata': metadata,
        'prediction': prediction,
        'paths': {
            'original': original_path,
            'processed': processed_path
        }
    }
    
    # ===== NEW: Add OpenCV defect detection =====
    print("  Running OpenCV defect detection...")
    defect_detector = MangoDefectDetector()
    defect_analysis = defect_detector.detect_defects(roi)
    
    print(f"  OpenCV defects found: {defect_analysis.defect_count}")
    print(f"  Total defect area: {defect_analysis.total_defect_percentage:.2f}%")
    print(f"  Color uniformity: {defect_analysis.color_uniformity_score:.1f}/100")
    print(f"  Export grade impact: {defect_analysis.export_grade_impact}")
    
    # Save OpenCV defect visualization
    defect_viz = defect_detector.visualize_defects(roi, defect_analysis)
    defect_viz_path = save_frame(defect_viz, f"peak_{frame_id}_opencv_defects.jpg", output_dir)
    
    # Add OpenCV results to result dictionary
    result['opencv_defects'] = {
        'defect_count': defect_analysis.defect_count,
        'total_defect_percentage': defect_analysis.total_defect_percentage,
        'color_uniformity_score': defect_analysis.color_uniformity_score,
        'surface_quality_score': defect_analysis.surface_quality_score,
        'export_grade_impact': defect_analysis.export_grade_impact,
        'processing_time': defect_analysis.processing_time,
        'defect_regions': defect_analysis.defect_regions  # for detailed analysis
    }
    result['paths']['opencv_defects'] = defect_viz_path
    # ===== END NEW CODE =====
    
    # Check if segmentation should be performed (your existing code)
    should_segment = (config.ENABLE_SEGMENTATION and 
                     segmentation_model is not None)
    
    if should_segment:
        print("  Running ML segmentation...")
        
        # Segment disease regions (your existing code)
        segmentation_result = segment_disease(segmentation_model, roi)
        
        print(f"  ML disease coverage: {segmentation_result['disease_percentage']:.2f}%")
        print(f"  Segmentation time: {segmentation_result['segmentation_time']:.3f}s")
        
        # Get disease statistics (your existing code)
        stats = get_disease_statistics(segmentation_result['mask'])
        print(f"  Number of disease regions: {stats['num_regions']}")
        
        # ===== NEW: Combined analysis of ML + OpenCV results =====
        combined_defect_pct = max(
            segmentation_result['disease_percentage'], 
            defect_analysis.total_defect_percentage
        )
        
        print(f"  Combined defect coverage: {combined_defect_pct:.2f}%")
        
        # Generate comprehensive quality assessment
        quality_issues = []
        if combined_defect_pct > 5.0:
            quality_issues.append("High defect coverage")
        if defect_analysis.color_uniformity_score < 70:
            quality_issues.append("Poor color uniformity")
        if len([d for d in defect_analysis.defect_regions if d.severity == 'severe']) > 0:
            quality_issues.append("Severe defects present")
        
        # Determine export grade
        if combined_defect_pct <= 2.0 and defect_analysis.color_uniformity_score >= 85:
            export_grade = "Grade A (Premium Export)"
        elif combined_defect_pct <= 5.0:
            export_grade = "Grade B (Standard Export)"
        else:
            export_grade = "Grade C (Local Market/Processing)"
        
        print(f"  Export grade assessment: {export_grade}")
        
        result['combined_analysis'] = {
            'combined_defect_percentage': combined_defect_pct,
            'export_grade': export_grade,
            'quality_issues': quality_issues
        }
        # ===== END NEW CODE =====
        
        # Save segmentation results (your existing code)
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
        
        # Create final annotated result with segmentation info (enhanced)
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
            f"ML Disease: {segmentation_result['disease_percentage']:.1f}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        # NEW: Add OpenCV defect info
        cv2.putText(
            annotated,
            f"OpenCV Defects: {defect_analysis.total_defect_percentage:.1f}%",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),  # Magenta for OpenCV results
            2
        )
        cv2.putText(
            annotated,
            f"Export Grade: {export_grade.split('(')[0].strip()}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),  # Yellow for grade
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
        # No ML segmentation - just save annotated classification result with OpenCV info
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
        # NEW: Add OpenCV defect info even without ML segmentation
        cv2.putText(
            annotated,
            f"OpenCV Defects: {defect_analysis.total_defect_percentage:.1f}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )
        cv2.putText(
            annotated,
            f"Export Impact: {defect_analysis.export_grade_impact}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        result_path = save_frame(annotated, f"peak_{frame_id}_result.jpg", output_dir)
        result['paths']['result'] = result_path
        result['segmentation'] = None
    
    return result


def generate_rag_feature_description(result):
    """
    Generate textual description for RAG system integration
    
    This function creates a structured description that can be fed to your RAG pipeline
    for export country recommendation.
    """
    prediction = result['prediction']
    opencv_defects = result.get('opencv_defects', {})
    combined_analysis = result.get('combined_analysis', {})
    segmentation = result.get('segmentation')
    
    # Start with basic classification
    description = f"Mango Classification: {prediction['class_name']} (confidence: {prediction['confidence']:.1%})"
    
    # Add ML disease information
    if segmentation:
        description += f"\nML Disease Analysis: {segmentation['disease_percentage']:.1f}% disease coverage"
    
    # Add OpenCV defect information
    if opencv_defects:
        description += f"\nOpenCV Defect Analysis:"
        description += f"\n- Surface defects: {opencv_defects['total_defect_percentage']:.1f}% coverage"
        description += f"\n- Color uniformity: {opencv_defects['color_uniformity_score']:.1f}/100"
        description += f"\n- Surface quality: {opencv_defects['surface_quality_score']:.1f}/100"
        description += f"\n- Export grade impact: {opencv_defects['export_grade_impact']}"
    
    # Add combined assessment
    if combined_analysis:
        description += f"\nOverall Assessment:"
        description += f"\n- Export grade: {combined_analysis['export_grade']}"
        if combined_analysis['quality_issues']:
            description += f"\n- Quality issues: {', '.join(combined_analysis['quality_issues'])}"
    
    description += "\n\nBased on this analysis, which export countries would be most suitable for this mango?"
    
    return description


# Example of how to modify your run_pipeline function summary
def enhanced_pipeline_summary(results):
    """
    Enhanced summary including OpenCV defect analysis
    """
    print("\n" + "=" * 50)
    print("Enhanced Pipeline Completed Successfully!")
    print(f"Processed: {len(results)} frames")
    print("=" * 50)
    
    # Print summary of predictions with OpenCV data
    print("\nEnhanced Prediction Summary:")
    diseased_count = 0
    segmented_count = 0
    high_defect_count = 0
    
    for idx, result in enumerate(results, 1):
        pred = result['prediction']
        meta = result['metadata']
        opencv_defects = result.get('opencv_defects', {})
        combined_analysis = result.get('combined_analysis', {})
        
        status_str = f"  Frame {meta['frame_idx']}: {pred['class_name']} ({pred['confidence']:.2%})"
        
        if pred['class_name'] != 'Healthy':
            diseased_count += 1
        
        # Add OpenCV defect info
        if opencv_defects:
            defect_pct = opencv_defects['total_defect_percentage']
            status_str += f" | OpenCV: {defect_pct:.1f}% defects"
            
            if defect_pct > 5.0:
                high_defect_count += 1
                status_str += " ⚠️"
        
        # Add export grade if available
        if combined_analysis and 'export_grade' in combined_analysis:
            grade = combined_analysis['export_grade'].split('(')[0].strip()
            status_str += f" | {grade}"
        
        # Count segmented frames
        if result['segmentation'] is not None:
            segmented_count += 1
            disease_pct = result['segmentation']['disease_percentage']
            status_str += f" | ML: {disease_pct:.1f}%"
        
        print(status_str)
    
    print(f"\nSummary Statistics:")
    print(f"  ML Diseased: {diseased_count}/{len(results)}")
    print(f"  High OpenCV defects (>5%): {high_defect_count}/{len(results)}")
    print(f"  Segmented: {segmented_count}/{len(results)}")
    
    # Generate RAG descriptions for all frames
    print(f"\n🤖 RAG Integration Ready:")
    print(f"  Generated feature descriptions for {len(results)} frames")
    print(f"  Ready for export country prediction via RAG pipeline")


# Example usage in main:
if __name__ == "__main__":
    print("Enhanced Pipeline with OpenCV Defect Detection")
    print("This example shows integration with your existing main.py")
    print("Copy the relevant functions to enhance your current pipeline!")