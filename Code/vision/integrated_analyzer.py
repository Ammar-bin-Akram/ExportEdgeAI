"""
Integration example for combining ML disease segmentation with OpenCV defect detection.

This script demonstrates how to integrate the new MangoDefectDetector with your 
existing vision pipeline for comprehensive mango quality assessment.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
import time

from vision.defect_detector import MangoDefectDetector, DefectAnalysis


class IntegratedMangoAnalyzer:
    """
    Integrated analyzer combining ML disease segmentation with OpenCV defect detection
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the integrated analyzer"""
        self.defect_detector = MangoDefectDetector(config)
        
    def analyze_mango_comprehensive(self, image: np.ndarray, ml_segmentation_result: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive mango analysis combining ML and OpenCV approaches
        
        Args:
            image: Input mango image (BGR format)
            ml_segmentation_result: Results from your existing ML segmentation
                Expected format: {
                    'disease_percentage': float,
                    'mask': np.ndarray,
                    'overlay': np.ndarray
                }
        
        Returns:
            Complete analysis results combining both approaches
        """
        start_time = time.time()
        
        # 1. OpenCV defect detection
        opencv_analysis = self.defect_detector.detect_defects(image)
        
        # 2. Combine with ML results if provided
        total_area = image.shape[0] * image.shape[1]
        
        if ml_segmentation_result:
            # Combine disease and defect areas (avoid double counting overlaps)
            disease_area_pct = ml_segmentation_result.get('disease_percentage', 0)
            defect_area_pct = opencv_analysis.total_defect_percentage
            
            # Simple approach: take maximum (conservative estimate)
            combined_defect_pct = max(disease_area_pct, defect_area_pct)
            
            # Alternative: Add them with overlap reduction (more accurate but complex)
            # This would require intersection analysis of masks
            
        else:
            combined_defect_pct = opencv_analysis.total_defect_percentage
        
        # 3. Generate comprehensive quality assessment
        quality_assessment = self._generate_quality_assessment(
            opencv_analysis, 
            ml_segmentation_result,
            combined_defect_pct
        )
        
        # 4. Export country predictions (placeholder for RAG integration)
        export_recommendations = self._predict_export_suitability(quality_assessment)
        
        processing_time = time.time() - start_time
        
        return {
            # OpenCV defect analysis
            'opencv_defects': {
                'total_defect_percentage': opencv_analysis.total_defect_percentage,
                'defect_count': opencv_analysis.defect_count,
                'defect_regions': opencv_analysis.defect_regions,
                'color_uniformity': opencv_analysis.color_uniformity_score,
                'surface_quality': opencv_analysis.surface_quality_score,
                'export_grade_impact': opencv_analysis.export_grade_impact
            },
            
            # ML disease analysis (if provided)
            'ml_disease': ml_segmentation_result if ml_segmentation_result else None,
            
            # Combined analysis
            'combined_analysis': {
                'total_defect_percentage': combined_defect_pct,
                'overall_quality_score': quality_assessment['overall_score'],
                'export_grade_category': quality_assessment['grade_category'],
                'quality_issues': quality_assessment['issues']
            },
            
            # Export recommendations
            'export_recommendations': export_recommendations,
            
            # Performance metrics
            'performance': {
                'opencv_processing_time': opencv_analysis.processing_time,
                'total_processing_time': processing_time
            }
        }
    
    def _generate_quality_assessment(self, opencv_result: DefectAnalysis, 
                                   ml_result: Dict, combined_defect_pct: float) -> Dict[str, Any]:
        """Generate comprehensive quality assessment"""
        
        # Collect all quality issues
        issues = []
        
        # Check defect percentage thresholds
        if combined_defect_pct > 5.0:
            issues.append("High defect coverage")
        elif combined_defect_pct > 2.0:
            issues.append("Moderate defect coverage")
        
        # Check color uniformity
        if opencv_result.color_uniformity_score < 70:
            issues.append("Poor color uniformity")
        elif opencv_result.color_uniformity_score < 85:
            issues.append("Moderate color uniformity issues")
        
        # Check surface quality
        if opencv_result.surface_quality_score < 70:
            issues.append("Poor surface quality")
        
        # Check for severe defects
        severe_defects = [d for d in opencv_result.defect_regions if d.severity == 'severe']
        if len(severe_defects) > 0:
            issues.append(f"{len(severe_defects)} severe defect(s)")
        
        # Determine overall grade category
        if combined_defect_pct <= 2.0 and opencv_result.color_uniformity_score >= 85:
            grade_category = "Grade A (Premium Export)"
        elif combined_defect_pct <= 5.0 and opencv_result.color_uniformity_score >= 70:
            grade_category = "Grade B (Standard Export)"
        elif combined_defect_pct <= 10.0:
            grade_category = "Grade C (Local Market)"
        else:
            grade_category = "Processing Grade"
        
        # Calculate overall score (0-100)
        overall_score = (
            0.4 * (100 - combined_defect_pct * 10) +  # Defect penalty
            0.3 * opencv_result.color_uniformity_score +
            0.3 * opencv_result.surface_quality_score
        )
        overall_score = max(0, min(100, overall_score))
        
        return {
            'overall_score': overall_score,
            'grade_category': grade_category,
            'issues': issues
        }
    
    def _predict_export_suitability(self, quality_assessment: Dict) -> Dict[str, Any]:
        """
        Predict export suitability for different markets
        This is a placeholder - you'll integrate with your RAG pipeline
        """
        grade = quality_assessment['grade_category']
        issues = quality_assessment['issues']
        
        # Simplified export recommendations based on grade
        if "Grade A" in grade:
            suitable_markets = ["USA", "EU", "Japan", "Australia"]
            restrictions = []
        elif "Grade B" in grade:
            suitable_markets = ["Middle East", "Asia", "Canada"]
            restrictions = ["Premium markets may reject"]
        elif "Grade C" in grade:
            suitable_markets = ["Local markets", "Regional export"]
            restrictions = ["International export not recommended"]
        else:
            suitable_markets = ["Processing only"]
            restrictions = ["Not suitable for fresh export"]
        
        return {
            'suitable_markets': suitable_markets,
            'restrictions': restrictions,
            'rag_query_needed': True,  # Flag for RAG integration
            'feature_description': self._generate_rag_feature_description(quality_assessment)
        }
    
    def _generate_rag_feature_description(self, quality_assessment: Dict) -> str:
        """
        Generate textual description for RAG query
        This will be used as input to your RAG system
        """
        grade = quality_assessment['grade_category']
        score = quality_assessment['overall_score']
        issues = quality_assessment['issues']
        
        description = f"""
        Mango Quality Analysis:
        - Classification Grade: {grade}
        - Overall Quality Score: {score:.1f}/100
        - Quality Issues: {', '.join(issues) if issues else 'None detected'}
        
        Please recommend suitable export countries based on this quality profile.
        """
        
        return description.strip()
    
    def create_comprehensive_visualization(self, image: np.ndarray, 
                                        analysis_result: Dict, 
                                        ml_overlay: np.ndarray = None) -> np.ndarray:
        """
        Create comprehensive visualization showing both ML and OpenCV results
        
        Args:
            image: Original image
            analysis_result: Result from analyze_mango_comprehensive
            ml_overlay: ML segmentation overlay (optional)
        
        Returns:
            Combined visualization image
        """
        # Start with OpenCV defect visualization
        opencv_viz = self.defect_detector.visualize_defects(
            image, 
            analysis_result['opencv_defects']
        )
        
        # If ML overlay provided, create side-by-side comparison
        if ml_overlay is not None:
            # Resize images to same height
            height = opencv_viz.shape[0]
            ml_resized = cv2.resize(ml_overlay, (opencv_viz.shape[1], height))
            
            # Create side-by-side image
            combined = np.hstack([ml_resized, opencv_viz])
            
            # Add labels
            cv2.putText(combined, "ML Disease Segmentation", (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "OpenCV Defect Detection", 
                       (ml_resized.shape[1] + 10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return combined
        
        return opencv_viz


def demo_integration(image_path: str, save_results: bool = True):
    """
    Demo function showing how to integrate with existing pipeline
    
    Args:
        image_path: Path to test mango image
        save_results: Whether to save result images
    """
    print("🍋 Mango Defect Detection Integration Demo")
    print("=" * 50)
    
    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Loaded image: {image.shape}")
    
    # Initialize analyzer
    analyzer = IntegratedMangoAnalyzer()
    
    # Simulate ML segmentation result (replace with actual ML pipeline output)
    mock_ml_result = {
        'disease_percentage': 3.5,
        'mask': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),  # Placeholder
        'overlay': image.copy()  # Placeholder
    }
    
    # Run comprehensive analysis
    print("\nRunning comprehensive analysis...")
    result = analyzer.analyze_mango_comprehensive(image, mock_ml_result)
    
    # Print results
    print("\nOpenCV Defect Detection Results:")
    opencv_defects = result['opencv_defects']
    print(f"  📊 Total defect percentage: {opencv_defects['total_defect_percentage']:.2f}%")
    print(f"  🔍 Number of defects: {opencv_defects['defect_count']}")
    print(f"  🎨 Color uniformity: {opencv_defects['color_uniformity']:.1f}/100")
    print(f"  ✨ Surface quality: {opencv_defects['surface_quality']:.1f}/100")
    print(f"  📈 Export grade impact: {opencv_defects['export_grade_impact']}")
    
    print("\nCombined Analysis:")
    combined = result['combined_analysis']
    print(f"  🏆 Overall quality score: {combined['overall_quality_score']:.1f}/100")
    print(f"  🥇 Grade category: {combined['grade_category']}")
    print(f"  ⚠️ Quality issues: {', '.join(combined['quality_issues']) if combined['quality_issues'] else 'None'}")
    
    print("\nExport Recommendations:")
    export_rec = result['export_recommendations']
    print(f"  🌍 Suitable markets: {', '.join(export_rec['suitable_markets'])}")
    print(f"  🚫 Restrictions: {', '.join(export_rec['restrictions']) if export_rec['restrictions'] else 'None'}")
    
    print("\n🤖 RAG Feature Description (for export country prediction):")
    print(export_rec['feature_description'])
    
    # Save visualization if requested
    if save_results:
        output_dir = Path("defect_detection_results")
        output_dir.mkdir(exist_ok=True)
        
        # Create and save visualization
        viz_image = analyzer.create_comprehensive_visualization(image, result)
        output_path = output_dir / "comprehensive_analysis.jpg"
        cv2.imwrite(str(output_path), viz_image)
        print(f"\n💾 Visualization saved to: {output_path}")
    
    print(f"\n⚡ Processing time: {result['performance']['total_processing_time']:.3f}s")
    print("=" * 50)


if __name__ == "__main__":
    # Demo with a test image
    # Replace with path to one of your mango images
    demo_integration("path/to/your/mango_image.jpg")