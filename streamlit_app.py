"""Streamlit interface for Mango Disease Detection - Backward compatibility wrapper"""
# This file now uses the new modular structure
import streamlit as st

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Mango Disease Detection",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show loading message while heavy imports load
with st.spinner("Loading dependencies..."):
    import cv2
    import numpy as np
    import os
    import tempfile
    from datetime import datetime

    # Import from new modular structure
    from config.settings import Settings
    from fruits.mango import MangoPipeline
    from vision import ROIExtractor, MotionDetector
    from vision.defect_detector import MangoDefectDetector
    from vision.mask_utils import create_mango_hsv_mask
    from vision.export_advisor import ExportAdvisor
    from vision.report_generator import ReportGenerator
    from vision.inspection_logger import log_inspection
    
    # Backward compatibility imports
    import config
    from video_utils import (
        create_background_subtractor,
        extract_roi,
        process_frame_for_motion
    )
    from preprocessing import preprocess_image
    from model_utils import load_model_weights, predict_disease
    from segmentation_utils import load_segmentation_model, segment_disease, get_disease_statistics


# Initialize session state
if 'started' not in st.session_state:
    st.session_state.started = False
if 'pipeline_loaded' not in st.session_state:
    st.session_state.pipeline_loaded = False
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
# Backward compatibility - keep old state variables
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'segmentation_model_loaded' not in st.session_state:
    st.session_state.segmentation_model_loaded = False
if 'segmentation_model' not in st.session_state:
    st.session_state.segmentation_model = None
if 'defect_detector' not in st.session_state:
    st.session_state.defect_detector = MangoDefectDetector()
if 'export_advisor' not in st.session_state:
    st.session_state.export_advisor = ExportAdvisor()  # RAG loaded lazily on first query
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator()
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = []

# Pre-load pipeline on app startup
@st.cache_resource
def initialize_pipeline():
    """Load and cache the mango pipeline on first run"""
    settings = Settings()
    pipeline = MangoPipeline(settings)
    pipeline.load_models()
    return pipeline

# Backward compatibility functions
@st.cache_resource
def initialize_model():
    """Load and cache the TFLite classification model on first run"""
    return load_model_weights()

@st.cache_resource
def initialize_segmentation_model():
    """Load and cache the segmentation model on first run"""
    try:
        return load_segmentation_model()
    except Exception as e:
        print(f"Could not load segmentation model: {e}")
        return None

# Load pipeline immediately in background
if not st.session_state.pipeline_loaded:
    try:
        st.session_state.pipeline = initialize_pipeline()
        st.session_state.pipeline_loaded = True
        # Set backward compatibility state — predict_disease() expects a
        # (interpreter, input_details, output_details) tuple, not the
        # ClassificationModel wrapper, so unwrap it here.
        clf = st.session_state.pipeline.classification_model
        st.session_state.model = (clf.interpreter, clf.input_details, clf.output_details)
        st.session_state.model_loaded = True
        # segment_disease() expects the raw PyTorch model, not the
        # SegmentationModel wrapper, so unwrap .model here.
        seg = st.session_state.pipeline.segmentation_model
        st.session_state.segmentation_model = seg.model if seg is not None else None
        st.session_state.segmentation_model_loaded = True
    except Exception as e:
        pass  # Will load on demand if this fails


def load_model():
    """Load the TFLite model (reuses main pipeline function)"""
    if not st.session_state.model_loaded:
        # Model already loaded on startup via cache, just update state
        if st.session_state.model is None:
            with st.spinner("Loading model..."):
                st.session_state.model = initialize_model()
                st.session_state.model_loaded = True
        st.success("Model ready!")


def process_video_with_ui(video_path, video_col, frame_col, status_placeholder):
    """
    Process video with real-time UI updates
    Uses functions from main pipeline (video_utils, preprocessing, model_utils)
    """
    
    cap = cv2.VideoCapture(video_path)
    fgbg = create_background_subtractor()
    
    frame_idx = 0
    in_motion = False
    motion_buffer = []
    low_motion_counter = 0
    detected_frames = []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create UI placeholders
    video_placeholder = video_col.empty()
    frame_placeholder = frame_col.empty()
    info_placeholder = frame_col.empty()
    prediction_placeholder = frame_col.empty()
    
    status_placeholder.info(f"📹 Processing video: {total_frames} frames at {fps} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        roi = extract_roi(frame)  # From video_utils
        
        if roi.size == 0:
            break
        
        # Use video_utils function for motion detection
        combined_mask, motion_area = process_frame_for_motion(frame, roi, fgbg)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Display current video frame (every 5 frames for performance)
        if frame_idx % 5 == 0:
            display_frame = frame.copy()
            x1, y1, x2, y2 = config.ROI_COORDS
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame_rgb, channels="RGB", width='stretch')
        
        # Motion detection logic (same as main pipeline)
        if not in_motion and motion_area > config.MOTION_AREA_THRESHOLD:
            in_motion = True
            motion_buffer = []
            low_motion_counter = 0
            status_placeholder.warning(f"🔍 Motion detected at frame {frame_idx}")
        
        if in_motion:
            motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))
            
            if motion_area < config.MOTION_AREA_THRESHOLD:
                low_motion_counter += 1
            else:
                low_motion_counter = 0
            
            if low_motion_counter >= config.MOTION_END_FRAMES:
                in_motion = False
                if motion_buffer:
                    best_frame, best_area, best_idx, best_time = max(
                        motion_buffer, key=lambda x: x[1]
                    )
                    
                    status_placeholder.info(f"✅ Peak frame selected: Frame {best_idx}")
                    
                    # Extract ROI
                    roi_crop = extract_roi(best_frame)  # From video_utils
                    
                    # Display original
                    roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(roi_rgb, caption="🥭 Detected Mango", 
                                          width='stretch')
                    
                    # Preprocess using main pipeline function
                    info_placeholder.info("⚙️ Preprocessing image...")
                    processed_image = preprocess_image(roi_crop)  # From preprocessing
                    
                    # Defect detection — compute HSV mask on raw ROI,
                    # resize to 224×224, then run detector on preprocessed
                    info_placeholder.info("🔎 Running defect detection...")
                    hsv_mask = create_mango_hsv_mask(roi_crop)
                    mask_224 = cv2.resize(hsv_mask, (224, 224),
                                          interpolation=cv2.INTER_NEAREST)
                    _, mask_224 = cv2.threshold(mask_224, 127, 255,
                                                cv2.THRESH_BINARY)
                    defect_result = st.session_state.defect_detector.detect_defects(
                        processed_image, mask=mask_224
                    )
                    
                    # Predict using main pipeline function
                    info_placeholder.info("🤖 Running classification...")
                    prediction = predict_disease(  # From model_utils
                        st.session_state.model, 
                        processed_image, 
                        return_probabilities=True
                    )
                    
                    # Perform segmentation for all mangoes
                    should_segment = (config.ENABLE_SEGMENTATION and 
                                    st.session_state.segmentation_model is not None)
                    
                    segmentation_result = None
                    if should_segment:
                        info_placeholder.info("🔬 Running segmentation...")
                        segmentation_result = segment_disease(
                            st.session_state.segmentation_model,
                            roi_crop
                        )
                        
                        # Display segmented overlay
                        overlay_rgb = cv2.cvtColor(segmentation_result['overlay'], cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(overlay_rgb, caption="🥭 Disease Segmentation", 
                                              width='stretch')
                        
                        # Display results with segmentation + defect info
                        prediction_placeholder.success(
                            f"**Prediction:** {prediction['class_name']}\n\n"
                            f"**Confidence:** {prediction['confidence']:.2%}\n\n"
                            f"**Disease Area:** {segmentation_result['disease_percentage']:.2f}%\n\n"
                            f"**Surface Quality:** {defect_result.surface_quality_score:.0f}/100\n\n"
                            f"**Export Impact:** {defect_result.export_grade_impact}\n\n"
                            f"**Defects:** {defect_result.defect_count} "
                            f"(dark: {defect_result.dark_spot_count}, brown: {defect_result.brown_spot_count})"
                        )
                    else:
                        # Display results without segmentation
                        prediction_placeholder.success(
                            f"**Prediction:** {prediction['class_name']}\n\n"
                            f"**Confidence:** {prediction['confidence']:.2%}\n\n"
                            f"**Surface Quality:** {defect_result.surface_quality_score:.0f}/100\n\n"
                            f"**Export Impact:** {defect_result.export_grade_impact}\n\n"
                            f"**Defects:** {defect_result.defect_count} "
                            f"(dark: {defect_result.dark_spot_count}, brown: {defect_result.brown_spot_count})"
                        )
                    
                    # Show probabilities
                    with frame_col.expander("📊 All Class Probabilities"):
                        for class_name, prob in prediction['probabilities'].items():
                            st.progress(prob, text=f"{class_name}: {prob:.2%}")
                    
                    # Generate defect visualisation
                    defect_vis = st.session_state.defect_detector.visualize_defects(
                        processed_image, defect_result
                    )
                    
                    # Show defect overlay in an expander
                    with frame_col.expander("🔎 Defect Analysis"):
                        vis_rgb = cv2.cvtColor(defect_vis, cv2.COLOR_BGR2RGB)
                        st.image(vis_rgb, caption="Defect Overlay", width='stretch')
                        st.write(f"**Dark spots:** {defect_result.dark_spot_count}")
                        st.write(f"**Brown spots:** {defect_result.brown_spot_count}")
                        st.write(f"**Defect area:** {defect_result.total_defect_percentage:.2f}%")
                        st.write(f"**Colour uniformity:** {defect_result.color_uniformity_score:.0f}/100")
                        st.write(f"**Surface quality:** {defect_result.surface_quality_score:.0f}/100")
                        st.write(f"**Export impact:** {defect_result.export_grade_impact}")
                    
                    # Store result
                    result_data = {
                        'frame_idx': best_idx,
                        'timestamp_ms': best_time,
                        'motion_area': best_area,
                        'prediction': prediction,
                        'original_roi': roi_crop,
                        'processed_roi': processed_image,
                        'defect_analysis': {
                            'defect_count': defect_result.defect_count,
                            'dark_spot_count': defect_result.dark_spot_count,
                            'brown_spot_count': defect_result.brown_spot_count,
                            'total_defect_percentage': defect_result.total_defect_percentage,
                            'color_uniformity_score': defect_result.color_uniformity_score,
                            'surface_quality_score': defect_result.surface_quality_score,
                            'export_grade_impact': defect_result.export_grade_impact,
                            'defect_visualization': defect_vis,
                        }
                    }
                    
                    if segmentation_result:
                        result_data['segmentation'] = {
                            'disease_percentage': segmentation_result['disease_percentage'],
                            'overlay': segmentation_result['overlay'],
                            'mask': segmentation_result['mask'],
                            'segmentation_time': segmentation_result['segmentation_time']
                        }
                    else:
                        result_data['segmentation'] = None
                    
                    detected_frames.append(result_data)
                
                motion_buffer = []
                low_motion_counter = 0
    
    # Handle edge case: motion still ongoing at end
    if in_motion and motion_buffer:
        best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
        roi_crop = extract_roi(best_frame)
        processed_image = preprocess_image(roi_crop)
        prediction = predict_disease(st.session_state.model, processed_image, 
                                    return_probabilities=True)
        
        # Defect detection on edge-case frame
        hsv_mask = create_mango_hsv_mask(roi_crop)
        mask_224 = cv2.resize(hsv_mask, (224, 224),
                              interpolation=cv2.INTER_NEAREST)
        _, mask_224 = cv2.threshold(mask_224, 127, 255, cv2.THRESH_BINARY)
        defect_result = st.session_state.defect_detector.detect_defects(
            processed_image, mask=mask_224
        )
        defect_vis = st.session_state.defect_detector.visualize_defects(
            processed_image, defect_result
        )
        
        result_data = {
            'frame_idx': best_idx,
            'timestamp_ms': best_time,
            'motion_area': best_area,
            'prediction': prediction,
            'original_roi': roi_crop,
            'processed_roi': processed_image,
            'defect_analysis': {
                'defect_count': defect_result.defect_count,
                'dark_spot_count': defect_result.dark_spot_count,
                'brown_spot_count': defect_result.brown_spot_count,
                'total_defect_percentage': defect_result.total_defect_percentage,
                'color_uniformity_score': defect_result.color_uniformity_score,
                'surface_quality_score': defect_result.surface_quality_score,
                'export_grade_impact': defect_result.export_grade_impact,
                'defect_visualization': defect_vis,
            },
            'segmentation': None
        }
        
        # Perform segmentation for all mangoes
        if (config.ENABLE_SEGMENTATION and 
            st.session_state.segmentation_model is not None):
            segmentation_result = segment_disease(
                st.session_state.segmentation_model,
                roi_crop
            )
            result_data['segmentation'] = {
                'disease_percentage': segmentation_result['disease_percentage'],
                'overlay': segmentation_result['overlay'],
                'mask': segmentation_result['mask'],
                'segmentation_time': segmentation_result['segmentation_time']
            }
        
        detected_frames.append(result_data)
    
    cap.release()
    status_placeholder.success(f"✅ Processing complete! Detected {len(detected_frames)} mangoes")
    
    return detected_frames


def main_page():
    """Main landing page with start button"""
    st.title("🥭 Mango Disease Detection System")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Mango Disease Detection System
    
    This application uses AI to detect and classify mango diseases in real-time.
    
    **Features:**
    - 🎥 Process video files or live camera feed
    - 🔍 Automatic mango detection using motion tracking
    - 🤖 AI-powered disease classification
    - 📊 Real-time visualization and results
    
    **Detected Diseases:**
    - Alternaria
    - Anthracnose
    - Black Mould Rot
    - Stem End Rot
    - Healthy
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Source selection
        source_type = st.radio(
            "Select Input Source:",
            ["Video File", "Camera Feed (Coming Soon)"],
            index=0
        )
        
        if source_type == "Video File":
            # Video file upload or selection
            use_default = st.checkbox("Use default video (live_recording1.mp4)", value=True)
            
            if use_default:
                video_path = config.VIDEO_SOURCE
                if os.path.exists(video_path):
                    st.success(f"✅ Using: {video_path}")
                else:
                    st.error(f"❌ Default video not found: {video_path}")
                    video_path = None
            else:
                uploaded_file = st.file_uploader("Upload a video file", 
                                                type=['mp4', 'avi', 'mov', 'mkv'])
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    st.success("✅ Video uploaded successfully!")
                else:
                    video_path = None
        else:
            st.info("📹 Camera feed support coming soon!")
            video_path = None
        
        st.markdown("---")
        
        # Start button
        if st.button("🚀 Start Detection", type="primary", width='stretch'):
            if video_path and os.path.exists(video_path):
                load_model()  # Load model using main pipeline function
                st.session_state.started = True
                st.session_state.video_path = video_path
                st.session_state.processing_complete = False
                st.rerun()
            else:
                st.error("❌ Please select a valid video source!")


def processing_page():
    """Processing page with video display and results"""
    st.title("🥭 Mango Disease Detection - Live Processing")
    
    # Back button
    if st.button("⬅️ Back to Home"):
        st.session_state.started = False
        st.session_state.processing_complete = False
        st.session_state.results = []
        st.rerun()
    
    st.markdown("---")
    
    # Create two columns for video and frame display
    video_col, frame_col = st.columns([2, 1])
    
    with video_col:
        st.subheader("📹 Video Feed")
    
    with frame_col:
        st.subheader("🖼️ Selected Frame & Results")
    
    # Status placeholder
    status_placeholder = st.empty()
    
    # Process video using integrated pipeline functions
    if not st.session_state.processing_complete:
        results = process_video_with_ui(
            st.session_state.video_path,
            video_col,
            frame_col,
            status_placeholder
        )
        
        st.session_state.results = results
        st.session_state.processing_complete = True

    # ── Always render results after processing is done ────────────────────
    # This block runs on every rerun (including button clicks) so that
    # widgets like the RAG recommendation button remain interactive.
    if st.session_state.processing_complete:
        results = st.session_state.results
        
        # Show summary
        st.markdown("---")
        st.subheader("📋 Detection Summary")
        
        if results:
            # Create summary table
            summary_data = []
            for idx, result in enumerate(results, 1):
                da = result.get('defect_analysis', {})
                summary_data.append({
                    'Detection #': idx,
                    'Frame': result['frame_idx'],
                    'Time (ms)': result['timestamp_ms'],
                    'Prediction': result['prediction']['class_name'],
                    'Confidence': f"{result['prediction']['confidence']:.2%}",
                    'Quality': f"{da.get('surface_quality_score', 0):.0f}/100",
                    'Defects': da.get('defect_count', 0),
                    'Export Impact': da.get('export_grade_impact', 'N/A'),
                })
            
            st.dataframe(summary_data, width='stretch')

            # ── Per-mango details & recommendations ───────────────
            st.markdown("---")
            st.subheader("🔍 Inspection Details")

            for idx, result in enumerate(results, 1):
                with st.expander(f"Detection #{idx} - {result['prediction']['class_name']}"):
                    # Defect analysis metrics
                    da = result.get('defect_analysis', {})
                    if da:
                        st.markdown("**🔎 Defect Analysis**")
                        d_col1, d_col2, d_col3 = st.columns(3)
                        with d_col1:
                            st.metric("Surface Quality", f"{da.get('surface_quality_score', 0):.0f}/100")
                        with d_col2:
                            st.metric("Colour Uniformity", f"{da.get('color_uniformity_score', 0):.0f}/100")
                        with d_col3:
                            st.metric("Export Impact", da.get('export_grade_impact', 'N/A'))
                        st.write(f"**Dark spots:** {da.get('dark_spot_count', 0)} | "
                                 f"**Brown spots:** {da.get('brown_spot_count', 0)} | "
                                 f"**Defect area:** {da.get('total_defect_percentage', 0):.2f}%")

                    # Segmentation disease %
                    has_seg = result.get('segmentation') is not None
                    if has_seg and result['segmentation']['disease_percentage'] > 0:
                        st.write(f"**Disease coverage:** {result['segmentation']['disease_percentage']:.2f}%")

                    # ── Export country recommendation via RAG ──────────────
                    st.markdown("---")
                    rec_key = f"rec_{idx}"
                    if st.button("🌍 Get Export Country Recommendation",
                                 key=f"btn_rec_{idx}"):
                        with st.spinner("Querying export regulations..."):
                            try:
                                seg_pct = (
                                    result['segmentation']['disease_percentage']
                                    if result.get('segmentation') else 0.0
                                )
                                metadata = st.session_state.export_advisor.build_metadata(
                                    defect_analysis=result.get('defect_analysis', {}),
                                    disease_percentage=seg_pct,
                                )
                                rec = st.session_state.export_advisor.get_recommendation(metadata)
                                st.session_state[rec_key] = rec
                            except Exception as e:
                                st.session_state[rec_key] = {
                                    "status": "error",
                                    "answer": f"Error: {e}",
                                    "sources": []
                                }

                    if rec_key in st.session_state:
                        rec = st.session_state[rec_key]
                        if rec.get('status') == 'success':
                            st.markdown("**🌍 Export Recommendation:**")
                            st.markdown(rec['answer'])
                            if rec.get('sources'):
                                with st.expander("📚 Regulatory Sources"):
                                    for s in rec['sources']:
                                        st.write(f"**{s['index']}.** {s['source']} — {s['section']}")
                                        st.caption(s['content_preview'])
                        else:
                            st.error(rec.get('answer', 'Unknown error'))

            # ── PDF Report (shown after recommendations area) ─────
            st.markdown("---")
            dl_col1, dl_col2 = st.columns([1, 3])
            with dl_col1:
                generate_report = st.button("📄 Generate PDF Report", type="primary")
            if generate_report:
                with st.spinner("Generating PDF report..."):
                    try:
                        # Collect any recommendations that have been fetched
                        recs = {}
                        for i in range(1, len(results) + 1):
                            rk = f"rec_{i}"
                            if rk in st.session_state:
                                recs[i] = st.session_state[rk]
                        video_name = os.path.basename(
                            st.session_state.get("video_path", "")
                        )
                        pdf_bytes = st.session_state.report_generator.generate(
                            results,
                            recommendations=recs,
                            video_name=video_name,
                        )
                        st.session_state["report_pdf"] = pdf_bytes
                        st.session_state["report_name"] = (
                            f"mango_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        )
                        # Auto-log inspection details to text file
                        try:
                            log_path = log_inspection(
                                results=results,
                                recommendations=recs,
                                video_name=video_name,
                            )
                            st.toast(f"Inspection logged to {log_path.name}")
                        except Exception as log_err:
                            st.warning(f"Logging failed: {log_err}")
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")

            if "report_pdf" in st.session_state:
                with dl_col2:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=st.session_state["report_pdf"],
                        file_name=st.session_state.get("report_name", "mango_report.pdf"),
                        mime="application/pdf",
                    )
        else:
            st.warning("No mangoes detected in the video.")


# Main app logic
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("### Model Configuration")
        st.info(f"**Model:** {config.MODEL_PATH.split('/')[-1]}")
        st.info(f"**Input Shape:** {config.INPUT_SHAPE[0]}x{config.INPUT_SHAPE[1]}")
        
        st.markdown("### Detection Parameters")
        st.info(f"**Motion Threshold:** {config.MOTION_AREA_THRESHOLD}")
        st.info(f"**Motion End Frames:** {config.MOTION_END_FRAMES}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system uses computer vision and deep learning 
        to detect and classify mango diseases automatically.
        
        **Pipeline Components:**
        - `video_utils.py` - Motion detection
        - `preprocessing.py` - Image processing
        - `model_utils.py` - TFLite inference
        - `main.py` - CLI pipeline
        - `streamlit_app.py` - Web UI
        """)
    
    # Route to appropriate page
    if not st.session_state.started:
        main_page()
    else:
        processing_page()


if __name__ == "__main__":
    main()
