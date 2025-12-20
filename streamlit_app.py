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

    # Import from new modular structure
    from config.settings import Settings
    from fruits.mango import MangoPipeline
    from vision import ROIExtractor, MotionDetector
    
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
        # Set backward compatibility state
        st.session_state.model = st.session_state.pipeline.classification_model
        st.session_state.model_loaded = True
        st.session_state.segmentation_model = st.session_state.pipeline.segmentation_model
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
            video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
        
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
                                          use_container_width=True)
                    
                    # Preprocess using main pipeline function
                    info_placeholder.info("⚙️ Preprocessing image...")
                    processed_image = preprocess_image(roi_crop)  # From preprocessing
                    
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
                                              use_container_width=True)
                        
                        # Display results with segmentation info
                        prediction_placeholder.success(
                            f"**Prediction:** {prediction['class_name']}\n\n"
                            f"**Confidence:** {prediction['confidence']:.2%}\n\n"
                            f"**Disease Area:** {segmentation_result['disease_percentage']:.2f}%\n\n"
                            f"**Inference Time:** {prediction['inference_time']:.3f}s\n\n"
                            f"**Segmentation Time:** {segmentation_result['segmentation_time']:.3f}s"
                        )
                    else:
                        # Display results without segmentation
                        prediction_placeholder.success(
                            f"**Prediction:** {prediction['class_name']}\n\n"
                            f"**Confidence:** {prediction['confidence']:.2%}\n\n"
                            f"**Inference Time:** {prediction['inference_time']:.3f}s"
                        )
                    
                    # Show probabilities
                    with frame_col.expander("📊 All Class Probabilities"):
                        for class_name, prob in prediction['probabilities'].items():
                            st.progress(prob, text=f"{class_name}: {prob:.2%}")
                    
                    # Store result
                    result_data = {
                        'frame_idx': best_idx,
                        'timestamp_ms': best_time,
                        'motion_area': best_area,
                        'prediction': prediction,
                        'original_roi': roi_crop,
                        'processed_roi': processed_image
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
        
        result_data = {
            'frame_idx': best_idx,
            'timestamp_ms': best_time,
            'motion_area': best_area,
            'prediction': prediction,
            'original_roi': roi_crop,
            'processed_roi': processed_image,
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
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
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
        
        # Show summary
        st.markdown("---")
        st.subheader("📋 Detection Summary")
        
        if results:
            # Create summary table
            summary_data = []
            for idx, result in enumerate(results, 1):
                summary_data.append({
                    'Detection #': idx,
                    'Frame': result['frame_idx'],
                    'Time (ms)': result['timestamp_ms'],
                    'Prediction': result['prediction']['class_name'],
                    'Confidence': f"{result['prediction']['confidence']:.2%}"
                })
            
            st.dataframe(summary_data, use_container_width=True)
            
            # Show detailed results
            st.markdown("---")
            st.subheader("🔍 Detailed Results")
            
            for idx, result in enumerate(results, 1):
                with st.expander(f"Detection #{idx} - {result['prediction']['class_name']}"):
                    # Show images - 3 columns if segmentation, 2 otherwise
                    if result['segmentation']:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.image(cv2.cvtColor(result['original_roi'], cv2.COLOR_BGR2RGB),
                                   caption="Original", use_container_width=True)
                        
                        with col2:
                            st.image(cv2.cvtColor(result['processed_roi'], cv2.COLOR_BGR2RGB),
                                   caption="Preprocessed", use_container_width=True)
                        
                        with col3:
                            st.image(cv2.cvtColor(result['segmentation']['overlay'], cv2.COLOR_BGR2RGB),
                                   caption="Segmented", use_container_width=True)
                    else:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(cv2.cvtColor(result['original_roi'], cv2.COLOR_BGR2RGB),
                                   caption="Original", use_container_width=True)
                        
                        with col2:
                            st.image(cv2.cvtColor(result['processed_roi'], cv2.COLOR_BGR2RGB),
                                   caption="Preprocessed", use_container_width=True)
                    
                    st.write(f"**Frame:** {result['frame_idx']}")
                    st.write(f"**Timestamp:** {result['timestamp_ms']} ms")
                    st.write(f"**Motion Area:** {result['motion_area']:.0f} pixels")
                    st.write(f"**Prediction:** {result['prediction']['class_name']}")
                    st.write(f"**Confidence:** {result['prediction']['confidence']:.2%}")
                    
                    # Show segmentation info if available
                    if result['segmentation']:
                        st.write(f"**Disease Area:** {result['segmentation']['disease_percentage']:.2f}%")
                        st.write(f"**Segmentation Time:** {result['segmentation']['segmentation_time']:.3f}s")
                    
                    st.write("**Probability Distribution:**")
                    for class_name, prob in result['prediction']['probabilities'].items():
                        st.progress(prob, text=f"{class_name}: {prob:.2%}")
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
