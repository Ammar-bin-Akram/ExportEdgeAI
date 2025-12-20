"""Streamlit web interface for mango processing pipeline"""
import streamlit as st

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Export Edge AI",
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
    
    from config.settings import Settings
    from fruits.mango import MangoPipeline
    from vision import ROIExtractor


# Initialize session state
if 'started' not in st.session_state:
    st.session_state.started = False
if 'pipeline_loaded' not in st.session_state:
    st.session_state.pipeline_loaded = False
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
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


# Load pipeline immediately in background
if not st.session_state.pipeline_loaded:
    try:
        st.session_state.pipeline = initialize_pipeline()
        st.session_state.pipeline_loaded = True
    except Exception as e:
        pass  # Will load on demand if this fails


def process_video_with_ui(video_path, video_col, frame_col, status_placeholder):
    """Process video with real-time UI updates using the pipeline"""
    
    pipeline = st.session_state.pipeline
    settings = pipeline.settings
    
    cap = cv2.VideoCapture(video_path)
    
    # Motion detection setup
    from vision import MotionDetector, ROIExtractor
    motion_detector = MotionDetector(settings)
    roi_extractor = ROIExtractor(settings)
    
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
        roi = roi_extractor.extract_roi(frame)
        
        if roi.size == 0:
            break
        
        # Motion detection
        combined_mask, motion_area = motion_detector.detect_motion(roi)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Display current video frame (every 5 frames for performance)
        if frame_idx % 5 == 0:
            display_frame = roi_extractor.visualize_roi(frame)
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
        
        # Motion detection logic
        if not in_motion and motion_detector.is_motion_detected(motion_area):
            in_motion = True
            motion_buffer = []
            low_motion_counter = 0
            status_placeholder.warning(f"🔍 Motion detected at frame {frame_idx}")
        
        if in_motion:
            motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))
            
            if motion_area < settings.MOTION_AREA_THRESHOLD:
                low_motion_counter += 1
            else:
                low_motion_counter = 0
            
            if low_motion_counter >= settings.MOTION_END_FRAMES:
                in_motion = False
                if motion_buffer:
                    best_frame, best_area, best_idx, best_time = max(
                        motion_buffer, key=lambda x: x[1]
                    )
                    
                    status_placeholder.info(f"✅ Peak frame selected: Frame {best_idx}")
                    
                    # Process through pipeline
                    info_placeholder.info("⚙️ Processing...")
                    result = pipeline.process_single_frame_detailed(best_frame)
                    
                    # Display original
                    roi_rgb = cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(roi_rgb, caption="🥭 Detected Mango", 
                                          use_container_width=True)
                    
                    # Check if segmentation was performed
                    if result['segmentation_overlay'] is not None:
                        # Display segmented overlay
                        overlay_rgb = cv2.cvtColor(result['segmentation_overlay'], cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(overlay_rgb, caption="🥭 Disease Segmentation", 
                                              use_container_width=True)
                        
                        # Display results with segmentation info
                        prediction_placeholder.success(
                            f"**Prediction:** {result['class_name']}\n\n"
                            f"**Confidence:** {result['confidence']:.2%}\n\n"
                            f"**Disease Area:** {result['disease_percentage']:.2f}%"
                        )
                        
                        # Show disease statistics
                        if 'disease_statistics' in result:
                            stats = result['disease_statistics']
                            with frame_col.expander("🔬 Disease Statistics"):
                                st.write(f"**Number of Regions:** {stats['num_regions']}")
                                st.write(f"**Largest Region:** {stats['largest_region_area']:.0f} pixels")
                                st.write(f"**Average Region:** {stats['average_region_area']:.0f} pixels")
                    else:
                        # Display results without segmentation
                        prediction_placeholder.success(
                            f"**Prediction:** {result['class_name']}\n\n"
                            f"**Confidence:** {result['confidence']:.2%}"
                        )
                    
                    # Show top-k predictions
                    if 'top_k_predictions' in result:
                        with frame_col.expander("📊 Top Predictions"):
                            for class_name, prob in result['top_k_predictions']:
                                st.progress(prob, text=f"{class_name}: {prob:.2%}")
                    
                    # Store result
                    result_data = {
                        'frame_idx': best_idx,
                        'timestamp_ms': best_time,
                        'motion_area': best_area,
                        'result': result
                    }
                    
                    detected_frames.append(result_data)
                
                motion_buffer = []
                low_motion_counter = 0
    
    # Handle edge case: motion still ongoing at end
    if in_motion and motion_buffer:
        best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
        result = pipeline.process_single_frame_detailed(best_frame)
        
        result_data = {
            'frame_idx': best_idx,
            'timestamp_ms': best_time,
            'motion_area': best_area,
            'result': result
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
    
    This system uses computer vision and deep learning to:
    - 🎥 **Extract frames** from video with motion detection
    - 🔍 **Classify** mango varieties and health status
    - 🔬 **Segment** diseased regions (if detected)
    - 📊 **Analyze** disease coverage and statistics
    
    **Supported Varieties:**
    - Anwar Ratool
    - Chaunsa (Black, Summer Bahisht, White)
    - Dosehri
    - Fajri
    - Langra
    - Sindhri
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Processing", use_container_width=True, type="primary"):
            if not st.session_state.pipeline_loaded:
                with st.spinner("Loading pipeline..."):
                    st.session_state.pipeline = initialize_pipeline()
                    st.session_state.pipeline_loaded = True
            st.session_state.started = True
            st.rerun()


def processing_page():
    """Processing page with video upload and results"""
    st.title("🥭 Mango Disease Detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model status
        st.subheader("🤖 Model Status")
        if st.session_state.pipeline_loaded:
            st.success("✅ Pipeline loaded")
            pipeline = st.session_state.pipeline
            st.info(f"🔬 Segmentation: {'Enabled' if pipeline.settings.ENABLE_SEGMENTATION else 'Disabled'}")
        else:
            st.warning("⚠️ Pipeline not loaded")
        
        st.markdown("---")
        
        # Navigation
        if st.button("⬅️ Back to Home"):
            st.session_state.started = False
            st.session_state.processing_complete = False
            st.session_state.results = []
            st.rerun()
    
    # Main content
    st.markdown("### 📤 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video containing mangoes on a conveyor belt"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ Process Video", use_container_width=True, type="primary"):
                st.session_state.processing_complete = False
                
                # Create layout for real-time display
                st.markdown("---")
                st.markdown("### 🎬 Real-time Processing")
                
                status_placeholder = st.empty()
                video_col, frame_col = st.columns(2)
                
                video_col.markdown("#### 📹 Video Stream")
                frame_col.markdown("#### 🥭 Detection Results")
                
                # Process video
                try:
                    results = process_video_with_ui(
                        video_path,
                        video_col,
                        frame_col,
                        status_placeholder
                    )
                    
                    st.session_state.results = results
                    st.session_state.processing_complete = True
                    
                    # Clean up temp file
                    os.unlink(video_path)
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    os.unlink(video_path)
                    return
        
        # Display results if processing complete
        if st.session_state.processing_complete and st.session_state.results:
            st.markdown("---")
            st.markdown("### 📊 Processing Summary")
            
            results = st.session_state.results
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Mangoes", len(results))
            
            # Count varieties
            variety_counts = {}
            for r in results:
                variety = r['result']['class_name']
                variety_counts[variety] = variety_counts.get(variety, 0) + 1
            
            with col2:
                st.metric("Varieties Detected", len(variety_counts))
            
            # Count diseased
            diseased_count = sum(1 for r in results if r['result']['disease_percentage'] > 0)
            
            with col3:
                st.metric("Diseased Mangoes", diseased_count)
            
            # Average disease coverage
            if diseased_count > 0:
                avg_disease = sum(r['result']['disease_percentage'] for r in results 
                                if r['result']['disease_percentage'] > 0) / diseased_count
                with col4:
                    st.metric("Avg Disease Coverage", f"{avg_disease:.1f}%")
            
            # Variety distribution
            st.markdown("#### 🥭 Variety Distribution")
            variety_data = {variety: count for variety, count in sorted(variety_counts.items())}
            st.bar_chart(variety_data)
            
            # Detailed results
            st.markdown("#### 📋 Detailed Results")
            
            for idx, r in enumerate(results, 1):
                result = r['result']
                
                with st.expander(f"Mango {idx} - {result['class_name']} ({result['confidence']:.1%})"):
                    # Create columns for images
                    img_cols = st.columns(3)
                    
                    # Original
                    with img_cols[0]:
                        st.markdown("**Original**")
                        orig_rgb = cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB)
                        st.image(orig_rgb, use_container_width=True)
                    
                    # Preprocessed
                    with img_cols[1]:
                        st.markdown("**Preprocessed**")
                        prep_rgb = cv2.cvtColor(result['preprocessed'], cv2.COLOR_BGR2RGB)
                        st.image(prep_rgb, use_container_width=True)
                    
                    # Segmentation (if available)
                    if result['segmentation_overlay'] is not None:
                        with img_cols[2]:
                            st.markdown("**Disease Segmentation**")
                            seg_rgb = cv2.cvtColor(result['segmentation_overlay'], cv2.COLOR_BGR2RGB)
                            st.image(seg_rgb, use_container_width=True)
                    
                    # Metadata
                    st.markdown("**Details:**")
                    info_cols = st.columns(2)
                    with info_cols[0]:
                        st.write(f"Frame Index: {r['frame_idx']}")
                        st.write(f"Timestamp: {r['timestamp_ms']/1000:.2f}s")
                        st.write(f"Confidence: {result['confidence']:.2%}")
                    
                    with info_cols[1]:
                        if result['disease_percentage'] > 0:
                            st.write(f"Disease Area: {result['disease_percentage']:.2f}%")
                            if 'disease_statistics' in result:
                                stats = result['disease_statistics']
                                st.write(f"Disease Regions: {stats['num_regions']}")


def main():
    """Main app entry point"""
    if not st.session_state.started:
        main_page()
    else:
        processing_page()


if __name__ == "__main__":
    main()
