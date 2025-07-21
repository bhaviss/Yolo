import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Space Debris Detection System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for space theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007acc;
    }
    .detection-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Space Debris Detection System</h1>
    <p>Advanced YOLOv8-based detection for orbital debris monitoring</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLOv8 model"""
    try:
        model = YOLO('best.pt')  # Your trained model
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure 'best.pt' is in your repository root directory")
        return None

def detect_debris(image, model, conf_threshold=0.5):
    """Perform debris detection on image"""
    if model is None:
        return None, None
    
    # Run detection
    results = model(image, conf=conf_threshold)
    
    # Get annotated image
    annotated_img = results[0].plot()
    
    # Extract detection info
    boxes = results[0].boxes
    detection_data = []
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            detection_data.append({
                'object_id': i + 1,
                'confidence': conf,
                'class': cls,
                'bbox': [x1, y1, x2, y2],
                'area': (x2 - x1) * (y2 - y1)
            })
    
    return annotated_img, detection_data

def main():
    # Sidebar
    st.sidebar.markdown("### 🎛️ Detection Settings")
    
    # Load model
    model = load_model()
    
    if model is not None:
        # Model info
        st.sidebar.success("✅ YOLOv8 Model Loaded Successfully")
        
        # Confidence threshold
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for debris detection"
        )
        
        # Detection mode
        st.sidebar.markdown("### 📊 Analysis Mode")
        show_stats = st.sidebar.checkbox("Show Detection Statistics", value=True)
        show_original = st.sidebar.checkbox("Show Original Image", value=True)
        
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Space Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload satellite imagery or space photos for debris detection"
        )
        
        # Demo images section
        st.markdown("### 🖼️ Or try demo images:")
        demo_option = st.selectbox(
            "Select demo image",
            ["None", "Satellite View", "Space Station", "Orbital Debris"],
            help="Select a demo image if you don't have your own"
        )
    
    with col2:
        st.markdown("### 🔍 Detection Results")
        results_container = st.container()
    
    # Process image
    if uploaded_file is not None and model is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        if show_original:
            st.markdown("### 📸 Original Image")
            st.image(image, caption="Original Space Image", use_container_width=True)
        
        # Perform detection
        with st.spinner("🔍 Analyzing image for space debris..."):
            annotated_img, detection_data = detect_debris(image, model, conf_threshold)
        
        if annotated_img is not None:
            # Display results
            with results_container:
                st.markdown("### 🎯 Detection Results")
                st.image(annotated_img, caption="Space Debris Detection Results", use_container_width=True)
                
                # Detection statistics
                if detection_data and show_stats:
                    st.markdown("### 📊 Detection Summary")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_detections = len(detection_data)
                    avg_confidence = np.mean([d['confidence'] for d in detection_data]) if detection_data else 0
                    high_conf_count = sum(1 for d in detection_data if d['confidence'] > 0.8)
                    total_area = sum(d['area'] for d in detection_data)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="detection-stats">
                            <h3>{total_detections}</h3>
                            <p>Total Debris Objects</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="detection-stats">
                            <h3>{avg_confidence:.2f}</h3>
                            <p>Average Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="detection-stats">
                            <h3>{high_conf_count}</h3>
                            <p>High Confidence (>0.8)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="detection-stats">
                            <h3>{total_area:.0f}</h3>
                            <p>Total Detection Area</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed results table
                    st.markdown("### 📋 Detailed Detection Results")
                    
                    if detection_data:
                        import pandas as pd
                        df = pd.DataFrame([
                            {
                                'Object ID': d['object_id'],
                                'Confidence': f"{d['confidence']:.3f}",
                                'Bounding Box': f"({d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, {d['bbox'][2]:.0f}, {d['bbox'][3]:.0f})",
                                'Area (pixels)': f"{d['area']:.0f}",
                                'Risk Level': 'High' if d['confidence'] > 0.8 else 'Medium' if d['confidence'] > 0.6 else 'Low'
                            }
                            for d in detection_data
                        ])
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Detection Results (CSV)",
                            data=csv,
                            file_name="space_debris_detection_results.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.info("No debris detected in the image with current confidence threshold.")
        else:
            st.error("Failed to process image. Please try again.")
    
    elif model is None:
        st.warning("⚠️ Model not loaded. Please ensure 'best.pt' is available.")
        st.info("""
        **Setup Instructions:**
        1. Train your YOLOv8 model for space debris detection
        2. Save the best weights as 'best.pt'
        3. Upload 'best.pt' to your repository root directory
        4. Deploy on Streamlit Cloud
        """)
    
    else:
        st.info("👆 Please upload an image to start debris detection analysis")
        
        # Information about the system
        st.markdown("""
        ### 🛰️ About This System
        
        This Space Debris Detection System uses advanced YOLOv8 computer vision technology to identify and locate debris objects in space imagery. 
        
        **Key Features:**
        - **Real-time Detection**: Fast and accurate debris identification
        - **Confidence Scoring**: Reliability assessment for each detection
        - **Statistical Analysis**: Comprehensive detection metrics
        - **Risk Assessment**: Categorization of detected debris by threat level
        
        **Applications:**
        - Satellite mission planning
        - Space situational awareness
        - Orbital debris monitoring
        - Collision avoidance systems
        """)

if __name__ == "__main__":
    main()
