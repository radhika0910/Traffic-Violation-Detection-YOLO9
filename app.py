import streamlit as st
import cv2
import yaml
import os
import time
import numpy as np
import pandas as pd
import json
import subprocess
from datetime import datetime
from PIL import Image

from src.detection import Detector
from src.ocr import PlateRecognizer
from src.violation_logic import check_violations
from src.cloud_db import CloudDatabase

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Violation Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS – premium theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* Custom Background fixing */
.stApp {
    background: linear-gradient(180deg, #f8faff 0%, #ffffff 100%);
    color: #1a1c1e;
}

@media (prefers-color-scheme: dark) {
    .stApp {
        background: radial-gradient(circle at top left, #0d1117 0%, #010409 100%);
        color: #f0f6fc;
    }
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.1);
}

/* Metric cards with glassmorphism */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(128,128,128,0.1);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
}

.header-strip {
    background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    color: white;
    box-shadow: 0 10px 30px rgba(31, 111, 235, 0.2);
}

.header-strip h1 {
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

/* Button override */
div.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #388bfd;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Config + Model caching
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner=False)
def load_models(_config):
    return Detector(_config), PlateRecognizer(_config)

@st.cache_resource(show_spinner=False)
def load_cloud_db():
    return CloudDatabase()

config = load_config()
os.makedirs(config['output']['save_dir'], exist_ok=True)
os.makedirs("outputs/temp", exist_ok=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "running" not in st.session_state: st.session_state.running = False
if "uploaded_video_path" not in st.session_state: st.session_state.uploaded_video_path = None

detector, ocr = load_models(config)
cloud_db = load_cloud_db()

if cloud_db.connected:
    st.sidebar.success("✅ Connected to Firebase Cloud")
else:
    st.sidebar.warning("⚠️ Using Local SQL (Cloud Disconnected)")

st.markdown("""
<div class="header-strip">
    <h1>🚦 IntelliTraffic System</h1>
    <p style="margin:0; color:#cfe2ff;">Advanced YOLOv9 Engine + Firebase Cloud Dashboard + Bulk Email NodeMailer</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎥 Video / Live Detection", "📊 SQL Database Dashboard", "📧 Bulk Email System"])

# ─────────────────────────────────────────────
# TAB 1: DETECTION
# ─────────────────────────────────────────────
with tab1:
    col_settings, col_feed = st.columns([1, 2.5])
    
    with col_settings:
        st.markdown("### ⚙️ Video Source")
        source_type = st.radio("Select Source:", ["Upload Video or Image", "Webcam (0)", "Stream/RTSP URL"])
        
        feed_source = None
        if source_type == "Webcam (0)":
            feed_source = 0
        elif source_type == "Upload Video or Image":
            uploaded_file = st.file_uploader("Drop an MP4/AVI or Image (JPG, PNG) file here", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                temp_path = os.path.join("outputs/temp", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_video_path = temp_path
                st.success(f"Video ready: {uploaded_file.name}")
            feed_source = st.session_state.uploaded_video_path
        else:
            feed_source = st.text_input("Stream URL:", value="")
            
        st.markdown("---")
        viol_thresh = st.slider("Violation Confidence", 0.1, 1.0, 0.85, 0.05)
        
        c1, c2 = st.columns(2)
        if c1.button("▶  Start Feed"):
            if feed_source is not None and feed_source != "":
                st.session_state.running = True
            else:
                st.error("Please provide a valid stream source.")
        if c2.button("⏹  Stop Feed"):
            st.session_state.running = False
            
    with col_feed:
        frame_placeholder = st.empty()
        if not st.session_state.running:
            idle_img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(idle_img, "SYSTEM IDLE - Select source & Start", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,130,200), 2)
            frame_placeholder.image(idle_img, channels="BGR", use_container_width=True)
            
    # Video Processing Loop
    if st.session_state.running and feed_source is not None:
        cap = cv2.VideoCapture(feed_source)
        if not cap.isOpened():
            st.error(f"Failed to open source: {feed_source}")
            st.session_state.running = False
        else:
             while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video stream ended.")
                    st.session_state.running = False
                    break
                    
                detections = detector.detect(frame)
                violations = check_violations(detections, viol_thresh)
                
                # Draw standard detections
                for d in detections:
                    x1, y1, x2, y2 = d['box']
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (200,200,200), 2)
                    cv2.putText(frame, d['class_name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Process specific violations
                for v in violations:
                    plate_text = None
                    plate_conf = 0.0
                    
                    # Case A: Custom model found a Plate box
                    if v.get('plate') is not None:
                        px1, py1, px2, py2 = v['plate']['box']
                        plate_img = frame[py1:py2, px1:px2]
                        if plate_img.size > 0:
                            plate_text, plate_conf = ocr.recognize(plate_img)
                            
                    # Case B: COCO mode - we need to hunt for the plate in the MC area
                    elif v.get('motorcycle') is not None:
                        mx1, my1, mx2, my2 = v['motorcycle']['box']
                        # Typical plate area: bottom 40% of the motorcycle box
                        h = my2 - my1
                        buffer = int(h * 0.6)
                        plate_crop_y1 = max(my1, my2 - buffer)
                        plate_area = frame[plate_crop_y1:my2, mx1:mx2]
                        
                        if plate_area.size > 0:
                            plate_text, plate_conf = ocr.recognize(plate_area)
                            # Optional: visualize the "hunt" area if debugging
                            # cv2.rectangle(frame, (mx1, plate_crop_y1), (mx2, my2), (255, 255, 0), 1)

                    # DETERMINE WHAT TO DRAW
                    box_to_draw = v['plate']['box'] if v.get('plate') else v['motorcycle']['box']
                    dx1, dy1, dx2, dy2 = box_to_draw
                    
                    # Always draw the visual highlight for the violation reason
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3)
                    
                    # LOGGING DECISION: Always log violations to database now, 
                    # but mark plate as 'UNKNOWN' if OCR fails to read it clearly.
                    if plate_text and len(plate_text) >= 4:
                        display_text = f"VIOLATION: {v['type']} [{plate_text}]"
                        # Background for text
                        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (dx1, dy1 - th - 20), (dx1 + tw + 10, dy1), (0,0,255), -1)
                        cv2.putText(frame, display_text, (dx1 + 5, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Log to database with actual plate
                        cloud_db.log_violation(plate_text, plate_conf, v['type'])
                    else:
                        display_text = f"VIOLATION: {v['type']} [NO PLATE]"
                        # Background for text (Orange for unrecognized plate)
                        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (dx1, dy1 - th - 20), (dx1 + tw + 10, dy1), (0,165,255), -1)
                        cv2.putText(frame, display_text, (dx1 + 5, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Still log to database, but with UNKNOWN plate text
                        cloud_db.log_violation("UNKNOWN", 0.0, v['type'])
                            
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
             cap.release()

# ─────────────────────────────────────────────
# TAB 2: SQL DASHBOARD
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### 🗄️ SQL Database Dashboard")
    
    if st.button("🔄 Refresh Data", key="refresh_db"):
        pass

    records = cloud_db.get_all_violations()
    if records:
        df = pd.DataFrame(records)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Records Detected", len(df))
        col2.metric("Recognized Plates", len(df[df['status'] == 'Recognized']))
        
        # Display specific columns
        display_df = df[['timestamp', 'license_plate', 'violation_type', 'confidence', 'status', 'email_sent']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No records found in the cloud database. Run the detection module to populate.")

# ─────────────────────────────────────────────
# TAB 3: BULK EMAIL SYSTEM
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### 📧 NodeMailer Bulk Operations")
    st.markdown("Upload a CSV file containing vehicle owner data (`plate`, `owner_name`, `email`). The system will query the cloud database for unrecognized violations and dispatch automated emails via the attached Node.js microservice.")
    
    csv_file = st.file_uploader("Upload Owner Database (CSV)", type=['csv'], key="csv_uploader")
    if csv_file:
        owners_df = pd.read_csv(csv_file)
        st.markdown("##### Preview of uploaded CSV:")
        st.dataframe(owners_df.head(), use_container_width=True)
        
        if 'plate' in owners_df.columns and 'email' in owners_df.columns:
            if st.button("Verify matches & Send Bulk Emails 🚀", type="primary"):
                st.info(f"📧 Preparing to send emails to {len(owners_df)} CSV owners...")
                
                # Get all violations from the database
                records = cloud_db.get_all_violations()
                violations_df = pd.DataFrame(records) if records else pd.DataFrame()
                
                # Prepare email data for ALL CSV owners
                export_data = []
                
                for idx, owner in owners_df.iterrows():
                    plate = owner.get('plate', 'UNKNOWN')
                    
                    # Look for any violation matching this plate
                    if not violations_df.empty:
                        plate_violations = violations_df[violations_df['license_plate'] == plate]
                        
                        if not plate_violations.empty:
                            # Send with actual violation details
                            for _, viol in plate_violations.iterrows():
                                export_data.append({
                                    "plate": plate,
                                    "owner_name": owner.get('owner_name', 'Vehicle Owner'),
                                    "email": owner['email'],
                                    "violation_type": viol['violation_type'],
                                    "timestamp": str(viol['timestamp']),
                                    "confidence": viol['confidence'],
                                    "doc_id": viol['id']
                                })
                        else:
                            # No violation for this plate - send with null data
                            export_data.append({
                                "plate": plate,
                                "owner_name": owner.get('owner_name', 'Vehicle Owner'),
                                "email": owner['email'],
                                "violation_type": None,
                                "timestamp": None,
                                "confidence": None,
                                "doc_id": None
                            })
                    else:
                        # No violations in DB at all - send to everyone with null
                        export_data.append({
                            "plate": plate,
                            "owner_name": owner.get('owner_name', 'Vehicle Owner'),
                            "email": owner['email'],
                            "violation_type": None,
                            "timestamp": None,
                            "confidence": None,
                            "doc_id": None
                        })
                
                # Send emails
                if export_data:
                    st.success(f"✅ Ready to send {len(export_data)} emails...")
                    
                    os.makedirs("outputs", exist_ok=True)
                    with open('outputs/pending_emails.json', 'w') as f:
                        json.dump(export_data, f)
                    
                    with st.spinner("Calling NodeMailer Microservice..."):
                        try:
                            print("\n" + "="*60)
                            print("📧 STARTING EMAIL SENDING PROCESS...")
                            print(f"📧 Sending to {len(export_data)} recipients...")
                            print("="*60 + "\n")
                            
                            result = subprocess.run(['node', 'mailer.js'], cwd='mailer', text=True, check=True)
                            
                            print("\n" + "="*60)
                            print("✅ EMAIL SENDING COMPLETED SUCCESSFULLY")
                            print("="*60 + "\n")
                            
                            st.success("✅ All emails sent successfully! Check terminal for detailed logs.")
                            
                            # Mark in CloudDB (only for matched violations with doc_id)
                            for item in export_data:
                                if item['doc_id']:  # Only update if this is a matched violation
                                    cloud_db.mark_email_sent(item['doc_id'])
                                   
                        except subprocess.CalledProcessError as e:
                            print("\n" + "="*60)
                            print("❌ EMAIL SENDING FAILED")
                            print("="*60 + "\n")
                            st.error(f"❌ NodeMailer executed with errors. Check terminal for details.")
                        except FileNotFoundError:
                            st.error("Node.js is not installed or not in PATH. Please install Node.js to use the mailer.")

        else:
            st.error("CSV must contain exactly the 'plate' and 'email' columns.")
