import streamlit as st
import numpy as np
import cv2
import pickle
import time
import base64

# 1. Page Configuration
st.set_page_config(page_title="SkyGuard Elite", page_icon="🛡️", layout="wide")

# 2. Base64 Image Encoder
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 3. Enhanced Tactical CSS
def set_design(image_file):
    try:
        bin_str = get_base64_of_bin_file(image_file)
        st.markdown(f'''
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600&display=swap');
        
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.8)), url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
        }}

        /* Header Styling */
        .main-title {{ font-size: 3.5rem; font-weight: 700; margin-bottom: 0; color: #ffffff; }}
        .sub-title {{ font-size: 1.2rem; color: #94a3b8; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2rem; }}


        /* Result Cards */
        .status-box {{
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid;
            margin-top: 10px;
        }}
        .secure {{ border-color: #22c55e; background: rgba(34, 197, 94, 0.1); color: #4ade80; }}
        .alert {{ border-color: #ef4444; background: rgba(239, 68, 68, 0.1); color: #f87171; }}

        /* Remove Streamlit branding */
        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        </style>
        ''', unsafe_allow_html=True)
    except:
        st.warning("Background 'drone_bg.jpg' not found. Please add it to your repo.")

set_design('drone_bg.jpg')

# 4. Load Assets
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: m = pickle.load(f)
        with open("scaler.pkl", "rb") as f: s = pickle.load(f)
        return m, s
    except: return None, None

model, scaler = load_assets()
inv_label_map = {0: 'No Drone', 1: 'DJI Inspire', 2: 'DJI Mavic', 3: 'DJI Phantom'}

# 5. UI Layout
st.markdown("<h1 class='main-title'>Drone_Detect.Ai</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Professional AI Aerial Intelligence</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="tactical-card">', unsafe_allow_html=True)
    st.subheader("📡 Input Sensor")
    uploaded_file = st.file_uploader("Upload Drone Imagery", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(img, channels="BGR", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="tactical-card">', unsafe_allow_html=True)
    st.subheader("🖥️ Detected_Report")
    
    if uploaded_file and model:
        with st.status("Initializing Diagnostics...", expanded=True) as status:
            time.sleep(0.5)
            # ML Processing
            gray = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY)
            features = gray.flatten().reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            label = inv_label_map[prediction]
            
            time.sleep(0.5)
            status.update(label="Analysis Verified", state="complete")

        # Visual Output based on prediction
        if prediction == 0:
            st.markdown(f'''
                <div class="status-box secure">
                    <h3>STATUS: SECURE</h3>
                    <p>✅ {label.upper()} DETECTED</p>
                    <small>Airspace cleared. No unauthorized objects found.</small>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="status-box alert">
                    <h3>STATUS: ALERT</h3>
                    <p>🚨 {label.upper()} IDENTIFIED</p>
                    <small>Unauthorized aerial vehicle logged in sector.</small>
                </div>
            ''', unsafe_allow_html=True)
            st.warning(f"Detection Type: {label}")
    else:
        st.info("Waiting for data input from Sensor...")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
    <div style='text-align: center; margin-top: 50px;'>
        <hr style='border: 0.5px solid rgba(255,255,255,0.1)'>
        <p style='color: #64748b; font-size: 0.8rem;'>SECURE AI ANALYTICS ENGINE</p>
        <p style='color: #ffffff; font-weight: 600;'>DEVELOPED BY BAGADI SANTHOSH KUMAR</p>
    </div>""", unsafe_allow_html=True)
