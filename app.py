import streamlit as st
import numpy as np
import cv2
import pickle


st.set_page_config(page_title="SkyGuard AI", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #f8fafc, #e2e8f0, #f1f5f9, #ffffff);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Professional Font & Headers */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Glassmorphic Container */
    .glass-panel {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }

    /* Image Preview Styling */
    .stImage > img {
        border-radius: 16px;
        border: 4px solid white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        transition: all 0.4s ease;
    }

    /* Slide-in Result Animations */
    @keyframes slideInUp {
        0% { transform: translateY(30px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }

    .result-card {
        animation: slideInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        margin-top: 2rem;
    }

    .safe-card {
        background: linear-gradient(135deg, #dcfce7 0%, #ffffff 100%);
        border: 1px solid #86efac;
        color: #166534;
    }

    .alert-card {
        background: linear-gradient(135deg, #fee2e2 0%, #ffffff 100%);
        border: 1px solid #fca5a5;
        color: #991b1b;
    }

    /* Developer Branding */
    .dev-footer {
        text-align: center;
        margin-top: 4rem;
        padding: 20px;
        border-top: 1px solid rgba(0,0,0,0.05);
    }

    .badge {
        background: linear-gradient(90deg, #1e293b, #334155);
        color: #f8fafc;
        padding: 6px 16px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            m = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            s = pickle.load(f)
        return m, s
    except:
        return None, None

model, scaler = load_assets()

inv_label_map = {0: 'no_drone', 1: 'DJI Inspire', 2: 'DJI Mavic', 3: 'DJI Phantom'}


st.markdown("<h1 style='text-align: center; color: #0f172a;'>SkyGuard Elite</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1rem;'>Professional AI Aerial Intelligence</p>", unsafe_allow_html=True)

# Centered Upload Glass Panel
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read and Display centered shrunken image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", width=380)
    
    # Advanced Status Simulation
    with st.status("Initializing AI Diagnostics...", expanded=False) as status:
        time.sleep(0.8)
        status.update(label="Scanning Image Geometry...", state="running")
        
        # ML Logic
        gray = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY)
        features = gray.flatten().reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        class_name = inv_label_map[prediction]
        
        time.sleep(0.7)
        status.update(label="Analysis Verified", state="complete")

    # Animated Result with Color Coding
    if prediction == 0:
        st.markdown(f"""
            <div class="result-card safe-card">
                <p style='text-transform: uppercase; font-weight: 600; letter-spacing: 1px; margin-bottom: 5px;'>Status: Secure</p>
                <h2 style='margin:0;'>✅ NO DRONE DETECTED</h2>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
            <div class="result-card alert-card">
                <p style='text-transform: uppercase; font-weight: 600; letter-spacing: 1px; margin-bottom: 5px;'>Status: Alert</p>
                <h2 style='margin:0;'>🚨 {class_name.upper()} DETECTED</h2>
                <p style='margin-top: 5px; opacity: 0.8;'>Unauthorized aerial object logged.</p>
            </div>
            """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
    <div class="dev-footer">
        <p style='color: #94a3b8; font-size: 0.8rem; margin-bottom: 8px;'>SECURE AI ANALYTICS ENGINE</p>
        <div class="badge">DEVELOPED BY BAGADI SANTHOSH KUMAR</div>
    </div>

    """, unsafe_allow_html=True)

