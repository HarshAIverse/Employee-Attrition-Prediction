import streamlit as st
import numpy as np

def init_page_config():
    st.set_page_config(
        page_title="Palo Alto Networks Analytics",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Animated Title Gradient */
        h1 {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            padding-bottom: 5px;
            animation: fadeIn 1s ease-in-out;
        }

        /* Glassmorphism Metric Cards */
        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.5rem;
            border-radius: 20px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease-in-out;
            animation: slideUp 0.8s ease-out forwards;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-8px);
            box-shadow: 0 10px 40px rgba(0, 210, 255, 0.2);
            border-color: rgba(0, 210, 255, 0.4);
        }

        /* Smooth Tab Transitions */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: rgba(0,0,0,0.1);
            padding: 10px 20px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: transparent;
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 210, 255, 0.1) !important;
            color: #00d2ff !important;
        }

        /* Card Styles for Employee Profile */
        .emp-card {
            background: linear-gradient(145deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-top: 20px;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a; 
        }
        ::-webkit-scrollbar-thumb {
            background: #334155; 
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569; 
        }

        /* Animations */
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

def get_risk_category(prob):
    # prob is between 0 and 1
    score = int(prob * 100)
    category = "Low Risk"
    color = "#10b981" # Emerald Green
    if score >= 60:
        category = "High Risk"
        color = "#ef4444" # Rose Red
    elif score >= 30:
        category = "Medium Risk"
        color = "#f59e0b" # Amber Orange
        
    return score, category, color

def generate_reason_codes(shap_values, feature_names, top_n=3):
    abs_shap = np.abs(shap_values)
    sorted_idx = np.argsort(abs_shap)[::-1]
    
    reasons = []
    for idx in sorted_idx[:top_n]:
        feat = feature_names[idx]
        val = shap_values[idx]
        if val > 0:
            reasons.append(f"Increases risk: Higher influence of {feat}")
        else:
            reasons.append(f"Decreases risk: Stabilizing influence natively from {feat}")
            
    return reasons
