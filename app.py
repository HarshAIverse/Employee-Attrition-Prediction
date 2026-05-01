import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

from utils import init_page_config, inject_custom_css, get_risk_category

init_page_config()
inject_custom_css()

@st.cache_resource
def load_models():
    model = joblib.load('best_model.pkl')
    feature_meta = joblib.load('feature_meta.pkl')
    return model, feature_meta

@st.cache_data
def load_data():
    df = pd.read_csv('processed_data.csv')
    return df

st.title("Palo Alto Networks Analytics 🛡️")
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -15px;'>Predictive Employee Attrition & Risk Engine</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not os.path.exists('best_model.pkl') or not os.path.exists('processed_data.csv'):
    st.error("⚠️ Models and data not found. Run `train_model.py` first.")
    st.stop()

model, feature_meta = load_models()
df = load_data()

tabs = st.tabs([
    "📈 Executive Dashboard", 
    "👤 Employee Risk Profile", 
    "🏢 Department Analytics", 
    "🎛️ Scenario Simulator", 
    "🧠 Explainable AI"
])

X_cols = feature_meta['all_cols']
X_data = df[X_cols]

probs = model.predict_proba(X_data)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_data)
df['Risk_Probability'] = probs
df['Risk_Score'] = (probs * 100).astype(int)
df['Risk_Category'] = pd.cut(df['Risk_Score'], bins=[-1, 30, 60, 101], labels=['Low Risk', 'Medium Risk', 'High Risk'])

# Unified charting template config
layout_config = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#f8fafc", size=13)
)

# --- 1. Executive Dashboard ---
with tabs[0]:
    st.markdown("### AI Driven Global Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Employees", len(df))
    with col2: 
        high_risk = len(df[df['Risk_Category'] == 'High Risk'])
        st.metric("High Risk Vectors", high_risk, delta_color="inverse")
    with col3: st.metric("Avg Enterprise Risk", f"{df['Risk_Score'].mean():.1f}/100")
    with col4: st.metric("Historical Attrition", f"{(df['Attrition'].sum() / len(df)) * 100:.1f}%")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        fig1 = px.pie(df, names='Risk_Category', hole=0.5, 
                      color='Risk_Category', color_discrete_map={'Low Risk':'#10b981', 'Medium Risk':'#f59e0b', 'High Risk':'#ef4444'})
        fig1.update_layout(**layout_config, title="Workforce Risk Distribution", showlegend=True)
        # hover effects natively included in plotly
        fig1.update_traces(hoverinfo='label+percent', textinfo='none', sort=False)
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        fig2 = px.histogram(df, x='Risk_Score', nbins=25,
                            color='Risk_Category', color_discrete_map={'Low Risk':'#10b981', 'Medium Risk':'#f59e0b', 'High Risk':'#ef4444'})
        fig2.update_layout(**layout_config, title="Firm-wide Risk Score Density", bargap=0.1)
        st.plotly_chart(fig2, use_container_width=True)

# --- 2. Employee Risk Profile ---
with tabs[1]:
    st.markdown("### High-Resolution Individual Profiling")
    emp_idx = st.selectbox("Search by Employee Profile Index", df.index.tolist())
    emp_data = df.iloc[emp_idx]
    
    prob = emp_data['Risk_Probability']
    score, category, color_code = get_risk_category(prob)
    
    st.markdown("<div class='emp-card'>", unsafe_allow_html=True)
    colA, colB = st.columns([1, 1.8])
    
    with colA:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            number = {'font': {'color': color_code, 'size': 50}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': color_code, 'thickness': 0.8},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 0,
                'steps' : [
                    {'range': [0, 30], 'color': "rgba(16,185,129,0.1)"},
                    {'range': [30, 60], 'color': "rgba(245,158,11,0.1)"},
                    {'range': [60, 100], 'color': "rgba(239,68,68,0.1)"}],
            }
        ))
        fig_gauge.update_layout(**layout_config, height=250, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(f"<h3 style='text-align: center; color: {color_code}; margin-top:-30px;'>{category}</h3>", unsafe_allow_html=True)
        
    with colB:
        st.markdown(f"#### 👤 HR Profiling | ID #{emp_idx}")
        st.markdown(f"**Department:** {emp_data['Department']}  |  **Role:** {emp_data['JobRole']}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Reason Codes
        reasons = []
        if emp_data['OverTime'] == 'Yes': reasons.append("High contextual stress: Consistent Overtime triggered.")
        if emp_data['JobSatisfaction'] <= 2: reasons.append("Negative Job Satisfaction identified.")
        if emp_data['YearsSinceLastPromotion'] > 3: reasons.append("Stagnation flag: 3+ years since last promotion.")
        if emp_data['WorkLifeBalance'] <= 2: reasons.append("Sub-optimal Work-Life Balance score.")
        
        if not reasons:
            st.success("✨ Stable metrics identified. No heavy risk markers flagged.")
        else:
            for r in reasons:
                st.markdown(f"🚩 <span style='color: #ef4444;'>{r}</span>", unsafe_allow_html=True)
                
    st.markdown("</div>", unsafe_allow_html=True)

# --- 3. Department Analytics ---
with tabs[2]:
    st.markdown("### Organizational Risk Topology")
    c_line1, c_line2 = st.columns([1.5, 1])
    
    with c_line1:
        # Boxplot with native plotly dark mode
        fig_dept = px.box(df, x="Department", y="Risk_Score", color="Department", points="all",
                          color_discrete_sequence=['#00d2ff', '#8b5cf6', '#ec4899'])
        fig_dept.update_layout(**layout_config, title="Risk Variance by Department")
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with c_line2:
        role_risk = df.groupby('JobRole')['Risk_Score'].mean().reset_index().sort_values(by='Risk_Score')
        fig_role = px.bar(role_risk, x='Risk_Score', y='JobRole', orientation='h', 
                          color='Risk_Score', color_continuous_scale='Inferno')
        fig_role.update_layout(**layout_config, title="Avg Risk Heatmap by Role")
        st.plotly_chart(fig_role, use_container_width=True)

# --- 4. Scenario Simulator ---
with tabs[3]:
    st.markdown("### Interactive Retention Simulator")
    st.markdown("Select an employee and adjust parameters dynamically to forecast retention probability shifts.")
    
    sim_emp_idx = st.selectbox("Target Employee", list(range(min(20, len(df)))), key='sim_emp')
    e_data = df.iloc[sim_emp_idx].copy()
    
    st.markdown(f"<div style='border-left: 4px solid #00d2ff; padding-left: 15px; margin-bottom: 20px;'><b>Baseline Risk:</b> {e_data['Risk_Score']} / 100</div>", unsafe_allow_html=True)
    
    colS1, colS2 = st.columns(2)
    with colS1:
        n_salary = st.slider("Percent Salary Hike", 10, 30, int(e_data['PercentSalaryHike']))
        n_overtime = st.selectbox("OverTime", ["Yes", "No"], index=0 if e_data['OverTime']=="Yes" else 1)
    with colS2:
        n_satisfy = st.slider("Job Satisfaction", 1, 4, int(e_data['JobSatisfaction']))
        n_prom = st.slider("Years Since Last Promotion", 0, 15, int(e_data['YearsSinceLastPromotion']))
        
    if st.button("Simulate Updated Attrition Risk", type="primary"):
        with st.spinner("Processing neural pipeline..."):
            t_df = e_data.to_frame().T
            t_df['PercentSalaryHike'], t_df['OverTime'], t_df['JobSatisfaction'], t_df['YearsSinceLastPromotion'] = n_salary, n_overtime, n_satisfy, n_prom
            
            # Recalc engineered
            t_df['Promotion_Delay_Flag'] = (t_df['YearsSinceLastPromotion'] > 3).astype(int)
            t_df['Engagement_Score'] = t_df['JobInvolvement'] + t_df['JobSatisfaction']
            t_df['Overtime_Num'] = 1 if n_overtime == 'Yes' else 0
            
            n_prob = model.predict_proba(t_df[X_cols])[:, 1][0] if hasattr(model, "predict_proba") else model.predict(t_df[X_cols])[0]
            n_score, n_cat, n_color = get_risk_category(n_prob)
            diff = n_score - e_data['Risk_Score']
            
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border: 1px solid {n_color}; margin-top: 20px;'>
                <h4 style='margin:0; color:{n_color};'>Forecasted Risk: {n_score}/100 ({n_cat})</h4>
                <p style='margin:0; color:#cbd5e1; font-size: 1.1rem;'>Shift in probability: <b>{diff:+} points</b></p>
            </div>
            """, unsafe_allow_html=True)

# --- 5. Explainable AI ---
with tabs[4]:
    st.markdown("### Algorithmic Decision Mapping")
    st.markdown("Understanding the neural weights deciding Palo Alto Networks' attrition indices.")
    
    try:
        clf = model.named_steps['classifier']
        if hasattr(clf, 'feature_importances_') or hasattr(clf, 'coef_'):
            importances = clf.feature_importances_ if hasattr(clf, 'feature_importances_') else np.abs(clf.coef_[0])
            pre = model.named_steps['preprocessor']
            fname = pre.get_feature_names_out(X_cols) if hasattr(pre, 'get_feature_names_out') else [f"F{i}" for i in range(len(importances))]
            
            fi_df = pd.DataFrame({'Feature': [f.split('__')[-1] for f in fname], 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)
            
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                            color='Importance', color_continuous_scale='tealgrn')
            fig_fi.update_layout(**layout_config, title='Top 15 Absolute Predictive Weights', yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("The selected algorithmic backbone operates outside direct feature tree extraction.")
    except Exception as e:
        st.warning("Explainability matrix failed standard extraction mapping.")
