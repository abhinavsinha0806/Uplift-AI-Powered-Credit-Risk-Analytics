import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import tempfile
from ml_engine import UpliftMLEngine
from ui_components import inject_custom_css, display_navbar, display_groww_metric, display_ticker_row, create_sidebar_section

# Try to import FPDF for PDF generation, handle if missing
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

st.set_page_config(
    page_title="Uplift | Credit Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
inject_custom_css()

# FORCE CACHE INVALIDATION
# CHANGED: Renamed to v4 to force Streamlit to reload the new ml_engine code
@st.cache_resource
def get_engine_v4():
    return UpliftMLEngine()

engine = get_engine_v4()

# --- HELPER: CLEAN CHARTS ---
def plot_clean_line(df, y_col, color):
    fig = px.line(df, y=y_col, markers=True)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=250,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        hovermode="x unified",
        font=dict(color='#2B3674')
    )
    fig.update_traces(line_color=color, line_width=3, marker_size=6)
    return fig

# --- HELPER: PDF GENERATOR ---
def create_credit_memo(profile_name, result):
    if not FPDF_AVAILABLE:
        return None
        
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Uplift | Official Credit Memo', 0, 1, 'C')
            self.ln(5)
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Helper to clean text for PDF (removes unicode characters like ₹)
    def clean_text(text):
        return str(text).replace('₹', 'Rs. ').encode('latin-1', 'replace').decode('latin-1')
    
    # Header Details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, clean_text(f"Applicant: {profile_name}"), ln=True)
    pdf.cell(200, 10, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)
    
    # Score Section
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 15, f"  Uplift Score: {result['uplift_score']}   |   Status: {'APPROVED' if result['prob_default'] < 0.4 else 'REJECTED'}", 1, 1, 'L', fill=True)
    pdf.ln(10)
    
    # Risk Metrics
    pdf.set_font("Arial", '', 12)
    pdf.cell(100, 10, f"Probability of Default: {result['prob_default']*100:.1f}%", 0, 1)
    pdf.cell(100, 10, f"Projected CIBIL: {result['simulated_cibil']}", 0, 1)
    pdf.cell(100, 10, f"Est. Runway: {result['financial_metrics']['survival_runway_days']} days", 0, 1)
    pdf.ln(10)
    
    # AI Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "AI Risk Assessment:", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, clean_text(f"Income Volatility: {result['financial_metrics']['volatility_category']}"))
    pdf.multi_cell(0, 8, clean_text(f"Digital Footprint Score: {result['battery_hygiene']['score']} ({result['battery_hygiene']['level']})"))
    pdf.multi_cell(0, 8, clean_text(f"Geo-Stability: {result['geo_stability']['level']}"))
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Key Recommendations:", 0, 1)
    pdf.set_font("Arial", '', 11)
    
    for plan in result['action_plan']:
        # Clean formatting and symbols
        clean_plan = plan.replace("**", "").replace("₹", "Rs. ")
        # Encode/decode to strip any other hidden unicode chars
        safe_plan = clean_plan.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, f"- {safe_plan}")
        
    return pdf.output(dest='S').encode('latin-1')

# ==================== NAVBAR ====================
display_navbar()

# ==================== SIDEBAR ====================
with st.sidebar:
    create_sidebar_section("NAVIGATION")
    page = st.radio(
        "Select Page:",
        ["Dashboard Overview", "Applicant Analysis", "Batch Analysis", "Uplift AI Simulator"],
        label_visibility="collapsed",
        key="navigation"
    )

    st.markdown("---")

    if "Applicant Analysis" in page:
        create_sidebar_section("DATA CONTROLS")
        st.markdown("#### Select Applicant")
        selected_profile = st.selectbox(
            "Choose Profile:",
            ["Vikram (Stable)", "Rahul (High Volatility)", "Priya (Reseller)", 
             "Karan (Crypto Trader)", "Zara (Influencer)", "Rohan (Weekend Hustler)"],
            label_visibility="collapsed",
            key="profile_selector"
        )
        st.markdown("---")
        
    elif "Batch Analysis" in page:
        create_sidebar_section("BATCH TOOLS")
        st.markdown("#### File Upload")
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
        st.caption("Required Cols: Income, Volatility, Battery_Health, Location_Score")
        st.markdown("---")

    elif "Uplift AI Simulator" in page:
        create_sidebar_section("SIMULATION CONTROLS")
        
        st.markdown("#### Income Parameters")
        sim_monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=5000, max_value=500000, value=40000, step=5000
        )
        
        sim_volatility = st.slider("Income Volatility (₹)", 0, 50000, 8000, 1000)

        st.markdown("#### Digital Footprint")
        sim_battery = st.slider("Battery Hygiene Score", 0, 100, 75)
        sim_location = st.slider("Geo-Stability Score", 0, 100, 70)

        st.markdown("#### Expenses")
        sim_monthly_expenses = st.number_input(
            "Monthly Expenses (₹)",
            min_value=5000, max_value=100000, value=25000, step=1000
        )

        st.markdown("---")
        if st.button("Run Simulation", use_container_width=True):
            st.session_state['run_simulation'] = True

    st.markdown("---")
    create_sidebar_section("SYSTEM INFO")
    st.caption("**Secure Connection**")
    st.caption("**Model:** v7.12 (Optimized)")
    st.caption("**Status:** Operational")

@st.cache_data
def fetch_profile_data(profile_name):
    df_f = engine.generate_financial_data(profile_name)
    df_m = engine.generate_metadata(profile_name)
    return df_f, df_m

# ==================== PAGE: DASHBOARD OVERVIEW ====================
if "Dashboard Overview" in page:
    st.markdown("## Daily Overview")

    col1, col2, col3 = st.columns(3)
    display_groww_metric(col1, "Total Applications", "1,248", "+12% vs yesterday", "#10B981")
    display_groww_metric(col2, "Approval Rate", "64.2%", "AI Model v7.12", "#10B981")
    display_groww_metric(col3, "Disbursed Amount", "₹4.2 Cr", "Avg Ticket: ₹35k", "#10B981")

    st.markdown("<br>", unsafe_allow_html=True)

    # Restored Original Applicant Data
    applicants_data = [
        {"Name": "Vikram Singh", "Role": "Salaried", "Score": "782", "Risk": "Low", "Status": "Approved"},
        {"Name": "Priya Sharma", "Role": "Reseller", "Score": "645", "Risk": "Medium", "Status": "Pending"},
        {"Name": "Rahul Kumar", "Role": "Freelancer", "Score": "521", "Risk": "High", "Status": "Rejected"},
        {"Name": "Karan Patel", "Role": "Crypto Trader", "Score": "598", "Risk": "Medium", "Status": "Approved"},
        {"Name": "Zara Khan", "Role": "Influencer", "Score": "712", "Risk": "Low", "Status": "Approved"},
    ]
    display_ticker_row(applicants_data)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Distribution")
        risk_data = pd.DataFrame({
            'Risk Tier': ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
            'Count': [456, 512, 218, 62]
        })
        fig = px.pie(risk_data, values='Count', names='Risk Tier',
                     color_discrete_sequence=['#10B981', '#3B82F6', '#F59E0B', '#EF4444'],
                     hole=0.6)
        fig.update_layout(showlegend=True, height=350, margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Approval Amounts")
        amount_data = pd.DataFrame({
            'Range': ['₹10k-25k', '₹25k-50k', '₹50k-75k', '₹75k+'],
            'Applications': [342, 468, 289, 149]
        })
        fig = px.bar(amount_data, x='Range', y='Applications',
                     color='Applications', color_continuous_scale=['#A7F3D0', '#10B981'])
        fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: BATCH ANALYSIS ====================
elif "Batch Analysis" in page:
    st.markdown("## Batch Analysis Module")
    st.markdown("Upload a CSV containing multiple applicants to run the Uplift AI model in bulk.")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
            
            required_cols = ['Income', 'Volatility', 'Battery_Health', 'Location_Score']
            missing_cols = [c for c in required_cols if c not in df_batch.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}. Please check your file.")
            else:
                st.success("✅ File uploaded successfully! Processing records...")
                
                results = []
                progress_bar = st.progress(0)
                
                for index, row in df_batch.iterrows():
                    pred = engine.model.predict_comprehensive(
                        income=row['Income'],
                        volatility=row['Volatility'],
                        battery=row['Battery_Health'],
                        location=row['Location_Score']
                    )
                    
                    # Intelligent Reason Logic
                    reason = "Strong Profile"
                    if pred['prob_default'] >= 0.4:
                        if row['Battery_Health'] < 70 or row['Location_Score'] < 60:
                            reason = "Digital Footprint Risk"
                        elif row['Volatility'] > (row['Income'] * 0.2):
                            reason = "High Income Volatility"
                        elif row['Income'] < 25000:
                            reason = "Insufficient Income"
                        else:
                            reason = "Composite Risk High"
                    
                    results.append({
                        'ID': row.get('Applicant_ID', index + 1),
                        'Name': row.get('Name', f"Applicant {index+1}"),
                        'Income': row['Income'],
                        'Score': pred['uplift_score'],
                        'Risk_Prob': f"{pred['prob_default']*100:.1f}%",
                        'Status': 'APPROVED' if pred['prob_default'] < 0.4 else 'REJECTED',
                        'Primary_Reason': reason
                    })
                    progress_bar.progress((index + 1) / len(df_batch))
                
                result_df = pd.DataFrame(results)
                
                # Show Summary
                col1, col2 = st.columns(2)
                with col1:
                    approved_count = result_df[result_df['Status'] == 'APPROVED'].shape[0]
                    st.markdown(f"### Approved: {approved_count} / {len(result_df)}")
                with col2:
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv,
                        file_name='uplift_batch_results.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                # Styling the dataframe with color maps
                st.dataframe(result_df.style.applymap(
                    lambda v: 'color: red; font-weight: bold' if v == 'REJECTED' else 'color: green; font-weight: bold' if v == 'APPROVED' else '', 
                    subset=['Status']
                ), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.info("👈 Please upload a file from the sidebar to begin.")
        st.markdown("""
        **Sample CSV Format:**
        ```csv
        Income,Volatility,Battery_Health,Location_Score
        45000,5000,85,70
        32000,12000,60,55
        ...
        ```
        """)

# ==================== PAGE: APPLICANT ANALYSIS ====================
elif "Applicant Analysis" in page:
    st.markdown(f"## Applicant Analysis: {selected_profile}")
    df_financial, df_metadata = fetch_profile_data(selected_profile)
    latest = df_financial.iloc[-1]
    
    # Call exact function from restored ML Engine
    result = engine.model.predict_comprehensive(
        income=latest['Monthly_Income'], 
        volatility=df_financial['Monthly_Income'].std() * 12,
        battery=df_metadata['Battery_Health'].mean(),
        location=df_metadata['Location_Stability'].mean(),
        expenses=latest['Monthly_Expenses']
    )

    # --- NEW PDF BUTTON ---
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown("") # Spacer
    with col_head2:
        if FPDF_AVAILABLE:
            pdf_bytes = create_credit_memo(selected_profile, result)
            st.download_button(
                label="📄 Download Credit Report",
                data=pdf_bytes,
                file_name=f"{selected_profile.split()[0]}_Credit_Memo.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("PDF Gen Unavailable (Install fpdf)")

    col1, col2, col3, col4 = st.columns(4)
    display_groww_metric(col1, "Uplift Score", str(result['uplift_score']), f"vs CIBIL: {result['simulated_cibil']}")
    display_groww_metric(col2, "Default Risk", f"{result['prob_default']*100:.1f}%", result['financial_metrics']['volatility_category'], 
                         "#EF4444" if result['prob_default'] > 0.5 else "#F59E0B" if result['prob_default'] > 0.3 else "#10B981")
    display_groww_metric(col3, "Monthly Savings", f"₹{result['financial_metrics']['monthly_savings']:,.0f}", f"Rate: {result['financial_metrics']['savings_rate']:.1f}%")
    display_groww_metric(col4, "Runway", f"{result['financial_metrics']['survival_runway_days']} days", "Financial Buffer")

    st.markdown("---")
    
    # --- NEW: PATH TO APPROVAL ACTION PLAN ---
    if result['prob_default'] > 0.3: # Show for everyone not perfect, or threshold > 0.3
        st.markdown("""
        <div style='background-color: #F0FDF4; border-left: 4px solid #10B981; padding: 20px; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>
            <h3 style='color: #065F46; margin-top: 0; font-size: 1.2rem;'>🚀 Steps to Improve Credit Score</h3>
        """, unsafe_allow_html=True)
        
        for item in result['action_plan']:
             st.markdown(f"<li style='color: #115E59; margin-bottom: 8px;'>{item}</li>", unsafe_allow_html=True)
             
        st.markdown("</div>", unsafe_allow_html=True)
    # -----------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(["Financial Analysis", "Feature Importance", "Digital Footprint", "Risk Factors"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Income Trend")
            st.plotly_chart(plot_clean_line(df_financial, 'Monthly_Income', '#10B981'), use_container_width=True)
        with col2:
            st.markdown("#### Expense Breakdown")
            st.plotly_chart(plot_clean_line(df_financial, 'Monthly_Expenses', '#F59E0B'), use_container_width=True)

    with tab2:
        st.markdown("#### SHAP Feature Importance")
        shap_df = pd.DataFrame({'Feature': list(result['shap_values'].keys()), 'Impact': list(result['shap_values'].values())})
        fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Impact', 
                     color_continuous_scale=['#10B981', '#EF4444'])
        # FIXED: Added font color explicitly to make text visible
        fig.update_layout(height=300, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#2B3674'), xaxis=dict(color='#2B3674'), yaxis=dict(color='#2B3674'))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style='background:white; padding:20px; border-radius:16px; border:1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
                <h4 style='margin:0; color: #64748B; font-size:0.9rem;'>BATTERY HYGIENE</h4>
                <h1 style='color:#10B981; font-size:3rem; margin:10px 0;'>{result['battery_hygiene']['score']:.0f}</h1>
                <p style='color:#64748B;'>{result['battery_hygiene']['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background:white; padding:20px; border-radius:16px; border:1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
                <h4 style='margin:0; color: #64748B; font-size:0.9rem;'>GEO-STABILITY</h4>
                <h1 style='color:#10B981; font-size:3rem; margin:10px 0;'>{result['geo_stability']['score']:.0f}</h1>
                <p style='color:#64748B;'>{result['geo_stability']['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        if result['outliers']:
            for outlier in result['outliers']:
                st.warning(f"⚠️ {outlier}")
        else:
            st.success("✅ No significant outliers detected")
        st.info(f"📍 Local Economic Risk Level: {result['local_risk']['level']} (PRI: {result['local_risk']['pri_score']:.1f})")

# ==================== PAGE: UPLIFT AI SIMULATOR ====================
elif "Uplift AI Simulator" in page:
    st.markdown("## Uplift AI Simulator")
    
    if st.session_state.get('run_simulation', False):
        sim_result = engine.model.predict_comprehensive(
            income=sim_monthly_income, volatility=sim_volatility,
            battery=sim_battery, location=sim_location, expenses=sim_monthly_expenses
        )

        col1, col2, col3, col4 = st.columns(4)
        display_groww_metric(col1, "Predicted Score", str(sim_result['uplift_score']), f"Credit Score: {sim_result['credit_score']}")
        display_groww_metric(col2, "Default Risk", f"{sim_result['prob_default']*100:.1f}%", sim_result['financial_metrics']['volatility_category'], "#EF4444" if sim_result['prob_default'] > 0.5 else "#10B981")
        display_groww_metric(col3, "Monthly Savings", f"₹{sim_result['financial_metrics']['monthly_savings']:,.0f}", f"Rate: {sim_result['financial_metrics']['savings_rate']:.1f}%")
        display_groww_metric(col4, "Runway", f"{sim_result['financial_metrics']['survival_runway_days']} days", "Buffer")

        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Feature Impact")
            shap_df = pd.DataFrame({'Feature': list(sim_result['shap_values'].keys()), 'Impact': list(sim_result['shap_values'].values())})
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Impact', 
                         color_continuous_scale=['#10B981', '#EF4444'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#2B3674'), xaxis=dict(color='#2B3674'), yaxis=dict(color='#2B3674'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Score Comparison")
            comp_df = pd.DataFrame({'Model': ['Traditional', 'Uplift AI'], 'Score': [sim_result['simulated_cibil'], sim_result['uplift_score']]})
            fig = px.bar(comp_df, x='Model', y='Score', color='Model', color_discrete_sequence=['#94A3B8', '#10B981'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#2B3674'), xaxis=dict(color='#2B3674'), yaxis=dict(color='#2B3674'))
            st.plotly_chart(fig, use_container_width=True)

        st.session_state['run_simulation'] = False
    else:
        st.info("👈 Adjust parameters in the sidebar and click 'Run Simulation'")
