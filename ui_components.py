import streamlit as st

def inject_custom_css():
    """
    Modern Soft Dashboard Theme
    Ref: High-end admin panels (Horizon UI / Soft UI)
    Colors: 
      - Background: #F4F7FE
      - Text Main: #2B3674
      - Text Sec: #A3AED0
      - Card: #FFFFFF with soft shadow
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    /* --- GLOBAL RESETS & FONT --- */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #2B3674; /* Deep Navy */
    }

    /* --- BACKGROUND --- */
    .stApp {
        background-color: #F4F7FE;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: none; /* Remove border for clean look */
        box-shadow: 0px 0px 20px rgba(0,0,0,0.02);
    }

    /* Sidebar Headings */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #2B3674;
        font-weight: 700;
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar Nav Items (Radio) */
    .stRadio > label { display: none; } /* Hide main label */
    
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        gap: 10px;
    }

    /* Radio Buttons -> Nav Links Look */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: transparent;
        padding: 12px 20px;
        border-radius: 12px;
        border: none;
        color: #A3AED0; /* Cool Gray */
        font-weight: 500;
        transition: all 0.2s ease;
        display: flex;
    }

    /* Hide the radio circle */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Hover */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        color: #4318FF; /* Brand Blue */
        background: #F4F7FE;
    }

    /* Selected */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: #4318FF;
        color: #FFFFFF !important;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(67, 24, 255, 0.2);
    }

    /* --- GLOBAL HEADERS --- */
    h1, h2, h3, h4 {
        color: #2B3674;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    p, div, span {
        color: #2B3674;
    }
    
    .small-text {
        color: #A3AED0 !important;
    }

    /* --- FLOATING CARD STYLE (Global Wrapper) --- */
    /* We apply this style to Metrics and Dataframes automatically via Streamlit classes */
    
    [data-testid="stMetric"], .stDataFrame {
        background: #FFFFFF !important;
        border-radius: 20px !important;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.03) !important;
        border: none !important;
        padding: 1.5rem !important;
    }

    /* --- METRICS SPECIFIC --- */
    [data-testid="stMetricLabel"] {
        color: #A3AED0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #2B3674 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricDelta"] svg {
        display: none; /* Hide default arrow if needed, or keep */
    }

    /* --- INPUTS --- */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div, 
    .stNumberInput > div > div > input {
        background-color: #FFFFFF;
        border-radius: 16px;
        border: 1px solid #E0E5F2;
        color: #2B3674;
        padding: 0.5rem 1rem;
    }
    
    .stTextInput > div > div > input:focus, 
    .stSelectbox > div > div:focus {
        border-color: #4318FF;
        box-shadow: 0 0 0 2px rgba(67, 24, 255, 0.1);
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background-color: #4318FF;
        color: white;
        border-radius: 16px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(67, 24, 255, 0.2);
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 25px rgba(67, 24, 255, 0.3);
        background-color: #3311DB;
        color: white;
    }

    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: none; 
    }

    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF;
        border-radius: 30px;
        padding: 8px 24px;
        border: none;
        color: #A3AED0;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }

    .stTabs [aria-selected="true"] {
        background: #4318FF !important;
        color: #FFFFFF !important;
        box-shadow: 0 5px 15px rgba(67, 24, 255, 0.3) !important;
    }

    /* --- CHARTS --- */
    /* Ensure Plotly charts have transparent bg in the container */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    </style>
    """, unsafe_allow_html=True)

def display_navbar():
    """Modern clean navbar"""
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0 30px 0;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="width: 45px; height: 45px; background: linear-gradient(135deg, #868CFF 0%, #4318FF 100%); 
                        border-radius: 14px; display: flex; align-items: center; justify-content: center; 
                        color: white; font-weight: 700; font-size: 1.5rem; box-shadow: 0 5px 15px rgba(67, 24, 255, 0.2);">
                U
            </div>
            <div style="line-height: 1.2;">
                <h3 style="margin: 0; color: #2B3674; font-size: 1.4rem; font-weight: 700;">Uplift</h3>
                <span style="color: #A3AED0; font-size: 0.85rem; font-weight: 500;">Financial Intelligence</span>
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background: #FFFFFF; color: #05CD99; padding: 8px 16px; border-radius: 30px; 
                        font-size: 0.8rem; font-weight: 700; box-shadow: 0 5px 15px rgba(0,0,0,0.05); display: flex; align-items: center; gap: 6px;">
                <div style="width: 8px; height: 8px; background: #05CD99; border-radius: 50%;"></div>
                LIVE SYSTEM
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_groww_metric(col, label, value, delta, color="#05CD99"):
    """
    Custom 'Floating Card' Metric.
    Matches global card style: radius 20px, white bg, soft shadow.
    """
    with col:
        st.markdown(f"""
        <div style="background-color: #FFFFFF; padding: 24px; border-radius: 20px; 
                    box-shadow: 0px 10px 30px rgba(0,0,0,0.03); border: none; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background: #F4F7FE; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #A3AED0; font-size: 12px;">📊</span>
                </div>
                <p style="color: #A3AED0; font-size: 0.9rem; font-weight: 500; margin: 0;">{label}</p>
            </div>
            <h2 style="color: #2B3674; font-size: 2rem; font-weight: 700; margin: 10px 0 5px 0;">{value}</h2>
            <p style="color: {color}; font-size: 0.85rem; font-weight: 700; margin: 0;">{delta}</p>
        </div>
        """, unsafe_allow_html=True)

def display_ticker_row(applicants_data):
    """
    Table styling -> Floating Card
    """
    st.markdown("""
    <div style="background: #FFFFFF; border-radius: 20px; padding: 24px; 
                box-shadow: 0px 10px 30px rgba(0,0,0,0.03); margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h4 style="margin: 0; color: #2B3674; font-size: 1.1rem;">Recent Applications</h4>
            <button style="background: #F4F7FE; border: none; padding: 8px 16px; border-radius: 10px; color: #4318FF; font-weight: 600; font-size: 0.8rem;">See All</button>
        </div>
    """, unsafe_allow_html=True)

    for applicant in applicants_data:
        # Modern pill colors
        if applicant["Status"] == "Approved":
            bg = "#E6FFF5"; text = "#05CD99"
        elif applicant["Status"] == "Pending":
            bg = "#FFF7E6"; text = "#FFB547"
        else:
            bg = "#FFEEF3"; text = "#EE5D50"

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 0; border-bottom: 1px solid #F4F7FE;">
            <div style="flex: 2; display: flex; align-items: center; gap: 12px;">
                 <div style="width: 40px; height: 40px; border-radius: 12px; background: #F4F7FE; display: flex; align-items: center; justify-content: center; font-weight: 700; color: #4318FF;">
                    {applicant["Name"][0]}
                 </div>
                 <div>
                    <div style="color: #2B3674; font-weight: 700; font-size: 0.95rem;">{applicant["Name"]}</div>
                    <div style="color: #A3AED0; font-size: 0.8rem;">{applicant["Role"]}</div>
                 </div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="color: #2B3674; font-weight: 700;">{applicant["Score"]}</div>
                <div style="color: #A3AED0; font-size: 0.75rem;">Score</div>
            </div>
            <div style="flex: 1; text-align: right;">
                <span style="background: {bg}; color: {text}; padding: 6px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">
                    {applicant["Status"]}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def create_sidebar_section(title):
    """Clean sidebar headers"""
    st.markdown(f"""
    <div style="margin-top: 20px; margin-bottom: 10px; padding-left: 10px;">
        <p style="color: #A3AED0; font-size: 0.75rem; font-weight: 700; 
                  text-transform: uppercase; letter-spacing: 1.2px;">
            {title}
        </p>
    </div>
    """, unsafe_allow_html=True)