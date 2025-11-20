import streamlit as st

def inject_custom_css():
    """
    Modern Light Theme CSS (Dimmed White) with Hidden Radio Dots.
    This applies to the main Dashboard.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

    /* --- ANIMATIONS --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Apply animations to main blocks */
    .element-container, [data-testid="stMetric"], .stDataFrame {
        animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* --- GLOBAL STYLING --- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #44475B;
    }

    .stApp {
        /* Changed to pure white to match the intro cards */
        background-color: #FFFFFF; 
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        /* Slightly lighter dimmed white for sidebar to distinguish it */
        background-color: #F7F8FA;
        border-right: 1px solid #E2E4E8;
        box-shadow: none;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #44475B;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Sidebar Sections */
    .sidebar-section {
        background: #EFF6FF; /* Light Blue tint */
        padding: 10px 12px;
        border-radius: 8px;
        margin: 12px 0;
        border-left: 3px solid #2563EB; /* Royal Blue */
        animation: slideInLeft 0.5s ease-out;
    }

    /* --- NAVIGATION (RADIO BUTTONS) --- */
    .stRadio > label { display: none; }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] { gap: 8px; }
    
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: transparent;
        padding: 10px 16px;
        border-radius: 6px;
        border: 1px solid transparent;
        color: #44475B;
        transition: all 0.2s ease;
        display: flex;
    }

    /* CRITICAL: Hide the Radio Button Dot */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Hover & Selected States */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: #EFF6FF;
        color: #2563EB;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.05);
    }

    /* Input Fields & Selectboxes */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        color: #44475B;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #2563EB;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    /* --- MAIN CONTENT AREA --- */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Headings */
    h1, h2, h3 { color: #1F2937; font-weight: 700; }
    h2 { border-bottom: none; font-size: 1.8rem; letter-spacing: -0.5px; }

    /* Cards (Metrics) */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    /* Buttons - Converted to Blue/Cyan Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #0891B2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #067490 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; border-bottom: 1px solid #E5E7EB; }
    .stTabs [data-baseweb="tab"] { background: transparent; border: none; color: #6B7280; font-weight: 500; padding-bottom: 12px; }
    .stTabs [aria-selected="true"] { color: #2563EB !important; border-bottom: 2px solid #2563EB; font-weight: 600; }
    
    /* Plotly Chart Backgrounds */
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

def inject_landing_css():
    """
    Specific CSS for the Landing Page (Light Mode High-End Polish).
    Separated to keep main CSS clean.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    /* Hide Streamlit Toolbar for clean landing */
    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    /* HERO TYPOGRAPHY */
    .landing-title {
        font-family: 'Inter', sans-serif;
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1.1;
        color: #111827; /* Deep Black-Blue */
        letter-spacing: -0.03em;
        margin-bottom: 1.5rem;
    }

    .landing-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        color: #4B5563; /* Slate 600 */
        line-height: 1.6;
        font-weight: 400;
        margin-bottom: 2.5rem;
        max-width: 90%;
    }

    .highlight-text {
        background: -webkit-linear-gradient(0deg, #2563EB 0%, #0891B2 100%); /* Blue to Cyan */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* MODERN BUTTON STYLING FOR LANDING PAGE ONLY */
    div.stButton > button:first-child {
        background: #111827 !important; /* Force Dark button */
        color: white !important;
        border-radius: 50px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 14px 0 rgba(0,0,0,0.1) !important;
    }
    div.stButton > button:first-child:hover {
        background: #000000 !important;
        transform: translateY(-2px);
    }

    /* FEATURE CARDS */
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        height: 100%;
        min-height: 240px; /* Enforces equal height */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #CBD5E1;
    }

    /* FLOATING CREDIT CARD ANIMATION */
    .floating-card-container {
        perspective: 1000px;
        width: 100%;
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .credit-card {
        width: 380px;
        height: 240px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        position: relative;
        transform: rotateY(-15deg) rotateX(10deg);
        animation: float 6s ease-in-out infinite;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    @keyframes float {
        0% { transform: rotateY(-15deg) rotateX(10deg) translateY(0px); }
        50% { transform: rotateY(-10deg) rotateX(5deg) translateY(-20px); }
        100% { transform: rotateY(-15deg) rotateX(10deg) translateY(0px); }
    }

    .chip {
        width: 50px;
        height: 35px;
        background: linear-gradient(135deg, #fbbf24, #d97706);
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .card-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        color: white;
        letter-spacing: 2px;
        margin-bottom: 40px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .card-details {
        display: flex;
        justify-content: space-between;
        color: rgba(255, 255, 255, 0.8);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

def display_landing_page():
    """
    Beautiful Light-Mode Landing Page.
    Clean, Crisp, Professional.
    """
    inject_landing_css()

    # --- Spacing ---
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

    # --- Main Layout (Split 60/40) ---
    col_content, col_visual = st.columns([1.4, 1], gap="large")

    with col_content:
        # Title with gradient highlight
        st.markdown("""
        <div class="landing-title">
            Credit Scoring for the <br>
            <span class="highlight-text">Gig Economy.</span>
        </div>
        <div class="landing-subtitle">
            Uplift uses <strong>alternative data</strong>—income volatility, digital footprints, 
            and behavioral stability—to approve borrowers that traditional banks reject.
        </div>
        """, unsafe_allow_html=True)
        
        # CTA Button
        st.markdown("<div style='margin-top: 20px; margin-bottom: 60px;'>", unsafe_allow_html=True)
        if st.button("Launch Dashboard", key="hero_cta"):
            st.session_state['page'] = 'dashboard'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_visual:
        # 3D Credit Card (Dark object floating on light background)
        st.markdown("""
        <div class="floating-card-container">
            <div class="credit-card">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <div class="chip"></div>
                    <div style="color:white; font-weight:bold; font-size:1.2rem;">UPLIFT</div>
                </div>
                <div style="height: 30px;"></div>
                <div class="card-number">4242 9900 1029 8410</div>
                <div class="card-details">
                    <div>
                        <div style="font-size:0.7rem; opacity:0.7;">CARD HOLDER</div>
                        <div>ABHINAV SINHA</div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; opacity:0.7;">EXPIRES</div>
                        <div>08/32</div>
                    </div>
                </div>
                <div style="position:absolute; top:0; left:0; right:0; bottom:0; background: linear-gradient(125deg, rgba(255,255,255,0) 40%, rgba(255,255,255,0.1) 45%, rgba(255,255,255,0) 50%); border-radius: 24px; pointer-events:none;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Features Section (Clean White Cards) ---
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="medium")
    
    def feature_card(title, desc):
        return f"""
        <div class="feature-card">
            <div style="height: 1rem;"></div>
            <h3 style="color: #111827; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;">{title}</h3>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.6;">{desc}</p>
        </div>
        """

    with c1:
        st.markdown(feature_card("TensorFlow Engine", 
                    "Deep neural networks identify non-linear patterns in income streams that simple regression models miss."), unsafe_allow_html=True)
    with c2:
        st.markdown(feature_card("Bias-Free Logic", 
                    "Algorithmic auditing ensures zero discrimination based on zip code, gender, or ethnicity."), unsafe_allow_html=True)
    with c3:
        st.markdown(feature_card("Real-Time Decision", 
                    "Optimized inference pipelines deliver credit approvals in under 200ms at the point of sale."), unsafe_allow_html=True)

    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

def display_navbar():
    """Navbar with 2 columns: Title flush left, Badge on right"""
    col_text, col_badge = st.columns([7, 1])
    with col_text:
        st.markdown("""
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <h3 style="margin: 0; font-size: 3.2rem; font-weight: 700; 
                       background: -webkit-linear-gradient(0deg, #2563EB 0%, #0891B2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Uplift
            </h3>
            <p style="margin: 0; color: #6B7280; font-size: 1.8rem; font-weight: 500;">Credit Risk Analytics</p>
        </div>
        """, unsafe_allow_html=True)

    with col_badge:
        st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 15px;">
            <div style="background: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 20px; 
                        font-size: 1rem; font-weight: 600; border: 1px solid #DBEAFE;">
                LIVE
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

def display_groww_metric(col, label, value, delta, color="#2563EB"):
    """Clean, white-card metric with shadow"""
    with col:
        st.markdown(f"""
        <div style="background-color: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease; animation: fadeIn 0.5s ease-out;">
            <p style="color: #6B7280; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">{label}</p>
            <div style="display: flex; align-items: baseline; gap: 8px;">
                <h2 style="color: #111827; font-size: 1.8rem; font-weight: 700; margin: 0;">{value}</h2>
            </div>
            <p style="color: {color}; font-size: 0.85rem; font-weight: 500; margin-top: 4px;">{delta}</p>
        </div>
        """, unsafe_allow_html=True)

def display_ticker_row(applicants_data):
    """Clean table row style"""
    st.markdown("""
    <div style="background: white; border-radius: 12px; border: 1px solid #E5E7EB; 
                padding: 16px; margin: 24px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 0.95rem;">Recent Applications</h4>
    """, unsafe_allow_html=True)

    for applicant in applicants_data:
        if applicant["Status"] == "Approved":
            bg = "#EFF6FF"; text = "#1E40AF" # Blue-50, Blue-800
        elif applicant["Status"] == "Pending":
            bg = "#FFFBEB"; text = "#D97706"
        else:
            bg = "#FEF2F2"; text = "#DC2626"

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #F3F4F6;">
            <div style="flex: 2;">
                <div style="color: #111827; font-weight: 600;">{applicant["Name"]}</div>
                <div style="color: #6B7280; font-size: 0.8rem;">{applicant["Role"]}</div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="color: #374151; font-weight: 700;">{applicant["Score"]}</div>
                <div style="color: #9CA3AF; font-size: 0.75rem;">Score</div>
            </div>
            <div style="flex: 1; text-align: right;">
                <span style="background: {bg}; color: {text}; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                    {applicant["Status"]}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def create_sidebar_section(title):
    """Sidebar section header"""
    st.markdown(f"""
    <div class="sidebar-section">
        <p style="margin: 0; color: #2563EB; font-size: 0.75rem; font-weight: 700; 
                  text-transform: uppercase; letter-spacing: 1px;">
            {title}
        </p>
    </div>
    """, unsafe_allow_html=True)
