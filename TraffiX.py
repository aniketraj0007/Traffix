import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Traffix",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #f5f7fb;
}

/* Main content */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

section[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}

/* Main text */
h1, h2, h3 {
    color: #172033 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e3e7ef;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0px 2px 8px rgba(20, 30, 50, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
}

[data-testid="stMetricValue"] {
    color: #172033 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: none;
    background-color: #2563eb;
    color: white !important;
    font-weight: 600;
    padding: 0.65rem 1rem;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white !important;
}

/* Select boxes and date inputs */
div[data-baseweb="select"] > div {
    border-radius: 8px;
}

input {
    border-radius: 8px !important;
}

/* Horizontal line */
hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 25px 0;
}

/* Info / alert boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# FILE PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "traffic_congestion_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "feature_columns.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "traffic.csv"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

except Exception as e:

    st.error("Machine learning model could not be loaded.")

    st.write("Make sure these files are present:")

    st.code(
        """
traffic_congestion_model.pkl
feature_columns.pkl
        """
    )

    st.stop()


# ==========================================================
# LOAD DATA
# ==========================================================

try:

    df = pd.read_csv(DATA_PATH)

except Exception:

    st.error("traffic.csv was not found.")

    st.write(
        "Keep traffic.csv in the same folder as Traffix.py."
    )

    st.stop()


# ==========================================================
# DATA PREPARATION
# ==========================================================

try:

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    df["Hour"] = df["DateTime"].dt.hour
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek

except Exception:

    st.error("Could not process the DateTime column.")
    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🚦 Traffix")

    st.caption("Smart Traffic Monitoring")

    st.divider()

    st.markdown("### Prediction")

    junction = st.selectbox(
        "Select Junction",
        [1, 2, 3, 4]
    )

    selected_date = st.date_input(
        "Select Date"
    )

    selected_time = st.time_input(
        "Select Time"
    )

    predict_button = st.button(
        "Predict Traffic",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.markdown("### System")

    st.write("Machine Learning")
    st.caption("Random Forest Classifier")

    st.write("Dataset")
    st.caption(f"{len(df):,} historical records")


# ==========================================================
# MAIN HEADER
# ==========================================================

st.title("🚦 Traffix")

st.write(
    "Smart Traffic Monitoring & Congestion Prediction"
)

st.caption(
    "Analyse historical traffic patterns and estimate congestion "
    "for selected junctions and time periods."
)


# ==========================================================
# TOP METRICS
# ==========================================================

total_records = len(df)

average_vehicles = df["Vehicles"].mean()

junction_average = (
    df.groupby("Junction")["Vehicles"]
    .mean()
)

busiest_junction = junction_average.idxmax()

busiest_value = junction_average.max()


st.markdown("### Traffic Overview")

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        label="Traffic Records",
        value=f"{total_records:,}",
        help="Total historical traffic observations"
    )

with m2:

    st.metric(
        label="Average Vehicles",
        value=f"{average_vehicles:.1f}",
        help="Average number of vehicles per observation"
    )

with m3:

    st.metric(
        label="Busiest Junction",
        value=f"Junction {busiest_junction}",
        delta=f"{busiest_value:.1f} vehicles"
    )


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    hour = selected_time.hour
    day = selected_date.day
    month = selected_date.month
    day_of_week = selected_date.weekday()

    input_data = pd.DataFrame({

        "Junction": [junction],

        "Hour": [hour],

        "Day": [day],

        "Month": [month],

        "DayOfWeek": [day_of_week]

    })

    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]

    # Probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        confidence = probabilities.max() * 100

    else:

        confidence = 0


    # Prediction interpretation

    if prediction == "Low":

        icon = "🟢"

        message = (
            "Traffic is expected to remain light. "
            "Normal travel conditions are likely."
        )

    elif prediction == "Medium":

        icon = "🟡"

        message = (
            "Traffic may be noticeable. "
            "Allow some additional travel time."
        )

    else:

        icon = "🔴"

        message = (
            "Heavy traffic is expected. "
            "Consider leaving earlier or using an alternate route."
        )


    # ======================================================
    # PREDICTION RESULT
    # ======================================================

    st.markdown("---")

    st.markdown("### Traffic Prediction")

    r1, r2 = st.columns([2, 1])

    with r1:

        st.success(
            f"{icon} Predicted Traffic Level: **{prediction}**"
        )

        st.write(message)

        st.caption(
            f"Junction {junction}  •  "
            f"{selected_date.strftime('%d %b %Y')}  •  "
            f"{selected_time.strftime('%I:%M %p')}"
        )

    with r2:

        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )

        st.progress(
            min(int(confidence), 100)
        )


# ==========================================================
# TRAFFIC TREND
# ==========================================================

st.markdown("---")

st.markdown("### Traffic Trend")

st.caption(
    "Average vehicle count across different hours of the day."
)


hourly_data = (
    df.groupby("Hour")["Vehicles"]
    .mean()
    .reset_index()
)


fig = px.line(
    hourly_data,
    x="Hour",
    y="Vehicles",
    markers=True
)


fig.update_layout(

    height=400,

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),

    plot_bgcolor="white",

    paper_bgcolor="#ffffff",

    font=dict(
        color="#4b5563"
    ),

    xaxis=dict(
        title="Hour of Day",
        dtick=1,
        gridcolor="#edf0f5"
    ),

    yaxis=dict(
        title="Average Vehicles",
        gridcolor="#edf0f5"
    ),

    hovermode="x unified"

)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================================
# JUNCTION ANALYSIS
# ==========================================================

st.markdown("### Junction Analysis")

st.caption(
    "Average traffic volume recorded at each monitored junction."
)


junction_data = (
    df.groupby("Junction")["Vehicles"]
    .mean()
    .reset_index()
)


junction_data["Junction"] = (
    "Junction "
    + junction_data["Junction"].astype(str)
)


fig2 = px.bar(
    junction_data,
    x="Junction",
    y="Vehicles",
    text_auto=".1f"
)


fig2.update_layout(

    height=380,

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),

    plot_bgcolor="white",

    paper_bgcolor="#ffffff",

    font=dict(
        color="#4b5563"
    ),

    xaxis=dict(
        title="",
        gridcolor="#edf0f5"
    ),

    yaxis=dict(
        title="Average Vehicles",
        gridcolor="#edf0f5"
    )

)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.markdown("---")

st.markdown("### About Traffix")

info1, info2 = st.columns(2)

with info1:

    st.info(
        """
        **Purpose**

        Traffix analyses historical traffic data and uses
        machine learning to estimate the expected congestion
        level for a selected junction and time.
        """
    )

with info2:

    st.info(
        """
        **Technology**

        Python · Pandas · Scikit-learn · Random Forest ·
        Plotly · Streamlit
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        Traffix · Smart Traffic Monitoring & Congestion Prediction
        <br>
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)