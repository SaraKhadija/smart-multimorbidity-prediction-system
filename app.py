import streamlit as st
import pandas as pd
import pickle
import xgboost
import shap
from xgboost import XGBClassifier
import pickle
import plotly.express as px
import plotly.express as px
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from reportlab.lib import colors
from io import BytesIO
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="SMART Multimorbidity Prediction System",
    page_icon="🏥",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #ffffff,
        #f3f6fa
    );
}

h1, h2, h3 {
    color: #0d47a1;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-weight: bold;
}

/* Risk cards */

[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:20px;
    box-shadow:0px 6px 16px rgba(0,0,0,0.08);
    padding:15px;
    border:1px solid #e0e0e0;
}

</style>
""", unsafe_allow_html=True)

    
st.markdown("""
    <div style="
    width:100%;
    background:linear-gradient(135deg,#0d47a1,#1976d2);
    padding:40px;
    border-radius:20px;
    color:white;
    text-align:center;
    margin-bottom:25px;
    ">

    <h1 style="font-size:48px; margin-bottom:15px;">
    SMART MULTIMORBIDITY PREDICTION AND DECISION SUPPORT SYSTEM
    </h1>

    <h3 style="margin-bottom:20px;">
    🧠 Predict • 🩺 Prevent • 🛡️ Protect
    </h3>

    <p style="font-size:18px;">
    Transforming healthcare risk assessment through Explainable AI and advanced XGBoost predictive modeling.
    </p>

    </div>
    """, unsafe_allow_html=True)

st.info(
    "AI-Powered Clinical Decision Support System for Diabetes, Heart Disease, and Cancer Risk Assessment"
)

# =====================================
# LOAD FILES
# =====================================

@st.cache_resource
def load_files():

    with open("xgb_final_models.pkl", "rb") as f:
        models = pickle.load(f)

    with open("xgb_best_thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)

    with open("encoded_feature_columns.pkl", "rb") as f:
        encoded_columns = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("numerical_cols.pkl", "rb") as f:
        numerical_cols = pickle.load(f)

    return (
        models,
        thresholds,
        encoded_columns,
        scaler,
        numerical_cols
    )

(
    models,
    thresholds,
    encoded_columns,
    scaler,
    numerical_cols
) = load_files()

# =====================================
# INPUT FORM
# =====================================

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:

    general_health = st.selectbox(
        "General Health",
        ["Excellent", "Very Good", "Good", "Fair", "Poor"]
    )

    checkup = st.selectbox(
        "Last Medical Checkup",
        [
            "Within the past year",
            "Within the past 2 years",
            "Within the past 5 years",
            "5 or more years ago",
            "Never"
        ]
    )

    exercise = st.checkbox(
        "Exercise During Past Month"
    )

    exercise = "Yes" if exercise else "No"

    depression = st.checkbox(
        "Ever Diagnosed with Depression"
    )
    depression = "Yes" if depression else "No"

    arthritis = st.checkbox(
        "Ever Diagnosed with Arthritis"
    )
    arthritis = "Yes" if arthritis else "No"

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    age_category = st.selectbox(
        "Age Category",
        [
            "18-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80+"
        ]
    )

with col2:

    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=170.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=70.0
    )

    smoking_history = st.checkbox(
        "Smoking History"
    )
    smoking_history = "Yes" if smoking_history else "No"

    alcohol = st.number_input(
        "Alcohol Consumption",
        min_value=0,
        value=1,
        step=1
    )

    fruit = st.number_input(
        "Fruit Consumption",
        min_value=0,
        value=30,
        step=1
    )

    vegetables = st.number_input(
        "Green Vegetables Consumption",
        min_value=0,
        value=30,
        step=1
    )

    fried_potato = st.number_input(
        "Fried Potato Consumption",
        min_value=0,
        value=5,
        step=1
    )

# =====================================
# PREDICTION
# =====================================

# Create SHAP explainers
explainers = {}

for target_name, model in models.items():
    explainers[target_name] = shap.TreeExplainer(model)

#Function PDF generator
def generate_pdf_report(
    patient_info,
    risk_results,
    status,
    predicted_conditions,
    recommendations,
    contributing_factors
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    title = Paragraph(
        "SMART CHRONIC DISEASE PREDICTION REPORT",
        styles["Title"]
    )

    story.append(title)
    story.append(Spacer(1, 12))

    report_id = datetime.now().strftime(
        "CDP-%Y%m%d-%H%M%S"
    )

    story.append(
        Paragraph(
            f"<b>Report ID:</b> {report_id}",
            styles["Normal"]
        )
    )

    generated_date = Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}",
        styles["Normal"]
    )

    story.append(generated_date)
    story.append(Spacer(1, 20))

    # PATIENT PROFILE

    story.append(
        Paragraph(
            "Patient Profile",
            styles["Heading2"]
        )
    )

    for key, value in patient_info.items():

        story.append(
            Paragraph(
                f"<b>{key}:</b> {value}",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 15))

    # RISK ASSESSMENT

    story.append(
        Paragraph(
            "Risk Assessment",
            styles["Heading2"]
        )
    )

    for result in risk_results:

        disease = (
            result["Disease"]
            .replace("_Target", "")
            .replace("_", " ")
        )

        risk_level = (
            "HIGH RISK"
            if result["Prediction"] == 1
            else "LOW RISK"
        )

        story.append(
            Paragraph(
                f"{disease}: {result['Probability']:.1%} ({risk_level})",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 15))

    # MULTIMORBIDITY

    story.append(
        Paragraph(
            "Multimorbidity Status",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            status,
            styles["Normal"]
        )
    )

    for condition in predicted_conditions:

        story.append(
            Paragraph(
                f"• {condition}",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 15))

    # FACTORS

    story.append(
        Paragraph(
            "Key Contributing Factors",
            styles["Heading2"]
        )
    )

    for factor in contributing_factors:

        story.append(
            Paragraph(
                f"• {factor}",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 15))

    # RECOMMENDATIONS

    story.append(
        Paragraph(
            "Personalized Recommendations",
            styles["Heading2"]
        )
    )

    for rec in recommendations:

        story.append(
            Paragraph(
                f"• {rec}",
                styles["Normal"]
            )
        )

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            "Disclaimer",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "This report is intended for educational and decision-support purposes only. "
            "The predictions generated by this system do not constitute medical diagnosis. "
            "Patients should consult qualified healthcare professionals for medical advice and treatment decisions.",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            "Generated by SMART CHRONIC DISEASES PREDICTION SYSTEM",
            styles["Italic"]
        )
    )

    story.append(
        Paragraph(
            "Centre for Mathematical Sciences, Universiti Malaysia Pahang Al-Sultan Abdullah",
            styles["Italic"]
        )
    )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf

if st.button("Predict Risk"):
    all_shap_features = []

    bmi = weight / ((height / 100) ** 2)
    patient_info = {
        "Age Category": age_category,
        "Sex": sex,
        "Height (cm)": height,
        "Weight (kg)": weight,
        "BMI": round(bmi, 2),
        "Smoking History": smoking_history,
        "Exercise": exercise,
        "Alcohol Consumption": alcohol,
        "Fruit Consumption": fruit,
        "Green Vegetables Consumption": vegetables,
        "Fried Potato Consumption": fried_potato
    }

    input_df = pd.DataFrame({
        "General_Health": [general_health],
        "Checkup": [checkup],
        "Exercise": [exercise],
        "Depression": [depression],
        "Arthritis": [arthritis],
        "Sex": [sex],
        "Age_Category": [age_category],
        "Height_(cm)": [height],
        "Weight_(kg)": [weight],
        "BMI": [bmi],
        "Smoking_History": [smoking_history],
        "Alcohol_Consumption": [alcohol],
        "Fruit_Consumption": [fruit],
        "Green_Vegetables_Consumption": [vegetables],
        "FriedPotato_Consumption": [fried_potato]
    })

    # Same encoding used during training
    input_encoded = pd.get_dummies(
        input_df
    )

    # Match training columns
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # Same scaling used during training
    input_scaled = input_encoded.copy()

    input_scaled[numerical_cols] = scaler.transform(
        input_scaled[numerical_cols]
    )

    st.header("Prediction Results")

    predictions = {}
    probabilities = {}

    risk_results = []

    for target_name, model in models.items():

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        threshold = thresholds[target_name]

        prediction = int(
            probability >= threshold
        )
        predictions[target_name] = prediction
        probabilities[target_name] = probability

        risk_results.append({
            "Disease": target_name,
            "Probability": probability,
            "Prediction": prediction
        })

        # SHAP collection for high-risk diseases
        if prediction == 1:

            explainer = explainers[target_name]

            shap_values = explainer.shap_values(
                input_scaled
            )

            shap_df = pd.DataFrame({
                "Feature": input_scaled.columns,
                "SHAP Value": shap_values[0]
            })

            top_positive = (
                shap_df[shap_df["SHAP Value"] > 0]
                .sort_values(
                    "SHAP Value",
                    ascending=False
                )
                .head(5)
            )

            all_shap_features.append(
                top_positive
            )

    st.markdown("---")
    st.header("Risk Assessment Dashboard")

    col1, col2, col3 = st.columns(3)

    card_columns = [col1, col2, col3]

    icons = {
        "Diabetes_Target": "🩸",
        "Heart_Disease_Target": "❤️",
        "Cancer_Target": "🎗️"
    }

    for idx, result in enumerate(risk_results):

        disease = result["Disease"]
        probability = result["Probability"]

        risk_level = (
            "HIGH RISK"
            if result["Prediction"] == 1
            else "LOW RISK"
        )

        color = (
            "#ff4b4b"
            if result["Prediction"] == 1
            else "#00c853"
        )

        with card_columns[idx]:

            disease_name = (
                disease
                .replace("_Target", "")
                .replace("_", " ")
            )

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div style='text-align:center'>
                        <h1>{icons.get(disease,'🏥')}</h1>
                        <h3>{disease_name}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    "<div style='text-align:center;'>Risk Probability</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<h1 style='text-align:center;'>{probability:.1%}</h1>",
                    unsafe_allow_html=True
                )
                if result["Prediction"] == 1:
                    progress_color = "#e53935"   # red

                else:
                    progress_color = "#43a047"   # green

                st.markdown(
                    f"""
                    <div style="
                        width:100%;
                        background:#eeeeee;
                        border-radius:10px;
                        height:12px;
                        margin-bottom:15px;
                    ">
                        <div style="
                            width:{probability*100:.1f}%;
                            background:{progress_color};
                            height:12px;
                            border-radius:10px;
                        ">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                if result["Prediction"] == 1:

                    st.markdown(
                        """
                        <div style='
                            background:#ffebee;
                            border-left:8px solid #e53935;
                            color:#d32f2f;
                            padding:15px;
                            border-radius:10px;
                            text-align:center;
                            font-weight:bold;
                            font-size:20px;
                        '>
                            HIGH RISK
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div style='
                            background:#e8f5e9;
                            border-left:8px solid #43a047;
                            color:#2e7d32;
                            padding:15px;
                            border-radius:10px;
                            text-align:center;
                            font-weight:bold;
                            font-size:20px;
                        '>
                            LOW RISK
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("Multimorbidity Status")

    predicted_conditions = []

    for disease, pred in predictions.items():

        if pred == 1:

            predicted_conditions.append(
                disease
                .replace("_Target", "")
                .replace("_", " ")
            )

    num_positive = len(predicted_conditions)

    if num_positive == 0:

        risk_color = "#4CAF50"
        status = "LOW RISK"
        recommendation = (
            "Excellent health profile detected. "
            "Continue maintaining your healthy lifestyle and regular health practices. "
            "Keep up the good work! 🌟"
        )

    elif num_positive == 1:

        risk_color = "#FF9800"
        status = "MODERATE RISK"
        recommendation = (
            "One chronic disease risk factor was detected. "
            "Lifestyle improvements and periodic health monitoring are recommended."
        )

    else:

        risk_color = "#F44336"
        status = "HIGH RISK"
        recommendation = (
            "Multimorbidity risk was detected."
            "A comprehensive medical assessment and appropriate lifestyle modifications are strongly recommended."
        )

    if len(predicted_conditions) == 0:

        predicted_conditions_html = """
        <p style='color:#4CAF50;font-weight:bold;'>
        None
        </p>
    """

    else:

        predicted_conditions_html = f"""
        <ul>
            {''.join([f'<li>{x}</li>' for x in predicted_conditions])}
        </ul>
        """

    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background:#f8f9fa;
            border-left:8px solid {risk_color};
            box-shadow:0px 4px 8px rgba(0,0,0,0.1);
        ">

        <h2>Risk Status: {status}</h2>

        <p><b>Number of Predicted Conditions:</b>
            {num_positive}</p>

        <p><b>Predicted Conditions:</b></p>

        {predicted_conditions_html}

        <p><b>Recommendation:</b></p>

        <p>{recommendation}</p>

        </div>
        """,
        unsafe_allow_html=True
    )
    

    st.markdown("---")
    st.header("Key Contributing Factors")

    if len(all_shap_features) > 0:

        combined = pd.concat(
            all_shap_features
        )

        feature_summary = (
            combined
            .groupby("Feature")
            ["SHAP Value"]
            .mean()
            .reset_index()
        )

        feature_summary = (
            feature_summary
            .sort_values(
                "SHAP Value",
                ascending=False
            )
            .head(10)
        )
    else:

        st.info(
            "No significant risk factors detected because no disease was predicted as high risk."
        )

        feature_summary = pd.DataFrame({
            "Feature": [],
            "SHAP Value": [],
            "Display": []
        })

    feature_names = {

        # General Health
        "General_Health_Poor": "Poor General Health",
        "General_Health_Fair": "Fair General Health",
        "General_Health_Good": "Good General Health",
        "General_Health_Very Good": "Very Good General Health",

        # Age
        "Age_Category_80+": "Age Above 80 Years",
        "Age_Category_75-79": "Age 75-79 Years",
        "Age_Category_70-74": "Age 70-74 Years",
        "Age_Category_65-69": "Age 65-69 Years",

        "Sex_Male": "Male Gender",
        "Sex_Female": "Female Gender",

        # Lifestyle & Health
        "Smoking_History_Yes": "Smoking History",
        "Depression_Yes": "Depression",
        "Arthritis_Yes": "Arthritis",
        "Exercise_Yes": "Regular Exercise",

        # Continuous Variables
        "BMI": "Body Mass Index (BMI)",
        "Weight_(kg)": "Body Weight",
        "Height_(cm)": "Height",
        "Alcohol_Consumption": "Alcohol Consumption",
        "Fruit_Consumption": "Fruit Consumption",
        "Green_Vegetables_Consumption": "Vegetable Consumption",
        "FriedPotato_Consumption": "Fried Potato Consumption"
    }

    feature_summary["Display"] = (
        feature_summary["Feature"]
        .map(feature_names)
        .fillna(
            feature_summary["Feature"]
        )
    )

    contributing_factors = (
        feature_summary["Display"]
        .tolist()
    )

    if not feature_summary.empty:

        fig = px.bar(
            feature_summary,
            x="SHAP Value",
            y="Display",
            orientation="h",
            title="Top Factors Associated with the Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        top_risk_factors = (
            feature_summary["Display"]
            .head(5)
            .tolist()
        )

        st.markdown(
            """
            <div style="
                background:#fafafa;
                border-left:8px solid #fb8c00;
                padding:20px;
                border-radius:12px;
                margin-top:20px;
                margin-bottom:20px;
            ">
                <h3>⚠️ Top Risk Drivers</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        for factor in top_risk_factors:

            st.markdown(
                f"""
                <div style="
                    background:#fafafa;
                    padding:12px;
                    margin-bottom:8px;
                    border-radius:10px;
                    border-left:5px solid #ef5350;
                    font-size:16px;
                    font-weight:500;
                ">
                    🔺 {factor}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Split factors into risk vs protective

        risk_factors = feature_summary[
            feature_summary["SHAP Value"] > 0
        ]

        protective_factors = feature_summary[
            feature_summary["SHAP Value"] < 0
        ]
        

    st.markdown("---")
    st.header("Personalized Health Recommendations")
    recommendations = []
    
    if smoking_history == "Yes":

        recommendations.append(
            "🚭 Reduce or quit smoking to lower the risk of chronic diseases."
        )

    if exercise == "No":

        recommendations.append(
            "🏃 Increase physical activity to at least 150 minutes per week."
        )

    if vegetables < 15:

        recommendations.append(
            "🥦 Increase green vegetable consumption."
        )

    if fruit < 15:

        recommendations.append(
            "🍎 Increase fruit consumption."
        )

    if bmi >= 30:

        recommendations.append(
            "⚖️ Maintain a healthier body weight through diet and exercise."
        )

    if age_category in [
        "65-69",
        "70-74",
        "75-79",
        "80+"
    ]:

        recommendations.append(
            "🩺 Schedule regular health screenings due to age-related risk."
        )

    if num_positive >= 2:

        recommendations.append(
            "🏥 Seek comprehensive medical assessment due to multimorbidity risk."
        )

    elif num_positive == 1:

        recommendations.append(
            "📅 Follow up with healthcare professionals regarding the identified risk."
        )

    for rec in recommendations:

        st.markdown(
            f"""
            <div style="
                padding:15px;
                margin-bottom:10px;
                border-radius:10px;
                background:#f8f9fa;
                border-left:5px solid #2196F3;
            ">
                {rec}
            </div>
            """,
            unsafe_allow_html=True
        )

    pdf_recommendations = [
        rec.replace("🚭", "")
        .replace("🏃", "")
        .replace("🥦", "")
        .replace("🍎", "")
        .replace("⚖️", "")
        .replace("🩺", "")
        .replace("🏥", "")
        .replace("📅", "")
        .strip()
        for rec in recommendations
    ]

    pdf_file = generate_pdf_report(
        patient_info=patient_info,
        risk_results=risk_results,
        status=status,
        predicted_conditions=predicted_conditions,
        recommendations=pdf_recommendations,
        contributing_factors=contributing_factors
    )

    st.markdown("---")

    st.download_button(
        label="📄 Download Health Risk Report",
        data=pdf_file,
        file_name="health_risk_report.pdf",
        mime="application/pdf"
    )

st.markdown("---")

with st.expander("ℹ️ About This System"):

    st.markdown("""
This system predicts the risk of:

- Diabetes
- Heart Disease
- Cancer

using machine learning models developed with XGBoost.

The system also provides:

- Multimorbidity assessment
- Explainable AI insights
- Personalized recommendations
- Downloadable health reports

This system is intended for educational and decision-support purposes only.
""")

st.markdown("---")

st.markdown("""
<div style="
    margin-top:50px;
    padding:25px;
    border-top:1px solid #dce3ea;
    text-align:center;
    color:#666;
    font-size:14px;
">

<h4 style="
    color:#0d47a1;
    margin-bottom:15px;
">
SMART MULTIMORBIDITY PREDICTION AND DECISION SUPPORT SYSTEM
</h4>

<p>
<b>Centre for Mathematical Sciences</b><br>
Universiti Malaysia Pahang Al-Sultan Abdullah
</p>

<p>
<b>XGBoost</b> |
<b>SHAP Explainable AI</b> |
<b>Streamlit</b>
</p>

<p>
Version 1.0 | © 2026 UMPSA
</p>

</div>
""", unsafe_allow_html=True)
