import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder

# Load your saved model and preprocessing objects with caching
@st.cache_resource
def load_objects():
    model = joblib.load('engagement_model.pkl')
    scaler = joblib.load('scaler.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, tfidf, label_encoders

model, scaler, tfidf, label_encoders = load_objects()

def preprocess(data: pd.DataFrame):
    categorical_cols = list(label_encoders.keys())

    def encode_with_fallback(le: LabelEncoder, values):
        known_classes = set(le.classes_)
        fallback = le.classes_[0]
        safe_values = [v if v in known_classes else fallback for v in values]
        return le.transform(safe_values)

    for col in categorical_cols:
        if col in data.columns:
            le = label_encoders[col]
            data[col] = encode_with_fallback(le, data[col].astype(str))
        else:
            st.error(f"Column '{col}' missing from input data.")
            return None

    # List all features required for numeric processing
    numeric_features = categorical_cols + [
        'likes_count', 'shares_count', 'comments_count', 'impressions',
        'sentiment_score', 'toxicity_score', 'user_past_sentiment_avg',
        'user_engagement_growth', 'buzz_change_rate'
    ]

    # Check for missing columns in numeric features
    missing_numeric = [col for col in numeric_features if col not in data.columns]
    if missing_numeric:
        st.error(f"Missing numeric columns: {missing_numeric}")
        return None

    X_numeric = data[numeric_features].fillna(0)
    X_numeric_scaled = scaler.transform(X_numeric)
    X_text = tfidf.transform(data['text_content'].astype(str))

    X_final = hstack([X_numeric_scaled, X_text])
    return X_final

st.title("Social Media Engagement Rate Prediction")

uploaded_file = st.file_uploader("Upload CSV file with posts data", type=['csv'])

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    st.write("Uploaded Data Preview:")
    st.dataframe(input_data.head())

    if st.button("Predict Engagement Rate"):
        X = preprocess(input_data)
        if X is not None:
            preds = model.predict(X)
            input_data['predicted_engagement_rate'] = preds

            st.write("Predictions:")
            cols_to_show = ['post_id', 'predicted_engagement_rate']
            if 'post_id' not in input_data.columns:
                cols_to_show = ['predicted_engagement_rate']
            st.dataframe(input_data[cols_to_show])

            # Plot interactive line chart if actual engagement_rate exists
            if 'engagement_rate' in input_data.columns:
                st.subheader("Actual vs Predicted Engagement Rate (Interactive)")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=input_data['engagement_rate'],
                    mode='lines+markers',
                    name='Actual Engagement Rate',
                    line=dict(color='blue'),
                    marker=dict(size=6)
                ))
                fig.add_trace(go.Scatter(
                    y=input_data['predicted_engagement_rate'],
                    mode='lines+markers',
                    name='Predicted Engagement Rate',
                    line=dict(color='orange', dash='dash'),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title='Actual vs Predicted Engagement Rate',
                    xaxis_title='Sample Index',
                    yaxis_title='Engagement Rate',
                    hovermode='x unified',
                    legend=dict(font_size=12),
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
