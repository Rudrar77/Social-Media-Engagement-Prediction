# Social Media Engagement Rate Prediction

This repository contains a machine learning project focused on predicting the engagement rate of social media posts. The model combines numeric features, categorical variables, and textual content to accurately estimate how much engagement (likes, shares, comments, etc.) a post is likely to receive.

## Overview

The approach uses a Random Forest Regressor trained on data that includes interaction metrics, sentiment scores, and user engagement trends. Textual data from posts is vectorized using TF-IDF to capture the semantic meaning. Categorical data such as platform, day of the week, and topic category are encoded to numeric values. Numeric features are scaled for improved model performance.

After training, the model's accuracy is assessed using RMSE and R² metrics. The project also includes visualization tools to analyze predicted versus actual engagement rates, as well as feature importance plots to interpret the model.

An interactive Streamlit app is provided to allow users to upload their own CSV data files of social media posts and receive engagement rate predictions instantly. The app supports visualization of predictions alongside actual values when available.

## Dataset

The dataset includes:
- Post content (text)
- Numeric features: likes, shares, comments, impressions
- Sentiment and toxicity scores
- User historical engagement metrics
- Categorical variables like platform, day_of_week, and topic_category
- Target variable: engagement_rate

## How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Rudrar77/Social-Media-Engagement-Prediction.git
   cd Social-Media-Engagement-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional if you want to retrain):**
   Run the training script to preprocess data, train the model, and save necessary artifacts.

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Upload your CSV file** with social media posts through the app interface to get predicted engagement rates.

## Project Structure

```
Social-Media-Engagement-Prediction/
├── SME.csv                     # Original dataset
├── train_model.py              # Script for data preprocessing and model training
├── app.py                      # Streamlit app for predictions
├── engagement_model.pkl        # Trained model saved
├── tfidf_vectorizer.pkl        # Saved text vectorizer
├── scaler.pkl                  # Saved scaler for numeric data
├── label_encoders.pkl          # Saved categorical encoders
├── requirements.txt            # Project dependencies
└── README.md                   # This documentation
```

## Features

- **Machine Learning Model**: Random Forest Regressor for accurate engagement prediction
- **Text Processing**: TF-IDF vectorization for semantic analysis
- **Data Preprocessing**: Automated scaling and encoding of features
- **Interactive App**: User-friendly Streamlit interface
- **Visualization**: Comprehensive plots for model interpretation
- **Model Evaluation**: RMSE and R² metrics for performance assessment

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib/Seaborn
- TF-IDF Vectorization

## Contact

For questions or collaboration, contact: rudrarathod29112005@gmail.com

You can explore the full project and download all files here: https://github.com/Rudrar77/Social-Media-Engagement-Prediction

## License

This project is open source and available under the [MIT License](LICENSE).
