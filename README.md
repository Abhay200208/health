# Health Drink Survey Streamlit Dashboard

This repository contains a Streamlit dashboard for analyzing the synthetic health drink survey dataset.

## Features

1. **Data Visualization**: Explore descriptive insights from the survey data.
2. **Classification**: Train and evaluate classification algorithms (KNN, Decision Tree, Random Forest, Gradient Boosting) and make predictions on new data.
3. **Clustering**: Perform K-Means clustering with dynamic cluster count.
4. **Association Rule Mining**: Generate association rules based on selected survey items.
5. **Regression**: Apply Linear, Ridge, Lasso, and Decision Tree regression models for numeric insights.

## Setup

1. Clone the repository.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. In `app.py`, update `DATA_URL` with your GitHub raw CSV URL.
4. Run locally:
    ```
    streamlit run app.py
    ```
5. Deploy on Streamlit Cloud:
   - Push this repo to GitHub.
   - Connect your GitHub repo in Streamlit Cloud.
