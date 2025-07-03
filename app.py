import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Health Drink Survey Dashboard", layout="wide")

st.sidebar.title("Configuration")
DATA_URL = st.sidebar.text_input("GitHub Raw CSV URL", "https://raw.githubusercontent.com/username/repo/main/health_drink_survey_1000.csv")

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

data = load_data(DATA_URL)
num_cols = data.select_dtypes(include=np.number)

tabs = st.tabs(["Data Visualization", "Classification", "Clustering", "Association Rules", "Regression"])

# 1. Data Visualization
with tabs[0]:
    st.header("Data Visualization")
    st.subheader("1. Gender Distribution")
    fig, ax = plt.subplots()
    data['gender'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Gender Distribution")
    st.pyplot(fig)
    st.write("Shows distribution by gender.")

    st.subheader("2. Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(data['age'], bins=10)
    ax.set_title("Age Distribution")
    st.pyplot(fig)
    st.write("Histogram of ages.")

    st.subheader("3. Monthly Income Distribution")
    fig, ax = plt.subplots()
    ax.hist(data['monthly_income'], bins=10)
    ax.set_title("Monthly Income (AED)")
    st.pyplot(fig)
    st.write("Income distribution.")

    st.subheader("4. Income vs. Average Spend")
    fig, ax = plt.subplots()
    ax.scatter(data['monthly_income'], data['avg_monthly_spend'], alpha=0.5)
    ax.set_xlabel("Income (AED)")
    ax.set_ylabel("Avg Spend (AED)")
    ax.set_title("Income vs Spend")
    st.pyplot(fig)
    st.write("Relationship between income and spend.")

    st.subheader("5. Primary Health Goals")
    goals = data['primary_health_goals'].explode().value_counts()
    fig, ax = plt.subplots()
    goals.plot(kind='barh', ax=ax)
    ax.set_title("Health Goals")
    st.pyplot(fig)
    st.write("Common health objectives.")

    st.subheader("6. Weekly Consumption")
    fig, ax = plt.subplots()
    data['health_drink_consumption_per_week'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Weekly Consumption")
    st.pyplot(fig)
    st.write("Number of servings per week.")

    st.subheader("7. Satisfaction Levels")
    fig, ax = plt.subplots()
    data['satisfaction'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Satisfaction")
    st.pyplot(fig)
    st.write("Current satisfaction with health drinks.")

    st.subheader("8. Flavor Preferences")
    flavors = data['flavor_profiles'].explode().value_counts()
    fig, ax = plt.subplots()
    flavors.plot(kind='bar', ax=ax)
    ax.set_title("Flavor Preferences")
    st.pyplot(fig)
    st.write("Preferred flavor profiles.")

    st.subheader("9. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(num_cols.corr(), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.write("Correlation matrix of numeric features.")

    st.subheader("10. Purchase Channels")
    channels = data['purchase_channels'].explode().value_counts()
    fig, ax = plt.subplots()
    channels.plot(kind='bar', ax=ax)
    ax.set_title("Purchase Channels")
    st.pyplot(fig)
    st.write("Where respondents buy drinks.")

# 2. Classification
with tabs[1]:
    st.header("Classification Models")
    target = 'would_try_new_brand'
    features = st.multiselect("Features", num_cols.columns.tolist(), default=['age','monthly_income'])
    X = data[features]
    y = LabelEncoder().fit_transform(data[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'Train Acc': accuracy_score(y_train, y_pred_train),
            'Test Acc': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    st.table(pd.DataFrame(results))

    sel = st.selectbox("Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[sel].predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax.plot(fpr, tpr, label=name)
    ax.plot([0,1],[0,1],'k--')
    ax.legend(); ax.set_title("ROC Curve")
    st.pyplot(fig)

    st.subheader("Predict New Data")
    upd = st.file_uploader("Upload CSV", type="csv")
    if upd:
        df_new = pd.read_csv(upd)
        X_new = scaler.transform(df_new[features])
        preds = models[sel].predict(X_new)
        df_new['prediction'] = preds
        st.download_button("Download Predictions", df_new.to_csv(index=False), file_name="predictions.csv")

# 3. Clustering
with tabs[2]:
    st.header("K-Means Clustering")
    k = st.slider("Clusters", 2, 10, 3)
    inertias = [KMeans(n_clusters=i, random_state=42).fit(num_cols).inertia_ for i in range(2,11)]
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertias, marker='o')
    ax.set_title("Elbow Method"); st.pyplot(fig)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(num_cols)
    data['cluster'] = kmeans.labels_
    st.subheader("Centroids")
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=num_cols.columns))
    st.subheader("Cluster Summary")
    st.dataframe(data.groupby('cluster')[num_cols.columns].mean())
    st.download_button("Download Clusters", data.to_csv(index=False), file_name="clustered.csv")

# 4. Association Rules
with tabs[3]:
    st.header("Association Rule Mining")
    cols = st.multiselect("Select fields", ['primary_health_goals','beverage_types','purchase_channels','flavor_profiles','main_barriers','promotions_sway'])
    support = st.slider("Min Support", 0.01, 0.5, 0.1)
    confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)
    if cols:
        trans = data[cols].apply(lambda row: sum(row.tolist(), []), axis=1).tolist()
        te = TransactionEncoder()
        te_arr = te.fit(trans).transform(trans)
        df_te = pd.DataFrame(te_arr, columns=te.columns_)
        freq = apriori(df_te, min_support=support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=confidence)
        st.write("Top 10 Association Rules")
        st.dataframe(rules.head(10))

# 5. Regression
with tabs[4]:
    st.header("Regression Models")
    reg_target = st.selectbox("Target", ['avg_monthly_spend','max_price_pay_250ml'])
    reg_features = st.multiselect("Features", num_cols.columns.drop(reg_target).tolist(), default=num_cols.columns.drop(reg_target).tolist())
    Xr = data[reg_features]; yr = data[reg_target]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)
    reg_models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Tree': DecisionTreeRegressor(random_state=42)
    }
    reg_res = []
    for name, m in reg_models.items():
        m.fit(Xr_train, yr_train)
        reg_res.append({'Model': name, 'Train R2': m.score(Xr_train, yr_train), 'Test R2': m.score(Xr_test, yr_test)})
    st.table(pd.DataFrame(reg_res))
