import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")
st.write("Upload your customer data or use the default dataset to discover purchasing behavior segments.")

# Data Upload
uploaded_file = st.file_uploader("Upload customer_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("customer_data.csv"):
    df = pd.read_csv("customer_data.csv")
else:
    st.error("No data found. Please run generate_data.py first or upload a CSV.")
    st.stop()

st.sidebar.header("Model Configuration")
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)

if st.button("Run Segmentation"):
    with st.spinner("Training K-Means Model..."):
        # Select Features
        X = df[['Annual_Income_k$', 'Spending_Score_1_100']]
        
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Modeling
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters
        
        # Evaluation
        score = silhouette_score(X_scaled, clusters)
        
        # Layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success("Model trained successfully!")
            st.metric("Silhouette Score", f"{score:.3f}", help="Closer to 1 means better defined clusters.")
            st.write("### Data Preview")
            st.dataframe(df.head())
            
        with col2:
            st.write("### Cluster Visualization")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='Annual_Income_k$', y='Spending_Score_1_100', hue='Cluster', 
                            data=df, palette='viridis', ax=ax, s=100)
            plt.title(f"Customer Segments (K={n_clusters})")
            st.pyplot(fig)
            
    st.info("💡 **Deployment tip:** You can host this dashboard for free on [Streamlit Community Cloud](https://streamlit.io/cloud) by pushing this code to a GitHub repository.")
