import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

st.set_page_config(page_title="Customer Segmentation Pro", layout="wide", page_icon="🎯")

st.title("🎯 Customer Segmentation Pro Dashboard")
st.markdown("Leverage machine learning to discover purchasing behavior segments and get AI-powered marketing strategies.")

# Data Upload
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload customer_data.csv", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    elif os.path.exists("customer_data.csv"):
        return pd.read_csv("customer_data.csv")
    return None

df = load_data(uploaded_file)

if df is None:
    st.error("No data found. Please run generate_data.py first or upload a CSV in the sidebar.")
    st.stop()

n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)
run_btn = st.sidebar.button("Run Segmentation", type="primary")

# Initialize session state for chat and clusters
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cluster_stats" not in st.session_state:
    st.session_state.cluster_stats = None
if "model_run" not in st.session_state:
    st.session_state.model_run = False

if run_btn or st.session_state.model_run:
    st.session_state.model_run = True
    
    # Preprocessing & Modeling
    X = df[['Annual_Income_k$', 'Spending_Score_1_100']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, df['Cluster'])
    
    # Compute cluster stats for LLM
    cluster_stats = df.groupby('Cluster').agg({
        'Annual_Income_k$': ['mean', 'min', 'max'],
        'Spending_Score_1_100': ['mean', 'min', 'max'],
        'CustomerID': 'count'
    }).round(2)
    st.session_state.cluster_stats = cluster_stats.to_markdown()

    # Layout Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Visualization", "📈 Cluster Analytics", "🤖 AI Marketing Assistant"])
    
    with tab1:
        st.subheader("Segmentation Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Clusters Formed", n_clusters)
        col3.metric("Silhouette Score", f"{score:.3f}", help="Closer to 1 means better defined clusters.")
        
        st.markdown("### Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Annual_Income_k$', 
            y='Spending_Score_1_100', 
            hue='Cluster', 
            data=df, 
            palette='Set2', 
            ax=ax, 
            s=120, 
            edgecolor='w'
        )
        plt.title(f"Customer Segments (K={n_clusters})", fontsize=14)
        plt.xlabel("Annual Income (k$)", fontsize=12)
        plt.ylabel("Spending Score (1-100)", fontsize=12)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Cluster Profiles")
        st.dataframe(cluster_stats, use_container_width=True)
        st.write("### Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
    with tab3:
        st.subheader("💬 AI Marketing Assistant")
        st.markdown("Ask our AI about how to target these clusters effectively.")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Chat Input
        if prompt := st.chat_input("E.g., How should I target Cluster 1?"):
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Real LLM Response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Generating marketing strategy... ▌")
                
                try:
                    from hf_llm import HFInferenceLLM
                    llm = HFInferenceLLM(model_id="distilgpt2", max_new_tokens=200)
                    
                    full_prompt = (
                        f"Stats: {st.session_state.cluster_stats}\n\n"
                        f"Question: {prompt}\n\n"
                        f"Answer:"
                    )
                    full_response = llm.invoke(full_prompt)
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"An error occurred: {str(e)}"
                    message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("👈 Please configure your model in the sidebar and click **Run Segmentation** to start.")
