import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_segmentation():
    if not os.path.exists("customer_data.csv"):
        print("Data not found. Run generate_data.py first.")
        return

    df = pd.read_csv("customer_data.csv")
    X = df[['Annual_Income_k$', 'Spending_Score_1_100']]
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Evaluation
    score = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score for 4 clusters: {score:.3f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual_Income_k$', y='Spending_Score_1_100', hue='Cluster', data=df, palette='viridis')
    plt.title("Customer Segments")
    plt.savefig("segmentation_results.png")
    print("Results saved to segmentation_results.png")
    
if __name__ == "__main__":
    run_segmentation()
