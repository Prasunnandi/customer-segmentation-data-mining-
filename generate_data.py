import pandas as pd
import numpy as np

def generate_mock_data(filepath="customer_data.csv"):
    np.random.seed(42)
    n_customers = 500
    
    data = {
        'CustomerID': range(1, n_customers + 1),
        'Annual_Income_k$': np.random.normal(60, 20, n_customers).astype(int),
        'Spending_Score_1_100': np.random.normal(50, 25, n_customers).astype(int),
        'Age': np.random.randint(18, 70, n_customers)
    }
    
    df = pd.DataFrame(data)
    # Clip values to realistic ranges
    df['Spending_Score_1_100'] = df['Spending_Score_1_100'].clip(1, 100)
    df['Annual_Income_k$'] = df['Annual_Income_k$'].clip(15, 150)
    df.to_csv(filepath, index=False)
    print(f"Mock data generated and saved to {filepath}")

if __name__ == "__main__":
    generate_mock_data()
