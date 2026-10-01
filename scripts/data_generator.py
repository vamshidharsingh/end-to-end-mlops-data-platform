import pandas as pd
import numpy as np
import os

def generate_data(num_samples=1000, output_path="data/raw/customers.csv"):
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    age = np.random.randint(18, 70, size=num_samples)
    income = np.random.randint(30000, 150000, size=num_samples)
    website_visits = np.random.randint(1, 50, size=num_samples)
    cart_adds = np.random.randint(0, 10, size=num_samples)
    time_on_site = np.random.uniform(1.0, 120.0, size=num_samples)
    
    # Simple logic for target
    prob = (age * 0.01) + (income / 100000) + (website_visits * 0.05) + (cart_adds * 0.2)
    prob = prob / prob.max() # Normalize
    purchased = (np.random.rand(num_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'customer_id': range(1, num_samples + 1),
        'age': age,
        'income': income,
        'website_visits': website_visits,
        'cart_adds': cart_adds,
        'time_on_site': time_on_site,
        'purchased': purchased
    })
    
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} records and saved to {output_path}")

if __name__ == "__main__":
    generate_data()
