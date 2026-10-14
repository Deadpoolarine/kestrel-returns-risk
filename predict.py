import pandas as pd
import pickle

# Load models and test data
test = pd.read_csv('test_unlabelled.csv')
customers = pd.read_csv('customers.csv')
products = pd.read_csv('products.csv')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# Merge and clean identically to training
df = test.merge(customers, on='customer_id', how='left')
df = df.merge(products, on='sku', how='left')
df['delivery_pincode'] = df['delivery_pincode'].fillna(0)
df['discount_pct'] = df['discount_pct'].fillna(0)

for col in encoders:
    # Handle unseen labels by filling with a default or mode
    df[col] = df[col].astype(str).apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
    df[col] = encoders[col].transform(df[col])

# Predict and save
X_test = df[features]
predictions = pd.DataFrame({
    'order_id': test['order_id'],
    'score': model.predict_proba(X_test)[:, 1]
})
predictions.to_csv('predictions.csv', index=False)
print("predictions.csv generated successfully.")