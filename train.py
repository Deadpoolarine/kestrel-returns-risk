import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load Data
train = pd.read_csv('train.csv')
customers = pd.read_csv('customers.csv')
products = pd.read_csv('products.csv')

# 2. Merge Data
df = train.merge(customers, on='customer_id', how='left')
df = df.merge(products, on='sku', how='left')

# 3. Clean & Feature Engineering
# DROP LEAKAGE COLUMNS - these happen after dispatch!
df = df.drop(columns=['last_service_event_type', 'pickup_scheduled_at', 'delivery_note'])

# Fill missing values
df['delivery_pincode'] = df['delivery_pincode'].fillna(0)
df['discount_pct'] = df['discount_pct'].fillna(0)

# Encode categorical variables (FIXED: Added 'is_gift' to this list)
cat_cols = ['sales_channel', 'payment_mode', 'source', 'city', 'state', 'shield_member', 'family', 'is_gift']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Define Features and Target
X = df.drop(columns=['order_id', 'order_placed_at', 'customer_id', 'sku', 'returned', 'model_name', 'signup_date', 'launch_date'])
y = df['returned']

# 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 5. Save Artifacts for API
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
    
print("Training complete. Models saved successfully.")