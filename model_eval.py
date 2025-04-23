import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1) Reload data and simulate contexts exactly as training
df = pd.read_csv("../dataset_generate/training_dataset_v2.csv")
np.random.seed(42)
df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)
df_sample['relationship_type'] = np.random.choice(['romance','friends','family','sibling'], len(df_sample))
df_sample['user_gender'] = np.random.choice(['m','f','o'], len(df_sample))
df_sample['partner_gender'] = np.random.choice(['m','f','o'], len(df_sample))
df_sample['user_age'] = np.random.randint(18,100, len(df_sample))
df_sample['partner_age'] = np.random.randint(18,100, len(df_sample))
df_sample['budget'] = np.random.randint(500,5000, len(df_sample))
# Bin ages
bins = [18,25,35,45,60,100]
labels = ['18-24','25-34','35-44','45-59','60+']
df_sample['user_age_bin'] = pd.cut(df_sample['user_age'], bins=bins, labels=labels, right=False)
df_sample['partner_age_bin'] = pd.cut(df_sample['partner_age'], bins=bins, labels=labels, right=False)

# 2) Prepare features and raw labels
features = ['relationship_type','user_gender','partner_gender','user_age_bin','partner_age_bin','budget','district']
X = df_sample[features]
y_raw = df_sample['interest']

# 3) Load trained artifacts
pipeline = joblib.load('model_pipeline.joblib')
le = joblib.load('label_encoder.joblib')

# 4) Encode raw labels and split test set
y_encoded = le.transform(y_raw)
_, X_test, _, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5) Predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
labels_test = le.inverse_transform(y_test)
labels_pred = le.inverse_transform(y_pred)

# 6) Evaluation metrics
print(f"Accuracy:       {accuracy_score(labels_test, labels_pred):.4f}")
print(f"Weighted F1:    {f1_score(labels_test, labels_pred, average='weighted'):.4f}")
print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, y_proba, k=3):.4f}\n")
print("Classification Report:\n", classification_report(labels_test, labels_pred))
print("Confusion Matrix:\n", confusion_matrix(labels_test, labels_pred))