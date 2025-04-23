import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# 1) Load scraped venue data
df = pd.read_csv("../dataset_generate/training_dataset_v2.csv")

# 2) Define age‑binning helper
def bin_age(age_series):
    bins = [18, 25, 35, 45, 60, 100]
    labels = ['18-24', '25-34', '35-44', '45-59', '60+']
    return pd.cut(age_series, bins=bins, labels=labels, right=False)

# 3) Simulate user contexts for training
np.random.seed(42)
df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)
df_sample['relationship_type'] = np.random.choice(['romance','friends','family','sibling'], len(df_sample))
df_sample['user_gender'] = np.random.choice(['m','f','o'], len(df_sample))
df_sample['partner_gender'] = np.random.choice(['m','f','o'], len(df_sample))
df_sample['user_age'] = np.random.randint(18,100, len(df_sample))
df_sample['partner_age'] = np.random.randint(18,100, len(df_sample))
df_sample['budget'] = np.random.randint(500,5000, len(df_sample))

# 4) Apply age bins
df_sample['user_age_bin'] = bin_age(df_sample['user_age'])
df_sample['partner_age_bin'] = bin_age(df_sample['partner_age'])

# 5) Prepare features and target
features = ['relationship_type','user_gender','partner_gender',
            'user_age_bin','partner_age_bin','budget','district']
X = df_sample[features]
y = df_sample['interest']

# 6) Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 8) Preprocessing pipeline
cat_feats = ['relationship_type','user_gender','partner_gender','user_age_bin','partner_age_bin','district']
bud_feat = ['budget']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats),
    ('bud', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'), bud_feat)
])

pipeline = Pipeline([
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# 9) Fit and save artifacts
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'model_pipeline.joblib')
joblib.dump(le, 'label_encoder.joblib')