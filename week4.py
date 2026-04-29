# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv('social.csv')

# Remove duplicate columns if any
df = df.loc[:, ~df.columns.duplicated()]

# Drop missing values
df = df.dropna()

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Extract time-based features
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Create engagement column
df['engagement'] = df['likes_count'] + df['shares_count'] + df['comments_count']

# Log-transform target to reduce skew
df['engagement_log'] = np.log1p(df['engagement'])

# ================================
# 3. Feature Engineering
# ================================

# Numeric Features
numeric_features = [
    'sentiment_score', 'toxicity_score', 'impressions',
    'user_past_sentiment_avg', 'user_engagement_growth', 'buzz_change_rate',
    'hour', 'day', 'month'
]
X_numeric = df[numeric_features]

# Categorical Features
categorical_features = ['platform', 'topic_category', 'campaign_phase']
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_categorical = encoder.fit_transform(df[categorical_features])

# Text Features (content + hashtags + keywords)
text_data = df['text_content'].fillna('') + ' ' + df['hashtags'].fillna('') + ' ' + df['keywords'].fillna('')
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_text = vectorizer.fit_transform(text_data)

# Combine all features
X = hstack([X_numeric.values, X_categorical, X_text])

# Target
y = df['engagement_log']  # log-transformed

# ================================
# 4. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 5. Train Model (Gradient Boosting)
# ================================
model = GradientBoostingRegressor(
    n_estimators=1000,      # More trees for better learning
    learning_rate=0.05,     # Moderate learning rate
    max_depth=8,            # Allow complex relationships
    random_state=42
)
model.fit(X_train, y_train)

# ================================
# 6. Predict and Evaluate
# ================================
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # convert back to original scale
y_test_original = np.expm1(y_test)

mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print("Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# ================================
# 7. Sample Prediction for New Post
# ================================
# Example new post features
sample_numeric = np.array([[0.1, 0.3, 20000, 0.05, 0.02, 5.0, 14, 15, 6]])
sample_categorical = encoder.transform(pd.DataFrame({
    'platform': ['Instagram'],
    'topic_category': ['Tech'],
    'campaign_phase': ['Launch']
}))
sample_text = vectorizer.transform(["Just tried the new Pixel phone #Tech #Launch"])

sample_features = hstack([sample_numeric, sample_categorical, sample_text])
predicted_engagement_log = model.predict(sample_features)
predicted_engagement = np.expm1(predicted_engagement_log)
print("\nPredicted Engagement for New Post:", predicted_engagement)

# ================================
# 8. Visualization
# ================================
# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test_original[:100], y_pred[:100], alpha=0.6)
plt.xlabel("Actual Engagement")
plt.ylabel("Predicted Engagement")
plt.title("Actual vs Predicted Engagement (Sample 100 posts)")
plt.show()