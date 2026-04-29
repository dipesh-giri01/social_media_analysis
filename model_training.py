import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
  classification_report,
  confusion_matrix,
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
)
import warnings

warnings.filterwarnings('ignore')

print('='*70)
print('MLP Model Training and Testing - AIT 500 Project')
print('='*70)

# Load the cleaned dataset
print('\n[1] Loading Dataset...')
df = pd.read_csv('train_cleaned.csv')
print(f'Dataset shape: {df.shape}')
print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')
print('\nFirst few rows:')
print(df.head())

# Display feature information
print('\n[2] Dataset Information:')
print(df.info())
print('\nDataset Statistics:')
print(df.describe())

# Check for missing values
print('\nMissing Values:')
print(df.isnull().sum())

# Define features and target
print('\n[3] Feature Selection and Engineering:')
print('Selected Features:')
print('  - platform_encoded (Platform: Twitter=0, Instagram=1, Facebook=2)')
print('  - post_type_encoded (Post Type: Image=0, Video=1, Text=2)')
print('  - likes (Number of likes)')
print('  - shares (Number of shares)')
print('  - comments (Number of comments)')
print('  - views (Number of views)')
print('  - year, month, day_of_week, week_of_year (Temporal features)')

# Select features for training
features = [
  'platform_encoded',
  'post_type_encoded',
  'likes',
  'shares',
  'comments',
  'views',
  'year',
  'month',
  'day_of_week',
  'week_of_year',
]

X = df[features]

# Create target variable: High engagement (1) vs Low engagement (0)
# Using median of likes as threshold for binary classification
engagement_threshold = df['likes'].median()
y = (df['likes'] > engagement_threshold).astype(int)

print(f'\nTarget Variable: High Engagement Classification')
print(f'Threshold (Median Likes): {engagement_threshold}')
print(f'Class Distribution:')
print(f'  Low Engagement (0): {(y == 0).sum()} samples')
print(f'  High Engagement (1): {(y == 1).sum()} samples')

# Feature Scaling
print('\n[4] Feature Scaling:')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('Applied StandardScaler to all features')
print(f'Scaled feature range: Mean≈0, Std≈1')
print(f'\nScaling parameters:')
for i, col in enumerate(features):
  print(f'  {col}: mean={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}')

# Train-Test Split
print('\n[5] Train-Test Split:')
X_train, X_test, y_train, y_test = train_test_split(
  X_scaled, y, test_size=0.2, random_state=42
)
print(f'Total samples: {len(X_scaled)}')
print(f'Training set: {len(X_train)} samples (80%)')
print(f'Testing set: {len(X_test)} samples (20%)')
print(f'\nTraining set class distribution:')
print(f'  Low Engagement: {(y_train == 0).sum()} samples')
print(f'  High Engagement: {(y_train == 1).sum()} samples')
print(f'\nTesting set class distribution:')
print(f'  Low Engagement: {(y_test == 0).sum()} samples')
print(f'  High Engagement: {(y_test == 1).sum()} samples')

# MLP Model Configuration
print('\n[6] MLP Model Configuration:')
print('Model Architecture:')
print('  Hidden layers: (100, 50)')
print('    - Layer 1: 100 neurons')
print('    - Layer 2: 50 neurons')
print('  Activation function: ReLU (Rectified Linear Unit)')
print('  Solver/Optimizer: Adam')
print('  Learning rate: Adaptive')
print('  Max iterations: 300')
print('  Random state: 42')

mlp_model = MLPClassifier(
  hidden_layer_sizes=(100, 50),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=300,
  random_state=42,
  early_stopping=True,
  validation_fraction=0.1,
  n_iter_no_change=20,
  verbose=0,
)

# Train the model
print('\n[7] Training the MLP Model:')
print('Training in progress...')
mlp_model.fit(X_train, y_train)
print('✓ Training completed!')
print(f'Number of iterations: {mlp_model.n_iter_}')
print(f'Loss: {mlp_model.loss_:.4f}')

# Make predictions
print('\n[8] Model Predictions:')
y_train_pred = mlp_model.predict(X_train)
y_test_pred = mlp_model.predict(X_test)

# Evaluate on training set
print('\n' + '='*70)
print('TRAINING SET PERFORMANCE')
print('='*70)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print(f'Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)')
print(f'Precision: {train_precision:.4f}')
print(f'Recall:    {train_recall:.4f}')
print(f'F1-Score:  {train_f1:.4f}')

print('\nClassification Report (Training Set):')
print(classification_report(y_train, y_train_pred, target_names=['Low Engagement', 'High Engagement']))

print('Confusion Matrix (Training Set):')
cm_train = confusion_matrix(y_train, y_train_pred)
print(cm_train)
print(f'  True Negatives: {cm_train[0, 0]}')
print(f'  False Positives: {cm_train[0, 1]}')
print(f'  False Negatives: {cm_train[1, 0]}')
print(f'  True Positives: {cm_train[1, 1]}')

# Evaluate on testing set
print('\n' + '='*70)
print('TESTING SET PERFORMANCE')
print('='*70)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f'Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
print(f'Precision: {test_precision:.4f}')
print(f'Recall:    {test_recall:.4f}')
print(f'F1-Score:  {test_f1:.4f}')

print('\nClassification Report (Testing Set):')
print(classification_report(y_test, y_test_pred, target_names=['Low Engagement', 'High Engagement']))

print('Confusion Matrix (Testing Set):')
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)
print(f'  True Negatives: {cm_test[0, 0]}')
print(f'  False Positives: {cm_test[0, 1]}')
print(f'  False Negatives: {cm_test[1, 0]}')
print(f'  True Positives: {cm_test[1, 1]}')

# Model comparison summary
print('\n' + '='*70)
print('MODEL PERFORMANCE SUMMARY')
print('='*70)
print(f'{"Metric":<20} {"Training":<15} {"Testing":<15} {"Difference":<15}')
print('-'*70)
print(f'{"Accuracy":<20} {train_accuracy:.4f}{"":<10} {test_accuracy:.4f}{"":<10} {abs(train_accuracy - test_accuracy):.4f}')
print(f'{"Precision":<20} {train_precision:.4f}{"":<10} {test_precision:.4f}{"":<10} {abs(train_precision - test_precision):.4f}')
print(f'{"Recall":<20} {train_recall:.4f}{"":<10} {test_recall:.4f}{"":<10} {abs(train_recall - test_recall):.4f}')
print(f'{"F1-Score":<20} {train_f1:.4f}{"":<10} {test_f1:.4f}{"":<10} {abs(train_f1 - test_f1):.4f}')

# Visualization - Individual images for each statistic
print('\n[9] Generating Individual Visualizations...')

# 1. Confusion Matrix - Training
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(
  cm_train,
  annot=True,
  fmt='d',
  cmap='Blues',
  ax=ax1,
  cbar=True,
  xticklabels=['Low Engagement', 'High Engagement'],
  yticklabels=['Low Engagement', 'High Engagement'],
  cbar_kws={'label': 'Count'},
)
ax1.set_title('Confusion Matrix - Training Set', fontweight='bold', fontsize=14)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('01_confusion_matrix_training.png', dpi=300, bbox_inches='tight')
print('✓ Image 1 saved: 01_confusion_matrix_training.png')
plt.close()

# 2. Confusion Matrix - Testing
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(
  cm_test,
  annot=True,
  fmt='d',
  cmap='Greens',
  ax=ax2,
  cbar=True,
  xticklabels=['Low Engagement', 'High Engagement'],
  yticklabels=['Low Engagement', 'High Engagement'],
  cbar_kws={'label': 'Count'},
)
ax2.set_title('Confusion Matrix - Testing Set', fontweight='bold', fontsize=14)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('02_confusion_matrix_testing.png', dpi=300, bbox_inches='tight')
print('✓ Image 2 saved: 02_confusion_matrix_testing.png')
plt.close()

# 3. Performance Metrics Comparison
fig3, ax3 = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy, train_precision, train_recall, train_f1]
test_scores = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax3.bar(x - width/2, train_scores, width, label='Training', alpha=0.8, color='steelblue')
bars2 = ax3.bar(x + width/2, test_scores, width, label='Testing', alpha=0.8, color='darkorange')

ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Model Performance Metrics Comparison', fontweight='bold', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.legend(fontsize=11)
ax3.set_ylim([0, 1.1])
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
  for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('03_performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
print('✓ Image 3 saved: 03_performance_metrics_comparison.png')
plt.close()

# 4. Feature Importance
fig4, ax4 = plt.subplots(figsize=(10, 7))
feature_importance = np.abs(mlp_model.coefs_[0]).mean(axis=1)
feature_names_short = [
  'Platform',
  'Post Type',
  'Likes',
  'Shares',
  'Comments',
  'Views',
  'Year',
  'Month',
  'Day of Week',
  'Week of Year',
]
sorted_idx = np.argsort(feature_importance)
colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_idx)))
ax4.barh(
  [feature_names_short[i] for i in sorted_idx],
  feature_importance[sorted_idx],
  alpha=0.8,
  color=colors,
)
ax4.set_xlabel('Average Absolute Weight', fontsize=12)
ax4.set_title('Feature Importance (Layer 1 Weights)', fontweight='bold', fontsize=14)
ax4.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('04_feature_importance.png', dpi=300, bbox_inches='tight')
print('✓ Image 4 saved: 04_feature_importance.png')
plt.close()

# 5. Training vs Testing Accuracy
fig5, ax5 = plt.subplots(figsize=(8, 6))
datasets = ['Training', 'Testing']
accuracies = [train_accuracy, test_accuracy]
colors_acc = ['#2ecc71', '#e74c3c']
bars = ax5.bar(datasets, accuracies, color=colors_acc, alpha=0.8, width=0.5)
ax5.set_ylabel('Accuracy', fontsize=12)
ax5.set_title('Overall Model Accuracy Comparison', fontweight='bold', fontsize=14)
ax5.set_ylim([0, 1.1])
ax5.grid(axis='y', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
  ax5.text(bar.get_x() + bar.get_width()/2., acc + 0.02,
          f'{acc:.4f}\n({acc*100:.2f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('05_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print('✓ Image 5 saved: 05_accuracy_comparison.png')
plt.close()

# 6. Precision Comparison
fig6, ax6 = plt.subplots(figsize=(8, 6))
precisions = [train_precision, test_precision]
bars = ax6.bar(datasets, precisions, color=colors_acc, alpha=0.8, width=0.5)
ax6.set_ylabel('Precision', fontsize=12)
ax6.set_title('Model Precision Comparison', fontweight='bold', fontsize=14)
ax6.set_ylim([0, 1.1])
ax6.grid(axis='y', alpha=0.3)
for i, (bar, prec) in enumerate(zip(bars, precisions)):
  ax6.text(bar.get_x() + bar.get_width()/2., prec + 0.02,
          f'{prec:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('06_precision_comparison.png', dpi=300, bbox_inches='tight')
print('✓ Image 6 saved: 06_precision_comparison.png')
plt.close()

# 7. Recall Comparison
fig7, ax7 = plt.subplots(figsize=(8, 6))
recalls = [train_recall, test_recall]
bars = ax7.bar(datasets, recalls, color=colors_acc, alpha=0.8, width=0.5)
ax7.set_ylabel('Recall', fontsize=12)
ax7.set_title('Model Recall Comparison', fontweight='bold', fontsize=14)
ax7.set_ylim([0, 1.1])
ax7.grid(axis='y', alpha=0.3)
for i, (bar, rec) in enumerate(zip(bars, recalls)):
  ax7.text(bar.get_x() + bar.get_width()/2., rec + 0.02,
          f'{rec:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('07_recall_comparison.png', dpi=300, bbox_inches='tight')
print('✓ Image 7 saved: 07_recall_comparison.png')
plt.close()

# 8. F1-Score Comparison
fig8, ax8 = plt.subplots(figsize=(8, 6))
f1_scores = [train_f1, test_f1]
bars = ax8.bar(datasets, f1_scores, color=colors_acc, alpha=0.8, width=0.5)
ax8.set_ylabel('F1-Score', fontsize=12)
ax8.set_title('Model F1-Score Comparison', fontweight='bold', fontsize=14)
ax8.set_ylim([0, 1.1])
ax8.grid(axis='y', alpha=0.3)
for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
  ax8.text(bar.get_x() + bar.get_width()/2., f1 + 0.02,
          f'{f1:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('08_f1_score_comparison.png', dpi=300, bbox_inches='tight')
print('✓ Image 8 saved: 08_f1_score_comparison.png')
plt.close()

# 9. All metrics in individual bar chart
fig9, ax9 = plt.subplots(figsize=(12, 6))
metrics_all = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_values = [train_accuracy, train_precision, train_recall, train_f1]
test_values = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics_all))
width = 0.35
bars1 = ax9.bar(x - width/2, train_values, width, label='Training', alpha=0.85, color='steelblue')
bars2 = ax9.bar(x + width/2, test_values, width, label='Testing', alpha=0.85, color='coral')

ax9.set_ylabel('Score', fontsize=12)
ax9.set_xlabel('Metrics', fontsize=12)
ax9.set_title('Complete Performance Metrics Comparison', fontweight='bold', fontsize=14)
ax9.set_xticks(x)
ax9.set_xticklabels(metrics_all)
ax9.legend(fontsize=11)
ax9.set_ylim([0, 1.15])
ax9.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
  for bar in bars:
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('09_all_metrics_combined.png', dpi=300, bbox_inches='tight')
print('✓ Image 9 saved: 09_all_metrics_combined.png')
plt.close()

print('\n✓ All individual visualization images generated successfully!')

# Additional analysis
print('\n[10] Additional Analysis:')
print('\nPrediction Statistics (Testing Set):')
print(f'Predictions - Low Engagement: {(y_test_pred == 0).sum()}')
print(f'Predictions - High Engagement: {(y_test_pred == 1).sum()}')
print(f'Actual - Low Engagement: {(y_test == 0).sum()}')
print(f'Actual - High Engagement: {(y_test == 1).sum()}')

# Model summary
print('\n' + '='*70)
print('MODEL SUMMARY')
print('='*70)
print(f'Model Type: Multi-layer Perceptron (MLPClassifier)')
print(f'Number of Features: {X_train.shape[1]}')
print(f'Number of Training Samples: {X_train.shape[0]}')
print(f'Number of Testing Samples: {X_test.shape[0]}')
print(f'Hidden Layer Sizes: (100, 50)')
print(f'Activation Function: ReLU')
print(f'Optimizer: Adam')
print(f'Number of Iterations: {mlp_model.n_iter_}')
print(f'Final Loss: {mlp_model.loss_:.4f}')
print(f'\nBest Performing Metric (Testing):')
best_metric = max(
  [
    ('Accuracy', test_accuracy),
    ('Precision', test_precision),
    ('Recall', test_recall),
    ('F1-Score', test_f1),
  ],
  key=lambda x: x[1],
)
print(f'  {best_metric[0]}: {best_metric[1]:.4f}')

print('\n' + '='*70)
print('Training and Testing Complete!')
print('='*70)
