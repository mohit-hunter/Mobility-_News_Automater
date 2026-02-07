"""
Baseline Multinomial Logistic Regression Classifier for News Classification

This script trains a baseline supervised classifier using:
- PRECOMPUTED semantic embeddings (from News_Semantic_Embedding.npy)
- Multinomial Logistic Regression with L2 regularization
- Class weighting to handle imbalanced classes (especially 'suppliers')

Training data: classified_news_3.csv (with 'category' labels)
Test data: test.csv (unlabeled)
Embeddings: News_Semantic_Embedding.npy (concatenated train + test embeddings)

Output:
- test_predictions.csv: Predictions with confidence scores
- baseline_model.pkl: Trained model for future use
- training_metrics.json: Training performance metrics
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(train_path, test_path, embeddings_path):
    """Load training data, test data, and precomputed embeddings."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load training data
    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    n_train = len(train_df)
    print(f"  [OK] Loaded {n_train} training samples")
    
    # Load test data
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    n_test = len(test_df)
    print(f"  [OK] Loaded {n_test} test samples")
    
    # Load precomputed embeddings
    print(f"\nLoading precomputed embeddings from: {embeddings_path}")
    all_embeddings = np.load(embeddings_path)
    print(f"  [OK] Loaded embeddings with shape: {all_embeddings.shape}")
    
    # Verify alignment
    expected_total = n_train + n_test
    if all_embeddings.shape[0] != expected_total:
        raise ValueError(
            f"Embedding count ({all_embeddings.shape[0]}) does not match "
            f"train + test count ({expected_total})"
        )
    print(f"  [OK] Embedding count matches train + test data")
    
    # Split embeddings into train and test portions
    train_embeddings = all_embeddings[:n_train]
    test_embeddings = all_embeddings[n_train:]
    
    print(f"\n  Training embeddings shape: {train_embeddings.shape}")
    print(f"  Test embeddings shape: {test_embeddings.shape}")
    
    # Show category distribution
    print("\n" + "-" * 50)
    print("TRAINING DATA CATEGORY DISTRIBUTION")
    print("-" * 50)
    category_counts = train_df['category'].value_counts()
    total = len(train_df)
    for cat, count in category_counts.items():
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {cat:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Highlight class imbalance
    min_class = category_counts.idxmin()
    min_count = category_counts.min()
    max_count = category_counts.max()
    imbalance_ratio = max_count / min_count
    print(f"\n  [!] Class imbalance detected:")
    print(f"      Most underrepresented: '{min_class}' with {min_count} samples")
    print(f"      Imbalance ratio: {imbalance_ratio:.1f}x")
    
    return train_df, test_df, train_embeddings, test_embeddings


def train_classifier(X_train, y_train):
    """
    Train multinomial logistic regression classifier with class weighting.
    
    Uses:
    - L2 regularization (appropriate for high-dimensional features)
    - lbfgs solver (efficient for multinomial classification)
    - class_weight='balanced' to handle class imbalance
    """
    print("\n" + "=" * 70)
    print("TRAINING MULTINOMIAL LOGISTIC REGRESSION")
    print("=" * 70)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    print(f"\nClass labels: {list(label_encoder.classes_)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Calculate class weights for display
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_encoded)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
    
    print("\n" + "-" * 50)
    print("CLASS WEIGHTS (balanced)")
    print("-" * 50)
    for cls_idx, weight in zip(classes, class_weights):
        cls_name = label_encoder.classes_[cls_idx]
        print(f"  {cls_name:12s}: {weight:.4f}")
    
    # Initialize classifier with balanced class weights
    print("\n[OK] Using class_weight='balanced' to handle imbalance")
    classifier = LogisticRegression(
        solver='lbfgs',             # Efficient for multinomial with L2
        penalty='l2',               # L2 regularization
        C=1.0,                      # Regularization strength (inverse)
        class_weight='balanced',    # CRITICAL: Handle class imbalance
        max_iter=1000,              # Sufficient iterations for convergence
        random_state=42
    )
    
    # Cross-validation for performance estimation
    print("\n" + "-" * 50)
    print("CROSS-VALIDATION (5-fold stratified)")
    print("-" * 50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X_train, y_encoded, cv=cv, scoring='accuracy')
    
    print(f"\n  Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Train final model on full training data
    print("\nTraining final model on full training data...")
    classifier.fit(X_train, y_encoded)
    print("  [OK] Model training complete")
    
    # Training set performance
    train_predictions = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_encoded, train_predictions)
    print(f"\nTraining set accuracy: {train_accuracy:.4f}")
    
    # Detailed classification report
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT (Training Data)")
    print("-" * 50)
    report = classification_report(
        y_encoded, 
        train_predictions, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    print(classification_report(
        y_encoded, 
        train_predictions, 
        target_names=label_encoder.classes_
    ))
    
    # Special attention to suppliers class
    print("-" * 50)
    print("SUPPLIERS CLASS PERFORMANCE (previously underrepresented)")
    print("-" * 50)
    if 'suppliers' in report:
        sup = report['suppliers']
        print(f"  Precision: {sup['precision']:.4f}")
        print(f"  Recall:    {sup['recall']:.4f}")
        print(f"  F1-score:  {sup['f1-score']:.4f}")
        print(f"  Support:   {int(sup['support'])}")
    
    # Confusion matrix
    print("\n" + "-" * 50)
    print("CONFUSION MATRIX (Training Data)")
    print("-" * 50)
    cm = confusion_matrix(y_encoded, train_predictions)
    print(f"\nClasses: {list(label_encoder.classes_)}")
    
    # Pretty print confusion matrix
    print("\nPredicted ->")
    header = "Actual     " + "".join([f"{c:>12}" for c in label_encoder.classes_])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{label_encoder.classes_[i]:10s} " + "".join([f"{v:>12}" for v in row])
        print(row_str)
    
    # Analyze confusion patterns
    print("\n" + "-" * 50)
    print("CONFUSION ANALYSIS")
    print("-" * 50)
    
    # Find the index for each class
    class_to_idx = {c: i for i, c in enumerate(label_encoder.classes_)}
    
    if 'automakers' in class_to_idx and 'suppliers' in class_to_idx:
        auto_idx = class_to_idx['automakers']
        sup_idx = class_to_idx['suppliers']
        
        # Automakers predicted as suppliers
        auto_as_sup = cm[auto_idx, sup_idx]
        sup_as_auto = cm[sup_idx, auto_idx]
        
        print(f"  'automakers' misclassified as 'suppliers': {auto_as_sup}")
        print(f"  'suppliers' misclassified as 'automakers': {sup_as_auto}")
    
    return classifier, label_encoder, cv_scores, cm, report


def predict_test(classifier, label_encoder, X_test, test_df):
    """Generate predictions for test data with confidence scores."""
    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)
    
    # Get predictions
    predictions_encoded = classifier.predict(X_test)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    # Get prediction probabilities
    probabilities = classifier.predict_proba(X_test)
    
    # Get confidence (probability of predicted class)
    confidence = np.max(probabilities, axis=1)
    
    print(f"\nGenerated predictions for {len(predictions)} test samples")
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['predicted_category'] = predictions
    results_df['prediction_confidence'] = confidence
    
    # Add individual class probabilities
    for i, class_name in enumerate(label_encoder.classes_):
        results_df[f'prob_{class_name}'] = probabilities[:, i]
    
    # Show prediction distribution
    print("\n" + "-" * 50)
    print("TEST PREDICTION DISTRIBUTION")
    print("-" * 50)
    pred_counts = pd.Series(predictions).value_counts()
    total = len(predictions)
    for cat, count in pred_counts.items():
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {cat:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Show confidence statistics
    print("\n" + "-" * 50)
    print("PREDICTION CONFIDENCE STATISTICS")
    print("-" * 50)
    print(f"  Mean:   {confidence.mean():.4f}")
    print(f"  Std:    {confidence.std():.4f}")
    print(f"  Min:    {confidence.min():.4f}")
    print(f"  Max:    {confidence.max():.4f}")
    print(f"  Median: {np.median(confidence):.4f}")
    
    # Per-class confidence
    print("\n  Per-class mean confidence:")
    for cat in label_encoder.classes_:
        mask = predictions == cat
        if mask.sum() > 0:
            cat_conf = confidence[mask].mean()
            print(f"    {cat:12s}: {cat_conf:.4f}")
    
    return results_df, predictions, confidence, probabilities


def save_outputs(results_df, classifier, label_encoder, cv_scores, cm, 
                 report, output_dir):
    """Save all outputs: predictions, model, and metrics."""
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    
    # Save predictions
    predictions_path = output_dir / 'test_predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    print(f"\n[OK] Saved predictions to: {predictions_path}")
    
    # Save model
    model_path = output_dir / 'baseline_model.pkl'
    model_bundle = {
        'classifier': classifier,
        'label_encoder': label_encoder,
        'model_type': 'Multinomial Logistic Regression',
        'class_weight': 'balanced',
        'embedding_source': 'News_Semantic_Embedding.npy',
        'created_at': datetime.now().isoformat()
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"[OK] Saved model to: {model_path}")
    
    # Prepare metrics
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    metrics = {
        'model_type': 'Multinomial Logistic Regression',
        'solver': 'lbfgs',
        'penalty': 'L2',
        'class_weight': 'balanced',
        'embedding_source': 'News_Semantic_Embedding.npy (precomputed)',
        'embedding_dimensions': 384,
        'cross_validation': {
            'folds': 5,
            'mean_accuracy': float(cv_scores.mean()),
            'std_accuracy': float(cv_scores.std()),
            'fold_scores': [float(s) for s in cv_scores]
        },
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in label_encoder.classes_],
        'per_class_metrics': convert_to_native(report),
        'created_at': datetime.now().isoformat(),
        'note': 'Baseline model with class weighting to handle imbalance'
    }
    
    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved training metrics to: {metrics_path}")
    
    return predictions_path, model_path, metrics_path


def main():
    """Main execution pipeline."""
    print("\n" + "=" * 70)
    print("BASELINE MULTINOMIAL LOGISTIC REGRESSION CLASSIFIER")
    print("(Using Precomputed Embeddings with Class Weighting)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = Path(__file__).parent
    train_path = script_dir / 'classified_news_3.csv'
    test_path = script_dir / 'test.csv'
    embeddings_path = script_dir / 'News_Semantic_Embedding.npy'
    
    # Step 1: Load data and embeddings
    train_df, test_df, train_embeddings, test_embeddings = load_data(
        train_path, test_path, embeddings_path
    )
    
    # Step 2: Train classifier
    classifier, label_encoder, cv_scores, cm, report = train_classifier(
        train_embeddings,
        train_df['category'].tolist()
    )
    
    # Step 3: Generate predictions
    results_df, predictions, confidence, probabilities = predict_test(
        classifier, label_encoder, test_embeddings, test_df
    )
    
    # Step 4: Save outputs
    predictions_path, model_path, metrics_path = save_outputs(
        results_df, classifier, label_encoder, cv_scores, cm, report, script_dir
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("BASELINE CLASSIFIER COMPLETE")
    print("=" * 70)
    print(f"\nKey improvements over previous baseline:")
    print(f"  1. Uses precomputed semantic embeddings (no TF-IDF/NMF)")
    print(f"  2. Class weighting to address 'suppliers' underrepresentation")
    print(f"  3. Better handling of class imbalance")
    
    print(f"\nOutputs created:")
    print(f"  1. {predictions_path}")
    print(f"  2. {model_path}")
    print(f"  3. {metrics_path}")
    
    print(f"\nThis baseline model should be used for comparison with future models.")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
