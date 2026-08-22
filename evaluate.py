"""
ArthroScan AI - Model Evaluation & Academic Report Suite
Author: Nikita Tuteja
Description: Evaluates the trained model on unseen holdout test data.
Generates publication-ready Confusion Matrix, Classification Report, Loss/Accuracy Curves,
and Grad-CAM visual heatmaps for final year project presentation & defense.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from src.preprocess import apply_clahe, CLEAN_CLASS_NAMES
from src.gradcam import get_gradcam_heatmap, generate_superimposed_gradcam

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def plot_confusion_matrix(y_true, y_pred, classes, output_path='reports/confusion_matrix.png'):
    """
    Plots and saves high-resolution raw and normalized Confusion Matrices.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=True)
    axes[0].set_title('Confusion Matrix (Raw Sample Counts)', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Predicted Severity Grade', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Ground Truth Grade', fontsize=11, fontweight='bold')

    # Normalized Percentages
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=axes[1], cbar=True)
    axes[1].set_title('Normalized Confusion Matrix (Recall %)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Predicted Severity Grade', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Ground Truth Grade', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Confusion Matrix saved to: {output_path}")

def plot_training_history(log_csv='reports/training_log.csv', output_path='reports/training_history.png'):
    """
    Plots training and validation accuracy and loss over all training epochs.
    """
    if not os.path.exists(log_csv) or os.path.getsize(log_csv) == 0:
        print(f"  ⚠️ Note: {log_csv} not populated yet, skipping history plot.")
        return

    try:
        df_log = pd.read_csv(log_csv)
    except Exception as e:
        print(f"  ⚠️ Warning reading {log_csv}: {e}")
        return

    if len(df_log) == 0 or 'accuracy' not in df_log.columns:
        print("  ⚠️ Training log contains no epoch entries yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    epochs = np.arange(1, len(df_log) + 1)

    # Accuracy Plot
    axes[0].plot(epochs, df_log['accuracy'], label='Training Accuracy', color='#3b82f6', linewidth=2.5, marker='o', markersize=4)
    if 'val_accuracy' in df_log.columns:
        axes[0].plot(epochs, df_log['val_accuracy'], label='Validation Accuracy', color='#10b981', linewidth=2.5, linestyle='--', marker='s', markersize=4)
    axes[0].set_title('Model Accuracy vs Epochs', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right')

    # Loss Plot
    axes[1].plot(epochs, df_log['loss'], label='Training Loss', color='#ef4444', linewidth=2.5, marker='o', markersize=4)
    if 'val_loss' in df_log.columns:
        axes[1].plot(epochs, df_log['val_loss'], label='Validation Loss', color='#f59e0b', linewidth=2.5, linestyle='--', marker='s', markersize=4)
    axes[1].set_title('Cross-Entropy Loss vs Epochs', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📈 Training Curves saved to: {output_path}")

def generate_sample_gradcam_grid(model, test_df, output_path='reports/sample_gradcam_predictions.png'):
    """
    Selects sample test X-rays from each severity class and generates side-by-side
    Original vs CLAHE vs Grad-CAM heatmap visualizations.
    """
    samples = []
    for c_name in CLEAN_CLASS_NAMES:
        class_samples = test_df[test_df['class_name'] == c_name]
        if len(class_samples) > 0:
            samples.append(class_samples.iloc[0])

    if not samples:
        print("  ⚠️ No test samples available for Grad-CAM grid.")
        return

    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 3.8 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, row in enumerate(samples):
        filepath = row['filepath']
        true_class = row['class_name']

        # Load & Preprocess
        raw_img = cv2.imread(filepath)
        if raw_img is None:
            continue
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img_resized = cv2.resize(raw_img, (256, 256))
        
        enhanced = apply_clahe(raw_img_resized) / 255.0
        input_tensor = np.expand_dims(enhanced, axis=0)

        # Inference & Grad-CAM
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = CLEAN_CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])

        heatmap = get_gradcam_heatmap(input_tensor, model, pred_index=pred_idx)
        superimposed = generate_superimposed_gradcam(raw_img_resized, heatmap, alpha=0.45)

        # Plot Original
        axes[i, 0].imshow(raw_img_resized)
        axes[i, 0].set_title(f"True Grade: {true_class}", fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')

        # Plot CLAHE Preprocessed
        axes[i, 1].imshow(np.clip(enhanced, 0.0, 1.0))
        axes[i, 1].set_title("CLAHE Contrast Enhanced", fontsize=11)
        axes[i, 1].axis('off')

        # Plot Grad-CAM
        is_correct = (pred_class == true_class)
        tag_color = '#15803d' if is_correct else '#b91c1c'
        axes[i, 2].imshow(superimposed)
        axes[i, 2].set_title(
            f"AI Focus: {pred_class} ({confidence*100:.1f}%)",
            fontsize=11, fontweight='bold', color=tag_color
        )
        axes[i, 2].axis('off')

    plt.suptitle("ArthroScan AI - Grad-CAM Joint Space Localization", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  🔥 Sample Grad-CAM visualizations saved to: {output_path}")

def main():
    print("="*70)
    print("🧪 ARTHROSCAN AI - MODEL EVALUATION & REPORT SUITE")
    print("="*70)

    model_path = 'models/best_model_improved.keras'
    test_csv = 'data_splits/test.csv'
    os.makedirs('reports', exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split not found at: {test_csv}. Please run train.py first.")

    print(f"\n[1/4] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    print(f"\n[2/4] Evaluating on Holdout Test Set ({test_csv})...")
    test_df = pd.read_csv(test_csv)
    print(f"Total Test Samples: {len(test_df)}")

    # Fast vectorized batch inference
    batch_size = 32
    all_images = []
    y_true = []
    
    print("Pre-loading test batch images with CLAHE enhancement...")
    for idx, row in test_df.iterrows():
        raw_img = cv2.imread(row['filepath'])
        if raw_img is None:
            continue
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img = cv2.resize(raw_img, (256, 256))
        enhanced = apply_clahe(raw_img) / 255.0
        all_images.append(enhanced)
        y_true.append(row['label'])

    X_test = np.array(all_images, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    print(f"Executing fast batched prediction across {len(X_test)} test samples...")
    y_probs = model.predict(X_test, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    # Metrics calculation
    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "="*50)
    print(f"🏆 OVERALL TEST ACCURACY:   {test_acc * 100:.2f}%")
    print(f"🎯 MACRO F1-SCORE:          {macro_f1:.4f}")
    print(f"⚖️ WEIGHTED F1-SCORE:       {weighted_f1:.4f}")
    print("="*50)

    # Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=CLEAN_CLASS_NAMES, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=CLEAN_CLASS_NAMES, zero_division=0)
    print("\nDetailed Classification Report:")
    print(report_text)

    # Save Reports
    with open('reports/classification_report.txt', 'w') as f:
        f.write("ARTHROSCAN AI - KNEE OSTEOARTHRITIS CLASSIFICATION REPORT\n")
        f.write("Author: Nikita Tuteja\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Macro F1-Score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score: {weighted_f1:.4f}\n\n")
        f.write(report_text)
    
    pd.DataFrame(report_dict).transpose().to_csv('reports/classification_report.csv')
    print("  📄 Classification reports saved to reports/classification_report.txt and .csv")

    print("\n[3/4] Generating Academic Visual Figures...")
    plot_confusion_matrix(y_true, y_pred, CLEAN_CLASS_NAMES, output_path='reports/confusion_matrix.png')
    plot_training_history(log_csv='reports/training_log.csv', output_path='reports/training_history.png')

    print("\n[4/4] Generating Explainable AI (Grad-CAM) Visualizations...")
    generate_sample_gradcam_grid(model, test_df, output_path='reports/sample_gradcam_predictions.png')

    print("\n" + "="*70)
    print("✅ ACADEMIC REPORT SUITE COMPLETE!")
    print("All figures and metrics saved in 'reports/' folder for your project presentation & defense.")
    print("="*70)

if __name__ == "__main__":
    main()
