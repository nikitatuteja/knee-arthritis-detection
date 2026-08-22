"""
ArthroScan AI - End-to-End Model Training Pipeline
Author: Nikita Tuteja
Description: Trains DenseNet-121 on Knee X-ray images with CLAHE contrast enhancement,
balanced class weighting, and two-stage transfer learning.
"""

import os
import sys
import argparse

# Ensure UTF-8 output encoding on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
from src.preprocess import collect_dataset_records, prepare_data_splits, get_data_generators
from src.model import build_densenet_model, compute_balanced_class_weights, train_model_two_phase

def main():
    parser = argparse.ArgumentParser(description="Train Knee Arthritis Detection AI")
    parser.add_argument("--img_size", type=int, default=256, help="Input image dimension (default: 256)")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for head (default: 5)")
    parser.add_argument("--finetune_epochs", type=int, default=20, help="Fine-tuning epochs (default: 20)")
    parser.add_argument("--model_save_path", type=str, default="models/best_model_improved.keras", help="Path to save best model")
    args = parser.parse_args()

    print("="*70)
    print("[ARTHROSCAN AI] KNEE OSTEOARTHRITIS MODEL TRAINING PIPELINE")
    print(f"Author: Nikita Tuteja | Target: {args.model_save_path}")
    print("="*70)

    # 1. Dataset Indexing
    print("\n[Step 1/5] Scanning and indexing dataset directories...")
    df = collect_dataset_records()
    if len(df) == 0:
        raise FileNotFoundError("No dataset images found in full code/Knee X-ray Images/")
    print(f"Found {len(df)} total X-ray images across 5 severity classes.")
    print("Class Distribution:")
    print(df['class_name'].value_counts())

    # 2. Stratified Data Split
    print("\n[Step 2/5] Creating Stratified Train (70%), Validation (15%), Test (15%) splits...")
    train_df, val_df, test_df = prepare_data_splits(df)

    # 3. Class Weights & Generators
    print("\n[Step 3/5] Setting up CLAHE Data Augmentation & Balanced Class Weights...")
    class_weights = compute_balanced_class_weights(train_df)
    print(f"Computed Class Weights: {class_weights}")

    train_gen, val_gen, test_gen = get_data_generators(
        train_df, val_df, test_df,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )

    # 4. Model Construction
    print("\n[Step 4/5] Building DenseNet-121 Architecture...")
    model, base_model = build_densenet_model(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=5
    )
    print(f"Model successfully constructed with {model.count_params():,} total parameters.")

    # 5. Two-Stage Training
    print("\n[Step 5/5] Launching Two-Stage Training...")
    trained_model, h_warmup, h_finetune = train_model_two_phase(
        model=model,
        base_model=base_model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights,
        model_save_path=args.model_save_path,
        reports_dir="reports",
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        unfreeze_layers=40
    )

    print("\n" + "="*70)
    print("[SUCCESS] Model Training concluded!")
    print(f"Checkpoint saved at: {args.model_save_path}")
    print("Next step: Run 'python evaluate.py' to generate academic reports & test metrics!")
    print("="*70)

if __name__ == "__main__":
    main()

