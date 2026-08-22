import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.utils import class_weight

def build_densenet_model(input_shape=(256, 256, 3), num_classes=5, dropout_rate=0.4):
    """
    Builds a Transfer Learning architecture using DenseNet-121 backbone
    with custom classification head optimized for medical X-ray feature extraction.
    """
    inputs = Input(shape=input_shape)
    
    # Pretrained DenseNet-121 Feature Extractor
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    base_model.trainable = False  # Initially freeze for warmup

    # Custom Radiological Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_head_1')(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4), name='dense_256')(x)
    x = Dropout(dropout_rate, name='dropout_1')(x)
    x = BatchNormalization(name='bn_head_2')(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name='dense_128')(x)
    x = Dropout(dropout_rate * 0.75, name='dropout_2')(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ArthroScan_DenseNet121')
    return model, base_model

def compute_balanced_class_weights(train_df):
    """
    Calculates balanced class weights to compensate for class imbalance across severity grades.
    """
    y_train = train_df['label'].values
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    return dict(enumerate(weights))

def train_model_two_phase(
    model,
    base_model,
    train_gen,
    val_gen,
    class_weights,
    model_save_path='models/best_model_improved.keras',
    reports_dir='reports',
    warmup_epochs=5,
    finetune_epochs=20,
    unfreeze_layers=40
):
    """
    Executes a two-phase transfer learning routine:
      - Phase 1: Warmup training of top classification head
      - Phase 2: Deep fine-tuning of top convolutional blocks
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    csv_logger = CSVLogger(os.path.join(reports_dir, 'training_log.csv'), append=False)
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # -------------------------------------------------------------
    # PHASE 1: WARMUP TRAINING (Head Only)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("[PHASE 1] WARMUP TRAINING (Classification Head)")
    print("="*60)
    base_model.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    history_warmup = model.fit(
        train_gen,
        epochs=warmup_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr, csv_logger],
        verbose=1
    )

    # -------------------------------------------------------------
    # PHASE 2: DEEP FINE-TUNING (Unfreeze top blocks)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print(f"[PHASE 2] FINE-TUNING (Unfreezing top {unfreeze_layers} layers of DenseNet-121)")
    print("="*60)
    base_model.trainable = True
    
    # Freeze all layers except the last unfreeze_layers
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        # Keep batch norm layers in inference mode for stability
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Recompile with smaller learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    csv_logger_finetune = CSVLogger(os.path.join(reports_dir, 'training_log.csv'), append=True)
    history_finetune = model.fit(
        train_gen,
        epochs=warmup_epochs + finetune_epochs,
        initial_epoch=warmup_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger_finetune],
        verbose=1
    )

    print("\n" + "="*60)
    print(f"[TRAINING COMPLETE] Best model saved to: {model_save_path}")
    print("="*60)
    return model, history_warmup, history_finetune

