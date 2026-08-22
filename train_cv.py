
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
import pandas as pd

# Paths
dataset_path = 'full code/Knee X-ray Images/MedicalExpert-I/MedicalExpert-I/'
model_save_dir = 'models/cv_folds/'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
K_FOLDS = 3

# Prepare data list
data = []
for class_name in sorted(os.listdir(dataset_path)):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({'filename': os.path.join(class_name, img_name), 'class': class_name})

df = pd.DataFrame(data)

# K-Fold Cross Validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

fold_no = 1
for train_index, val_index in kf.split(df):
    print(f'\n--- Training Fold {fold_no} ---')
    
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=dataset_path,
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=dataset_path,
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Model: Transfer Learning with MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False # Freeze base model
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Class weights to handle imbalance
    y_train = train_gen.classes
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    
    # Callbacks
    checkpoint = ModelCheckpoint(f'{model_save_dir}fold_{fold_no}.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)
    
    # Fine-tuning: Unfreeze top layers of base model after initial training
    print("Initial training of top layers...")
    model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        class_weight=cw_dict,
        callbacks=[checkpoint, reduce_lr]
    )
    
    print("Fine-tuning base model...")
    base_model.trainable = True
    # Only train from layer 100 onwards
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=0.0001), # Lower LR for fine-tuning
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        class_weight=cw_dict,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    fold_no += 1

print("\nK-Fold Cross Validation Complete. Models saved in models/cv_folds/")
