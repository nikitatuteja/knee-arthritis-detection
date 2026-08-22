import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd

# Paths
dataset_path = 'full code/Knee X-ray Images/MedicalExpert-I/MedicalExpert-I/'
model_save_path = 'models/best_model_improved.keras'

# Hyperparameters - Optimized for SPEED
IMG_SIZE = 256
BATCH_SIZE = 32
INITIAL_EPOCHS = 5  # Quick warm-up
FINE_TUNE_EPOCHS = 15 # Fast fine-tune
CLASS_NAMES = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']

# Prepare data
data = []
for class_name in CLASS_NAMES:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({'filename': os.path.join(class_name, img_name), 'class': class_name})

df = pd.DataFrame(data)
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

# Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, directory=dataset_path, x_col='filename', y_col='class',
    classes=CLASS_NAMES, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, directory=dataset_path, x_col='filename', y_col='class',
    classes=CLASS_NAMES, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Class weights
y_train = train_gen.classes
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))

# Model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False 
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Callbacks
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# FAST TRAINING
print("\n--- FAST TRAINING STARTED ---")
print("Step 1: Quick warm-up...")
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=INITIAL_EPOCHS, validation_data=val_gen, class_weight=cw_dict, callbacks=[checkpoint])

print("\nStep 2: Fast Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]: layer.trainable = False
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=FINE_TUNE_EPOCHS, validation_data=val_gen, class_weight=cw_dict, callbacks=[checkpoint, early_stop, reduce_lr])

print(f"\nFAST TRAINING COMPLETE. Model saved to {model_save_path}")
