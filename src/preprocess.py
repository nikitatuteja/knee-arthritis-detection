import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASS_NAMES = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLEAN_CLASS_NAMES = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']

def apply_clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance bone edges and joint space in X-ray images.
    """
    if len(image.shape) == 3:
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image).astype(np.float32)

def collect_dataset_records(base_dirs=None):
    """
    Scans all dataset folders and returns a clean DataFrame with image paths and labels.
    """
    if base_dirs is None:
        base_dirs = [
            'full code/Knee X-ray Images/MedicalExpert-I/MedicalExpert-I',
            'full code/Knee X-ray Images/MedicalExpert-II/MedicalExpert-II'
        ]
    
    records = []
    for bdir in base_dirs:
        if not os.path.exists(bdir):
            continue
        for cname in CLASS_NAMES:
            cdir = os.path.join(bdir, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('.'):
                    full_path = os.path.normpath(os.path.join(cdir, fname))
                    records.append({
                        'filepath': full_path,
                        'class_dir': cname,
                        'label': LABEL_MAP[cname],
                        'class_name': CLEAN_CLASS_NAMES[LABEL_MAP[cname]]
                    })
    
    df = pd.DataFrame(records)
    # Remove any potential duplicates based on filename
    df = df.drop_duplicates(subset=['filepath']).reset_index(drop=True)
    return df

def prepare_data_splits(df, output_dir='data_splits', test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits the dataset into Stratified Train (70%), Validation (15%), and Test (15%) sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First split: Train + Val vs Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: Train vs Val (relative proportion)
    val_rel_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Dataset Split Complete:")
    print(f"  Training Set:   {len(train_df)} samples")
    print(f"  Validation Set: {len(val_df)} samples")
    print(f"  Test Set:       {len(test_df)} samples")
    print(f"  Total Images:   {len(df)} samples")
    
    return train_df, val_df, test_df

def get_data_generators(train_df, val_df, test_df, img_size=(256, 256), batch_size=32):
    """
    Creates high-performance image data generators with CLAHE preprocessing and data augmentation.
    """
    def custom_preprocessing(img):
        # Apply CLAHE then rescale to [0, 1]
        enhanced = apply_clahe(img)
        return enhanced / 255.0

    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )
    
    eval_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='class_name',
        classes=CLEAN_CLASS_NAMES,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = eval_datagen.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='class_name',
        classes=CLEAN_CLASS_NAMES,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = eval_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='class_name',
        classes=CLEAN_CLASS_NAMES,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator
