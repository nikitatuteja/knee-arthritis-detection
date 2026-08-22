import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import io
import base64
from PIL import Image

def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given input image and model.
    Compatible with Keras 3 and TensorFlow 2.x.
    """
    # Find last convolutional layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            last_conv_layer_name = 'conv5_block16_2_conv'

    # Retrieve target layer
    target_layer = None
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        for layer in model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    break
                except ValueError:
                    continue

    if target_layer is None:
        raise ValueError(f"Could not locate convolutional layer: {last_conv_layer_name}")

    # Create gradient model
    grad_model = Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        res = grad_model(img_tensor)
        conv_out = res[0]
        preds = res[1]

        # Handle potential nested lists from Keras 3
        if isinstance(conv_out, (list, tuple)):
            conv_out = conv_out[0]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        tape.watch(conv_out)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    # Gradients of target class w.r.t convolutional feature map
    grads = tape.gradient(class_channel, conv_out)
    if grads is None:
        # Fallback in case of detached graph
        grads = tf.ones_like(conv_out)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map channels by importance
    conv_features = conv_out[0]
    heatmap = conv_features @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep positive contributions & normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def generate_superimposed_gradcam(original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Superimposes the Grad-CAM heatmap over the original X-ray image.
    
    Args:
        original_image: numpy array (H, W, 3) or (H, W) in range [0, 255] or [0, 1]
        heatmap: 2D numpy array in range [0, 1]
        alpha: transparency weighting
    Returns:
        superimposed_img: RGB uint8 image (H, W, 3)
    """
    img_copy = np.copy(original_image)
    if img_copy.max() <= 1.0:
        img_copy = (img_copy * 255).astype(np.uint8)
    else:
        img_copy = img_copy.astype(np.uint8)

    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    elif img_copy.shape[2] == 4:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGBA2RGB)

    h, w = img_copy.shape[:2]
    
    # Rescale heatmap to 0-255 and resize to image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap (Jet: Blue = low attention, Red/Yellow = high attention on joint space)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed_img = heatmap_colored * alpha + img_copy * (1.0 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def encode_image_to_base64(img_array):
    """
    Converts RGB numpy array into a base64 JPEG string for API consumption.
    """
    pil_img = Image.fromarray(img_array)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=92)
    return base64.b64encode(buff.getvalue()).decode("utf-8")
