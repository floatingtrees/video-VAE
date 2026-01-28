import os
# Set backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import keras_cv
import numpy as np

# 1. Load a generic Video Classifier Backbone (e.g., VideoMAE or equivalent)
# KerasCV manages the JAX translation automatically.
# Note: Specific presets change; 'videomae_v2_b_16_k400' is a common naming pattern
# If this preset fails, check keras_cv.models.VideoClassifier.presets()
try:
    model = keras_cv.models.VideoClassifier.from_preset(
        "videomae_v2_b_16_k400", 
        num_classes=400
    )
    # To get just the feature extractor (hidden states equivalent):
    backbone = model.backbone
except Exception:
    # Fallback if preset name is different in your version
    print("Available presets:", keras_cv.models.VideoClassifier.presets())
    raise

# 2. Create Dummy Data (Batch, Frames, H, W, Channels)
# Note: KerasCV usually expects 'channels_last' for JAX
dummy_video = np.random.uniform(0, 1, (1, 16, 224, 224, 3))

# 3. Forward Pass (JAX Native)
features = backbone.predict(dummy_video)

print(f"Output shape: {features.shape}")