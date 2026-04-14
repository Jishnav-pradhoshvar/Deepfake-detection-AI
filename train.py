import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16          # safe for 4 GB RAM
EPOCHS     = 20          # EarlyStopping will halt if no improvement
os.makedirs("model", exist_ok=True)

# ── Data generators ──────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.15,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.85, 1.15],   # deepfakes often have lighting artifacts
    shear_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False          # important: keep order for evaluation
)

# ── CRITICAL: verify label mapping ──────────────────────────────────────────
print("Class indices:", train_data.class_indices)
# Should print: {'fake': 0, 'real': 1}  — if reversed, rename your folders!

# ── Build model ──────────────────────────────────────────────────────────────
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Phase 1: freeze entire base
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=4,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint('model/best_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

# ── Phase 1 training (frozen base) ───────────────────────────────────────────
print("\n=== Phase 1: Training top layers only ===")
history1 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    callbacks=callbacks
)

# ── Phase 2: Fine-tune last 30 layers ────────────────────────────────────────
print("\n=== Phase 2: Fine-tuning last 30 layers ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Lower LR is critical for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Save & plot ───────────────────────────────────────────────────────────────
model.save("model/deepfake_model.h5")
print("Model saved!")

# Plot accuracy
plt.figure(figsize=(12, 4))
all_acc     = history1.history['accuracy']     + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

plt.subplot(1, 2, 1)
plt.plot(all_acc,     label='Train Accuracy')
plt.plot(all_val_acc, label='Val Accuracy')
plt.axvline(x=len(history1.history['accuracy'])-1,
            color='r', linestyle='--', label='Fine-tune start')
plt.title('Accuracy'); plt.legend()

all_loss     = history1.history['loss']     + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.subplot(1, 2, 2)
plt.plot(all_loss,     label='Train Loss')
plt.plot(all_val_loss, label='Val Loss')
plt.axvline(x=len(history1.history['loss'])-1,
            color='r', linestyle='--', label='Fine-tune start')
plt.title('Loss'); plt.legend()

plt.tight_layout()
plt.savefig('model/training_plot.png')
print("Plot saved to model/training_plot.png")