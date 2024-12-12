import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV3Small

DATASET_NAME = "organmnist3d"
BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (180, 96, 96, 3)
INPUT_SHAPE_2D = (96, 96, 3)
MAX_FRAMES = 180
NUM_CLASSES = 2

# OPTIMIZER
LEARNING_RATE = 1e-4

# TRAINING
EPOCHS = 30
DATA_NUM = 500

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 12

# MOBILENETV3
MOBILENET_NUM_LAYER = 40

def load_video(path, max_frames=MAX_FRAMES, resize=(96, 96), dtype=np.float16):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame.astype(dtype))
            frame_count += 1
    finally:
        cap.release()

    while len(frames) < max_frames:
        frames.append(frames[-1].astype(dtype) if frames else np.zeros(resize + (3,), dtype=dtype))

    return np.array(frames)

def prepare_dataset(folder_path):
    class_names = ["NonViolence", "Violence"]
    x, y = [], []
    
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        video_cnt = 0
        print(f"Processing {class_name} videos...")
        
        for video_file in tqdm(os.listdir(class_folder)):
            if video_cnt >= DATA_NUM:
                break
            video_path = os.path.join(class_folder, video_file)
            frames = load_video(video_path)
            x.append(frames)
            y.append(class_index)
            video_cnt += 1

    return np.array(x), keras.utils.to_categorical(y, num_classes=len(class_names))

class MobileNetV3FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, num_frames, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames

        self.mobilenetv3 = MobileNetV3Small(
            input_shape=INPUT_SHAPE_2D, 
            include_top=False, 
            weights='imagenet'
        )
        self.reduced_mobilenetv3_model = tf.keras.Model(
            inputs=self.mobilenetv3.input,
            outputs=self.mobilenetv3.layers[MOBILENET_NUM_LAYER].output
        )

        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(PROJECTION_DIM)

    def call(self, inputs):
        frames = tf.split(inputs, num_or_size_splits=self.num_frames, axis=1)
        frame_features = [
            self.dense(self.pooling(self.reduced_mobilenetv3_model(tf.squeeze(frame, axis=1))))
            for frame in frames
        ]
        features = tf.stack(frame_features, axis=1)
        return features

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
    
def create_vivit_model(
    feature_extractor,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES
):
    inputs = layers.Input(shape=input_shape)
    features = feature_extractor(inputs)
    encoded_patches = positional_encoder(features)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=layer_norm_eps)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    representation = layers.Dropout(0.1)(representation)

    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    return keras.Model(inputs=inputs, outputs=outputs)

print("Loading dataset...")
dataset_path = "Real Life Violence Dataset"
x, y = prepare_dataset(dataset_path)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=123)

print("Creating model...")
model = create_vivit_model(
    feature_extractor=MobileNetV3FeatureExtractor(num_frames=MAX_FRAMES),
    positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="loss", patience=10, restore_best_weights=True
)

reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.6, patience=5,
    min_lr=0.00005, verbose=1
)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("Training model...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping_callback, reduce_lr_callback]
)

print("Evaluating model...")
ts_loss, ts_precision, ts_recall = model.evaluate(test_dataset)
print(f"Test Loss: {ts_loss}")
print(f"Test Precision: {ts_precision}")
print(f"Test Recall: {ts_recall}")

f1_score = 2 * (ts_precision * ts_recall) / (ts_precision + ts_recall)
print(f"F1 Score: {f1_score}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['precision'])
plt.plot(history.history['recall'])
plt.title('Model Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(['Precision', 'Recall'])
plt.tight_layout()
plt.show()