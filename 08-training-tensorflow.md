---
title: "Training with TensorFlow"
teaching: 25
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you set up TensorFlow for GPU training on Sagehen?
- What are the differences between PyTorch and TensorFlow workflows?
- How do you use Keras for rapid model development?
- When should you choose TensorFlow over PyTorch (and vice versa)?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Write TensorFlow training code with Keras
- Use tf.data pipelines for efficient I/O on Sagehen
- Apply transfer learning with pre-trained models
- Monitor training with TensorFlow callbacks
- Decide between PyTorch and TensorFlow for a research project
::::::::::::::::::::::::::::::::::::::::::::::::

## TensorFlow vs PyTorch: Which to Choose

Both frameworks run well on Sagehen and both expose roughly the same capabilities. The choice usually comes down to ecosystem and team rather than raw performance:

```
PyTorch:             Imperative, dominant in research, dynamic graphs by default,
                     simpler debugging, larger university adoption, default for
                     transformers and most cutting-edge model implementations.

TensorFlow/Keras:    Was declarative (TF1) and now hybrid (TF2 eager), strong in
                     production deployment, best-in-class TFLite/TF Serving for
                     edge and serving, integrates cleanly with Google Cloud,
                     Keras API is genuinely friendly for newcomers.
```

Most ML coursework at Pomona and most arXiv papers since 2022 ship in PyTorch. If your collaborators or advisor use TensorFlow, or your downstream pipeline needs TFLite or TF Serving, TensorFlow is the right call. Otherwise default to PyTorch and revisit only if there is a specific reason.

## Setting Up TensorFlow on Sagehen

The cluster has a system module for TensorFlow, but creating your own conda environment is usually safer because version pinning is in your control:

```bash
# TensorFlow comes from a conda environment -- there is no tensorflow module on Sagehen
module load miniconda3
conda create -n tf_env python=3.11
conda activate tf_env
pip install tensorflow[and-cuda]==2.15
```

Confirm the install picked up the GPU before queuing a long job:

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Important: set memory growth to avoid grabbing the entire GPU at startup
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

::::::::::::::::::::::::::::::::::::: callout
**Why memory_growth matters on shared GPUs**

By default TensorFlow tries to allocate nearly all GPU memory at process startup. On a shared cluster this is antisocial: even if your model only needs 4 GB, TF locks 80 GB on an A100 and other tenants OOM. With `set_memory_growth(True)`, TF allocates incrementally as needed.

Set this before any other TF call. If you create an op or variable first, TF has already grabbed the memory and the flag does nothing.
::::::::::::::::::::::::::::::::::::::::::::::::

## Keras High-Level API

Keras is the standard way to write TensorFlow models. The API is sequence-of-layers for simple models and a functional API for branching architectures:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                   validation_data=(x_val, y_val))
```

`compile` configures the optimizer, loss, and metrics. `fit` runs the actual training loop and `history` is a dict-like object with per-epoch losses and metrics that you can plot or save.

The Keras API hides the loop entirely, which is great for prototyping and frustrating when you need fine control. For custom training (custom losses with multiple terms, GAN-style alternating updates, distributed strategies) drop to a `tf.GradientTape` loop:

```python
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function  # compile to a graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
```

The `@tf.function` decorator traces the Python code into a static computation graph the first time it runs. Subsequent calls skip Python and run pure C++ operations, which on Sagehen GPUs is typically 1.5 to 3 times faster than eager execution.

## Efficient Data Loading with tf.data

`tf.data.Dataset` is the equivalent of PyTorch's DataLoader. The same overlapping principles apply: read from storage, transform on CPU, hand to GPU, all in parallel.

```python
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # overlap loading with training

model.fit(dataset, epochs=10)
```

The two arguments named `AUTOTUNE` tell TF to dynamically pick parallelism levels based on observed throughput. This is almost always the right choice on Sagehen. Manual tuning with explicit numbers is only useful when you are debugging a specific bottleneck.

For datasets too big to fit in memory, read directly from disk with `tf.data.TFRecordDataset` or `tf.data.Dataset.from_generator`. Stage the data files on `/scratch` first, exactly as you would for a PyTorch DataLoader.

::::::::::::::::::::::::::::::::::::: callout
**A subtle gotcha: shuffle buffer size**

`shuffle(buffer_size=1000)` only shuffles within a window of 1000 examples. If your dataset is sorted by class (label 0, then label 1, then label 2, ...), this is not enough randomness. For ML on HPC where you load 100k or 1M examples, set buffer_size to at least 10x your batch size and ideally close to your full dataset size.

If memory is tight, shuffle the input file list before reading and rely on per-shard randomness instead of one giant buffer.
::::::::::::::::::::::::::::::::::::::::::::::::

## Transfer Learning with Pre-trained Models

Most ML projects on Sagehen are not training from scratch. Transfer learning starts from a model pre-trained on ImageNet, COCO, or a large text corpus and fine-tunes it on your dataset. This converges 10 to 100 times faster than starting from random weights and typically gives better accuracy with limited data.

```python
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base weights

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=5)

# Stage 2: Fine-tune base model with a tiny learning rate
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy')
model.fit(train_dataset, epochs=10)
```

The two-stage pattern is important. First, freeze the base and train only the new head. The base provides good features and the head learns to map them to your task. Second, unfreeze the base and continue training with a much smaller learning rate (typically 1e-5 instead of 1e-3). Skipping stage one and unfreezing immediately can destroy the pretrained weights with large gradients.

## TensorFlow Callbacks

Callbacks run logic at specific points during training without modifying the loop. Three callbacks are worth knowing for HPC training runs:

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f'/scratch/{os.environ["USER"]}/best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7
)

model.fit(train_dataset, validation_data=val_dataset, epochs=20,
         callbacks=[checkpoint, early_stop, lr_schedule])
```

`ModelCheckpoint` writes the model after every epoch where validation loss improved. Without `save_best_only=True` it writes every epoch, which fills `/scratch` quickly with hundreds of megabytes of redundant snapshots.

`EarlyStopping` halts training when validation loss has not improved for `patience` epochs. On a job with `--time=24:00:00`, early stopping reliably saves you from wasting GPU-hours on a model that has converged.

`ReduceLROnPlateau` halves the learning rate when validation plateaus. This is the "learning rate decay" pattern many tutorials handcode; the callback is cleaner and resilient to checkpoint resumption.

::::::::::::::::::::::::::::::::::::: callout
**Multi-GPU training: when and how**

Sagehen GPU nodes typically host 2 GPUs each. Training across both with `tf.distribute.MirroredStrategy` is straightforward:

```python
strategy = tf.distribute.MirroredStrategy()
print(f'Devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_dataset, epochs=10)
```

Submit with `--gres=gpu:2` (or `--gres=gpu:a100:2` for both A100s on the same node). The strategy replicates the model on each GPU, splits each batch, and averages gradients via NCCL all-reduce.

In practice, two-GPU training gives 1.6 to 1.9x speedup, not 2x, because of synchronization overhead and uneven batch boundaries. It is rarely worth requesting two GPUs for a model that already fits and trains in reasonable time on one.
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Transfer Learning on CIFAR-10**

Using TensorFlow/Keras, load CIFAR-10 with tf.data, use a pre-trained ResNet50, and train for 5 epochs with GPU memory growth enabled.

::::::::::::::::::::::::::::::::::::: solution

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
import os

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

def preprocess(x, y):
    return tf.image.resize(x, [224, 224]) / 255.0, y

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model, layers.GlobalAveragePooling2D(), layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=5)
```

Notice the resize from 32x32 to 224x224 in the preprocessing function. ResNet50 expects ImageNet-sized inputs. Skipping this resize will fail with a shape mismatch on the first batch.

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Pick a Framework for Three Projects**

For each scenario, recommend PyTorch or TensorFlow and explain why in one sentence:

A. A first-year graduate student fine-tuning BERT for sentiment classification on a corpus of student course evaluations.

B. A senior thesis project deploying a trained classifier to an embedded device that runs at the edge of a sensor network.

C. A computational biology lab implementing a recently-published transformer variant whose code only ships in PyTorch.

::::::::::::::::::::::::::::::::::::: solution

A. PyTorch. The HuggingFace Transformers library is PyTorch-first; nearly all BERT fine-tuning tutorials and pretrained checkpoints are easier to use in PyTorch.

B. TensorFlow. TFLite is the standard path for converting trained models to run on edge hardware (Raspberry Pi, microcontrollers, mobile). The TF ecosystem for deployment is meaningfully better than PyTorch's.

C. PyTorch. Re-implementing a paper's code in another framework is rarely worth the time, and you risk subtle bugs that affect reproducibility.

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- TensorFlow with Keras provides a high-level API for fast prototyping; tf.GradientTape gives full control
- Always set memory_growth=True on shared GPUs; otherwise TF grabs the whole device
- tf.data with AUTOTUNE is the equivalent of PyTorch DataLoader for parallel I/O
- Transfer learning is the default starting point: train the head first, then fine-tune
- Callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau) save GPU-hours on long runs
- MirroredStrategy enables 2-GPU training but the speedup is sub-linear
- Choose framework based on ecosystem fit, not micro-benchmarks
::::::::::::::::::::::::::::::::::::::::::::::::
