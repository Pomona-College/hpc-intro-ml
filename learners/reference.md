# Reference: Introduction to Machine Learning on HPC

## Sagehen HPC Cluster Quick Reference

### Login
```bash
ssh your_netid@sagehen.hpc.pomona.edu
```

### Module Commands
```bash
module avail              # List available modules
module load miniconda3   # PyTorch comes from conda: conda activate pytorch_env   # Load specific module
module list               # Show loaded modules
module unload pytorch     # Unload module
```

### Storage Paths
```bash
/rhome/$USER              # Home directory (100 GB, backed up, slower)
/bigdata/$USER            # Group storage (multi-TB, moderate speed)
/scratch/$USER            # Node-local SSD (in-job only, deleted at job end)
```

### SLURM Job Commands
```bash
# Submit job
sbatch script.sh

# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# Get detailed job info
scontrol show job <job_id>

# Check partition availability
sinfo
```

## GPU Job Submission Templates

### Single GPU PyTorch Job
```bash
#!/bin/bash
#SBATCH --job-name=pytorch_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00

module load miniconda3   # PyTorch comes from conda: conda activate pytorch_env
python3 train.py
```

### Multi-GPU PyTorch Job
```bash
#!/bin/bash
#SBATCH --job-name=pytorch_multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00

module load miniconda3   # PyTorch comes from conda: conda activate pytorch_env
torchrun --nproc_per_node=4 train.py
```

### Single GPU TensorFlow Job
```bash
#!/bin/bash
#SBATCH --job-name=tf_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

module load miniconda3
conda activate tf_env   # no tensorflow module -- use your conda env
python3 train.py
```

## Python Code Snippets

### Check GPU Availability
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

### Move Model to GPU
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

### Monitor GPU Memory (PyTorch)
```python
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
```

### Save Checkpoint
```python
checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pt')
```

### Load Checkpoint
```python
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
```

## Data Preprocessing Patterns

### Normalization (PyTorch)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### PyTorch DataLoader
```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    # training code
```

### TensorFlow Data Pipeline
```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

model.fit(dataset, epochs=10)
```

## Common PyTorch Patterns

### Training Loop
```python
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
```

### Validation
```python
model.eval()
total_loss = 0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        total_loss += loss.item()
avg_loss = total_loss / len(val_loader)
```

## Common TensorFlow Patterns

### Sequential Model
```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
```

### Custom Training Loop
```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss
```

## Debugging Commands

### Check GPU Usage in Real-time
```bash
# Login to compute node and monitor
watch -n 1 nvidia-smi
```

### Check Job Details
```bash
# While job is running
scontrol show job <job_id>

# View job log
tail -f slurm-<job_id>.out
```

### Profile Python Code
```python
import cProfile
cProfile.run('your_function()', sort='cumulative')
```

## Hyperparameter Tuning Reference

### Grid Search Parameters
```python
param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64],
    'dropout': [0.2, 0.3, 0.4]
}
```

### Typical Learning Rates by Task
```
Text/NLP:       0.0001 - 0.001
Vision:         0.001 - 0.01
Tabular:        0.01 - 0.1
Transfer Learn: 1e-5 - 1e-3 (small rate for fine-tuning)
```

## Useful External Resources

### Official Documentation
- **PyTorch**: https://pytorch.org/docs
- **TensorFlow**: https://tensorflow.org/guide
- **SLURM**: https://slurm.schedmd.com/quickstart.html
- **Sagehen**: [Introduction to HPC Systems](https://pomona-college.github.io/hpc-intro/) · [SLURM Job Scheduling](https://pomona-college.github.io/hpc-slurm-scheduling/) · [GPU Computing](https://pomona-college.github.io/hpc-gpu-computing/)
- **Pomona ITS**: <https://www.pomona.edu/its/> · <its-hpc@pomona.edu>

### Tutorials
- PyTorch Tutorials: https://pytorch.org/tutorials
- TensorFlow Tutorials: https://tensorflow.org/tutorials
- Hugging Face Course: https://huggingface.co/course

### Papers and Benchmarks
- ArXiv ML Papers: https://arxiv.org/list/cs.LG
- Papers with Code: https://paperswithcode.com
- ML Benchmarks: https://www.paperswithcode.com/benchmarks

## Contact and Support
- **HPC Questions**: its-hpc@pomona.edu
- **Technical Issues**: File ticket at [ticket system]
- **Dataset Advice**: Research librarians available