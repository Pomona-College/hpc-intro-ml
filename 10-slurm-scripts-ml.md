---
title: "SLURM Scripts for ML"
teaching: 10
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you write SLURM scripts for GPU training jobs?
- How do you request GPU resources correctly?
- How do you launch multi-GPU training with SLURM?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Write SLURM batch scripts for GPU training
- Request appropriate GPU, CPU, and memory resources
- Launch distributed training with torchrun and SLURM
::::::::::::::::::::::::::::::::::::::::::::::::

## Single-GPU SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=training_%j.log

module load miniconda3/py313_26.3.2-2   # then activate your conda env -- no pytorch module exists

python3 << 'EOF'
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Your training code here
EOF
```

Submit and monitor:

```bash
sbatch train_job.sh
squeue -u $USER -l
```

![Submitting the training job and confirming with `squeue -l` that it's RUNNING on a GPU node.](fig/10-sbatch-squeue-gpu-training.jpg){alt='Terminal showing sbatch submitting train_job.sh as batch job 38943, followed by squeue long-format output. The pytorch job runs on the gpu partition in RUNNING state, a few seconds into its limit, on one node named gpu001.'}

## TensorFlow SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=tf_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=tf_training_%j.log

module load miniconda3/py313_26.3.2-2   # then activate your conda env -- no tensorflow module exists

python3 << 'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Your TensorFlow training code here
EOF
```

## Multi-GPU SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=training_%j.log

module load miniconda3/py313_26.3.2-2   # then activate your conda env -- no pytorch module exists

# Set master node details for distributed training
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_GPUS_PER_NODE

# Launch with torchrun (automatic launcher)
torchrun --nproc_per_node=4 train_distributed.py
```

## Interactive GPU Sessions

### Command-Line Interactive Session

```bash
# Request interactive GPU session
srun -p gpu --gres=gpu:1 --pty bash

# Now you're on a GPU node
python3 test_pytorch.py
```

### OnDemand Web Interface

Access via https://ondemand.hpc.pomona.edu:

1. Log in with Pomona credentials
2. Click "Jupyter Lab" or "Terminal"
3. Select GPU and duration
4. Upload your code

## Resource Guidelines

| Workload | GPUs | CPUs | Memory | Time |
|----------|------|------|--------|------|
| Quick test | 1 | 4 | 16 GB | 30 min |
| Single-GPU training | 1 | 4 | 16-32 GB | 2-8 hours |
| Multi-GPU training | 4 | 16 | 64 GB | 4-24 hours |
| Hyperparameter sweep | 1 per job | 4 | 16 GB | 2 hours each |

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Write a Training Job Script**

Write a SLURM script that requests 1 GPU, loads your conda environment, copies data from /bigdata to /scratch, runs training, and saves the model back to /bigdata.

::::::::::::::::::::::::::::::::::::: solution

```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=training_%j.log

module load miniconda3
conda activate pytorch_env

# Copy data to fast storage
cp -r /bigdata/lab/<labname>/dataset /scratch/$USER/dataset

# Run training
python3 train.py --data /scratch/$USER/dataset --output /scratch/$USER/model.pt

# Save results back to persistent storage
cp /scratch/$USER/model.pt /bigdata/lab/<labname>/models/
```

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

**Important: Always Test First**

Before submitting a large GPU job:
1. Run on CPU with small dataset (10 iterations)
2. Check GPU memory usage with an interactive session
3. Verify checkpointing works
4. Then scale up to full training

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- Use `--partition=gpu` and `--gres=gpu:N` to request GPU resources
- Request 4 CPUs per GPU as a rule of thumb
- Use torchrun for launching distributed PyTorch training
- Test interactively with `srun` before submitting batch jobs
- Copy data to /scratch for faster I/O during training
::::::::::::::::::::::::::::::::::::::::::::::::

<!-- highlight <labname>/<myusername> placeholders in code blocks; remove if the varnish theme handles this natively -->
<script>(function(){var CSS='.sh-placeholder{color:#c2410c;font-weight:700}[data-bs-theme="dark"] .sh-placeholder,html.dark .sh-placeholder{color:#fdba74}@media (prefers-color-scheme: dark){[data-bs-theme="auto"] .sh-placeholder{color:#fdba74}}';var RX=/<labname>|<myusername>/g;function firstMatch(el){var w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null),nodes=[],full='';while(w.nextNode()){nodes.push({n:w.currentNode,s:full.length});full+=w.currentNode.nodeValue;}RX.lastIndex=0;var m;while((m=RX.exec(full))){var s=m.index,e=s+m[0].length,inSpan=false,parts=[];for(var j=0;j<nodes.length;j++){var ns=nodes[j].s,ne=ns+nodes[j].n.nodeValue.length;if(ne<=s||ns>=e)continue;parts.push({node:nodes[j].n,a:Math.max(s-ns,0),b:Math.min(e-ns,nodes[j].n.nodeValue.length)});var p=nodes[j].n.parentNode;while(p&&p!==el){if(p.classList&&p.classList.contains('sh-placeholder')){inSpan=true;break;}p=p.parentNode;}}if(!inSpan&&parts.length)return parts;}return null;}function wrapParts(parts){for(var i=parts.length-1;i>=0;i--){var t=parts[i].node,txt=t.nodeValue,a=parts[i].a,b=parts[i].b;var span=document.createElement('span');span.className='sh-placeholder';span.textContent=txt.slice(a,b);var f=document.createDocumentFragment();if(a>0)f.appendChild(document.createTextNode(txt.slice(0,a)));f.appendChild(span);if(b<txt.length)f.appendChild(document.createTextNode(txt.slice(b)));t.parentNode.replaceChild(f,t);}}function run(){var st=document.createElement('style');st.textContent=CSS;document.head.appendChild(st);document.querySelectorAll('pre,code').forEach(function(el){var guard=0,parts;while((parts=firstMatch(el))&&guard++<500){wrapParts(parts);}});}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',run);}else{run();}})();</script>
