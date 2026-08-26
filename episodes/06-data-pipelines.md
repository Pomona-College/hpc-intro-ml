---
title: "Data Pipelines for HPC"
teaching: 25
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you process datasets larger than RAM?
- How do you build efficient data loading pipelines for GPU training?
- How do you run preprocessing at scale with SLURM?
- When should preprocessing run on CPU nodes vs GPU nodes?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Use Dask for parallel preprocessing of large datasets
- Build PyTorch DataLoader pipelines that keep the GPU fed
- Write SLURM jobs for data preprocessing at scale
- Diagnose I/O bottlenecks during training
::::::::::::::::::::::::::::::::::::::::::::::::

## Why Data Pipelines Matter on HPC

A modern A100 80GB GPU on Sagehen HPC can process roughly 5,000 to 20,000 training images per second once data sits in GPU memory. Reading those same images from `/bigdata` (BeeGFS, a shared parallel filesystem, ~200 MB/s for this kind of many-small-file access) into RAM, decoding them, and copying them to the device can take ten times longer than the forward and backward pass. The result is a $20,000 GPU sitting at 5% utilization while you wait for disk.

A good data pipeline does three things at once. It moves data from persistent storage onto fast scratch storage. It transforms raw inputs into model-ready tensors using as many CPU cores as the node provides. And it overlaps that transformation with GPU compute so the next batch is ready by the time the current batch finishes. When the pipeline is set up correctly, GPU utilization climbs past 90%. When it is not, your job spends most of its allocated time waiting on I/O.

This episode covers the two halves of that pipeline separately. First, large-scale preprocessing with Dask, run as a CPU-only batch job before training. Second, the per-batch loading machinery that PyTorch and TensorFlow use during training itself.

## Large-Scale Preprocessing with Dask

When the raw dataset is larger than a single node's 512 GB of RAM, or when the cleaning step itself is too slow on one core, Dask gives you parallel pandas-like operations across many cores or many nodes. On a single Sagehen `amd` node Dask can use all 128 cores. The pattern is almost always the same: read lazily, chain transformations, and only call `.compute()` or `.to_parquet()` when you actually need the output.

```bash
module load miniconda3
conda activate pytorch_env   # any env works -- add dask with: conda install dask
```

```python
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

with LocalCluster(n_workers=8, threads_per_worker=4, memory_limit='8GB') as cluster:
    with Client(cluster) as client:
        # Load with Dask (lazy evaluation)
        ddf = dd.read_csv('/bigdata/lab/<labname>/datasets/large_data_*.csv')

        # Preprocessing operations (lazy, not computed yet)
        ddf = ddf[ddf['value'] > 0]
        ddf['scaled'] = (ddf['value'] - ddf['value'].mean()) / ddf['value'].std()

        # Compute only when needed
        result = ddf.compute()
```

Two design choices in that snippet are worth understanding. The `n_workers` x `threads_per_worker` product is your effective parallelism. With 8 workers x 4 threads = 32 cores you leave headroom on a 128-core node, which is sensible if your data fits in 8 x 8 GB = 64 GB of memory. Bumping workers to 32 with 4 threads each saturates the node but pushes memory pressure toward 256 GB. Watch the Dask dashboard or `seff` output to find the right balance.

The `memory_limit='8GB'` is per-worker. Workers spill to disk when they exceed this limit, which slows the job dramatically. If you see a lot of spilling in the dashboard, either increase per-worker memory or process fewer partitions at a time.

::::::::::::::::::::::::::::::::::::: callout
**When to choose Dask over pandas**

Use plain pandas when the dataset fits comfortably in node RAM (less than ~50 GB after loading). Pandas is faster than Dask for in-memory work because it skips the scheduling overhead.

Use Dask when:

- The raw dataset is larger than 50 GB and you need the whole thing in one transform.
- Preprocessing is CPU-bound and parallelizable across rows or files.
- You want one script that works on a 1 GB sample on your laptop and a 1 TB dataset on Sagehen.

Avoid Dask for graph-shaped or sequence-shaped data where rows are not independent. The overhead is rarely worth it.
::::::::::::::::::::::::::::::::::::::::::::::::

## Building Efficient Data Pipelines for Training

During training, data flows through three stages: storage to RAM (I/O), RAM to tensors (CPU), and RAM to GPU (PCIe transfer). Each stage runs on different hardware and each can become the bottleneck. The PyTorch `DataLoader` is designed to overlap all three.

### PyTorch DataLoader Pipeline

```python
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class SensorDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = torch.tensor([sample['temp'], sample['humidity']],
                               dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        if self.transform:
            features = self.transform(features)
        return features, label

# Create dataset and dataloader
dataset = SensorDataset('/scratch/$USER/sensor_data_clean.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True)
```

The `num_workers` argument forks subprocesses that prepare the next batches while the GPU works on the current one. With `num_workers=0` (the default), data loading is synchronous and your GPU will idle between batches. Setting it to 4 to 8 is a good starting point on Sagehen GPU nodes; going higher only helps when your `__getitem__` does heavy work like image decoding or tokenization.

`pin_memory=True` allocates the staging buffers in page-locked host memory so the CPU-to-GPU copy can use direct memory access without an extra copy through pageable memory. The speedup is usually 10 to 30 percent on PCIe transfers and costs a small amount of RAM.

`persistent_workers=True` keeps the worker processes alive between epochs. Without it, PyTorch tears down and recreates workers every epoch, which can cost 5 to 30 seconds depending on your `__init__`.

::::::::::::::::::::::::::::::::::::: callout
**Diagnosing data pipeline bottlenecks**

If `nvidia-smi` shows your GPU oscillating between 0% and 100% utilization, your pipeline is starving the GPU. Try:

1. Increase `num_workers` first.
2. Move the dataset to `/scratch` if it lives on `/bigdata` or `/rhome`.
3. Pre-decode images and serialize as torch tensors or NumPy memory-mapped arrays.
4. Check that `__getitem__` does not call expensive Python libraries that hold the GIL.
5. Profile with `torch.profiler` to see where time is actually spent.

If GPU utilization is steady at 100%, your pipeline is fine and the model is the bottleneck.
::::::::::::::::::::::::::::::::::::::::::::::::

## Where Data Should Live During a Job

Sagehen has three storage tiers and the right tier depends on the job phase:

| Stage | Storage | Why |
|-------|---------|-----|
| Original raw data, indefinite | `/bigdata/lab/<labname>/` | Persistent, 1 TB per lab, readable from every node |
| Anything a later job needs | `/bigdata/lab/<labname>/` | The only tier that survives the end of a job |
| Active training data, this job only | `/scratch` | Node-local NVMe SSD, ~1 GB/s, no backups |
| Job scripts, configs | `/rhome` | Backed up, small (100 GB) |
| Model checkpoints | Write to `/scratch`, copy the ones you want to keep to `/bigdata` **before the job ends** | Avoids hammering /bigdata with intermediate writes, without losing the final result |

::::::::::::::::::::::::::::::::::::: callout
**`/scratch` does not outlive the job**

`/scratch` is local to the compute node your job is running on, and it is cleared when
that job ends. Two consequences that catch people out:

- Another job cannot read your `/scratch` — even a job that starts seconds later, because
  SLURM may place it on a different node.
- Anything still in `/scratch` when the job finishes is gone. There is no grace period.

So `/scratch` is working space *within* a single job. Every artifact you want afterwards —
a preprocessed dataset, a final checkpoint, a metrics file — has to be copied to
`/bigdata` before the job exits.
::::::::::::::::::::::::::::::::::::::::::::::::

A common mistake is to point `DataLoader` directly at `/bigdata`. This works but wastes throughput, especially with many workers all reading concurrently from the same BeeGFS mount. Stage data to `/scratch` at job start using `rsync` or `cp -r`. The first few minutes feel like overhead but they pay back ten times during training.

## SLURM Job for Preprocessing at Scale

```bash
#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=preprocess_%j.log

module load miniconda3
conda activate ml_env

python3 << 'EOF'
import dask.dataframe as dd
import os
from dask.distributed import Client, LocalCluster

with LocalCluster(n_workers=8, threads_per_worker=4, memory_limit='8GB') as cluster:
    with Client(cluster) as client:
        ddf = dd.read_parquet('/bigdata/lab/<labname>/raw_data/')
        ddf = ddf[ddf['value'] > 0]
        ddf['scaled'] = (ddf['value'] - ddf['value'].mean()) / ddf['value'].std()
        ddf.to_parquet('/bigdata/lab/<labname>/processed_data')

print("Preprocessing complete!")
EOF
```

Note that this job uses `--partition=amd`, not `--partition=gpu`. Preprocessing is CPU-bound and pure Dask code does not touch the GPU. Running it on a GPU node burns expensive GPU-hours for nothing.

The standard pattern is one CPU job to preprocess, then a second GPU job to train, chained with `--dependency=afterok:JOBID` so the training job waits:

```bash
JOBID=$(sbatch --parsable preprocess.sh)
sbatch --dependency=afterok:$JOBID train.sh
```

The handoff between them has to go through `/bigdata`, which is why the job above writes its Parquet output there rather than to `/scratch`. The training job then stages that dataset onto its *own* node's `/scratch` as its first step:

```bash
# first lines of train.sh, before training starts
mkdir -p /scratch/$USER/$SLURM_JOB_ID
rsync -a /bigdata/lab/<labname>/processed_data/ /scratch/$USER/$SLURM_JOB_ID/data/
```

That gives you both things at once: the fast node-local reads during training, and a preprocessed dataset that still exists when the second job starts. Writing the handoff to `/scratch` instead is the classic version of this bug — the preprocessing job succeeds, the training job starts on a different node, and it fails with `FileNotFoundError` on a path that very definitely existed a minute ago.

::::::::::::::::::::::::::::::::::::: callout
**Output formats: CSV, Parquet, or memory-mapped arrays?**

CSV is human-readable but ~5x slower to parse than Parquet and uses ~3x more disk. Use Parquet for any preprocessed dataset bigger than a few hundred megabytes. Parquet keeps types, supports column-wise reads, and compresses well.

For ML training where you read the same data many epochs, consider serializing the final tensor representation to a `.npy` (NumPy memory-mapped) or `.pt` (PyTorch) file. Skipping the parse step every epoch can shave minutes off each epoch on large datasets.
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Create a Scalable Data Pipeline**

Build a SLURM job that loads 10 GB of raw data from `/bigdata`, preprocesses using Dask (4 workers), and saves processed chunks in Parquet format for a later training job to use. It should run in less than 30 minutes. Think about where the output has to go for that later job to find it.

::::::::::::::::::::::::::::::::::::: solution

```bash
#!/bin/bash
#SBATCH --job-name=dask_preprocess
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=preprocess_%j.log

module load miniconda3
conda activate ml_env

python3 << 'EOF'
import dask.dataframe as dd
import os
from dask.distributed import Client, LocalCluster

with LocalCluster(n_workers=4, threads_per_worker=4, memory_limit='4GB') as cluster:
    with Client(cluster) as client:
        ddf = dd.read_csv('/bigdata/lab/<labname>/raw_data/*.csv')
        ddf = ddf.dropna()
        numeric_cols = [c for c in ddf.columns if ddf[c].dtype in ['float64', 'int64']]
        for col in numeric_cols:
            ddf[col] = (ddf[col] - ddf[col].mean()) / ddf[col].std()
        ddf.to_parquet('/bigdata/lab/<labname>/processed_data')
EOF
```

The 16 `cpus-per-task` matches 4 workers x 4 threads. Time limit of 30 minutes is conservative; 10 GB of CSV typically processes in 5 to 10 minutes on this configuration.

The output goes to `/bigdata`, not `/scratch`, because a *later* job has to read it — and `/scratch` will not exist by then. If the lab quota is tight and the intermediate stages are large, you can do the intermediate shuffling on `/scratch` and copy only the final compressed Parquet to `/bigdata` at the end of the job, but that copy has to happen before the job exits.

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Profile a Slow DataLoader**

You are training a vision model on Sagehen. `nvidia-smi` shows the GPU at 30% utilization. Your `DataLoader` reads JPEGs from `/bigdata`, decodes with PIL, and applies torchvision transforms. List three changes you would try, in order, and predict the effect of each.

::::::::::::::::::::::::::::::::::::: solution

1. Stage the JPEG directory to `/scratch` at job start. Expected: GPU utilization climbs to 60 to 80% if I/O was the bottleneck. Cost: a few minutes of `rsync` at job start.

2. Increase `num_workers` from default to 8 (matching half of `--cpus-per-task=16`). Expected: GPU climbs further if PIL decoding was the bottleneck. Cost: 8 worker processes use proportionally more RAM.

3. Pre-decode the JPEGs once into a torch tensor file or use `torchvision.io.read_image` (faster than PIL). Expected: largest gain if decoding dominates; worth profiling first to confirm.

The order matters. Cheap changes (storage, workers) come before expensive refactors (pre-decoding the dataset).

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- A good data pipeline overlaps storage I/O, CPU transforms, and GPU compute
- Use Dask for datasets larger than RAM that need parallel preprocessing
- Stage active training data on /scratch; keep /bigdata for source-of-truth
- /scratch is node-local and cleared at job end — anything a later job needs must be written to /bigdata before the job exits
- DataLoader settings that matter most: num_workers, pin_memory, persistent_workers
- Run preprocessing as a CPU job on the amd partition, training as a GPU job
- If GPU utilization oscillates, you have a data pipeline problem, not a model problem
::::::::::::::::::::::::::::::::::::::::::::::::

<!-- highlight <labname>/<myusername> placeholders in code blocks; remove if the varnish theme handles this natively -->
<script>(function(){var CSS='.sh-placeholder{color:#c2410c;font-weight:700}[data-bs-theme="dark"] .sh-placeholder,html.dark .sh-placeholder{color:#fdba74}@media (prefers-color-scheme: dark){[data-bs-theme="auto"] .sh-placeholder{color:#fdba74}}';var RX=/<labname>|<myusername>/g;function firstMatch(el){var w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null),nodes=[],full='';while(w.nextNode()){nodes.push({n:w.currentNode,s:full.length});full+=w.currentNode.nodeValue;}RX.lastIndex=0;var m;while((m=RX.exec(full))){var s=m.index,e=s+m[0].length,inSpan=false,parts=[];for(var j=0;j<nodes.length;j++){var ns=nodes[j].s,ne=ns+nodes[j].n.nodeValue.length;if(ne<=s||ns>=e)continue;parts.push({node:nodes[j].n,a:Math.max(s-ns,0),b:Math.min(e-ns,nodes[j].n.nodeValue.length)});var p=nodes[j].n.parentNode;while(p&&p!==el){if(p.classList&&p.classList.contains('sh-placeholder')){inSpan=true;break;}p=p.parentNode;}}if(!inSpan&&parts.length)return parts;}return null;}function wrapParts(parts){for(var i=parts.length-1;i>=0;i--){var t=parts[i].node,txt=t.nodeValue,a=parts[i].a,b=parts[i].b;var span=document.createElement('span');span.className='sh-placeholder';span.textContent=txt.slice(a,b);var f=document.createDocumentFragment();if(a>0)f.appendChild(document.createTextNode(txt.slice(0,a)));f.appendChild(span);if(b<txt.length)f.appendChild(document.createTextNode(txt.slice(b)));t.parentNode.replaceChild(f,t);}}function run(){var st=document.createElement('style');st.textContent=CSS;document.head.appendChild(st);document.querySelectorAll('pre,code').forEach(function(el){var guard=0,parts;while((parts=firstMatch(el))&&guard++<500){wrapParts(parts);}});}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',run);}else{run();}})();</script>
