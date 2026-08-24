# Instructor Notes: Introduction to Machine Learning on HPC

## Workshop Overview

**Duration**: 2 days (approximately 10 hours contact time)
**Format**: Carpentries-style workshop with live coding demonstrations
**Target Audience**: Pomona researchers with basic Python knowledge, little to no HPC/GPU experience

**Learning Outcomes**: By the end, learners can:
- Submit GPU jobs to Sagehen HPC system
- Write PyTorch and TensorFlow training code for GPUs
- Optimize data loading and preprocessing at scale
- Use multiple GPUs for distributed training
- Implement hyperparameter search for ML models

## Episode-by-Episode Teaching Notes

### Episode 1: Why Machine Learning on HPC?
**Time**: 45 min teaching + 45 min exercises
**Key Points**:
- HPC is NOT just for ML, but ML is a great use case
- Speedup/efficiency metrics are critical
- Amdahl's law explains why parallelism has limits

**Live Coding**:
- Demo speedup calculation for their own laptops
- Show typical training time reduction (100h → 10h with 4 GPUs)

**Common Misconceptions**:
- "More GPUs = more speed" (false: communication overhead)
- "My laptop doesn't need HPC" (depends on dataset size and time)

**Exercises**:
- Have them calculate speedup for sample problems
- Discuss whether parallelism is worthwhile for their research

**Troubleshooting**:
- Some may have never done manual speedup calculations; walk through formulas

---

### Episode 2: Environment Setup for ML
**Time**: 20 min teaching + 15 min exercises
**Key Points**:
- Two paths: pre-built modules vs. conda environments
- Reproducibility matters: pin versions
- Testing before big jobs saves time

**Live Coding**:
- Create a conda environment live
- Install PyTorch with GPU support
- Test with simple matrix multiplication

**Common Issues**:
- CUDA version mismatches (module load first!)
- Conda slow downloads (normal, be patient)
- Import errors (verify torch.cuda.is_available())

**Exercises**:
- Have everyone create their own environment
- Run the verification script

**Troubleshooting**:
- Slow conda? Use `mamba` (faster)
- CUDA errors? Check: did they `module load miniconda3` and `conda activate` their environment first? (There is no pytorch module.)
- GPU not detected? Try: `python -c "import torch; print(torch.cuda.is_available())"`

---

### Episode 3: Data Preparation and Preprocessing
**Time**: 20 min teaching + 15 min exercises
**Key Points**:
- I/O bottlenecks are real (GPUs idle while waiting for data)
- Preprocessing should be mostly in /scratch (not /bigdata)
- Always separate train/val/test for hyperparameter selection

**Live Coding**:
- Load a CSV with pandas
- Show memory usage patterns
- Implement standard scaling
- Show Dask for large data

**Common Mistakes**:
- Computing normalization statistics on full dataset (data leakage!)
- Not actually timing I/O (they don't see the bottleneck until they profile)
- Forgetting to save preprocessing objects (scaler)

**Exercises**:
- Have them preprocess a real dataset
- Challenge: Load with both pandas and Dask, compare performance

**Troubleshooting**:
- Memory errors? Reduce batch size, use Dask
- Slow loading? Check storage tier (using /bigdata? Move to /scratch)

---

### Episode 4: GPU Training with PyTorch
**Time**: 20 min teaching + 15 min exercises
**Key Points**:
- GPU debugging is different (backward() errors can be cryptic)
- Mixed precision training is practical (2x faster, same accuracy)
- GPU memory grows during backward pass (save during forward)

**Live Coding**:
- Simple model: dense → ReLU → output
- Move to GPU: model.to(device)
- Training loop with optimizer.zero_grad()
- GPU memory monitoring

**Common Issues**:
- "CUDA out of memory": Reduce batch size, use mixed precision, gradient accumulation
- "Loss is NaN": Learning rate too high, bad initialization
- "Backward slower than forward": Normal, backward is ~2x cost

**Exercises**:
- Train on small dataset (MNIST/CIFAR-10 first 100 samples)
- Measure speedup: CPU vs single GPU

**Tips for Live Coding**:
- Pre-download datasets (internet can be slow in classroom)
- Have backup laptop in case of connection issues
- Start with CPU version, then GPU (shows difference)

---

### Episode 5: GPU Training with TensorFlow/Keras
**Time**: 20 min teaching + 15 min exercises
**Key Points**:
- High-level Keras API masks GPU complexity
- tf.data pipelines are important for performance
- Custom training loops for research flexibility

**Live Coding**:
- Sequential model (simplest)
- Functional API (for complex models)
- Custom training with tf.GradientTape
- Callbacks for monitoring

**Compare to PyTorch**:
- TensorFlow more declarative, PyTorch more imperative
- TensorFlow faster out-of-box, PyTorch easier to debug
- Both produce same results

**Exercises**:
- Train same model in TensorFlow vs PyTorch (compare code)
- Implement callbacks for early stopping

**Troubleshooting**:
- tf.function compilation errors? Remove @tf.function, debug, re-add
- Graph execution vs eager? Explain: @tf.function compiles for speed

---

### Episode 6: Distributed Training Across GPUs
**Time**: 25 min teaching + 20 min exercises
**Key Points**:
- Data parallelism most common (replicate model on each GPU)
- Communication overhead limits scaling (rarely >3x on 4 GPUs)
- DistributedDataParallel better than DataParallel

**Live Coding**:
- DataParallel (simple but inefficient)
- DistributedDataParallel with torchrun (professional)
- Measure speedup across 1, 2, 4 GPUs

**Important Notes**:
- Many students think "4 GPUs = 4x speed" (it's actually ~3x)
- Showing the overhead convinces them of importance
- Load balancing matters (some GPUs may stall)

**Exercises**:
- Modify single-GPU script for multi-GPU
- Submit job with --gres=gpu:4
- Collect speedup measurements

**Common Issues**:
- All prints on all ranks (use `if rank == 0`)
- Hanging processes (timeout=30min in init_process_group)
- Different results per rank (set seed, use sampler.set_epoch)

**Advanced Note**:
- Model parallelism is rare at Pomona (models fit on GPU)
- Mention for completeness

---

### Episode 7: Hyperparameter Tuning and Best Practices
**Time**: 25 min teaching + 20 min exercises
**Key Points**:
- Three approaches: grid, random, Bayesian search
- Bayesian most efficient (Ray Tune)
- Reproducibility requires version control + config files

**Live Coding**:
- Simple grid search loop
- Ray Tune example
- SLURM array job for parallel sweeps
- Tensorboard for visualizing results

**Reproducibility Focus**:
- Save config.yaml, code, timestamp, git hash
- Someone should be able to rerun and get same results
- Document hyperparameter choices

**Exercises**:
- Tune learning rate on small dataset
- Submit 10-job array for parallel search
- Visualize results (loss vs LR)

**Advanced Topics** (optional):
- Optuna for Bayesian tuning
- Population-based training for long jobs
- Early stopping via validation curves

**Troubleshooting**:
- GPU memory leaks during sweeps: Add `gc.collect(); torch.cuda.empty_cache()`
- Jobs not running: Check queue (squeue -u $USER)
- Hyperparameters not having effect: Verify they're actually used in code!

---

## Logistics and Setup

### Pre-Workshop Checklist
- [ ] Notify ITS to enable GPU access for all participants
- [ ] Pre-stage datasets on /bigdata (large downloads are slow)
- [ ] Create shared folder for example code
- [ ] Test all SLURM commands (sbatch, squeue, scancel)
- [ ] Prepare solution scripts for each challenge
- [ ] Check node availability (sinfo)

### During Workshop
- **Facilitator**: Live coding, narrate decisions
- **TA (if available)**: Monitor chat/questions, help individuals
- **Emphasis**: Iterate fast (test on CPU first, then GPU)

### Room Setup
- Projector for instructor demos
- Learners should have laptops
- Access to Sagehen (SSH client: terminal on Mac, PowerShell/WSL on Windows)

---

## Common Learner Challenges

### Challenge 1: "My GPU job is taking forever to run"
**Likely Causes**:
1. Long queue (check: `squeue | head`)
2. I/O bottleneck (data loading is slow)
3. GPU not actually being used (check nvidia-smi)
4. Job time limit approaching (check: scontrol show job)

**Teaching Point**: Profile before optimizing!

### Challenge 2: "My model trained locally but crashes on GPU"
**Likely Causes**:
1. GPU memory too small for batch size
2. Floating point differences (CPU uses float64, GPU float32)
3. Missing .to(device) somewhere in code

**Teaching Point**: Always test on small batch first!

### Challenge 3: "Multi-GPU training only 2x faster with 4 GPUs"
**Expected**: This is normal (70-80% efficiency is good)

**Teaching Point**: Communication overhead matters!

---

## Assessment and Feedback

### Ways to Assess Learning
- **Exercises within episodes**: Do learners' code run without error?
- **Challenges**: Can they modify existing code?
- **Final project** (optional): Can they train a model relevant to their research?

### Feedback Survey
Include questions:
- Did you understand when/why to use GPU?
- Can you now write a job submission script?
- Do you feel confident asking questions of HPC team?
- Would you recommend this workshop?

---

## Extensions and Follow-Up

### If You Have Extra Time
- Episode on Jupyter notebooks on Sagehen (OnDemand)
- Deep dive into profiling (Python profiler, GPU timelines)
- Model serving/inference considerations

### Recommended Next Steps for Learners
- Start small: Run your actual research on 1 GPU
- Measure: Always time before and after optimization
- Ask questions: ITS is here to help!
- Read: Sagehen documentation for advanced topics

---

## Resources for Instructors

### External Tutorials
- PyTorch Distributed Training: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- TensorFlow Distributed: https://tensorflow.org/guide/distributed_training
- Sagehen User Guide: [institution URL]

### Example Datasets
- MNIST: `torchvision.datasets.MNIST`
- CIFAR-10: `torchvision.datasets.CIFAR10`
- ImageNet subset: Available on /bigdata

### Benchmarking Tools
- NVIDIA GPU Monitoring: `nvidia-smi`, `nvidia-smi dmon`
- Python Profiler: `cProfile`
- PyTorch Profiler: `torch.profiler`

---

## Notes from Previous Runs

*(Update after each workshop)*

### What Went Well
- Live coding demonstrations helped people understand GPU concepts
- Challenges were appropriately difficult
- SLURM job submission was less intimidating after practice

### What to Improve
- Spend more time on data loading (I/O often overlooked)
- Pre-stage datasets (network transfers during workshop slow things down)
- Emphasize: "Test locally first" (saves GPU hours)

### Participant Feedback Summary
- Most appreciated: Real examples relevant to their research
- Most challenging: Understanding distributed training scaling
- Most useful: Templates for job submission