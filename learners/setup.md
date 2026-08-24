# Setup: Introduction to Machine Learning on HPC

## Before the Workshop

### Computer and Internet
- A laptop or desktop with internet connection
- SSH access to Sagehen (Pomona's HPC cluster)
- A terminal application (Terminal on Mac, PowerShell or WSL on Windows)

### Accounts
- Active Pomona College account
- Access to Sagehen HPC system
  - Request at: its-hpc@pomona.edu
  - Takes 1-2 business days to set up

### Optional: Local Environment
For quick testing before running on GPU:
- Install Python 3.10+
- Install pip
- Optional: Miniconda for conda environments

## During the Workshop

### Access Sagehen
```bash
# Log in
ssh your_netid@sagehen.hpc.pomona.edu

# There are no pytorch/tensorflow modules on Sagehen -- ML frameworks
# live in conda environments (created in episode 4):
module load miniconda3
conda activate pytorch_env    # or tf_env
```

### Download Workshop Materials
```bash
# The example data and scripts live in the lesson repository
git clone https://github.com/Pomona-College/hpc-intro-ml.git
cd hpc-intro-ml/episodes/data
```

You only need this if you want the example datasets locally. Everything in the
episodes can also be typed directly, and the small sample files are reproduced
inline where they are used.

### Create Your Working Directory
```bash
# Create personal workspace in your home directory
mkdir -p ~/ml-workshop
cd ~/ml-workshop

# Create subdirectories
mkdir -p data models logs
```

`/scratch` is **not** writable from the login node — `mkdir -p /scratch/$USER/...`
there fails with `Permission denied`. Create scratch directories inside a job
(`srun` or `sbatch`) only, and treat anything left there as gone once the job ends.

## Hardware Available on Sagehen

Sagehen provides 10 GPUs across A100/L40S/RTX PRO 6000 generations (confirmed May 2026); see Workshop 16 for full hardware details.

### GPU Nodes
- **A100 (80 GB, ×4)**: most powerful, for the largest models
- **L40S (48 GB, ×4)**: good for standard deep learning
- **RTX PRO 6000 (96 GB ECC, ×2)**: the largest GPU memory on Sagehen; very large models and pro workloads

### Storage
- **/rhome**: 100 GB (your home, slower, backed up)
- **/bigdata**: Large group storage (moderate speed)
- **/scratch**: Fast node-local SSD, writable only inside a job and deleted when the job ends

### Example Specs
- **Per GPU node**: 4 CPUs, 32 GB system RAM
- **Partition**: `gpu` (for GPU jobs)
- **Typical limits**: 4 GPUs per account, 2-hour default job limit

## Software Stack

### Pre-installed Modules
```bash
# Check available modules
module avail

# Modules you will actually use:
module load miniconda3          # Python/conda (frameworks install into conda envs)
module load anaconda3           # alternative conda distribution
module load cuda/12.2.1         # CUDA toolkit (current default)
```

### Python Packages
Common ML packages available in conda:
- PyTorch, TensorFlow, scikit-learn
- NumPy, pandas, matplotlib
- Jupyter notebooks (with GPU support via OnDemand)

## Getting Help

### During Workshop
- Instructors and TAs available
- Check #ml-hpc channel on community Slack (if available)

### After Workshop
- **HPC Support**: its-hpc@pomona.edu
- **Sagehen documentation**: the Pomona College HPC Workshop Series — start with
  [Introduction to HPC Systems](https://pomona-college.github.io/hpc-intro/)
- **Pomona ITS**: <https://www.pomona.edu/its/>
- **Framework Documentation**:
  - PyTorch: https://pytorch.org
  - TensorFlow: https://tensorflow.org
  - Scikit-learn: https://scikit-learn.org

### Troubleshooting Common Issues

**Can't connect to Sagehen?**
- Check VPN is connected
- Verify NetID and password
- Contact ITS at its-hpc@pomona.edu

**GPU not available?**
- Check: `squeue -u $USER`
- May need to wait for job queue
- Start with smaller job to test

**Module not found?**
- Run: `module avail`
- Try: `module load cuda/11.8.0` (a different version)
- Ask instructors

## Workshop Outline

1. **Episode 1**: Why Machine Learning on HPC?
2. **Episode 2**: Environment Setup (conda, modules)
3. **Episode 3**: Data Preparation and Preprocessing
4. **Episode 4**: GPU Training with PyTorch
5. **Episode 5**: GPU Training with TensorFlow/Keras
6. **Episode 6**: Distributed Training Across GPUs
7. **Episode 7**: Hyperparameter Tuning and Best Practices

## Recommended Reading
- Sagehen Quick Start Guide (available in first episode)
- PyTorch GPU Tutorial (link in Episode 4)
- TensorFlow Performance Guide (link in Episode 5)