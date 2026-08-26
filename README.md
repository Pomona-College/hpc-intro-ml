# Introduction to Machine Learning on HPC

Pomona College HPC Workshop Series

## Overview

This workshop teaches machine learning practitioners when and how to use HPC clusters for model training and inference. Participants learn to identify machine learning bottlenecks that HPC solves, including large datasets, long training times, and hyperparameter searches. The workshop covers environment setup, data preparation, training with PyTorch and TensorFlow on Sagehen HPC's GPU nodes, submitting long-running jobs, and following best practices for reproducible ML research.

## Episodes

1. **Why Machine Learning on HPC**: Understand when HPC is necessary for ML, identify computational bottlenecks, compare CPU vs GPU approaches, and learn Sagehen's ML capabilities.
2. **Environment Setup**: Configure ML software environments, install Python ML frameworks, use conda or virtual environments, and manage dependencies on HPC systems.
3. **Data Preparation**: Load and preprocess large datasets, manage data storage on /bigdata and /scratch, use streaming and batch processing, and handle data pipelines at scale.
4. **Training Models with PyTorch**: Set up PyTorch on GPU, implement distributed training, use DataParallel for multi-GPU work, and optimize training performance.
5. **Training Models with TensorFlow**: Configure TensorFlow for GPU usage, use Keras for model building, implement multi-GPU distributed training, and monitor training with TensorBoard.
6. **Managing Long-Running Jobs**: Submit multi-hour or multi-day training jobs, use checkpointing and resumable training, monitor job progress, and handle job failures gracefully.
7. **Best Practices for ML on HPC**: Implement reproducibility with seeds and logging, organize code and results, use version control, benchmark performance, and document experiments.

## Prerequisites

- Active Sagehen HPC cluster account
- Linux command line proficiency
- SLURM job submission experience
- Python programming knowledge
- Familiarity with machine learning concepts and frameworks (PyTorch, TensorFlow, or scikit-learn)
- Basic understanding of data formats (CSV, HDF5, NumPy)

## Learning Objectives

After completing this workshop, learners will be able to:
- Identify machine learning bottlenecks that HPC clusters solve
- Set up ML software environments on Sagehen
- Prepare and manage large datasets for training
- Train models efficiently using GPUs and multiple cores
- Submit long-running jobs and monitor progress
- Implement checkpointing and recovery strategies
- Adopt best practices for reproducible ML research

## Target Audience

Graduate students, postdocs, and researchers applying machine learning to their research. Ideal for those working with large datasets, complex models, or hyperparameter searches that require more computational resources than laptops provide.

## Duration

Approximately 4-5 hours, including hands-on training exercises with PyTorch and TensorFlow on GPU nodes.

## Technical Requirements

- Sagehen HPC cluster account with GPU allocation
- SSH access and SLURM job submission capability
- Python 3.7 or later
- PyTorch and/or TensorFlow installed in conda environment
- CUDA toolkit (compatible with GPU hardware)
- Text editor for scripts and configuration files
- Storage allocation on /bigdata or /scratch for training data

## Contact

- **Email**: its-hpc@pomona.edu
- **Workshop Author**: Andrew Wilson, Director of Research Computing

## License

This workshop is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

Wilson, A. (2026). *Introduction to Machine Learning on HPC*. Pomona College ITS Research Computing.

## Acknowledgments

**Andrew Wilson** — Director of Research Computing and Digital Scholarship,
Pomona College. Workshop design and development.

**Andrei Motchenko** — testing, editing, cleanup and screenshots across the
Pomona College HPC Workshop Series.
