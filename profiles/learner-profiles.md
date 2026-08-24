---
title: "Introduction to Machine Learning on HPC - Learner Profiles"
---

# Introduction to Machine Learning on HPC - Learner Profiles

## Maya Chen - Computational Biology Undergraduate

**Role:** Junior Biology Major with Research Focus

**Background:** Maya is a third-year biology major at Pomona College with a strong interest in computational genomics. She has taken two semesters of general biology lab courses and completed an intro computer science course in Python. Last summer, she worked in a plant biology lab where she generated next-generation sequencing (NGS) data but struggled to analyze the large datasets; her laptop could barely handle the files. She has attended one data carpentry workshop on basic Unix and R.

**Current ML Experience:** Maya has learned some basic machine learning concepts in her intro CS course (decision trees, basic classification) and tried using scikit-learn on small datasets. However, she has never applied ML to real biological data and doesn't understand how to scale her analysis beyond toy examples. She's heard of neural networks but views them as a "black box" and is unsure when they'd be appropriate for genomics work.

**Goals:** Maya wants to apply machine learning to genomics datasets, specifically using convolutional neural networks to classify DNA sequences and predict regulatory regions from ChIP-seq data. She needs to understand how to prepare large genomic datasets, train models efficiently on the HPC cluster, and interpret results biologically rather than just chasing high accuracy metrics. She also wants to share reproducible workflows with her lab group.

**Special Considerations:** Maya worries that HPC systems are only for computer scientists and fears making mistakes that might disrupt shared resources. She learns best through biological examples and needs clear connections between ML concepts and their meaning in genomics. She has limited Linux experience and gets anxious around command-line interfaces. Supporting her comfort with HPC environment setup and emphasizing the iterative nature of ML experimentation will help her engage fully.

---

## Dr. Rajesh Patel - Physics Graduate Student

**Role:** PhD Candidate, Experimental High Energy Physics

**Background:** Rajesh is a fourth-year physics doctoral student working on the ATLAS experiment analyzing particle collision data from CERN. He has strong experience with Python, ROOT (CERN's data analysis framework), and has written several hundred lines of scientific code. He completed an intro computational physics course covering numerical methods but has not formally studied machine learning. His physics background gives him deep domain knowledge about particle physics, detector systems, and statistical analysis.

**Current ML Experience:** Rajesh has recently discovered that neural networks are transforming high-energy physics analysis; colleagues are publishing papers using graph neural networks and recurrent networks for jet tagging. He's completed some online tutorials on PyTorch basics and tried running a simple model on his laptop, but had no guidance on structuring large training jobs, managing datasets, or understanding training dynamics at scale. He's uncertain about GPU utilization and hasn't used HPC systems for deep learning before.

**Goals:** Rajesh needs to train deep neural networks on millions of particle collision events to identify rare decays and classify jets. He wants to understand how to manage large training runs on the Sagehen HPC cluster, use GPUs efficiently for model training, structure data pipelines for high-volume physics data, and track experiments systematically to reproduce results and interpret model decisions. He also wants to collaborate with a physics colleague who's skeptical of "black box" ML methods.

**Special Considerations:** Rajesh brings sophisticated domain knowledge and high self-directed learning ability. He's comfortable with command-line interfaces and Linux but may initially view HPC and ML as separate domains to be combined rather than as integrated tools. He values rigorous statistical validation and will be skeptical of high accuracy claims without proper cross-validation. He learns best through physics-specific examples and will be highly motivated by real ATLAS analysis problems. Connecting ML interpretability to fundamental physics principles will strengthen his engagement.

---

## Professor Amanda Okafor - Social Science Faculty

**Role:** Associate Professor, Sociology Department

**Background:** Amanda is a tenured sociology faculty member with 8 years of teaching experience. Her research focuses on social movements and collective action. She's comfortable with quantitative social science methods (regression, survey analysis in R and Stata) but has always worked with structured datasets from surveys and census data. She has never written a Python script and completed a basic data literacy workshop two years ago. She's increasingly interested in computational approaches to understanding text and social dynamics but feels intimidated by technical tools.

**Current ML Experience:** Amanda has read popular articles about text mining and topic modeling in research blogs and attended a conference panel on NLP applications in social science. She's never trained a machine learning model and has no experience with Python or HPC systems. She understands statistical significance at a conceptual level but is unsure how ML "hypothesis testing" differs from traditional approaches. She has questions about what NLP can realistically do for her work.

**Goals:** Amanda wants to use natural language processing techniques to analyze hundreds of open-ended survey responses and Twitter data about social movements. She's interested in sentiment analysis, topic modeling, and identifying emergent themes without manually reading thousands of responses. She needs to understand how to format social science data for ML pipelines, what NLP techniques are appropriate for her research questions, how to validate that models produce sociologically meaningful results, and how to write reproducible workflows that her research assistants can use.

**Special Considerations:** Amanda comes from a non-technical background and may benefit from reassurance that she doesn't need to become a programmer. She values understanding *why* methods work and their limitations; abstract math is less important than interpretability and practical validity. She learns best through concrete social science examples and needs clear, non-technical explanations before diving into code. She'll be more motivated by research impact than technical sophistication. Providing working examples of NLP on social science data, explaining model validation through her existing statistical knowledge, and emphasizing reproducibility and team workflows will help her succeed and feel included in computational research.

---

## James Rodriguez - Computer Science Student

**Role:** Senior CS Major, Strong ML Background

**Background:** James is a fourth-year CS major with a GPA focus on machine learning. He has completed courses in machine learning algorithms, deep learning, computer vision, and natural language processing. He has written hundreds of thousands of lines of code in Python, C++, and Go. His coursework emphasized theory: backpropagation derivations, optimization algorithms, regularization techniques. He has trained neural networks on standard datasets (CIFAR-10, ImageNet, Penn Treebank) on his personal laptop or Google Colab, where he's comfortable with PyTorch and TensorFlow.

**Current ML Experience:** James deeply understands ML theory and can implement networks from scratch. However, he has never trained a model on real data at scale, never used GPUs in a shared HPC environment, and has no experience with the practical challenges of long-running training jobs, dataset pipelines, memory management, or troubleshooting distributed training failures. He hasn't thought carefully about hyperparameter tuning strategies for real problems or how to structure reproducible experiments. He sometimes conflates theoretical accuracy with practical model performance.

**Goals:** James wants to bridge the gap between "homework-scale" ML and real research problems. He needs to understand how to set up and execute large training jobs on HPC systems, manage GPU memory and compute resources responsibly, structure data pipelines for real datasets, debug training failures and convergence issues, and design experiments that answer real questions, not just maximize test accuracy. He's interested in pushing performance boundaries and wants to learn best practices from researchers who regularly train models at scale.

**Special Considerations:** James is self-directed and technically sophisticated; he can quickly grasp new concepts and troubleshoot independently. However, he may initially underestimate practical challenges or assume that scaling up involves only multiplying batch sizes. He'll benefit from exposure to workflows and "folk wisdom" accumulated by practitioners. He may be impatient with basic explanations but will engage deeply with advanced optimization problems and reproducibility challenges. Connect theoretical insights with practical consequences (e.g., how learning rate theory translates to tuning on real data), emphasize best practices and reproducibility, and present realistic failure scenarios he'll encounter on HPC systems.