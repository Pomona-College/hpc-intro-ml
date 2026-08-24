---
title: "ML Best Practices"
teaching: 25
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you ensure reproducibility of ML experiments?
- What are common pitfalls in HPC ML workflows?
- How do you track experiments and deploy models?
- What habits separate research-ready ML from one-off scripts?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Follow MLOps best practices for reproducibility
- Monitor and log experiments systematically
- Avoid common HPC ML pitfalls
- Export trained models for deployment and sharing
::::::::::::::::::::::::::::::::::::::::::::::::

## Why Best Practices Matter More on HPC

A research project that produces a single trained model is a one-off script. A project that produces a dozen variants over six months, with multiple co-authors and a paper deadline, is a system. The difference between the two is whether you can answer questions like "which version of the code produced figure 3?" and "what hyperparameters did the model in checkpoint 47 use?".

On HPC the stakes are higher than on a laptop because you spend real GPU-hours per experiment. Losing track of which run produced a result means redoing the run, which is sometimes a day of A100 time. The habits below are the ones that pay back many times over their setup cost.

## Reproducibility and MLOps

### 1. Version Control Everything

```bash
git init
git add *.py
git commit -m "Initial ML pipeline"
```

This is the absolute minimum. Beyond it, commit at every meaningful experiment boundary, tag the commit that produced each significant result, and never store data files in git (use `.gitignore` for `/data` and `/results`). Six months later, the only reliable way to know exactly what code produced a specific result is to read its git hash off the saved checkpoint.

### 2. Configuration Management

```yaml
# config.yaml - Never hardcode hyperparameters
model:
  architecture: "resnet50"
  pretrained: true
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
data:
  path: "/bigdata/lab/<labname>/datasets/cifar10"
  train_split: 0.8
```

```python
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
lr = config['training']['learning_rate']
```

Hardcoded hyperparameters scattered through your code mean tweaking one value can require touching ten files. Externalizing them to a config file means experiments differ from each other only in the YAML, which is easy to diff, easy to share, and easy to commit.

For sweeps, generate one config file per trial and save it alongside the result. Future-you will thank present-you when figuring out what produced run 47.

### 3. Experiment Tracking with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='/scratch/$USER/runs/exp_lr0.001_bs32')
for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.close()
```

```bash
tensorboard --logdir=/scratch/$USER/runs
```

TensorBoard is the simplest experiment tracker for PyTorch and TensorFlow workflows. Run it locally or via OnDemand to compare 30 sweep runs in one chart and immediately spot the trial that converged twice as fast as the others. For more sophisticated tracking, Weights and Biases (`wandb`) and MLflow are popular choices that add cloud-hosted dashboards and richer artifact tracking.

### 4. Complete Checkpoints

```python
import subprocess
from datetime import datetime

checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'hyperparameters': config,
    'timestamp': datetime.now().isoformat(),
    'git_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
}
torch.save(checkpoint, f'/scratch/$USER/checkpoints/model_epoch_{epoch}.pt')
```

Saving the git hash inside the checkpoint is the single most useful reproducibility habit. Six months later, when reviewers ask which version of the code trained your headline model, you can `git checkout HASH` and see exactly that state.

## Common HPC ML Pitfalls

::::::::::::::::::::::::::::::::::::: callout

**Pitfall 1: Forgetting to Scale Learning Rate**

When you increase batch size, adjust learning rate:

```python
base_lr = 0.001
lr = base_lr * np.sqrt(batch_size / 32)
```

Larger batches give more accurate gradients, which permits larger steps. Square-root scaling is a reasonable rule for most cases; linear scaling (lr ~ batch_size) is more aggressive and works for some setups.

**Pitfall 2: Tuning on Test Set**

```python
# WRONG: Using test set for hyperparameter selection
# CORRECT: Use three separate sets
train, val, test = split_data(dataset, [0.6, 0.2, 0.2])
# Tune on val_data, report final metrics on test_data once
```

This is research integrity 101 but still happens regularly under deadline pressure. Touch the test set exactly once, at the end, when reporting the final number. Tuning on the test set inflates apparent performance and the gap will not survive deployment.

**Pitfall 3: GPU Memory Leaks During Sweeps**

```python
import gc
for lr in learning_rates:
    model = MyModel()
    train(model, lr)
    del model
    gc.collect()
    torch.cuda.empty_cache()
```

Without explicit cleanup, the GPU accumulates state from previous iterations and eventually OOMs. The pattern above is the safe boilerplate.

**Pitfall 4: Not Monitoring Training**

```bash
watch -n 1 nvidia-smi          # GPU usage
squeue -u $USER -l              # Job status
tail -f training.log            # Log output
```

If your loss has been NaN for the last 20 epochs and you do not notice until the job finishes, you have wasted GPU-hours. Set up monitoring early and check it at least once per training run.

::::::::::::::::::::::::::::::::::::::::::::::::

## Model Deployment

### Export for Production

```python
# PyTorch: TorchScript for production
scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')

# TensorFlow: SavedModel format
model.save('model_saved_model/')
```

TorchScript and SavedModel are the framework-native serialization formats. They preserve the model architecture and weights together, so the consumer does not need access to your Python code to run inference. Use them whenever you want to share a model with someone outside your immediate codebase.

### Prediction Pipeline

```python
def predict(image_path, model_path):
    model = MyModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = transform(Image.open(image_path)).unsqueeze(0).cuda()
    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)
        predicted_class = torch.argmax(probs, dim=1)
    return predicted_class.item(), probs[0].cpu().numpy()
```

Two details matter for any inference pipeline: `model.eval()` switches off dropout and uses running statistics for batch norm, and `torch.no_grad()` disables gradient tracking which both speeds up inference and roughly halves memory use.

::::::::::::::::::::::::::::::::::::: callout

## A research-grade run log

For each significant experiment, record:

- Date and SLURM job ID
- Git hash of the code
- Path to the config.yaml used
- Path to the dataset (with version or checksum)
- Wall clock time and final epoch reached
- Best metric on validation
- Headline observation in one sentence

Keep this in a single Markdown file (or a spreadsheet) per project. Three months later, when a co-author asks "what was the best run we did with the new architecture?", you have an answer in 30 seconds instead of 30 minutes of grep through SLURM logs.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Create a Reproducible Experiment**

Set up a complete experiment workflow:
1. Create a config.yaml with model hyperparameters
2. Add TensorBoard logging to your training loop
3. Save checkpoints that include the git hash
4. Export the final model for inference

::::::::::::::::::::::::::::::::::::: solution

Create `config.yaml`, load it in your training script, log with TensorBoard, save checkpoints with git hash (as shown above), and export:

```python
# After training
torch.jit.script(model).save('model_production.pt')

# Verify
loaded = torch.jit.load('model_production.pt')
output = loaded(test_input)
```

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Audit a Project for Reproducibility Gaps**

You inherit a colleague's ML project. The repo has `train.py` (uses hardcoded hyperparameters), `model.pt` (no metadata), and a folder of result CSVs. List four things you would do, in order, to make this reproducible.

::::::::::::::::::::::::::::::::::::: solution

1. Initialize git in the project and commit the current state. At minimum you can now refer to "the version that produced these results" later.

2. Externalize hyperparameters into a `config.yaml` and refactor `train.py` to read it. This makes future experiments diff-able.

3. Add an experiment-log Markdown file documenting what each existing CSV came from (best effort) and start logging new experiments going forward.

4. Re-train the model with the new pipeline so the resulting checkpoint includes git hash, hyperparameters, and date. The old `model.pt` is essentially unreproducible; it stays for reference but is not authoritative.

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- Track experiments with TensorBoard, config files, and version control
- Always use separate train/val/test sets; touch test exactly once at the end
- Scale learning rate with batch size (sqrt or linear)
- Monitor GPU memory to avoid leaks during sweeps
- Save complete checkpoints including hyperparameters, git hash, and date
- Export models with TorchScript or SavedModel for deployment
- Keep a per-project run log so you can answer "which run produced figure 3?"
::::::::::::::::::::::::::::::::::::::::::::::::

<!-- highlight <labname>/<myusername> placeholders in code blocks; remove if the varnish theme handles this natively -->
<script>(function(){var CSS='.sh-placeholder{color:#c2410c;font-weight:700}[data-bs-theme="dark"] .sh-placeholder,html.dark .sh-placeholder{color:#fdba74}@media (prefers-color-scheme: dark){[data-bs-theme="auto"] .sh-placeholder{color:#fdba74}}';var RX=/<labname>|<myusername>/g;function firstMatch(el){var w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null),nodes=[],full='';while(w.nextNode()){nodes.push({n:w.currentNode,s:full.length});full+=w.currentNode.nodeValue;}RX.lastIndex=0;var m;while((m=RX.exec(full))){var s=m.index,e=s+m[0].length,inSpan=false,parts=[];for(var j=0;j<nodes.length;j++){var ns=nodes[j].s,ne=ns+nodes[j].n.nodeValue.length;if(ne<=s||ns>=e)continue;parts.push({node:nodes[j].n,a:Math.max(s-ns,0),b:Math.min(e-ns,nodes[j].n.nodeValue.length)});var p=nodes[j].n.parentNode;while(p&&p!==el){if(p.classList&&p.classList.contains('sh-placeholder')){inSpan=true;break;}p=p.parentNode;}}if(!inSpan&&parts.length)return parts;}return null;}function wrapParts(parts){for(var i=parts.length-1;i>=0;i--){var t=parts[i].node,txt=t.nodeValue,a=parts[i].a,b=parts[i].b;var span=document.createElement('span');span.className='sh-placeholder';span.textContent=txt.slice(a,b);var f=document.createDocumentFragment();if(a>0)f.appendChild(document.createTextNode(txt.slice(0,a)));f.appendChild(span);if(b<txt.length)f.appendChild(document.createTextNode(txt.slice(b)));t.parentNode.replaceChild(f,t);}}function run(){var st=document.createElement('style');st.textContent=CSS;document.head.appendChild(st);document.querySelectorAll('pre,code').forEach(function(el){var guard=0,parts;while((parts=firstMatch(el))&&guard++<500){wrapParts(parts);}});}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',run);}else{run();}})();</script>
