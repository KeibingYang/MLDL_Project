# MLDL_Project - **DocFloQA-FreezeTune**
The repository is the project belonged to Kaibing Yang and its theme is about finetuning the multimodal model to adpat specific task. Specically, the project finetunes the Florence-2 modle to better address the DocVQA problem.

In this project, we explored the related work of DocVQA and fine-tuned it on DocVQA2020 based on Florence-2 with satisfactory results

One prefix token + 80% tunable params ⇨ near-SOTA DocVQA.

# 1. Overview  

**DocFloQA-FreezeTune** shows that a large-scale vision–language model can
solve DocVQA without any task-specific layers.  We  

* prepend a single task prefix `<DocVQA>` to every question;  
* freeze (or partially unfreeze) the DaViT visual encoder, tuning only
  the decoder head;  
* analyse four freezing ratios **0 / 33 / 66 / 100 %**.

<p align="center">
  <img src="figures\Florence_model_pipeline.pdf" width="820">
</p>

# 2. Environment  

| component | version | note |
|-----------|---------|------|
| Python    | ≥ 3.9   | tested on 3.10 |
| PyTorch   | ≥ 2.1   | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| CUDA      | 11.8 / 12.x | 1 × A100 (40 GB) / H100 (80 GB) |
| Transformers | ≥ 4.41 | `pip install transformers` |
| Datasets  | ≥ 2.19 | `pip install datasets` |
| seaborn / pandas | plotting |

# 3. Dataset
For the fine-tuned dataset, we mainly used the DocVQA2020 dataset(https://huggingface.co/datasets/lmms-lab/DocVQA)

The dataset is also available through the Baidu Disk ()

# 4.Model
For model selection, we chose the Florence-2 model due to the consideration of computational resources(https://huggingface.co/microsoft/Florence-2-base-ft/tree/main)

The dataset is also availabel throught the Baidu Disk()

# 5. Usage  

## 5.1 Install 
git clone https://github.com/KeibingYang/MLDL_Project.git
cd MLDL_Project
pip install -r requirements.txt

## Quick Start
Once you download the dataset and model already, you can run the follow instruction to finetune the Florence-2.

python finetune.py

There are four visual trainable ratios in the  freezing_levels and you can change the ratio as you need to achieve your goals. 
Once the code starts, it will train the given ration in the freezing_levels continuely and output the log and json in the given path.

There are some exapmle outputs like train_log_all_frozen_config.json, train_log_all_unfrozen_config.json and train_log_all frozen_config.json which are the configration during the training. There will be other logs and visualizations results after the all visual trainable ratios executing.

# 6. Results  

## 6.1 Fine-tuning curve (frozen all visual encoder - trainable ratio = 0)

| epoch | accuracy % | EM % | F1 % | val loss |
|------:|-----------:|-----:|-----:|---------:|
| 0 | 15.20 | 0.2437 | -    |   -  |     -    |
| 1 | 36.20 | 0.5014 | 36.32 | 44.60 | 62.37 |
| 2 | 38.00 | 0.5164 | 38.83 | 47.37 | 59.66 |
| 3 | 40.20 | 0.5264 | 40.12 | 48.87 | 58.11 |
| 4 | 41.80 | 0.5449 | 40.74 | 49.50 | 57.97 |
| 5 | 43.20 | 0.5632 | 41.50 | 50.32 | 57.36 |
| 6 | 44.40 | 0.5670 | 42.14 | 50.99 | 57.11 |
| 7 | **45.80** | 0.5791 | 42.29 | 51.06 | **57.25** |
| 8 | 45.60 | **0.5816** | **42.40** | **51.13** | **57.25** |

<p align="center">
  <img src="figures/epoch_progress.pdf" width="820">
</p>

## 6.2 Freezing ratio ablation (epoch 8)

| vision trainable | accuracy % | ANLS | EM % | F1 % | val loss |
|------------------|-----------:|------:|-----:|-----:|---------:|
| 0 % | 45.6 | 0.582 | 42.4 | 51.1 | 0.573 |
| 33 % | 49.3 | 0.602 | 46.7 | 55.2 | 0.557 |
| **66 %** | **51.1** | **0.611** | **48.1** | **56.8** | **0.553** |
| 100 % | 50.2 | 0.607 | 47.5 | 55.9 | 0.560 |

66 % selective unfreezing => best accuracy / stability trade-off.

---

Vector PDFs saved to `figures/`.

---

# 7. Limitations  

* Validation split = 500 samples → wide CI.  
* EM/F1 still behind latest specialised models.  
* Robustness to severe OCR noise needs work
* 

# 8.Paper
Related results can be found in Final/3022234232-杨凯冰-机器学习和深度学习-课程报告.pdf
