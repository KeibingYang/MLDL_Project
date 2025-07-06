# MLDL_Project
The repository is the project belonged to Kaibing Yang and its theme is about finetuning the multimodal model to adpat specific task. Specically, the project finetunes the Florence-2 modle to better address the DocVQA problem.

In this project, we explored the related work of DocVQA and fine-tuned it on DocVQA2020 based on Florence-2 with satisfactory results

One prefix token + 0.8 % tunable params ⇨ near-SOTA DocVQA.

# 1. Overview  

DocFloQA-FreezeTune shows that a large-scale vision–language model can
solve DocVQA without any task-specific layers.  We  

* prepend a single task prefix `<DocVQA>` to every question;  
* freeze (or partially unfreeze) the DaViT visual encoder, tuning only
  the decoder head;  
* analyse four freezing ratios **0 / 33 / 66 / 100 %**.

<p align="center">
  <img src="assets/pipeline.png" width="820">
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

# 4.Model
For model selection, we chose the Florence-2 model due to the consideration of computational resources(https://huggingface.co/microsoft/Florence-2-base-ft/tree/main)

# 5. Usage  

## 5.1 Install 
git clone https://github.com/<user>/DocFloQA-FreezeTune.git
cd DocFloQA-FreezeTune
pip install -r requirements.txt

# 6. Results  

## 6.1 Fine-tuning curve (frozen visual encoder)

| epoch | accuracy % | ANLS | EM % | F1 % | val loss |
|------:|-----------:|------:|-----:|-----:|---------:|
| 0 | 15.20 | 0.244 | — | — | — |
| 8 | **45.60** | **0.582** | 42.4 | 51.1 | 0.573 |

## 6.2 Freezing ratio ablation (epoch 8)

| vision trainable | accuracy % | ANLS | EM % | F1 % | val loss |
|------------------|-----------:|------:|-----:|-----:|---------:|
| 0 % | 45.6 | 0.582 | 42.4 | 51.1 | 0.573 |
| 33 % | 49.3 | 0.602 | 46.7 | 55.2 | 0.557 |
| **66 %** | **51.1** | **0.611** | **48.1** | **56.8** | **0.553** |
| 100 % | 50.2 | 0.607 | 47.5 | 55.9 | 0.560 |

66 % selective unfreezing => best accuracy / stability trade-off.

---

Vector PDFs saved to `figs/`.

---

# 7. Limitations  

* Validation split = 500 samples → wide CI.  
* EM/F1 still behind latest specialised models.  
* Robustness to severe OCR noise needs work
* 
# Result
Related results can be found in Final/3022234232杨凯冰-课程报告.pdf
