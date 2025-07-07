"""
Florence模型在DocVQA数据集上的多层次微调训练脚本
=======================================================

支持四个训练层次的视觉编码器冻结/解冻策略：
1. 全部冻结 (0% 视觉编码器可训练)
2. 解冻1/3 (33% 视觉编码器可训练)  
3. 解冻2/3 (66% 视觉编码器可训练)
4. 全部解冻 (100% 视觉编码器可训练)

每个层次包含：
- 详细的训练日志记录
- 验证集性能评估
- 预测结果可视化
- 统计指标计算

符合科研规范，包含完整的实验记录和分析
"""

from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from rapidfuzz.distance import Levenshtein
import shutil
import numpy as np
import json
import time
from datetime import datetime


def compute_levenshtein_similarity(preds, refs):
    """
    计算预测结果与参考答案之间的Levenshtein相似度
    
    Args:
        preds (list[str] or str): 预测答案
        refs (list[str] or str): 参考答案
    
    Returns:
        tuple: (相似度数组, 平均相似度)
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(refs, str):
        refs = [refs]
    
    sims = []
    for p, r in zip(preds, refs):
        # 文本预处理
        p_norm = str(p).strip().lower()
        r_norm = str(r).strip().lower()
        
        # 计算Levenshtein距离
        dist = Levenshtein.distance(p_norm, r_norm)
        
        # 归一化为相似度
        denom = max(len(p_norm), len(r_norm)) or 1
        sims.append(1 - dist / denom)
    
    sims = np.round(np.array(sims), 4)
    return sims, float(np.round(sims.mean(), 4))


def set_vision_encoder_trainable_ratio(model, ratio):
    """
    根据ratio设置视觉编码器的可训练比例
    
    Args:
        model: Florence模型实例
        ratio (float): 可训练比例 (0.0=全部冻结, 1.0=全部解冻)
    
    Returns:
        dict: 包含视觉编码器参数统计信息
    """
    vision_params = []
    for name, param in model.named_parameters():
        if 'vision_tower' in name:
            vision_params.append(param)
    
    total_vision_params = len(vision_params)
    num_to_unfreeze = int(total_vision_params * ratio)
    
    # 首先冻结所有视觉编码器参数
    for param in vision_params:
        param.requires_grad = False
    
    # 解冻指定比例的参数（从后往前，通常后面的层更重要）
    if num_to_unfreeze > 0:
        for param in vision_params[-num_to_unfreeze:]:
            param.requires_grad = True
    
    info = {
        'total_vision_params': total_vision_params,
        'unfrozen_vision_params': num_to_unfreeze,
        'vision_trainable_ratio': ratio
    }
    
    print(f"🎯 Vision encoder trainable ratio set to {ratio:.2f}")
    print(f"   Unfrozen {num_to_unfreeze} of {total_vision_params} vision parameters")
    
    return info


def get_trainable_params_info(model):
    """
    获取模型详细的可训练参数信息
    
    Args:
        model: 模型实例
    
    Returns:
        dict: 详细的参数统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vision_params = sum(p.numel() for name, p in model.named_parameters() if 'vision_tower' in name)
    vision_trainable = sum(p.numel() for name, p in model.named_parameters() 
                          if 'vision_tower' in name and p.requires_grad)
    language_params = total_params - vision_params
    language_trainable = trainable_params - vision_trainable
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params,
        'vision_params': vision_params,
        'vision_trainable': vision_trainable,
        'vision_trainable_ratio': vision_trainable / vision_params if vision_params > 0 else 0,
        'language_params': language_params,
        'language_trainable': language_trainable,
        'language_trainable_ratio': language_trainable / language_params if language_params > 0 else 0
    }
    
    print(f"📊 Model Parameters Info:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({info['trainable_ratio']:.2%})")
    print(f"   Frozen: {info['frozen_params']:,}")
    print(f"   Vision Total: {vision_params:,}")
    print(f"   Vision Trainable: {vision_trainable:,} ({info['vision_trainable_ratio']:.2%})")
    print(f"   Language Total: {language_params:,}")
    print(f"   Language Trainable: {language_trainable:,} ({info['language_trainable_ratio']:.2%})")
    
    return info


def compute_em_f1(preds, refs):
    """
    计算EM (Exact Match) 和 F1 分数
    
    Args:
        preds (list[str]): 预测答案列表
        refs (list[str]): 参考答案列表
    
    Returns:
        tuple: (EM分数, F1分数)
    """
    try:
        from evaluate import load
        squad_metric = load("squad")
        
        formatted_preds = [{"id": str(i), "prediction_text": str(p)} for i, p in enumerate(preds)]
        formatted_refs = [{"id": str(i), "answers": {"text": [str(a)], "answer_start": [0]}} for i, a in enumerate(refs)]
        
        results = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)
        return results["exact_match"], results["f1"]
    except Exception as e:
        print(f"Warning: Could not compute EM/F1 using evaluate library: {e}")
        # 简单的EM计算作为备选
        em_count = sum(1 for p, r in zip(preds, refs) if str(p).strip().lower() == str(r).strip().lower())
        em_score = em_count / len(preds) * 100
        return em_score, em_score


def train_model_with_logging(train_loader, val_loader, model, processor, device, epochs=2, lr=1e-6, 
                             vision_trainable_ratio=0.0, log_file_path="train_log.txt", stage_name=""):
    """
    训练模型并记录详细日志
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 模型实例
        processor: 处理器
        device: 计算设备
        epochs: 训练轮数
        lr: 学习率
        vision_trainable_ratio: 视觉编码器可训练比例
        log_file_path: 日志文件路径
        stage_name: 训练阶段名称
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training Stage: {stage_name}")
    print(f"{'='*60}")
    
    # 设置训练开始时间
    start_time = time.time()
    
    # 设置视觉编码器冻结比例
    vision_info = set_vision_encoder_trainable_ratio(model, vision_trainable_ratio)
    
    # 获取并显示参数信息
    params_info = get_trainable_params_info(model)
    
    # 只优化可训练参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=lr)
    
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    # 创建日志文件
    log_file = open(log_file_path, "w", encoding='utf-8')
    
    # 记录训练配置
    config_info = {
        "timestamp": datetime.now().isoformat(),
        "stage_name": stage_name,
        "vision_trainable_ratio": vision_trainable_ratio,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "device": str(device),
        "params_info": params_info,
        "vision_info": vision_info
    }
    
    log_file.write("=" * 60 + "\n")
    log_file.write(f"Training Stage: {stage_name}\n")
    log_file.write(f"Timestamp: {config_info['timestamp']}\n")
    log_file.write("=" * 60 + "\n")
    log_file.write(f"Vision Trainable Ratio: {vision_trainable_ratio:.2f}\n")
    log_file.write(f"Total Parameters: {params_info['total_params']:,}\n")
    log_file.write(f"Trainable Parameters: {params_info['trainable_params']:,} ({params_info['trainable_ratio']:.2%})\n")
    log_file.write(f"Vision Parameters: {params_info['vision_params']:,}\n")
    log_file.write(f"Vision Trainable: {params_info['vision_trainable']:,} ({params_info['vision_trainable_ratio']:.2%})\n")
    log_file.write(f"Language Parameters: {params_info['language_params']:,}\n")
    log_file.write(f"Language Trainable: {params_info['language_trainable']:,} ({params_info['language_trainable_ratio']:.2%})\n")
    log_file.write(f"Learning Rate: {lr}\n")
    log_file.write(f"Epochs: {epochs}\n")
    log_file.write(f"Batch Size: {train_loader.batch_size}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write("-" * 50 + "\n")
    
    # 保存配置为JSON
    config_file = log_file_path.replace('.txt', '_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            
            # 处理答案格式
            processed_answers = []
            for ans in answers:
                if isinstance(ans, list):
                    processed_answers.append(ans[0] if ans else "")
                else:
                    processed_answers.append(str(ans))
            
            labels = processor.tokenizer(
                text=processed_answers, 
                return_tensors="pt", 
                padding=True, 
                return_token_type_ids=False
            ).input_ids.to(device)
            
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        epoch_train_time = time.time() - epoch_start_time
        
        # 验证阶段
        model.eval()
        val_loss = 0
        predictions = []
        ground_truths = []
        val_batches = 0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, answers = batch
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                
                # 处理答案格式
                processed_answers = []
                for ans in answers:
                    if isinstance(ans, list):
                        processed_answers.append(ans[0] if ans else "")
                    else:
                        processed_answers.append(str(ans))
                
                labels = processor.tokenizer(
                    text=processed_answers, 
                    return_tensors="pt", 
                    padding=True, 
                    return_token_type_ids=False
                ).input_ids.to(device)
                
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
                val_batches += 1
                
                # 生成预测答案
                gen_ids = model.generate(
                    input_ids=input_ids, 
                    pixel_values=pixel_values, 
                    max_new_tokens=128,
                    num_beams=3
                )
                gen_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
                
                predictions.extend(gen_texts)
                ground_truths.extend(processed_answers)
        
        val_time = time.time() - val_start_time
        
        # 计算评估指标
        avg_val_loss = val_loss / val_batches
        em, f1 = compute_em_f1(predictions, ground_truths)
        _, avg_similarity = compute_levenshtein_similarity(predictions, ground_truths)
        
        # 记录epoch结果
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "em": em,
            "f1": f1,
            "avg_similarity": avg_similarity,
            "train_time": epoch_train_time,
            "val_time": val_time
        }
        
        # 打印和写入日志
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, EM: {em:.2f}, F1: {f1:.2f}, Avg Sim: {avg_similarity:.4f}")
        log_file.write(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, EM: {em:.2f}, F1: {f1:.2f}, Avg Sim: {avg_similarity:.4f}\n")
        log_file.write(f"           Train Time: {epoch_train_time:.2f}s, Val Time: {val_time:.2f}s\n")
        
        # 保存epoch详细信息
        epoch_file = log_file_path.replace('.txt', f'_epoch_{epoch+1}.json')
        with open(epoch_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_info, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - start_time
    log_file.write(f"\nTotal Training Time: {total_time:.2f}s\n")
    log_file.write("=" * 60 + "\n")
    log_file.close()


def visualize_and_log(dataset, model, processor, device, num_samples=100, save_dir="./visualization", stage_name=""):
    """
    可视化预测结果并记录详细数据
    
    Args:
        dataset: 数据集
        model: 模型实例
        processor: 处理器
        device: 计算设备
        num_samples: 可视化样本数量
        save_dir: 保存目录
        stage_name: 阶段名称
    
    Returns:
        tuple: (准确率, 平均相似度)
    """
    print(f"\n📊 Starting Visualization for {stage_name}")
    
    model.eval()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    
    # 初始化计数器和记录
    correct_count = 0
    total_similarity = 0
    prediction_details = []
    
    log_file_path = os.path.join(save_dir, f"predictions_log_{stage_name}.txt")
    log_file = open(log_file_path, "w", encoding='utf-8')
    
    # 写入头部信息
    log_file.write("=" * 60 + "\n")
    log_file.write(f"Prediction Results for {stage_name}\n")
    log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    log_file.write(f"Total Samples: {min(num_samples, len(dataset))}\n")
    log_file.write("=" * 60 + "\n")
    
    results_summary = []
    
    for idx in tqdm(range(min(num_samples, len(dataset))), desc=f"Visualizing {stage_name}"):
        try:
            question, gt_answer, image = dataset[idx]
            
            # 处理答案格式
            if isinstance(gt_answer, list):
                gt_answer = gt_answer[0] if gt_answer else ""
            gt_answer = str(gt_answer)
            
            # 保存图像
            image_path = os.path.join(save_dir, "images", f"image_{stage_name}_{idx}.png")
            image.save(image_path)
            
            # 进行推理
            prompt = question
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    num_beams=3
                )
            
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = generated_texts[0] if generated_texts else ""
            
            # 计算相似度
            _, similarity = compute_levenshtein_similarity([generated_text], [gt_answer])
            
            # 判断是否正确
            is_correct = gt_answer.strip().lower() in generated_text.strip().lower()
            if is_correct:
                correct_count += 1
            
            total_similarity += similarity
            
            # 记录详细信息
            result_info = {
                'idx': idx,
                'question': question,
                'gt_answer': gt_answer,
                'prediction': generated_text,
                'similarity': similarity,
                'is_correct': is_correct,
                'image_path': image_path
            }
            results_summary.append(result_info)
            prediction_details.append(result_info)
            
            # 写入日志
            log_file.write(f"\n--- Example {idx+1} ---\n")
            log_file.write(f"Question: {question}\n")
            log_file.write(f"GT Answer: {gt_answer}\n")
            log_file.write(f"Predicted: {generated_text}\n")
            log_file.write(f"Similarity: {similarity:.4f}\n")
            log_file.write(f"Status: {'✔ Correct' if is_correct else '✘ Incorrect'}\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            log_file.write(f"\n--- Example {idx+1} ---\n")
            log_file.write(f"ERROR: {str(e)}\n")
    
    # 计算最终指标
    num_processed = min(num_samples, len(dataset))
    accuracy = correct_count / num_processed if num_processed > 0 else 0
    avg_similarity = total_similarity / num_processed if num_processed > 0 else 0
    
    # 创建统计汇总
    summary_stats = {
        "stage_name": stage_name,
        "timestamp": datetime.now().isoformat(),
        "total_samples": num_processed,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "average_similarity": avg_similarity,
        "metrics": {
            "accuracy_percentage": accuracy * 100,
            "error_rate": (1 - accuracy) * 100,
            "similarity_std": np.std([r['similarity'] for r in prediction_details]) if prediction_details else 0
        }
    }
    
    summary_text = f"""
Evaluation Summary for {stage_name}:
==========================================
Total Samples: {num_processed}
Correct Predictions: {correct_count}
Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
Average Similarity: {avg_similarity:.4f}
Similarity Std: {summary_stats['metrics']['similarity_std']:.4f}
Error Rate: {summary_stats['metrics']['error_rate']:.2f}%
"""
    
    # 保存汇总信息
    summary_path = os.path.join(save_dir, f"summary_{stage_name}.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(summary_text)
    
    # 保存详细JSON数据
    details_path = os.path.join(save_dir, f"details_{stage_name}.json")
    with open(details_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary_stats,
            "predictions": prediction_details
        }, f, indent=2, ensure_ascii=False)
    
    print(summary_text)
    log_file.write(summary_text)
    log_file.close()
    
    # 创建可视化图表
    create_visualization_grids(results_summary, save_dir, stage_name)
    create_metrics_charts(prediction_details, save_dir, stage_name)
    
    return accuracy, avg_similarity


def create_visualization_grids(results_summary, save_dir, stage_name):
    """
    创建预测结果的可视化网格图
    
    Args:
        results_summary: 结果汇总列表
        save_dir: 保存目录
        stage_name: 阶段名称
    """
    def create_single_grid(results_batch, save_dir, batch_idx):
        """创建单个网格图"""
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle(f'Predictions Visualization - {stage_name} (Batch {batch_idx + 1})', fontsize=16, fontweight='bold')
        
        for i, result in enumerate(results_batch):
            if i >= 10:
                break
            
            row = i // 5
            col = i % 5
            
            try:
                # 加载和显示图像
                img = Image.open(result['image_path'])
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                
                # 创建标题
                status_color = 'green' if result['is_correct'] else 'red'
                status_symbol = '✓' if result['is_correct'] else '✗'
                
                title = f"{status_symbol} Sim: {result['similarity']:.3f}\n"
                
                # 截断长文本
                gt_text = result['gt_answer'][:25] + "..." if len(result['gt_answer']) > 25 else result['gt_answer']
                pred_text = result['prediction'][:25] + "..." if len(result['prediction']) > 25 else result['prediction']
                
                title += f"GT: {gt_text}\n"
                title += f"Pred: {pred_text}"
                
                axes[row, col].set_title(title, fontsize=10, color=status_color, fontweight='bold')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"Error loading\nimage {result['idx']}\n{str(e)}", 
                                  ha='center', va='center', transform=axes[row, col].transAxes,
                                  fontsize=10, color='red')
                axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(results_batch), 10):
            row = i // 5
            col = i % 5
            axes[row, col].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"visualization_{stage_name}_batch_{batch_idx + 1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # 分批次创建可视化图
    batch_size = 10
    for batch_idx in range((len(results_summary) + batch_size - 1) // batch_size):
        batch_results = results_summary[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        if batch_results:
            create_single_grid(batch_results, save_dir, batch_idx)


def create_metrics_charts(prediction_details, save_dir, stage_name):
    """
    创建指标统计图表
    
    Args:
        prediction_details: 预测详细信息
        save_dir: 保存目录
        stage_name: 阶段名称
    """
    if not prediction_details:
        return
    
    # 提取数据
    similarities = [p['similarity'] for p in prediction_details]
    accuracies = [1 if p['is_correct'] else 0 for p in prediction_details]
    
    # 创建多子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Metrics Analysis - {stage_name}', fontsize=16, fontweight='bold')
    
    # 相似度分布直方图
    axes[0, 0].hist(similarities, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Similarity Distribution')
    axes[0, 0].set_xlabel('Similarity Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    axes[0, 0].legend()
    
    # 准确率趋势
    window_size = max(1, len(accuracies) // 20)
    rolling_acc = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
    axes[0, 1].plot(rolling_acc, color='green', linewidth=2)
    axes[0, 1].set_title(f'Accuracy Trend (Rolling Window: {window_size})')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Rolling Accuracy')
    axes[0, 1].set_ylim(0, 1)
    
    # 相似度 vs 准确率散点图
    colors = ['red' if acc == 0 else 'green' for acc in accuracies]
    axes[1, 0].scatter(similarities, accuracies, c=colors, alpha=0.6)
    axes[1, 0].set_title('Similarity vs Accuracy')
    axes[1, 0].set_xlabel('Similarity Score')
    axes[1, 0].set_ylabel('Accuracy (0/1)')
    axes[1, 0].set_ylim(-0.1, 1.1)
    
    # 统计汇总
    stats_text = f"""
Statistics Summary:
Total Samples: {len(prediction_details)}
Correct: {sum(accuracies)}
Accuracy: {np.mean(accuracies):.3f}
Mean Similarity: {np.mean(similarities):.3f}
Std Similarity: {np.std(similarities):.3f}
Min Similarity: {np.min(similarities):.3f}
Max Similarity: {np.max(similarities):.3f}
"""
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"metrics_analysis_{stage_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_all_training_stages(train_loader, val_loader, model, processor, device, epochs=2, lr=1e-6, num_samples=100):
    """
    运行所有四个训练阶段
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 模型实例
        processor: 处理器
        device: 计算设备
        epochs: 每个阶段的训练轮数
        lr: 学习率
        num_samples: 可视化样本数量
    
    Returns:
        dict: 所有阶段的结果汇总
    """
    # 定义四个冻结层次
    freezing_levels = [0, 0.33, 0.66, 1.0]
    stage_names = ["all_frozen", "one_third_unfrozen", "two_thirds_unfrozen", "all_unfrozen"]

    results = {}
    overall_start_time = time.time()
    
    print(f"\n🚀 Starting Multi-Stage Training Pipeline")
    print(f"   Stages: {len(stage_names)}")
    print(f"   Epochs per stage: {epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Visualization samples: {num_samples}")
    
    for stage_idx, (ratio, stage) in enumerate(zip(freezing_levels, stage_names)):
        stage_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🎯 Stage {stage_idx + 1}/4: {stage} (Vision Trainable: {ratio:.0%})")
        print(f"{'='*80}")
        
        # 训练阶段
        log_file = f"train_log_{stage}.txt"
        train_model_with_logging(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            processor=processor,
            device=device,
            epochs=epochs,
            lr=lr,
            vision_trainable_ratio=ratio,
            log_file_path=log_file,
            stage_name=stage
        )
        
        # 可视化和评估阶段
        save_dir = f"./visualization_{stage}"
        acc, sim = visualize_and_log(
            dataset=val_loader.dataset,
            model=model,
            processor=processor,
            device=device,
            num_samples=num_samples,
            save_dir=save_dir,
            stage_name=stage
        )
        
        stage_time = time.time() - stage_start_time
        
        # 记录结果
        results[stage] = {
            'accuracy': acc,
            'similarity': sim,
            'vision_trainable_ratio': ratio,
            'stage_time': stage_time,
            'stage_index': stage_idx + 1
        }
        
        print(f"✅ Stage {stage} completed in {stage_time:.2f}s")
        print(f"   Accuracy: {acc:.4f}, Similarity: {sim:.4f}")
        
        # 保存中间结果
        intermediate_results_path = f"intermediate_results_{stage}.json"
        with open(intermediate_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 创建总体结果汇总
    total_time = time.time() - overall_start_time
    final_results = {
        "pipeline_info": {
            "total_time": total_time,
            "num_stages": len(stage_names),
            "epochs_per_stage": epochs,
            "learning_rate": lr,
            "num_visualization_samples": num_samples,
            "completion_time": datetime.now().isoformat()
        },
        "stage_results": results
    }
    
    # 保存最终结果
    final_results_path = "final_results_all_stages.json"
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 创建对比图表
    create_comparison_charts(results, "./")
    
    print(f"\n🎉 All training stages completed in {total_time:.2f}s")
    print(f"📊 Results saved to: {final_results_path}")
    
    return results


def create_comparison_charts(results, save_dir):
    """
    创建各阶段对比图表
    
    Args:
        results: 所有阶段的结果
        save_dir: 保存目录
    """
    stage_names = list(results.keys())
    accuracies = [results[stage]['accuracy'] for stage in stage_names]
    similarities = [results[stage]['similarity'] for stage in stage_names]
    ratios = [results[stage]['vision_trainable_ratio'] for stage in stage_names]
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Stage Training Comparison', fontsize=16, fontweight='bold')
    
    # 准确率对比
    axes[0, 0].bar(stage_names, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, max(accuracies) * 1.1)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
    
    # 相似度对比
    axes[0, 1].bar(stage_names, similarities, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Similarity Comparison')
    axes[0, 1].set_ylabel('Average Similarity')
    axes[0, 1].set_ylim(0, max(similarities) * 1.1)
    for i, v in enumerate(similarities):
        axes[0, 1].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
    
    # 性能趋势
    axes[1, 0].plot(ratios, accuracies, 'bo-', label='Accuracy', linewidth=2, markersize=8)
    axes[1, 0].plot(ratios, similarities, 'ro-', label='Similarity', linewidth=2, markersize=8)
    axes[1, 0].set_title('Performance vs Vision Trainable Ratio')
    axes[1, 0].set_xlabel('Vision Trainable Ratio')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 改进统计
    acc_improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
    sim_improvements = [similarities[i] - similarities[i-1] for i in range(1, len(similarities))]
    
    x_pos = np.arange(len(acc_improvements))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, acc_improvements, width, label='Accuracy Δ', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, sim_improvements, width, label='Similarity Δ', alpha=0.7)
    axes[1, 1].set_title('Stage-to-Stage Improvements')
    axes[1, 1].set_xlabel('Stage Transition')
    axes[1, 1].set_ylabel('Improvement')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'{stage_names[i]} → {stage_names[i+1]}' for i in range(len(acc_improvements))], 
                              rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "multi_stage_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ========== 数据集类定义 ==========
class DocVQADataset(Dataset):
    """
    DocVQA数据集封装类
    """
    def __init__(self, split_data):
        self.data = split_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<DocVQA>" + example["question"]
        answer = example["answers"]
        image = example["image"]
        
        # 确保图像为RGB格式
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return question, answer, image


# ========== 主程序入口 ==========
if __name__ == "__main__":
    print("🚀 Starting Florence DocVQA Multi-Stage Fine-tuning Pipeline")
    print("=" * 80)
    
    # ========== 配置参数 ==========
    model_path = "/seu_nvme/home/fenglei/213240634/Florence/Model/Florence"
    dataset_path = "/seu_nvme/home/fenglei/213240634/Florence/dataset_20250615144366/dataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用cuda:0
    
    # 训练超参数
    batch_size = 6
    epochs = 8  # 每个阶段的训练轮数
    learning_rate = 1e-6
    num_samples_visualize = 100  # 可视化样本数量
    
    print(f"📱 Device: {device}")
    print(f"📊 Batch size: {batch_size}")
    print(f"🔄 Epochs per stage: {epochs}")
    print(f"📚 Learning rate: {learning_rate}")
    print(f"🎨 Visualization samples: {num_samples_visualize}")
    
    # ========== 加载模型和处理器 ==========
    print("\n📥 Loading model and processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            revision='refs/pr/6'
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            revision='refs/pr/6'
        ).to(device)
        print("✅ Model and processor loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)
    
    # ========== 加载数据集 ==========
    print("\n📂 Loading dataset...")
    try:
        data = load_dataset(dataset_path)
        train_dataset = DocVQADataset(data["train"])
        val_dataset = DocVQADataset(data["validation"])
        print(f"✅ Dataset loaded successfully")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        exit(1)
    
    # ========== 数据整理器 ==========
    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(
            text=list(questions), 
            images=list(images), 
            return_tensors="pt", 
            padding=True
        ).to(device)
        return inputs, answers
    
    # ========== 创建数据加载器 ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"✅ Data loaders created")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # ========== 开始多阶段训练 ==========
    print("\n🏋️ Starting multi-stage training pipeline...")
    
    try:
        results = run_all_training_stages(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            processor=processor,
            device=device,
            epochs=epochs,
            lr=learning_rate,
            num_samples=num_samples_visualize
        )
        
        # ========== 打印最终结果 ==========
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Final Results Summary:")
        print("-" * 40)
        
        for stage, metrics in results.items():
            print(f"Stage: {stage}")
            print(f"  Vision Trainable: {metrics['vision_trainable_ratio']:.0%}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Similarity: {metrics['similarity']:.4f}")
            print(f"  Time: {metrics['stage_time']:.2f}s")
            print("-" * 40)
        
        print("\n📁 Files Generated:")
        print("  - Training logs: train_log_*.txt")
        print("  - Visualization: visualization_*/")
        print("  - Final results: final_results_all_stages.json")
        print("  - Comparison charts: multi_stage_comparison.png")
        
        print("\n✅ All training stages completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training pipeline: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
