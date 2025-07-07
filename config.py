# ========== 路径配置 ==========
# 模型路径 - Florence-2预训练模型存储位置
MODEL_PATH = "/seu_nvme/home/fenglei/213240634/Florence/Model/Florence"

# 数据集路径 - DocVQA数据集存储位置
DATASET_PATH = "/seu_nvme/home/fenglei/213240634/Florence/dataset_20250615144366/dataset"

# ========== 设备配置 ==========
# 计算设备配置 (cuda:0, cuda:1, cpu等)
DEVICE = "cuda:0"

# ========== 训练超参数 ==========
# 批处理大小
BATCH_SIZE = 6

# 每个阶段的训练轮数
EPOCHS = 8

# 学习率
LEARNING_RATE = 1e-6

# 可视化样本数量
NUM_SAMPLES_VISUALIZE = 100

# ========== 视觉编码器冻结策略 ==========
# 四个冻结比例：0%, 33%, 66%, 100%
FREEZING_LEVELS = [0, 0.33, 0.66, 1.0]
STAGE_NAMES = ["all_frozen", "one_third_unfrozen", "two_thirds_unfrozen", "all_unfrozen"]

# ========== 输出配置 ==========
# 日志和结果保存配置
OUTPUT_DIR = "./"
LOG_PREFIX = "train_log"
VISUALIZATION_PREFIX = "visualization"
RESULTS_FILE = "final_results_all_stages.json"

# ========== 模型配置 ==========
# Florence-2模型特定配置
MODEL_REVISION = "refs/pr/6"
TRUST_REMOTE_CODE = True

# 生成参数
MAX_NEW_TOKENS = 128
NUM_BEAMS = 3

# ========== 数据加载配置 ==========
# DataLoader参数
NUM_WORKERS = 0
SHUFFLE_TRAIN = True

# ========== 优化器配置 ==========
# 学习率调度器
LR_SCHEDULER_NAME = "linear"
NUM_WARMUP_STEPS = 0
