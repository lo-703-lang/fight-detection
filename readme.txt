暴力行为检测系统 - 完整运行说明

1 系统概述

本系统是基于MindSpore框架开发的暴力行为检测系统，使用Vision Transformer (ViT)模型对视频中的暴力行为进行识别和分类。系统包含两个主要模块：
1. 训练模块（train.py）：训练ViT模型
2. 预测模块（video_predict.py）：对输入视频进行暴力行为检测

2 系统要求

2.1 硬件要求
- CPU: Intel/AMD 4核以上
- 内存: 8GB以上
- GPU: 推荐NVIDIA GPU(4GB显存以上)以加速训练
- 存储空间: 至少10GB可用空间

2.2 软件要求
- Python: 3.7-3.9
- CUDA: 11.1或11.6(如使用GPU)

3 环境配置

3.1创建conda虚拟环境：


conda create -n ms_py38 python=3.8 -y
conda activate ms_py38


3.2 安装依赖包:

   pip install mindspore==2.0.0 mindcv==0.3.0 opencv-python pillow numpy matplotlib tqdm


   *如使用GPU，请安装对应CUDA版本的MindSpore:

   pip install mindspore==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

3.3 验证安装:

   python -c "import mindspore; print(mindspore.__version__)"


4 数据准备

4.1训练数据
1. 创建以下目录结构:

   data/
   ├── nonviolence/  # 非暴力图像
   └── violence/     # 暴力图像


2. 将训练图像按类别放入对应文件夹，支持.jpg/.png格式

4.2测试视频
1. 创建测试目录:

   test/
   └── in1/
       └── video1.mp4  # 测试视频


5 使用指南

5.1 训练模型

运行训练脚本:
bash：
python train.py \
    --data-path ./data \
    --num-classes 2 \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001 \
    --weights ./model.ckpt \
    --freeze-layers partial \
    --device GPU  # 或CPU


参数说明:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-path` | 训练数据路径 | ./data |
| `--num-classes` | 分类数量 | 2 |
| `--epochs` | 训练轮数 | 10 |
| `--batch-size` | 批处理大小 | 32 |
| `--lr` | 初始学习率 | 0.001 |
| `--weights` | 预训练权重路径 | ./model.ckpt |
| `--freeze-layers` | 冻结策略(partial/head_only/none) | partial |
| `--device` | 训练设备(CPU/GPU) | CPU |

训练输出:
- 模型权重: `weights1/model-{epoch}.ckpt`
- 类别索引: `class_indices.json`

5.2 视频预测

运行预测脚本:

python video_predict.py


默认配置:
- 输入视频: `test/in1/video1.mp4`
- 输出视频: `test/in1/out_video1.mp4`
- 帧标签: `test/in1/frame_labels.txt`

自定义配置:
修改video_predict.py中的以下变量:

video_path = "your/video/path.mp4"  # 输入视频路径
output_path = "output/video.mp4"    # 输出视频路径
label_file_path = "labels.txt"      # 标签文件路径
model_weight_path = "model.ckpt"    # 模型权重路径


输出说明:
1. 输出视频: 带有预测标签和置信度的标注视频
2. 标签文件: 每帧的预测结果(0:非暴力, 1:暴力)




6 版本信息

关键组件版本信息：

| 组件 | 版本 |
|------|------|
| MindSpore | 2.0.0 |
| MindCV | 0.3.0 |
| OpenCV | 4.5.5 |
| Pillow | 9.3.0 |
| NumPy | 1.21.6 |


所有组件版本信息：

Package            Version
------------------ -----------
asttokens          3.0.0
astunparse         1.6.3
certifi            2025.7.14
charset-normalizer 3.4.2
colorama           0.4.6
cycler             0.12.1
filelock           3.16.1
fonttools          4.57.0
fsspec             2025.3.0
huggingface-hub    0.33.4
idna               3.10
Jinja2             3.1.6
kiwisolver         1.4.7
MarkupSafe         2.1.5
matplotlib         3.5.3
mindcv             0.3.0
mindspore          2.0.0
mpmath             1.3.0
networkx           3.1
numpy              1.21.6
opencv-python      4.5.5.64
packaging          25.0
Pillow             9.3.0
pip                24.3.1
protobuf           3.20.3
psutil             7.0.0
pyparsing          3.1.4
python-dateutil    2.9.0.post0
PyYAML             6.0.2
requests           2.32.4
scipy              1.10.1
setuptools         75.3.0
six                1.17.0
sympy              1.13.3
timm               0.6.12
torch              2.4.1
torchvision        0.19.1
tqdm               4.67.1
typing_extensions  4.13.2
urllib3            2.2.3
wheel              0.45.1
