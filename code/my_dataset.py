# my_dataset.py

import numpy as np
import mindspore.dataset as ds
from PIL import Image


class MyDataSet:
    def __init__(self, images_path, images_label):
        self.images_path = images_path
        self.images_label = images_label
        print("\n数据集初始化检查（前5个样本）：")
        for i in range(min(5, len(images_path))):
            print(f"样本 {i}: 路径={images_path[i]}, 标签={images_label[i]}")

    def __getitem__(self, item):
        try:
            img = Image.open(self.images_path[item]).convert('RGB')
            img_np = np.array(img)
            label = int(self.images_label[item])
            # print(f"加载图像 {item}: 原始形状 {img_np.shape}")  # 调试输出
            return img_np, label
        except Exception as e:
            print(f"跳过损坏图像 {self.images_path[item]}: {str(e)}")
            return np.zeros((224, 224, 3)), 0  # 返回默认形状的空白图像

    def __len__(self):
        return len(self.images_path)


def manual_transform(img_pil, img_size=224):
    """手动实现预处理步骤，便于调试"""
    # print("\n开始手动转换...")

    # 1. 转换为PIL图像（确保输入）
    if not isinstance(img_pil, Image.Image):
        img_pil = Image.fromarray(img_pil)
    # print(f"步骤1: PIL图像模式={img_pil.mode}, 大小={img_pil.size}")

    # 2. 调整大小
    resize_size = int(img_size * 1.14)
    img_resized = img_pil.resize((resize_size, resize_size))
    # print(f"步骤2: 调整大小后={img_resized.size}")

    # 3. 中心裁剪
    left = (resize_size - img_size) // 2
    top = (resize_size - img_size) // 2
    right = left + img_size
    bottom = top + img_size
    img_cropped = img_resized.crop((left, top, right, bottom))
    # print(f"步骤3: 裁剪后={img_cropped.size}")

    # 4. 转换为numpy数组
    img_np = np.array(img_cropped, dtype=np.float32)
    # print(f"步骤4: 转numpy后形状={img_np.shape}")

    # 5. 检查并调整通道顺序
    if img_np.ndim == 2:  # 灰度图
        img_np = np.stack([img_np] * 3, axis=0)
    elif img_np.shape[2] == 3:  # HWC转CHW
        img_np = img_np.transpose(2, 0, 1)
    # print(f"步骤5: 调整通道后形状={img_np.shape}")

    # 6. 归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_normalized = (img_np / 255.0 - mean) / std
    # print(f"步骤6: 归一化后形状={img_normalized.shape}")
    # print(f"归一化后范围: [{img_normalized.min()}, {img_normalized.max()}]")

    return img_normalized


def create_dataset(images_path, images_class, transform=None, batch_size=32, shuffle=True):
    # 创建原始数据集
    raw_dataset = MyDataSet(images_path, images_class)

    # 打印样本示例
    sample_img, sample_label = raw_dataset[0]
    print("\n数据集样本示例（预处理前）：")
    print(f"图像形状: {sample_img.shape}, 类型: {type(sample_img)}")
    print(f"标签: {sample_label}, 类型: {type(sample_label)}")

    # 创建MindSpore数据集
    dataset = ds.GeneratorDataset(
        raw_dataset,
        column_names=["image", "label"],
        shuffle=shuffle,
        python_multiprocessing=False,
        num_parallel_workers=1
    )

    # 应用转换
    if transform is None:
        print("警告: 未提供transform，使用手动转换")
        dataset = dataset.map(
            operations=lambda x: manual_transform(x),
            input_columns=["image"],
            python_multiprocessing=False,
            num_parallel_workers=1
        )
    else:
        print("使用提供的transform")
        dataset = dataset.map(
            operations=transform,
            input_columns=["image"],
            python_multiprocessing=False,
            num_parallel_workers=1
        )

    # 批处理
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset