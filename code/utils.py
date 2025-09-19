# utils.py

import os
import json
import pickle
import random
import glob
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from PIL import Image
import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历每个类别文件夹，递归查找所有图片
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 递归获取所有支持的图片路径（包括子目录）
        images = []
        for ext in supported:
            images.extend(glob.glob(os.path.join(cla_path, '**', f'*{ext}'), recursive=True))

        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print(f"{sum(every_class_num)} images were found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    # 在read_split_data后添加
    for img_path in train_images_path[:100]:  # 检查前100个样本
        try:
            img = Image.open(img_path).convert('RGB')
            img.verify()  # 验证图像完整性
        except Exception as e:
            print(f"损坏图像: {img_path} - {str(e)}")
            train_images_path.remove(img_path)  # 从训练集中移除

    return train_images_path, train_images_label, val_images_path, val_images_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, epoch):
    model.set_train()

    # 定义前向计算和损失函数
    def forward_fn(data, label):
        logits = model(data)
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        loss = loss_fn(logits, label)
        return loss, logits

    # 创建梯度函数
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    total_loss = 0
    total_acc = 0
    total_samples = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        # 确保数据格式正确
        if data.shape[1] != 3:  # 如果通道不在第二维度
            data = data.transpose(0, 3, 1, 2)  # 转换为(batch, channel, height, width)

        # 计算损失和梯度
        (loss, logits), grads = grad_fn(data, label)

        # 更新参数
        optimizer(grads)

        # 计算准确率
        preds = ops.Argmax(output_type=mindspore.int32)(logits)
        acc = ops.ReduceMean()((preds == label).astype(mindspore.float32))

        # 累计统计
        batch_size = len(data)
        total_loss += loss.asnumpy() * batch_size
        total_acc += acc.asnumpy() * batch_size
        total_samples += batch_size

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.asnumpy():.4f}, Acc: {acc.asnumpy():.4f}")

    return total_loss / total_samples, total_acc / total_samples


def evaluate(model, data_loader, epoch):
    model.set_train(False)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    total_loss = 0
    total_acc = 0
    total_samples = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        # 确保数据格式正确
        if data.shape[1] != 3:
            data = data.transpose(0, 3, 1, 2)

        logits = model(data)
        loss = loss_fn(logits, label)

        # 计算准确率
        preds = ops.Argmax(output_type=mindspore.int32)(logits)
        acc = ops.ReduceMean()((preds == label).astype(mindspore.float32))

        batch_size = len(data)
        total_loss += loss.asnumpy() * batch_size
        total_acc += acc.asnumpy() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples