
import os
import argparse
import mindspore
import mindspore.nn as nn
from mindspore import context
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindcv.models.vit import vit_b_16_224
import numpy as np
import sys

from my_dataset import create_dataset,manual_transform
from utils import read_split_data, train_one_epoch, evaluate


def validate_tensor_shape(tensor, img_size):
    """验证张量形状是否为(C,H,W)"""
    if tensor.ndim != 3:
        raise ValueError(f"需要3D张量，得到{tensor.ndim}D")
    if tensor.shape[0] != 3:
        raise ValueError(f"需要3通道，得到{tensor.shape[0]}通道")
    if tensor.shape[1] != img_size or tensor.shape[2] != img_size:
        raise ValueError(f"需要尺寸({img_size},{img_size})，得到({tensor.shape[1]},{tensor.shape[2]})")
    return tensor


def main(args):
    # 设置运行环境
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    # 创建权重保存目录
    os.makedirs("./weights1", exist_ok=True)

    # 读取数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    print(f"Train images: {len(train_images_path)}, Labels: {len(train_images_label)}")
    print(f"Val images: {len(val_images_path)}, Labels: {len(val_images_label)}")

    # 数据预处理
    img_size = 224
    data_transform = {
        "make": transforms.Compose([
            vision.ToPIL(),  # 确保输入为PIL图像
            vision.Resize(int(img_size * 1.14)),
            vision.CenterCrop(img_size),
            # 手动转换函数替代原有流程
            lambda x: manual_transform(x, img_size)  # 使用我们的手动转换函数
        ]),
        "val": transforms.Compose([
            vision.ToPIL(),
            vision.Resize(int(img_size * 1.14)),
            vision.CenterCrop(img_size),
            lambda x: manual_transform(x, img_size)  # 使用我们的手动转换函数
        ])
    }

    # 创建数据集前检查
    print("\n正在创建训练数据集...")
    train_dataset = create_dataset(
        train_images_path,
        train_images_label,
        transform=data_transform["make"],
        batch_size=args.batch_size,
        shuffle=True
    )

    print("\n正在创建验证数据集...")
    val_dataset = create_dataset(
        val_images_path,
        val_images_label,
        transform=data_transform["val"],
        batch_size=args.batch_size,
        shuffle=False
    )

    # 数据检查函数
    def check_dataset(dataset, name):
        try:
            sample = next(dataset.create_tuple_iterator())
            images, labels = sample[0], sample[1]
            print(f"\n{name} 数据集检查:")
            print(f"图像形状: {images.shape} (应为 [batch, 3, 224, 224])")
            print(f"像素范围: [{images.asnumpy().min():.4f}, {images.asnumpy().max():.4f}]")

            # 检查前3个样本
            for i in range(min(3, images.shape[0])):
                img = images[i]
                print(f"样本 {i}: 形状 {img.shape}, 通道数 {img.shape[0]}")
                print(f"像素范围: {img.asnumpy().min():.4f}-{img.asnumpy().max():.4f}")

            return True
        except Exception as e:
            print(f"\n{name} 检查失败: {str(e)}")
            return False

    # 执行检查
    if not check_dataset(train_dataset, "训练集"):
        print("训练集数据格式异常，程序终止！")
        sys.exit(1)

    if not check_dataset(val_dataset, "验证集"):
        print("验证集数据格式异常，程序终止！")
        sys.exit(1)

    # 创建模型
    model = vit_b_16_224(num_classes=args.num_classes, pretrained=False)

#############---------------------------------################
    # 加载预训练权重，调试时使用
    # if args.weights != "":
    #     assert os.path.exists(args.weights), f"权重文件不存在: '{args.weights}'"
    #
    #     # 加载转换后的权重
    #     param_dict = mindspore.load_checkpoint(args.weights)
    #
    #     # 获取模型参数名列表
    #     model_params = {p.name: p for p in model.get_parameters()}
    #
    #     # 打印调试信息
    #     print("\n===== 权重加载调试信息 =====")
    #     print("转换后的权重中的参数名示例:")
    #     for name in list(param_dict.keys())[:5]:
    #         print(f"  - {name}")
    #
    #     print("\n模型期望的参数名示例:")
    #     for name in list(model_params.keys())[:5]:
    #         print(f"  - {name}")
    #
    #     # # 筛选可加载的参数
    #     not_loaded = []
    #     loaded_params = {}
    #     for name, param in param_dict.items():
    #         if name in model_params:
    #             if param.shape == model_params[name].shape:
    #                 loaded_params[name] = param
    #             else:
    #                 not_loaded.append((name, f"形状不匹配 {param.shape} vs {model_params[name].shape}"))
    #         else:
    #             not_loaded.append((name, "参数名不匹配"))
    #
    #     # 加载匹配的参数
    #     mindspore.load_param_into_net(model, loaded_params)
    #
    #     # 打印加载结果
    #     print(f"\n成功加载 {len(loaded_params)}/{len(param_dict)} 个参数")
    #     if not_loaded:
    #         print("\n未加载的参数及原因:")
    #         for name, reason in not_loaded[:10]:  # 只打印前10个以免太多
    #             print(f"  - {name}: {reason}")
    #
    #     # 检查关键层是否加载
    #     critical_layers = ['patch_embed.proj.weight', 'head.weight']
    #     print("\n关键层加载状态:")
    #     for layer in critical_layers:
    #         if layer in loaded_params:
    #             print(f"  ✓ {layer} 已加载")
    #         else:
    #             print(f"  ✗ {layer} 未加载")

    # 加载预训练权重（核心逻辑，必须保留）
    if args.weights != "":
        assert os.path.exists(args.weights), f"权重文件不存在: '{args.weights}'"

        # 加载转换后的权重
        param_dict = mindspore.load_checkpoint(args.weights)

        # 过滤分类头参数（避免形状不匹配）
        filtered_params = {k: v for k, v in param_dict.items() if not k.startswith('head.')}

        # 加载特征提取层参数
        mindspore.load_param_into_net(model, filtered_params, strict_load=False)

        # 简化输出：只打印关键信息
        print(f"\n成功加载 {len(filtered_params)}/{len(param_dict)} 个特征提取层参数")
        print("分类头参数将重新训练（正常现象）")
  ##########---------------------------------#############


    # === 修复：分类头初始化 ===

    if hasattr(model, 'head'):
        print("\n=== 初始化分类头 ===")
        print(f"原始形状 - weight: {model.head.weight.shape}, bias: {model.head.bias.shape}")

        # 方法1：正态分布初始化
        model.head.weight.set_data(
            mindspore.Tensor(np.random.normal(0, 0.01, model.head.weight.shape).astype(np.float32))
        )
        model.head.bias.set_data(
            mindspore.Tensor(np.zeros(model.head.bias.shape).astype(np.float32))
        )

        # 验证初始化
        print(f"初始化后范围 - weight: [{model.head.weight.asnumpy().min():.4f}, {model.head.weight.asnumpy().max():.4f}]")



    # --- 参数冻结策略 ---
    # 方案1：基础冻结（只训练head）
    if args.freeze_layers == "head_only":
        for param in model.get_parameters():
            param.requires_grad = False
        for param in model.head.get_parameters():
            param.requires_grad = True

    # 方案2：部分冻结（训练最后几层+head）
    elif args.freeze_layers == "partial":
        for name, param in model.parameters_and_names():
            if 'blocks.10' in name or 'blocks.11' in name or 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 方案3：不冻结
    else:
        for param in model.get_parameters():
            param.requires_grad = True



    # --- 优化器设置 ---
    # 分层学习率设置
    head_params = [p for p in model.trainable_params() if "head" in p.name]
    backbone_params = [p for p in model.trainable_params() if "head" not in p.name]

    optimizer = nn.SGD([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # 主干网络较小学习率
        {'params': head_params, 'lr': args.lr}  # 分类头较大学习率
    ], momentum=0.9, weight_decay=5e-5)


    # 训练循环
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataset,
            epoch=epoch
        )

        # 验证
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_dataset,
            epoch=epoch
        )

        # 打印结果
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存模型
        mindspore.save_checkpoint(model, f"./weights1/model-{epoch}.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str,
                        default=r"D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\test\data")
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--weights', type=str,
                        default=r'D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\model.ckpt',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=str, default="partial",
                       choices=["head_only", "partial", "none"],
                       help='冻结策略: head_only(仅训练head), partial(最后几层+head), none(全训练)')
    parser.add_argument('--device', default='CPU', help='device id (i.e. GPU or CPU)')

    args = parser.parse_args()
    main(args)