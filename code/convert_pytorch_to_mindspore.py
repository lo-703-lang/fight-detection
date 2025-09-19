import torch
import mindspore as ms
import numpy as np
from collections import OrderedDict


def convert_pytorch_to_mindspore(pytorch_ckpt_path, mindspore_ckpt_path):
    # 加载PyTorch权重
    pytorch_ckpt = torch.load(pytorch_ckpt_path, map_location='cpu', weights_only=True)

    # 创建MindSpore参数列表
    ms_params = []

    # 150
    name_mapping = {
        # 基础组件（完全一致，直接保留）
        'patch_embed.proj.weight': 'patch_embed.proj.weight',
        'patch_embed.proj.bias': 'patch_embed.proj.bias',
        'cls_token': 'cls_token',
        'pos_embed': 'pos_embed',

        # 注意力层归一化（PyTorch的norm1 → MindSpore的norm1）
        'blocks.0.norm1.weight': 'blocks.0.norm1.gamma',
        'blocks.0.norm1.bias': 'blocks.0.norm1.beta',
        'blocks.1.norm1.weight': 'blocks.1.norm1.gamma',
        'blocks.1.norm1.bias': 'blocks.1.norm1.beta',
        'blocks.2.norm1.weight': 'blocks.2.norm1.gamma',
        'blocks.2.norm1.bias': 'blocks.2.norm1.beta',
        'blocks.3.norm1.weight': 'blocks.3.norm1.gamma',
        'blocks.3.norm1.bias': 'blocks.3.norm1.beta',
        'blocks.4.norm1.weight': 'blocks.4.norm1.gamma',
        'blocks.4.norm1.bias': 'blocks.4.norm1.beta',
        'blocks.5.norm1.weight': 'blocks.5.norm1.gamma',
        'blocks.5.norm1.bias': 'blocks.5.norm1.beta',
        'blocks.6.norm1.weight': 'blocks.6.norm1.gamma',
        'blocks.6.norm1.bias': 'blocks.6.norm1.beta',
        'blocks.7.norm1.weight': 'blocks.7.norm1.gamma',
        'blocks.7.norm1.bias': 'blocks.7.norm1.beta',
        'blocks.8.norm1.weight': 'blocks.8.norm1.gamma',
        'blocks.8.norm1.bias': 'blocks.8.norm1.beta',
        'blocks.9.norm1.weight': 'blocks.9.norm1.gamma',
        'blocks.9.norm1.bias': 'blocks.9.norm1.beta',
        'blocks.10.norm1.weight': 'blocks.10.norm1.gamma',
        'blocks.10.norm1.bias': 'blocks.10.norm1.beta',
        'blocks.11.norm1.weight': 'blocks.11.norm1.gamma',
        'blocks.11.norm1.bias': 'blocks.11.norm1.beta',

        # 注意力层QKV与投影（参数名完全一致，直接保留）
        'blocks.0.attn.qkv.weight': 'blocks.0.attn.qkv.weight',
        'blocks.0.attn.qkv.bias': 'blocks.0.attn.qkv.bias',
        'blocks.0.attn.proj.weight': 'blocks.0.attn.proj.weight',
        'blocks.0.attn.proj.bias': 'blocks.0.attn.proj.bias',
        'blocks.1.attn.qkv.weight': 'blocks.1.attn.qkv.weight',
        'blocks.1.attn.qkv.bias': 'blocks.1.attn.qkv.bias',
        'blocks.1.attn.proj.weight': 'blocks.1.attn.proj.weight',
        'blocks.1.attn.proj.bias': 'blocks.1.attn.proj.bias',
        # ... 其余blocks的attn参数名完全一致，无需修改（省略重复部分）

        # FFN层归一化（PyTorch的norm2 → MindSpore的norm2）
        'blocks.0.norm2.weight': 'blocks.0.norm2.gamma',
        'blocks.0.norm2.bias': 'blocks.0.norm2.beta',
        'blocks.1.norm2.weight': 'blocks.1.norm2.gamma',
        'blocks.1.norm2.bias': 'blocks.1.norm2.beta',
        'blocks.2.norm2.weight': 'blocks.2.norm2.gamma',
        'blocks.2.norm2.bias': 'blocks.2.norm2.beta',
        # ... 其余blocks的norm2参数以此类推（省略重复部分）

        # FFN层（参数名完全一致，直接保留）
        'blocks.0.mlp.fc1.weight': 'blocks.0.mlp.fc1.weight',
        'blocks.0.mlp.fc1.bias': 'blocks.0.mlp.fc1.bias',
        'blocks.0.mlp.fc2.weight': 'blocks.0.mlp.fc2.weight',
        'blocks.0.mlp.fc2.bias': 'blocks.0.mlp.fc2.bias',
        'blocks.1.mlp.fc1.weight': 'blocks.1.mlp.fc1.weight',
        'blocks.1.mlp.fc1.bias': 'blocks.1.mlp.fc1.bias',
        # ... 其余blocks的mlp参数名完全一致（省略重复部分）

        # 最终归一化层（PyTorch的norm → MindSpore的norm）
        'norm.weight': 'norm.gamma',
        'norm.bias': 'norm.beta',

        # 分类头（参数名完全一致，直接保留）
        'head.weight': 'head.weight',
        'head.bias': 'head.bias',

        # 针对所有blocks的norm2层（包括3~11）
        'blocks.3.norm2.weight': 'blocks.3.norm2.gamma',
        'blocks.3.norm2.bias': 'blocks.3.norm2.beta',
        'blocks.4.norm2.weight': 'blocks.4.norm2.gamma',
        'blocks.4.norm2.bias': 'blocks.4.norm2.beta',
        'blocks.5.norm2.weight': 'blocks.5.norm2.gamma',
        'blocks.5.norm2.bias': 'blocks.5.norm2.beta',
        'blocks.6.norm2.weight': 'blocks.6.norm2.gamma',
        'blocks.6.norm2.bias': 'blocks.6.norm2.beta',
        'blocks.7.norm2.weight': 'blocks.7.norm2.gamma',
        'blocks.7.norm2.bias': 'blocks.7.norm2.beta',
        'blocks.8.norm2.weight': 'blocks.8.norm2.gamma',
        'blocks.8.norm2.bias': 'blocks.8.norm2.beta',
        'blocks.9.norm2.weight': 'blocks.9.norm2.gamma',
        'blocks.9.norm2.bias': 'blocks.9.norm2.beta',
        'blocks.10.norm2.weight': 'blocks.10.norm2.gamma',
        'blocks.10.norm2.bias': 'blocks.10.norm2.beta',
        'blocks.11.norm2.weight': 'blocks.11.norm2.gamma',
        'blocks.11.norm2.bias': 'blocks.11.norm2.beta'


    }

    for pt_name, pt_param in pytorch_ckpt.items():
        # 初始化MindSpore参数名
        ms_name = pt_name

        # 应用名称映射
        for k, v in name_mapping.items():
            if k in ms_name:
                ms_name = ms_name.replace(k, v)

        # 特殊处理：调整参数形状（如果需要）
        if 'qkv.weight' in pt_name or 'qkv.bias' in pt_name:
            # 可能需要将qkv权重拆分为q,k,v三个部分
            pass  # 根据实际模型结构调整

        # 转换参数值
        param_dict = {
            'name': ms_name,
            'data': ms.Tensor(pt_param.numpy())
        }
        ms_params.append(param_dict)

    # 保存为MindSpore格式
    ms.save_checkpoint(ms_params, mindspore_ckpt_path)
    print(f"成功转换 {len(ms_params)}/{len(pytorch_ckpt)} 个参数")
    print(f"权重已保存到: {mindspore_ckpt_path}")


if __name__ == "__main__":
    pytorch_ckpt_path = r"D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\pytorch_model.bin"
    mindspore_ckpt_path = r"D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\pytorch_model.ckpt"
    convert_pytorch_to_mindspore(pytorch_ckpt_path, mindspore_ckpt_path)

    ms_params = ms.load_checkpoint(mindspore_ckpt_path)
    for name, param in ms_params.items():
        print(f"{name}: {param.shape}")