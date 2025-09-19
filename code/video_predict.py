# video_predict.py

import os
import json
import cv2
import numpy as np
from PIL import Image
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
from mindcv.models import create_model
from mindspore import context


def main():
    # 设置运行环境
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="GPU" if mindspore.context.get_context("device_target") == "GPU" else "CPU")

    img_size = 224

    # 修改后的数据预处理流程
    def manual_transform(img):
        # 转换为PIL图像
        img = Image.fromarray(img)

        # 调整大小和裁剪
        resize_size = int(img_size * 1.14)
        img = img.resize((resize_size, resize_size))
        left = (resize_size - img_size) // 2
        top = (resize_size - img_size) // 2
        right = left + img_size
        bottom = top + img_size
        img = img.crop((left, top, right, bottom))

        # 转换为numpy数组并归一化
        img_np = np.array(img, dtype=np.float32) / 255.0
        if len(img_np.shape) == 2:  # 灰度图处理
            img_np = np.stack([img_np] * 3, axis=-1)

        # 确保通道顺序为 (H, W, C)
        if img_np.shape[2] != 3:
            img_np = img_np.transpose(1, 2, 0)

        # 手动归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        # 转换为 (C, H, W) 顺序
        img_np = img_np.transpose(2, 0, 1)

        return img_np

    # 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = create_model(
        model_name="vit_b_16_224",
        pretrained=False,
        num_classes=2
    )

    # 加载权重
    model_weight_path = "./weights/model-2.ckpt"
    if os.path.exists(model_weight_path):
        param_dict = mindspore.load_checkpoint(model_weight_path)
        mindspore.load_param_into_net(model, param_dict, strict_load=False)
        print(f"已加载权重: {model_weight_path}")

    model.set_train(False)

    # 视频处理
    video_path = r"D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\test\test\in1\video1.mp4"
    assert os.path.exists(video_path), f"file: '{video_path}' does not exist."

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = r'D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\test\test\in1\out_video2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    label_file_path = r'D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\test\test\in1\frame_label2.txt'

    window_size = 15
    prediction_window = []

    try:
        with open(label_file_path, 'w') as label_file:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"视频结束于第 {frame_index} 帧")
                    break

                if frame_index % 100 == 0:
                    progress = frame_index / total_frames * 100
                    print(f"已处理 {frame_index}/{total_frames} 帧 ({progress:.2f}%)")

                # 转换为RGB并预处理
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = manual_transform(img)

                # 检查形状
                if frame_index == 0:
                    print(f"预处理后视频帧形状: {img_tensor.shape}")  # 应为 (3, 224, 224)

                # 转换为Tensor并增加batch维度
                img_tensor = Tensor(img_tensor[np.newaxis, ...])

                # 预测
                output = model(img_tensor).asnumpy()[0]
                predict = ops.softmax(Tensor(output), axis=0).asnumpy()

                # 滑动窗口处理
                prediction_window.append(predict)
                if len(prediction_window) > window_size:
                    prediction_window.pop(0)

                avg_predict = np.mean(np.array(prediction_window), axis=0)
                prob_non_fighting = avg_predict[0]
                prob_fighting = avg_predict[1]

                # 决策和输出
                threshold = 0.5
                if prob_fighting > threshold:
                    predict_cla = 1
                    result = "Fighting"
                    probability = prob_fighting
                else:
                    predict_cla = 0
                    result = "Non-fighting"
                    probability = prob_non_fighting

                print(f"Frame {frame_index}: {result} ({probability:.3f})")
                label_file.write(f"{predict_cla}\n")

                # 在帧上绘制结果
                cv2.putText(frame, f"Class: {result}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if predict_cla == 0 else (0, 0, 255),
                            2, cv2.LINE_AA)
                cv2.putText(frame, f"Prob: {probability:.3f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),
                            2, cv2.LINE_AA)

                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_index += 1

    except Exception as e:
        print(f"发生异常: {str(e)}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("处理完成")


if __name__ == '__main__':
    main()