import os
import numpy as np
import cv2
import chardet


def detect_file_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']


def parse_annotations(annotation_path, target_video):
    """解析标注文件"""
    encoding = detect_file_encoding(annotation_path)
    print(f"检测到文件编码: {encoding}")

    time_ranges = []
    with open(annotation_path, 'r', encoding=encoding, errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == target_video:
                i = 1
                while 2 * i + 1 < len(parts):
                    try:
                        start = int(parts[2 * i])
                        end = int(parts[2 * i + 1])
                        if start != -1 and end != -1:
                            time_ranges.append((start, end))
                    except ValueError:
                        pass
                    i += 1
    return time_ranges


def generate_frame_labels(video_path, frame_ranges):
    """生成逐帧标签"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    labels = np.zeros(total_frames, dtype=np.uint8)

    for start, end in frame_ranges:
        start = max(0, min(start, total_frames - 1))
        end = max(0, min(end, total_frames - 1))
        labels[start:end + 1] = 1

    return labels, fps, total_frames


def save_simple_labels(video_name, labels, output_dir):
    """
    保存简化版标签文件（只有0/1）
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(video_name)[0]

    # 保存为TXT（纯0/1）
    txt_path = os.path.join(output_dir, f"{base_name}_labels.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label}\n")  # 每行只有一个数字

    print(f"标签文件已保存: {txt_path}")
    return txt_path


def process_video(annotation_file, video_name, video_dir, output_dir):
    """处理单个视频"""
    try:
        print(f"\n{'=' * 40}\n处理视频: {video_name}")

        # 1. 解析标注
        frame_ranges = parse_annotations(annotation_file, video_name)
        if not frame_ranges:
            print(f"警告: 未找到 {video_name} 的标注信息")
            return None

        print(f"找到 {len(frame_ranges)} 个异常时间段:")
        for i, (start, end) in enumerate(frame_ranges, 1):
            print(f"  {i}. 帧 {start}-{end}")

        # 2. 生成标签
        video_path = os.path.join(video_dir, video_name)
        labels, fps, total_frames = generate_frame_labels(video_path, frame_ranges)

        # 3. 保存简化版标签
        label_file = save_simple_labels(video_name, labels, output_dir)

        return {
            'video': video_name,
            'total_frames': total_frames,
            'violent_frames': sum(labels),
            'label_file': label_file
        }

    except Exception as e:
        print(f"\n处理失败: {str(e)}")
        return None


if __name__ == "__main__":
    # 配置参数
    annotation_file = "Temporal_Anomaly_Annotation.txt"
    video_name = "Assault011_x264.mp4"
    video_dir = "test_videos"
    output_dir = "output_labels"

    # 执行处理
    result = process_video(annotation_file, video_name, video_dir, output_dir)

    if result:
        print("\n处理成功完成!")
        print(f"生成的标签文件: {result['label_file']}")