import os
import cv2


def extract_frames(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0
    while True:
        # 逐帧读取
        ret, frame = cap.read()
        if not ret:
            break

        # 保存帧为图片
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # 释放资源
    cap.release()
    print(f"已从 {video_path} 提取 {frame_count} 帧到 {output_folder}")


def process_videos_in_folder(input_folder, output_root):
    # 确保输出根目录存在
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # 检查是否是视频文件（可以根据需要扩展支持的格式）
        if os.path.isfile(filepath) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # 为每个视频创建输出子文件夹（使用文件名作为文件夹名）
            video_name = os.path.splitext(filename)[0]
            output_folder = os.path.join(output_root, video_name)

            # 提取帧
            extract_frames(filepath, output_folder)


if __name__ == "__main__":
    # 设置输入文件夹（包含视频文件）和输出根目录
    input_folder = "path_to_your_videos"  # 替换为你的视频文件夹路径
    output_root = "output_frames"  # 替换为你想要的输出根目录

    process_videos_in_folder(input_folder, output_root)