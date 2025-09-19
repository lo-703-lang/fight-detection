import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_labels(file_path):
    """加载标签文件（每行只有一个0或1）"""
    with open(file_path, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip() in ['0', '1']]
    return np.array(labels)


def evaluate_predictions(true_labels_path, pred_labels_path):
    """
    评估预测结果

    参数:
        true_labels_path: 真实标签文件路径
        pred_labels_path: 模型预测标签文件路径

    返回评估结果字典
    """
    # 加载标签
    y_true = load_labels(true_labels_path)
    y_pred = load_labels(pred_labels_path)

    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # 计算指标
    metrics = {
        'total_frames': min_len,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    return metrics


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Violence'],
                yticklabels=['Normal', 'Violence'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def save_evaluation_report(metrics, output_path):
    """保存评估报告"""
    with open(output_path, 'w') as f:
        f.write("Video Violence Detection Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Frames: {metrics['total_frames']}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"True Negative (TN): {metrics['confusion_matrix'][0][0]}\n")
        f.write(f"False Positive (FP): {metrics['confusion_matrix'][0][1]}\n")
        f.write(f"False Negative (FN): {metrics['confusion_matrix'][1][0]}\n")
        f.write(f"True Positive (TP): {metrics['confusion_matrix'][1][1]}\n")


def main():
    # 文件路径配置
    true_labels_path = r"D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\test\make\output_labels\Assault011_x264_labels.txt"  # 真实标签
    pred_labels_path = r"D:\school_code\workspace_pycharm\PycharmProjects\PythonProject8\test\test\in7\frame_label9.txt"  # 模型预测结果
    report_path = "evaluation_report02.txt"  # 评估报告保存路径

    # 评估预测结果
    print("正在评估模型性能...")
    metrics = evaluate_predictions(true_labels_path, pred_labels_path)

    # 打印结果
    print("\n评估结果:")
    print(f"总帧数: {metrics['total_frames']}")
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"精确率: {metrics['precision']:.2%}")
    print(f"召回率: {metrics['recall']:.2%}")
    print(f"F1分数: {metrics['f1_score']:.2%}")

    # 显示混淆矩阵
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    plot_confusion_matrix(metrics['confusion_matrix'])

    # 保存评估报告
    save_evaluation_report(metrics, report_path)
    print(f"\n评估报告已保存到: {report_path}")


if __name__ == "__main__":
    main()