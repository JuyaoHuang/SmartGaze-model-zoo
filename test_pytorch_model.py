"""
测试原始 PyTorch 模型的输出特征向量 Norm
用于诊断问题是否出在模型训练或 RKNN 转换上
"""
import cv2
import torch
import numpy as np
from model import MobileFaceNet
import sys

def test_pytorch_model(checkpoint_path, test_image_path):
    """
    测试 PyTorch 模型输出的特征向量 Norm

    Args:
        checkpoint_path: .pth 模型权重文件
        test_image_path: 测试图片（112x112 人脸）
    """
    print("=" * 60)
    print("PyTorch MobileFaceNet 模型测试")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = MobileFaceNet(embedding_size=512)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    print("✓ 模型加载成功")

    # 2. 准备输入（与训练时相同的预处理）
    print(f"\n加载测试图片: {test_image_path}")

    # 使用 OpenCV 读取图片（BGR格式）
    img_bgr = cv2.imread(test_image_path)
    if img_bgr is None:
        print(f"错误: 无法读取图片 {test_image_path}")
        sys.exit(1)

    # 转换为 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"图片尺寸: {img_rgb.shape[:2][::-1]}  (W, H)")

    # 如果不是 112x112，需要 resize
    if img_rgb.shape[:2] != (112, 112):
        print(f"警告: 图片尺寸不是 112x112，正在 resize...")
        img_rgb = cv2.resize(img_rgb, (112, 112))

    # 手动实现预处理（等价于 torchvision.transforms）
    # 1. 转换为浮点数并归一化到 [0, 1] (等价于 ToTensor)
    img_float = img_rgb.astype(np.float32) / 255.0

    # 2. 归一化到 [-1, 1] (等价于 Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    # 公式: output = (input - mean) / std
    img_normalized = (img_float - 0.5) / 0.5

    # 3. 转换为 (C, H, W) 格式并转为 Tensor
    img_chw = np.transpose(img_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)  # (1, 3, 112, 112)

    print(f"输入 Tensor 形状: {img_tensor.shape}")
    print(f"输入 Tensor 范围: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

    # 3. 推理
    print("\n开始推理...")
    with torch.no_grad():
        output = model(img_tensor)

    print(f"✓ 推理成功")
    print(f"输出形状: {output.shape}")

    # 4. 计算 Norm（L2 Norm，归一化前）
    feature = output.cpu().numpy()[0]  # (512,)

    # 计算原始 Norm（这是关键指标！）
    raw_norm = np.linalg.norm(feature)

    print("\n" + "=" * 60)
    print("特征向量统计")
    print("=" * 60)
    print(f"原始 Norm (归一化前): {raw_norm:.4f}")
    print(f"特征范围: [{feature.min():.4f}, {feature.max():.4f}]")
    print(f"特征均值: {feature.mean():.4f}")
    print(f"特征标准差: {feature.std():.4f}")

    # L2 归一化
    feature_normalized = feature / (raw_norm + 1e-10)
    normalized_norm = np.linalg.norm(feature_normalized)

    print(f"\nL2 归一化后 Norm: {normalized_norm:.4f}")
    print(f"归一化后范围: [{feature_normalized.min():.4f}, {feature_normalized.max():.4f}]")

    # 5. 诊断结果
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)

    if raw_norm < 1.0:
        print("❌ 严重问题: Raw Norm < 1.0")
        print("   -> 模型权重文件可能损坏或未训练")
        print("   -> 建议检查 .pth 文件是否正确")
    elif raw_norm < 5.0:
        print("⚠️  异常: Raw Norm < 5.0")
        print(f"   -> 当前 Norm = {raw_norm:.4f}")
        print("   -> 模型可能训练不足或权重有问题")
    elif raw_norm < 10.0:
        print("⚠️  略低: Raw Norm < 10.0")
        print(f"   -> 当前 Norm = {raw_norm:.4f}")
        print("   -> 模型基本正常，但可能还有优化空间")
    else:
        print(f"✓ 正常: Raw Norm = {raw_norm:.4f}")
        print("   -> 模型输出正常")
        print("   -> 问题可能出在 ONNX 转换或 RKNN 部署上")

    print("\n前 20 个特征值:")
    print(feature[:20])

    return raw_norm, feature


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python test_pytorch_model.py <model.pth> <test_image.jpg>")
        print("\n示例:")
        print("  python test_pytorch_model.py work_space/models/mobilefacenet_final.pth test_face.jpg")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    test_image_path = sys.argv[2]

    raw_norm, feature = test_pytorch_model(checkpoint_path, test_image_path)

    print("\n" + "=" * 60)
    if raw_norm >= 10.0:
        print("结论: PyTorch 模型正常，问题在 ONNX/RKNN 转换")
    else:
        print("结论: PyTorch 模型本身有问题，需要重新训练")
    print("=" * 60)
