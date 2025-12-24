"""
测试 MobileFaceNet 在 L2 归一化之前的特征向量Norm
用于判断模型训练是否真的有问题
"""
import cv2
import torch
import numpy as np
from model import MobileFaceNet
import sys


def test_model_before_normalization(checkpoint_path, test_image_path):
    """
    Hook到bn层之后，获取L2归一化之前的特征向量

    Args:
        checkpoint_path: .pth 模型权重文件
        test_image_path: 测试图片（112x112 人脸）
    """
    print("=" * 60)
    print("MobileFaceNet 归一化前特征测试")
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

    # 2. 准备输入
    print(f"\n加载测试图片: {test_image_path}")

    img_bgr = cv2.imread(test_image_path)
    if img_bgr is None:
        print(f"错误: 无法读取图片 {test_image_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"图片尺寸: {img_rgb.shape[:2][::-1]}  (W, H)")

    if img_rgb.shape[:2] != (112, 112):
        print(f"警告: 图片尺寸不是 112x112，正在 resize...")
        img_rgb = cv2.resize(img_rgb, (112, 112))

    # 预处理
    img_float = img_rgb.astype(np.float32) / 255.0
    img_normalized = (img_float - 0.5) / 0.5
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)

    print(f"输入 Tensor 形状: {img_tensor.shape}")
    print(f"输入 Tensor 范围: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

    # 3. 使用Hook获取bn层之后的特征（L2归一化之前）
    print("\n开始推理（Hook到bn层）...")

    feature_before_norm = None

    def hook_fn(module, input, output):
        nonlocal feature_before_norm
        feature_before_norm = output.detach().cpu().numpy()[0]  # (512,)

    # 注册hook到bn层（model.bn是最后的BatchNorm1d层）
    hook_handle = model.bn.register_forward_hook(hook_fn)

    with torch.no_grad():
        output_after_norm = model(img_tensor)  # 这个是L2归一化后的

    hook_handle.remove()

    if feature_before_norm is None:
        print("错误: 未能捕获bn层输出")
        return None, None

    print(f"✓ 推理成功")

    # 4. 分析归一化前的特征
    raw_norm_before = np.linalg.norm(feature_before_norm)

    print("\n" + "=" * 60)
    print("【归一化前】特征向量统计（bn层输出）")
    print("=" * 60)
    print(f"⭐ Raw Norm (这才是真正的原始模长): {raw_norm_before:.4f}")
    print(f"特征范围: [{feature_before_norm.min():.4f}, {feature_before_norm.max():.4f}]")
    print(f"特征均值: {feature_before_norm.mean():.4f}")
    print(f"特征标准差: {feature_before_norm.std():.4f}")
    print(f"前20个特征值: {feature_before_norm[:20]}")

    # 5. 分析归一化后的特征（作为对比）
    feature_after_norm = output_after_norm.cpu().numpy()[0]
    raw_norm_after = np.linalg.norm(feature_after_norm)

    print("\n" + "=" * 60)
    print("【归一化后】特征向量统计（model输出）")
    print("=" * 60)
    print(f"Raw Norm: {raw_norm_after:.4f}  (应该约为1.0)")
    print(f"特征范围: [{feature_after_norm.min():.4f}, {feature_after_norm.max():.4f}]")
    print(f"前20个特征值: {feature_after_norm[:20]}")

    # 6. 诊断结果
    print("\n" + "=" * 60)
    print("🔍 诊断结果")
    print("=" * 60)

    if raw_norm_before < 1.0:
        print("❌ 严重问题: 归一化前 Norm < 1.0")
        print("   -> 模型权重几乎全为0，可能是：")
        print("      1. 模型未训练（随机初始化）")
        print("      2. 权重文件损坏")
        print("      3. 加载了错误的checkpoint")
        status = "FAILED"
    elif raw_norm_before < 5.0:
        print("⚠️  异常: 归一化前 Norm < 5.0")
        print(f"   -> 当前 Norm = {raw_norm_before:.4f}")
        print("   -> 可能原因：")
        print("      1. 模型训练严重不足（epoch太少）")
        print("      2. 训练时loss没有收敛")
        print("      3. 学习率设置不当")
        print("\n   ⚠️  RKNN转换后问题会更严重!")
        status = "WARNING"
    elif raw_norm_before < 10.0:
        print("⚠️  略低: 归一化前 Norm < 10.0")
        print(f"   -> 当前 Norm = {raw_norm_before:.4f}")
        print("   -> 模型基本可用，但训练可能不充分")
        print("   -> 建议继续训练或使用预训练权重")
        status = "OK"
    else:
        print(f"✅ 正常: 归一化前 Norm = {raw_norm_before:.4f}")
        print("   -> 模型训练状态良好")
        print("   -> 如果RKNN部署有问题，应该是转换/预处理的问题")
        status = "GOOD"

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    if status == "FAILED":
        print("❌ PyTorch模型本身有严重问题，需要：")
        print("   1. 检查.pth文件是否正确")
        print("   2. 确认模型架构与checkpoint匹配")
        print("   3. 重新训练模型")
    elif status == "WARNING":
        print("⚠️  PyTorch模型训练不足，建议：")
        print("   1. 继续训练更多epoch")
        print("   2. 使用预训练权重fine-tune")
        print("   3. RKNN转换可能会让问题更明显")
    elif status == "OK":
        print("✓ PyTorch模型基本正常")
        print("  如果RKNN推理不准确，重点排查：")
        print("  - 预处理参数配置")
        print("  - 量化校准数据集")
    else:  # GOOD
        print("✅ PyTorch模型训练良好!")
        print("  如果RKNN推理不准确，问题在于：")
        print("  1. ONNX转换精度损失")
        print("  2. RKNN量化精度损失")
        print("  3. C++端预处理不匹配（已修复）")
    print("=" * 60)

    return raw_norm_before, feature_before_norm


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python test_model_before_norm.py <model.pth> <test_image.jpg>")
        print("\n示例:")
        print("  python test_model_before_norm.py work_space/models/mobilefacenet_final.pth ../project/face_app/imgs/1.jpg")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    test_image_path = sys.argv[2]

    raw_norm, feature = test_model_before_normalization(checkpoint_path, test_image_path)
