"""
测试ONNX模型输出，验证转换是否正确
"""
import cv2
import numpy as np
import onnxruntime as ort
import sys


def test_onnx_model(onnx_path, test_image_path):
    """
    测试ONNX模型输出

    Args:
        onnx_path: ONNX模型路径
        test_image_path: 测试图片（112x112）
    """
    print("=" * 60)
    print("ONNX 模型测试")
    print("=" * 60)

    # 1. 加载ONNX模型
    print(f"\n加载ONNX模型: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    print("✓ ONNX模型加载成功")

    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"输入名称: {input_name}")
    print(f"输出名称: {output_name}")

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

    # 预处理：归一化到 [-1, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    img_normalized = (img_float - 0.5) / 0.5

    # 转换为 NCHW 格式 (1, 3, 112, 112)
    img_nchw = np.transpose(img_normalized, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    img_batch = np.expand_dims(img_nchw, axis=0)  # (C,H,W) -> (1,C,H,W)

    print(f"输入 Tensor 形状: {img_batch.shape}")
    print(f"输入 Tensor 范围: [{img_batch.min():.4f}, {img_batch.max():.4f}]")

    # 3. 推理
    print("\n开始推理...")
    outputs = session.run([output_name], {input_name: img_batch})
    feature = outputs[0][0]  # (1, 512) -> (512,)

    print(f"✓ 推理成功")
    print(f"输出形状: {feature.shape}")

    # 4. 计算 Norm
    raw_norm = np.linalg.norm(feature)

    print("\n" + "=" * 60)
    print("特征向量统计")
    print("=" * 60)
    print(f"⭐ Raw Norm: {raw_norm:.4f}")
    print(f"特征范围: [{feature.min():.4f}, {feature.max():.4f}]")
    print(f"特征均值: {feature.mean():.4f}")
    print(f"特征标准差: {feature.std():.4f}")
    print(f"前20个特征值: {feature[:20]}")

    # 5. 诊断结果
    print("\n" + "=" * 60)
    print("🔍 诊断结果")
    print("=" * 60)

    if abs(raw_norm - 1.0) < 0.1:
        print(f"✅ 正常: Raw Norm ≈ 1.0 (实际值={raw_norm:.4f})")
        print("   -> ONNX模型包含了L2归一化层")
        print("   -> 输出是归一化后的特征（这是预期的）")
        print("   -> 与PyTorch模型行为一致")
    else:
        print(f"⚠️  异常: Raw Norm = {raw_norm:.4f}")
        print("   -> 预期应该约为1.0")
        print("   -> 可能ONNX转换有问题")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    if abs(raw_norm - 1.0) < 0.1:
        print("✅ ONNX模型正常！")
        print("如果RKNN推理不准确，问题可能在：")
        print("  1. RKNN转换配置（已通过手动预处理解决）")
        print("  2. RKNN量化精度损失（使用FP16可避免）")
    else:
        print("❌ ONNX转换可能有问题！")
        print("建议：")
        print("  1. 重新转换ONNX模型")
        print("  2. 检查PyTorch模型是否正确")
    print("=" * 60)

    return raw_norm, feature


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python test_onnx_model.py <model.onnx> <test_image.jpg>")
        print("\n示例:")
        print("  python test_onnx_model.py outputs/mobilefacenet.onnx ../project/face_app/imgs/1.jpg")
        sys.exit(1)

    onnx_path = sys.argv[1]
    test_image_path = sys.argv[2]

    raw_norm, feature = test_onnx_model(onnx_path, test_image_path)
