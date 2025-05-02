# 使用示例

import torch
import numpy as np
from walker_training import ImprovedWalkerDataset, split_dataset, EnhancedWalkerGNN, main

# 1. 准备您的数据
# 假设您已经有了数据格式为[(left_leg_data, right_leg_data), ...]
# 其中每个left_leg_data和right_leg_data都是三维数组

# 如果您尚未准备好数据，可以使用以下代码生成模拟数据
def generate_synthetic_data(n_samples=10000, noise_level=0.2):
    """
    生成模拟的Walker2D关节数据
    """
    # 基础数据
    left_legs = np.random.randn(n_samples, 3) * 0.5  # 3个关节
    
    # 添加一些非线性关系
    base = left_legs.copy()
    right_legs = np.zeros_like(base)
    
    # 关节0: 与左腿关节0有反向关系，加上一些噪声
    right_legs[:, 0] = -0.8 * base[:, 0] + 0.2 * base[:, 1] + np.random.randn(n_samples) * noise_level
    
    # 关节1: 与左腿关节0和1的非线性组合
    right_legs[:, 1] = 0.7 * base[:, 1] + 0.1 * np.sin(base[:, 0] * 2) + np.random.randn(n_samples) * noise_level
    
    # 关节2: 与左腿关节2有强相关性，但有噪声
    right_legs[:, 2] = 0.9 * base[:, 2] + 0.1 * base[:, 0] + np.random.randn(n_samples) * noise_level
    
    # 将数据格式化为[(left, right), ...]格式
    data = [(left, right) for left, right in zip(left_legs, right_legs)]
    
    return data

# 生成或加载您的数据
# 方法1: 生成模拟数据
data = generate_synthetic_data(n_samples=10000, noise_level=0.2)

# 方法2: 从文件加载您的真实数据
"""
# 如果您有保存的数据，可以这样加载
import pickle
with open('your_walker_data.pkl', 'rb') as f:
    data = pickle.load(f)
"""

import pickle
with open('fixed_walker_data.pkl', 'rb') as f:
    data = pickle.load(f)



# 2. 运行完整训练
# 方法1: 使用集成的main函数
model, results = main(
    data,
    use_enhanced_model=True,  # 使用增强型模型
    batch_size=32,            # 批处理大小
    hidden_dim=32,            # 隐藏层维度
    lr=0.001,                 # 学习率
    weight_decay=1e-5,        # 权重衰减
    epochs=3000,              # 训练epoch数
    save_dir="walker_training_results"  # 结果保存目录
)

print(f"训练完成! 测试MSE: {results['test_results']['mse']:.6f}")

# 3. 仅使用测试功能
"""
# 如果您已经有训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据集和加载器
dataset = ImprovedWalkerDataset(data, normalize=True)
_, _, test_loader = split_dataset(dataset, batch_size=32)

# 加载模型
model = EnhancedWalkerGNN(input_dim=3, hidden_dim=32, output_dim=3)
model.load_state_dict(torch.load('walker_training_results/best_model.pth')['model_state_dict'])
model.to(device)
model.eval()

# 导入测试函数
from walker_training import test, analyze_predictions

# 运行测试
test_results = test(model, test_loader, device)
worst_examples = analyze_predictions(test_results, "walker_test_results")
"""

# 4. 使用模型进行预测
"""
# 用训练好的模型进行单个预测
model.eval()

# 准备输入数据 (例如左腿数据)
left_leg_data = torch.tensor([[0.5, 0.2, -0.3]], dtype=torch.float32).to(device)

# 如果您使用了标准化，需要与训练数据用同样的方式标准化
# 假设您保存了数据集的标准化参数
left_leg_normalized = (left_leg_data - torch.tensor(dataset.median, dtype=torch.float32).to(device)) / torch.tensor(dataset.iqr + 1e-8, dtype=torch.float32).to(device)

# 进行预测
with torch.no_grad():
    predicted_right_leg = model(left_leg_normalized)

# 如果需要，反标准化预测结果
predicted_right_leg_denormalized = predicted_right_leg * torch.tensor(dataset.iqr + 1e-8, dtype=torch.float32).to(device) + torch.tensor(dataset.median, dtype=torch.float32).to(device)

print("预测的右腿关节角度:", predicted_right_leg_denormalized.cpu().numpy())
"""