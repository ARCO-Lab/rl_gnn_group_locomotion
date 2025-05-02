import pickle
with open('walker_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data))

for i in range(len(data)):
    print(data[i])
    break



import pickle
import torch
import numpy as np

# 假设您的原始数据存储在变量data中
# 创建新的修复后的数据列表
fixed_data = []

for item in data:
    # 处理张量格式
    left_tensor, right_tensor = item
    
    # 去除多余维度并转换为NumPy数组
    left_array = left_tensor.squeeze().detach().cpu().numpy()
    right_array = right_tensor.squeeze().detach().cpu().numpy()
    
    # 创建元组格式并添加到新列表
    fixed_data.append((left_array, right_array))

# 打印第一个元素以验证格式
print("转换前:", data[0])
print("转换后:", fixed_data[0])

# 保存修复后的数据
with open('fixed_walker_data.pkl', 'wb') as f:
    pickle.dump(fixed_data, f)

print(f"已保存修复后的数据到fixed_walker_data.pkl，共{len(fixed_data)}个样本")
with open('fixed_walker_data.pkl', 'rb') as f:
    new_data = pickle.load(f)

print("新数据第一个样本:", new_data[0])
print("数据类型:", type(new_data[0]))
print("左腿数据类型:", type(new_data[0][0]))
print("左腿数据形状:", new_data[0][0].shape)


# import numpy as np

# def generate_synthetic_data(n_samples=10000, noise_level=0.2):
#     """
#     生成模拟的Walker2D关节数据
#     """
#     # 基础数据
#     left_legs = np.random.randn(n_samples, 3) * 0.5  # 3个关节
    
#     # 添加一些非线性关系
#     base = left_legs.copy()
#     right_legs = np.zeros_like(base)
    
#     # 关节0: 与左腿关节0有反向关系，加上一些噪声
#     right_legs[:, 0] = -0.8 * base[:, 0] + 0.2 * base[:, 1] + np.random.randn(n_samples) * noise_level
    
#     # 关节1: 与左腿关节0和1的非线性组合
#     right_legs[:, 1] = 0.7 * base[:, 1] + 0.1 * np.sin(base[:, 0] * 2) + np.random.randn(n_samples) * noise_level
    
#     # 关节2: 与左腿关节2有强相关性，但有噪声
#     right_legs[:, 2] = 0.9 * base[:, 2] + 0.1 * base[:, 0] + np.random.randn(n_samples) * noise_level
    
#     # 将数据格式化为[(left, right), ...]格式
#     data = [(left, right) for left, right in zip(left_legs, right_legs)]
    
#     return data

# # 生成或加载您的数据
# # 方法1: 生成模拟数据
# data = generate_synthetic_data(n_samples=10000, noise_level=0.2)
# print(data[0])