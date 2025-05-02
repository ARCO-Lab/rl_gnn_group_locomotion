import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import time
import os

# 设置随机种子以确保结果可重现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

#-------------------------------------------------------------
# 1. 增强的模型架构
#-------------------------------------------------------------

class EnhancedWalkerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(EnhancedWalkerGNN, self).__init__()
        # 更深更宽的网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 添加残差连接
        self.shortcut = nn.Linear(input_dim, hidden_dim)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 主路径
        main = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        main = self.dropout(main)
        main = F.leaky_relu(self.bn2(self.fc2(main)), negative_slope=0.1)
        main = self.dropout(main)
        main = F.leaky_relu(self.bn3(self.fc3(main)), negative_slope=0.1)
        
        # 残差连接
        shortcut = F.leaky_relu(self.shortcut(x), negative_slope=0.1)
        
        # 合并
        combined = main + shortcut
        output = self.fc4(combined)
        
        return output

# 原始模型，作为比较
class WalkerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WalkerGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#-------------------------------------------------------------
# 2. 改进的数据集处理
#-------------------------------------------------------------

class ImprovedWalkerDataset(Dataset):
    def __init__(self, data, normalize=True, add_noise=False, noise_level=0.01):
        self.data = data
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_level = noise_level
        
        if normalize:
            # 收集所有数据
            all_data = []
            for left, right in data:
                all_data.append(left)
                all_data.append(right)
            all_data = np.vstack(all_data)
            
            # 计算更稳健的统计量
            self.median = np.median(all_data, axis=0)
            self.q75 = np.percentile(all_data, 75, axis=0)
            self.q25 = np.percentile(all_data, 25, axis=0)
            self.iqr = self.q75 - self.q25
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        left, right = self.data[idx]
        
        # 转换为张量
        left_tensor = torch.tensor(left, dtype=torch.float32)
        right_tensor = torch.tensor(right, dtype=torch.float32)
        
        if self.normalize:
            # 使用稳健的标准化方法
            left_tensor = (left_tensor - torch.tensor(self.median, dtype=torch.float32)) / torch.tensor(self.iqr + 1e-8, dtype=torch.float32)
            right_tensor = (right_tensor - torch.tensor(self.median, dtype=torch.float32)) / torch.tensor(self.iqr + 1e-8, dtype=torch.float32)
        
        if self.add_noise:
            # 添加小随机噪声
            left_tensor = left_tensor + torch.randn_like(left_tensor) * self.noise_level
            
        return left_tensor, right_tensor
    
    def to(self, device):
        # 注意：这个方法并不会实际移动数据集本身，只是用于示例
        print(f"数据集将被加载到 {device}")
        return self

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, num_workers=4):
    """
    将数据集划分为训练集、验证集和测试集
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为1"
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # 打乱索引
    np.random.shuffle(indices)
    
    # 计算分割点
    train_split = int(np.floor(train_ratio * dataset_size))
    val_split = int(np.floor((train_ratio + val_ratio) * dataset_size))
    
    # 创建数据加载器索引
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # 使用SubsetRandomSampler创建数据加载器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=test_sampler, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

#-------------------------------------------------------------
# 3. 数据关系分析
#-------------------------------------------------------------

def analyze_data_relationship(dataset):
    """分析左右腿数据的内在关系"""
    all_lefts = []
    all_rights = []
    
    # 收集所有数据
    for i in range(len(dataset)):
        left, right = dataset[i]
        all_lefts.append(left.numpy())
        all_rights.append(right.numpy())
    
    all_lefts = np.vstack(all_lefts)
    all_rights = np.vstack(all_rights)
    
    # 计算相关性
    correlations = []
    for i in range(all_lefts.shape[1]):
        for j in range(all_rights.shape[1]):
            corr = np.corrcoef(all_lefts[:, i], all_rights[:, j])[0, 1]
            correlations.append((i, j, corr))
    
    # 按相关性排序
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("左右腿关节角度相关性分析:")
    for i, j, corr in correlations[:9]:  # 显示所有相关性
        print(f"左腿关节{i} 与 右腿关节{j}: 相关系数 = {corr:.4f}")
    
    # 计算理论上的最佳线性预测MSE
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(all_lefts, all_rights)
    predictions = reg.predict(all_lefts)
    linear_mse = np.mean((predictions - all_rights) ** 2)
    
    print(f"\n最佳线性模型MSE: {linear_mse:.6f}")
    print(f"R^2 分数: {reg.score(all_lefts, all_rights):.6f}")
    
    return linear_mse, reg

#-------------------------------------------------------------
# 4. 梯度健康检查
#-------------------------------------------------------------

def check_gradient_health(model, epoch):
    """综合检查梯度健康状态"""
    print(f"\n===== 第 {epoch} 轮梯度健康检查 =====")
    
    # 检查是否有NaN或Inf
    has_nan = False
    has_inf = False
    
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
            
        grad = param.grad
        
        # 检查NaN和Inf
        if torch.isnan(grad).any():
            has_nan = True
            print(f"❌ 错误: {name} 存在NaN梯度")
            
        if torch.isinf(grad).any():
            has_inf = True
            print(f"❌ 错误: {name} 存在Inf梯度")
        
        # 计算基本统计量
        grad_norm = grad.norm().item()
        param_norm = param.norm().item()
        update_ratio = grad_norm / (param_norm + 1e-8)
        
        layer_stats[name] = {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "update_ratio": update_ratio
        }
    
    if has_nan or has_inf:
        print("❌ 严重问题: 存在NaN或Inf梯度，建议降低学习率或检查数据")
        return False
    
    # 分析统计量
    grad_norms = [stats["grad_norm"] for stats in layer_stats.values()]
    update_ratios = [stats["update_ratio"] for stats in layer_stats.values()]
    
    avg_grad_norm = np.mean(grad_norms)
    max_grad_norm = np.max(grad_norms)
    min_grad_norm = np.min(grad_norms)
    avg_update_ratio = np.mean(update_ratios)
    
    print("\n梯度健康摘要:")
    print(f"平均梯度范数: {avg_grad_norm:.6f}")
    print(f"最大梯度范数: {max_grad_norm:.6f}")
    print(f"最小梯度范数: {min_grad_norm:.6f}")
    print(f"平均更新比例: {avg_update_ratio:.6f}")
    
    # 健康状态判断
    health_status = "良好"
    
    if max_grad_norm > 10.0:
        health_status = "警告"
        print("⚠️ 梯度可能过大，考虑降低学习率")
    
    if min_grad_norm < 1e-3:
        health_status = "警告"
        print("⚠️ 某些层梯度可能过小，网络部分区域可能停止学习")
    
    if avg_update_ratio > 0.1:
        health_status = "警告" 
        print("⚠️ 更新比例过大，可能导致不稳定，考虑降低学习率")
    
    if avg_update_ratio < 1e-4:
        health_status = "警告"
        print("⚠️ 更新比例过小，学习速度可能太慢，考虑增大学习率")
    
    print(f"\n总体健康状态: {health_status}")
    
    return health_status == "良好"

#-------------------------------------------------------------
# 5. 验证函数
#-------------------------------------------------------------

def validate(model, val_loader, loss_fn, device):
    """
    验证模型性能
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # 禁用梯度计算
        for left_leg_data, right_leg_data in val_loader:
            # 将数据移到GPU
            left_leg_data = left_leg_data.to(device)
            right_leg_data = right_leg_data.to(device)
            
            # 前向传播
            output = model(left_leg_data)
            
            # 计算损失
            loss = loss_fn(output, right_leg_data)
            total_loss += loss.item()
            num_batches += 1
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    
    model.train()  # 恢复训练模式
    return avg_loss

#-------------------------------------------------------------
# 6. 测试与分析函数
#-------------------------------------------------------------

def test(model, test_loader, device):
    """
    测试模型性能并计算多种指标
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():  # 禁用梯度计算
        for left_leg_data, right_leg_data in test_loader:
            # 将数据移到GPU
            left_leg_data = left_leg_data.to(device)
            right_leg_data = right_leg_data.to(device)
            
            # 前向传播
            output = model(left_leg_data)
            
            # 收集预测和目标
            all_inputs.append(left_leg_data.cpu().numpy())
            all_preds.append(output.cpu().numpy())
            all_targets.append(right_leg_data.cpu().numpy())
    
    # 转换为numpy数组
    all_inputs = np.vstack(all_inputs)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 计算各种指标
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(mse)
    
    # 计算关节角度误差
    joint_mse = np.mean((all_preds - all_targets) ** 2, axis=0)
    joint_mae = np.mean(np.abs(all_preds - all_targets), axis=0)
    
    print("测试结果:")
    print(f"整体 MSE: {mse:.6f}")
    print(f"整体 MAE: {mae:.6f}")
    print(f"整体 RMSE: {rmse:.6f}")
    print("\n各关节误差:")
    for i in range(len(joint_mse)):
        print(f"关节 {i}: MSE = {joint_mse[i]:.6f}, MAE = {joint_mae[i]:.6f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'joint_mse': joint_mse,
        'joint_mae': joint_mae,
        'predictions': all_preds,
        'targets': all_targets,
        'inputs': all_inputs
    }

def analyze_predictions(test_results, save_dir="."):
    """详细分析预测结果，寻找模型瓶颈"""
    all_inputs = test_results['inputs']
    all_preds = test_results['predictions']
    all_targets = test_results['targets']
    
    # 分析每个关节的误差
    joint_errors = np.mean((all_preds - all_targets) ** 2, axis=0)
    
    print("\n关节预测误差深入分析:")
    for i, error in enumerate(joint_errors):
        print(f"关节 {i}: MSE = {error:.6f}")
    
    # 找出预测最差的样本
    sample_errors = np.mean((all_preds - all_targets) ** 2, axis=1)
    worst_idx = np.argsort(sample_errors)[-5:]  # 最差的5个
    
    print("\n预测最差的5个样本:")
    for idx in worst_idx:
        print(f"样本 {idx}:")
        print(f"  输入: {all_inputs[idx]}")
        print(f"  目标: {all_targets[idx]}")
        print(f"  预测: {all_preds[idx]}")
        print(f"  误差: {sample_errors[idx]:.6f}")
    
    # 绘制预测vs目标散点图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].scatter(all_targets[:, i], all_preds[:, i], alpha=0.5)
        
        # 添加理想线
        min_val = min(np.min(all_targets[:, i]), np.min(all_preds[:, i]))
        max_val = max(np.max(all_targets[:, i]), np.max(all_preds[:, i]))
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axes[i].set_xlabel('实际值')
        axes[i].set_ylabel('预测值')
        axes[i].set_title(f'关节 {i}')
        
        # 计算和显示相关系数
        corr = np.corrcoef(all_targets[:, i], all_preds[:, i])[0, 1]
        axes[i].annotate(f'r = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'))
    print(f"已保存预测分析图表到 {os.path.join(save_dir, 'prediction_analysis.png')}")
    
    # 误差直方图
    plt.figure(figsize=(10, 6))
    plt.hist(sample_errors, bins=50)
    plt.xlabel('样本MSE')
    plt.ylabel('频率')
    plt.title('预测误差分布')
    plt.savefig(os.path.join(save_dir, 'error_histogram.png'))
    print(f"已保存误差直方图到 {os.path.join(save_dir, 'error_histogram.png')}")
    
    return worst_idx

#-------------------------------------------------------------
# 7. 可视化训练过程
#-------------------------------------------------------------

def plot_training_curves(train_losses, val_losses, save_dir="."):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 使用对数尺度更好地查看小的变化
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    print(f"已保存损失曲线到 {os.path.join(save_dir, 'loss_curves.png')}")

#-------------------------------------------------------------
# 8. 训练主函数
#-------------------------------------------------------------

def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, 
               device, epochs=5000, early_stop_patience=300, grad_check_interval=500,
               lr_reset_patience=200, save_dir="checkpoints"):
    """
    训练模型的主函数
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练准备
    start_time = time.time()
    best_val_loss = float('inf')
    no_improve = 0
    train_losses = []
    val_losses = []
    lr_history = []
    
    # 用于学习率重置
    lr_no_improve = 0
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # 训练一个周期
        for left_leg_data, right_leg_data in train_loader:
            optimizer.zero_grad()
            
            # 将数据移到GPU
            left_leg_data = left_leg_data.to(device)
            right_leg_data = right_leg_data.to(device)
            
            # 前向传播
            output = model(left_leg_data)
            
            # 计算损失
            loss = loss_fn(output, right_leg_data)
            epoch_loss += loss.item()
            num_batches += 1
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度检查与调整（可选，用于更激进的训练）
            if epoch < 100:  # 仅在早期训练阶段
                for p in model.parameters():
                    if p.grad is not None and torch.max(torch.abs(p.grad)) < 0.001:
                        p.grad = p.grad * 1.5  # 放大小梯度
            
            optimizer.step()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 在验证集上评估
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        # 更新学习率（如果使用ReduceLROnPlateau）
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 检查是否有改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            lr_no_improve = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': avg_train_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            
        else:
            no_improve += 1
            lr_no_improve += 1
        
        # 打印训练进度
        if epoch % 100 == 0 or epoch == epochs-1:
            time_spent = (time.time() - start_time) / 60.0
            time_per_epoch = time_spent / (epoch + 1) * 60
            est_time_left = (epochs - epoch - 1) * time_per_epoch / 60
            
            print(f"Epoch {epoch}/{epochs-1}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {time_spent:.1f}m, "
                  f"Est. Remaining: {est_time_left:.1f}m")
            
        # 检查梯度是否健康
        if epoch % grad_check_interval == 0 and epoch > 0:
            check_gradient_health(model, epoch)
        
        # 早停
        if no_improve > early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        # 学习率重置检查（如果验证损失长期不下降）
        if lr_no_improve > lr_reset_patience:
            print(f"重置优化器，学习率从 {current_lr:.6f} 提高到 0.005")
            
            # 加载最佳模型
            checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 重新初始化优化器，使用更大的学习率
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
            
            # 重置调度器
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
            else:
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-5)
                
            lr_no_improve = 0
    
    # 训练结束，绘制训练曲线
    plot_training_curves(train_losses, val_losses, save_dir)
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 返回训练历史和最佳模型
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'lr_history': lr_history,
        'best_epoch': checkpoint['epoch'],
    }

#-------------------------------------------------------------
# 9. 主函数
#-------------------------------------------------------------

def main(data, use_enhanced_model=True, batch_size=32, hidden_dim=32, lr=0.001,
         weight_decay=1e-5, epochs=5000, save_dir="walker_training"):
    """
    主函数，集成所有步骤
    """
    print(f"{'='*20} 开始Walker GNN实验 {'='*20}")
    start_time = time.time()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据集
    print("\n[1/6] 准备数据集...")
    dataset = ImprovedWalkerDataset(data, normalize=True, add_noise=True, noise_level=0.005)
    train_loader, val_loader, test_loader = split_dataset(
        dataset, batch_size=batch_size, num_workers=4
    )
    print(f"数据集大小: {len(dataset)}, 训练批次: {len(train_loader)}, "
          f"验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
    
    # 分析数据关系
    print("\n[2/6] 分析数据关系...")
    linear_mse, linear_model = analyze_data_relationship(dataset)
    
    # 初始化模型
    print("\n[3/6] 初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if use_enhanced_model:
        model = EnhancedWalkerGNN(input_dim=3, hidden_dim=hidden_dim, output_dim=3, dropout_rate=0.2)
        print("使用增强型 Walker GNN模型")
    else:
        model = WalkerGNN(input_dim=3, hidden_dim=hidden_dim, output_dim=3)
        print("使用基本 Walker GNN模型")
        
    model.to(device)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 使用周期性学习率调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6)
    # 可选用自适应调度器
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
    
    loss_fn = nn.MSELoss()
    
    # 训练模型
    print("\n[4/6] 开始训练模型...")
    training_results = train_model(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn, device,
        epochs=epochs, save_dir=save_dir
    )
    
    # 测试模型
    print("\n[5/6] 测试模型性能...")
    test_results = test(model, test_loader, device)
    
    # 分析预测
    print("\n[6/6] 分析预测结果...")
    worst_examples = analyze_predictions(test_results, save_dir)
    
    # 保存完整结果
    final_results = {
        'linear_mse': linear_mse,
        'test_results': test_results,
        'training_results': {
            'train_losses': training_results['train_losses'],
            'val_losses': training_results['val_losses'],
            'best_val_loss': training_results['best_val_loss'],
            'lr_history': training_results['lr_history'],
            'best_epoch': training_results['best_epoch'],
        }
    }
    
    # 打印比较结果
    print("\n" + "="*50)
    print(f"训练完成! 总用时: {(time.time() - start_time) / 60:.2f} 分钟")
    print(f"最佳验证损失: {training_results['best_val_loss']:.6f} (Epoch {training_results['best_epoch']})")
    print(f"测试MSE: {test_results['mse']:.6f}")
    print(f"线性基准MSE: {linear_mse:.6f}")
    print(f"相对线性基准改进: {(linear_mse - test_results['mse']) / linear_mse * 100:.2f}%")
    print("="*50)
    
    return model, final_results

#-------------------------------------------------------------
# 示例用法
#-------------------------------------------------------------

# 假设您已经有了数据
"""
# 例如，您的数据可能是这样的
import numpy as np

# 生成一些假数据用于演示
n_samples = 1000
left_legs = np.random.randn(n_samples, 3) * 0.5  # 3个关节
right_legs = 0.8 * left_legs + np.random.randn(n_samples, 3) * 0.2  # 添加一些噪声

data = [(left, right) for left, right in zip(left_legs, right_legs)]

# 运行训练
model, results = main(data, epochs=2000)
"""