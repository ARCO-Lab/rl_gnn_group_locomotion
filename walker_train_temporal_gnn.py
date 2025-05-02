import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import time
import os

# 导入图神经网络相关库
try:
    from torch_geometric.nn import GCNConv, GATConv, RGCNConv
    from torch_geometric.data import Data, Batch
    # 导入时序GNN处理工具
    from torch_geometric.nn import GatedGraphConv
    # 移除TGCN导入，该模块可能在你的PyTorch Geometric版本中不可用
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    raise ImportError("请安装 PyTorch Geometric 库以使用图神经网络。运行: pip install torch-geometric torch-scatter torch-sparse")

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
# 1. 时序图结构定义
#-------------------------------------------------------------

class TemporalWalkerGraphBuilder:
    def __init__(self, sequence_length=5):
        """
        初始化时序图结构生成器，定义人体腿部关节的连接关系
        
        参数:
        - sequence_length: 时间序列长度，表示考虑多少个历史时间步
        """
        # 定义图的结构 - 假设我们有6个节点（左右腿各3个关节）
        self.num_nodes = 6
        self.sequence_length = sequence_length
        
        # 定义空间边连接 - 按照人体骨骼结构
        self.spatial_edge_index_src = [
            0, 1,  # 左腿：大腿→小腿，小腿→脚
            3, 4,  # 右腿：大腿→小腿，小腿→脚
            0, 3,  # 左右腿大腿之间的关系
            1, 4,  # 左右腿小腿之间的关系
            2, 5   # 左右腿脚之间的关系
        ]
        
        # 目标节点
        self.spatial_edge_index_dst = [
            1, 2,  # 左腿连接
            4, 5,  # 右腿连接
            3, 0,  # 左右腿大腿连接（双向）
            4, 1,  # 左右腿小腿连接（双向）
            5, 2   # 左右腿脚连接（双向）
        ]
        
        # 转换为PyTorch张量 - 空间边
        self.spatial_edge_index = torch.tensor([self.spatial_edge_index_src, self.spatial_edge_index_dst], 
                                           dtype=torch.long)
        
        # 创建时间边 - 连接不同时间步的相同节点
        self._build_temporal_edges()
    
    def _build_temporal_edges(self):
        """构建时间边连接不同时间步的相同节点"""
        temporal_edges_src = []
        temporal_edges_dst = []
        
        # 对于每个时间步t和每个节点，连接到t+1时间步的相同节点
        for t in range(self.sequence_length - 1):
            base_t = t * self.num_nodes  # 当前时间步的基础索引
            base_t_next = (t + 1) * self.num_nodes  # 下一个时间步的基础索引
            
            for node in range(self.num_nodes):
                # t时间步的节点连接到t+1时间步的相同节点
                temporal_edges_src.append(base_t + node)
                temporal_edges_dst.append(base_t_next + node)
                
                # 可选：添加更多时间边连接，例如连接到t+2, t+3等
                # 这里只连接到t+1
        
        self.temporal_edge_index = torch.tensor([temporal_edges_src, temporal_edges_dst], 
                                               dtype=torch.long)
        
        # 边的类型: 0=空间边, 1=时间边
        self.edge_type = torch.zeros(len(self.spatial_edge_index_src) + len(temporal_edges_src), 
                                    dtype=torch.long)
        self.edge_type[len(self.spatial_edge_index_src):] = 1
        
        # 组合空间和时间边
        self.combined_edge_index = torch.cat([
            self.spatial_edge_index, 
            self.temporal_edge_index
        ], dim=1)
    
    def build_temporal_graph_from_sequence(self, left_leg_sequence, right_leg_sequence):
        """
        将左右腿的时间序列数据构建为时序图
        
        参数:
        - left_leg_sequence: 左腿序列数据 [seq_len, 3]
        - right_leg_sequence: 右腿序列数据 [seq_len, 3]
        
        返回:
        - torch_geometric.data.Data 对象
        """
        # 确保输入是正确的格式和长度
        if isinstance(left_leg_sequence, torch.Tensor):
            left_leg_sequence = left_leg_sequence.detach().cpu().numpy()
        if isinstance(right_leg_sequence, torch.Tensor):
            right_leg_sequence = right_leg_sequence.detach().cpu().numpy()
            
        # 截断或填充序列以匹配所需的序列长度
        seq_len = min(len(left_leg_sequence), self.sequence_length)
        
        # 准备节点特征 - 每个时间步的每个节点一个特征
        node_features = []
        
        for t in range(seq_len):
            # 获取当前时间步的左右腿数据
            left_leg_t = left_leg_sequence[t]
            right_leg_t = right_leg_sequence[t]
            
            # 添加当前时间步的节点特征
            node_features.append(torch.tensor(left_leg_t, dtype=torch.float32).view(-1, 1))
            node_features.append(torch.tensor(right_leg_t, dtype=torch.float32).view(-1, 1))
        
        # 如果序列长度小于所需长度，用最后一个时间步的数据填充
        if seq_len < self.sequence_length:
            last_left = left_leg_sequence[-1]
            last_right = right_leg_sequence[-1]
            
            for _ in range(self.sequence_length - seq_len):
                node_features.append(torch.tensor(last_left, dtype=torch.float32).view(-1, 1))
                node_features.append(torch.tensor(last_right, dtype=torch.float32).view(-1, 1))
        
        # 合并所有节点特征
        x = torch.cat(node_features, dim=0)
        
        # 创建图数据对象，包括时间信息
        graph = Data(
            x=x, 
            edge_index=self.combined_edge_index,
            edge_type=self.edge_type,
            num_nodes=x.size(0)
        )
        
        return graph
    
    def build_batch_temporal_graphs(self, left_leg_sequences, right_leg_sequences, device=None):
        """
        批量构建时序图并合并为一个批次
        
        参数:
        - left_leg_sequences: 批量左腿序列数据 [batch_size, seq_len, 3]
        - right_leg_sequences: 批量右腿序列数据 [batch_size, seq_len, 3]
        - device: 设备 (CPU/GPU)
        
        返回:
        - batched_graphs: 批处理后的图数据
        - stacked_targets: 批量右腿标签数据（最后一个时间步）
        """
        batch_size = left_leg_sequences.size(0)
        graphs = []
        targets = []
        
        for i in range(batch_size):
            left_seq = left_leg_sequences[i]
            right_seq = right_leg_sequences[i]
            
            # 构建单个时序图
            graph = self.build_temporal_graph_from_sequence(left_seq, right_seq)
            graphs.append(graph)
            
            # 保存目标（右腿序列最后一个时间步作为标签）
            if isinstance(right_seq, torch.Tensor):
                targets.append(right_seq[-1])
            else:
                targets.append(torch.tensor(right_seq[-1], dtype=torch.float32))
        
        # 合并为批次图
        batched_graphs = Batch.from_data_list(graphs)
        stacked_targets = torch.stack(targets)
        
        # 移动到指定设备
        if device is not None:
            batched_graphs = batched_graphs.to(device)
            stacked_targets = stacked_targets.to(device)
        
        return batched_graphs, stacked_targets

#-------------------------------------------------------------
# 2. 时序GNN模型定义 (替代原TGCN实现)
#-------------------------------------------------------------

class TemporalWalkerGNN(nn.Module):
    """时序图神经网络模型，处理行走时序数据"""
    
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=3, num_nodes=6, sequence_length=5,
                num_relations=2, num_layers=2):
        super(TemporalWalkerGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.total_nodes = num_nodes * sequence_length
        
        # 使用关系型GCN处理不同类型的边（空间边和时间边）
        self.rgcn1 = RGCNConv(input_dim, hidden_dim, num_relations=num_relations)
        
        # 多层RGCN以增强模型表达能力
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            for _ in range(num_layers - 1)
        ])
        
        # 时间注意力机制 - 在不同时间步之间的注意力
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 最终的输出层
        self.out = nn.Linear(hidden_dim, output_dim)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 应用关系型GCN层
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        
        # 应用额外的RGCN层
        for layer in self.rgcn_layers:
            x = F.relu(layer(x, edge_index, edge_type))
        
        # 处理批次数据
        if batch is not None:
            batch_size = batch.max().item() + 1
            outputs = []
            
            for i in range(batch_size):
                # 找出当前批次样本的所有节点
                mask = (batch == i)
                batch_x = x[mask]
                
                # 检查并打印张量形状信息
                total_expected_nodes = self.sequence_length * self.num_nodes
                
                # 重塑为 [sequence_length, num_nodes, hidden_dim]
                try:
                    batch_x = batch_x.view(self.sequence_length, self.num_nodes, -1)
                except RuntimeError:
                    # 如果形状不匹配，打印详细信息并尝试修复
                    print(f"Shape error - batch_x: {batch_x.shape}, expected: [{self.sequence_length}, {self.num_nodes}, hidden_dim]")
                    # 确保节点数量是预期的
                    if batch_x.size(0) < total_expected_nodes:
                        # 填充缺失节点
                        pad_size = total_expected_nodes - batch_x.size(0)
                        padding = torch.zeros(pad_size, batch_x.size(1), device=batch_x.device)
                        batch_x = torch.cat([batch_x, padding], dim=0)
                    elif batch_x.size(0) > total_expected_nodes:
                        # 截断多余节点
                        batch_x = batch_x[:total_expected_nodes]
                    
                    batch_x = batch_x.view(self.sequence_length, self.num_nodes, -1)
                
                # 应用时间注意力 - 计算每个时间步的权重
                time_scores = self.time_attention(batch_x.mean(dim=1))  # [sequence_length, 1]
                time_weights = F.softmax(time_scores, dim=0)  # [sequence_length, 1]
                
                # 加权聚合所有时间步 - 修复维度不匹配问题
                time_weights = time_weights.unsqueeze(1)  # [sequence_length, 1, 1]
                weighted_batch = batch_x * time_weights  # 使用广播 [sequence_length, num_nodes, hidden_dim]
                weighted_representation = torch.sum(weighted_batch, dim=0)  # [num_nodes, hidden_dim]
                
                # 获取右腿节点（节点3，4，5）
                right_leg_representation = weighted_representation[3:6]  # [3, hidden_dim]
                
                # 聚合右腿节点特征
                aggregated_right_leg = torch.mean(right_leg_representation, dim=0)  # [hidden_dim]
                
                # 应用输出层
                out = self.out(aggregated_right_leg)
                outputs.append(out)
            
            # 堆叠所有输出
            return torch.stack(outputs)
        else:
            # 单个样本处理
            batch_x = x.view(self.sequence_length, self.num_nodes, -1)
            
            # 应用时间注意力
            time_scores = self.time_attention(batch_x.mean(dim=1))
            time_weights = F.softmax(time_scores, dim=0)
            
            # 加权聚合 - 同样修复维度不匹配问题
            time_weights = time_weights.unsqueeze(1)  # [sequence_length, 1, 1]
            weighted_batch = batch_x * time_weights
            weighted_representation = torch.sum(weighted_batch, dim=0)
            
            # 获取右腿节点
            right_leg_representation = weighted_representation[3:6]
            
            # 聚合
            aggregated_right_leg = torch.mean(right_leg_representation, dim=0)
            
            # 输出
            return self.out(aggregated_right_leg).unsqueeze(0)

#-------------------------------------------------------------
# 3. 时序数据集实现
#-------------------------------------------------------------

class TemporalWalkerDataset(Dataset):
    def __init__(self, data, sequence_length=5, stride=1, normalize=True, add_noise=False, noise_level=0.01):
        """
        构建行走时序数据集
        
        参数:
        - data: 原始数据列表 [(left, right), ...]
        - sequence_length: 使用的时间序列长度
        - stride: 滑动窗口步长
        - normalize: 是否标准化数据
        - add_noise: 是否添加噪声
        - noise_level: 噪声水平
        """
        self.original_data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_level = noise_level
        
        # 创建时序序列
        self._create_sequences()
        
        if normalize:
            self._compute_normalization_stats()
    
    def _create_sequences(self):
        """从原始数据创建时间序列"""
        self.sequences = []
        
        # 确保有足够数据形成序列
        if len(self.original_data) < self.sequence_length:
            raise ValueError(f"数据长度 {len(self.original_data)} 小于序列长度 {self.sequence_length}")
        
        # 使用滑动窗口创建序列
        for i in range(0, len(self.original_data) - self.sequence_length + 1, self.stride):
            left_seq = []
            right_seq = []
            
            for j in range(self.sequence_length):
                left, right = self.original_data[i + j]
                
                # 处理不同的数据格式
                if isinstance(left, torch.Tensor):
                    left = left.squeeze().detach().cpu().numpy()
                if isinstance(right, torch.Tensor):
                    right = right.squeeze().detach().cpu().numpy()
                
                left_seq.append(left)
                right_seq.append(right)
            
            self.sequences.append((np.array(left_seq), np.array(right_seq)))
    
    def _compute_normalization_stats(self):
        """计算数据标准化的统计量"""
        all_data = []
        
        for left_seq, right_seq in self.sequences:
            all_data.append(left_seq.reshape(-1, left_seq.shape[-1]))
            all_data.append(right_seq.reshape(-1, right_seq.shape[-1]))
        
        all_data = np.vstack(all_data)
        
        # 使用更稳健的统计量
        self.median = np.median(all_data, axis=0)
        self.q75 = np.percentile(all_data, 75, axis=0)
        self.q25 = np.percentile(all_data, 25, axis=0)
        self.iqr = self.q75 - self.q25
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        left_seq, right_seq = self.sequences[idx]
        
        # 转为PyTorch张量
        left_tensor = torch.FloatTensor(left_seq)
        right_tensor = torch.FloatTensor(right_seq)
        
        if self.normalize:
            # 应用稳健的标准化
            left_tensor = (left_tensor - torch.FloatTensor(self.median)) / torch.FloatTensor(self.iqr + 1e-8)
            right_tensor = (right_tensor - torch.FloatTensor(self.median)) / torch.FloatTensor(self.iqr + 1e-8)
        
        if self.add_noise:
            # 添加随机噪声
            left_tensor = left_tensor + torch.randn_like(left_tensor) * self.noise_level
        
        return left_tensor, right_tensor

#-------------------------------------------------------------
# 4. 优化的时序验证与测试函数
#-------------------------------------------------------------

def temporal_validate(model, val_loader, loss_fn, device, graph_builder):
    """
    验证时序模型性能
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for left_leg_seqs, right_leg_seqs in val_loader:
            # 构建批量时序图
            batched_graphs, stacked_targets = graph_builder.build_batch_temporal_graphs(
                left_leg_seqs, right_leg_seqs, device
            )
            
            # 前向传播
            outputs = model(batched_graphs)
            
            # 确保维度匹配
            if outputs.shape != stacked_targets.shape:
                if len(outputs.shape) > len(stacked_targets.shape):
                    outputs = outputs.squeeze()
                else:
                    stacked_targets = stacked_targets.view(outputs.shape)
            
            # 计算损失
            loss = loss_fn(outputs, stacked_targets)
            total_loss += loss.item()
            num_batches += 1
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    model.train()
    return avg_loss

def temporal_test(model, test_loader, device, graph_builder):
    """
    测试时序模型性能并计算多种指标
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_last_inputs = []  # 保存每个序列的最后一个输入
    
    with torch.no_grad():
        for left_leg_seqs, right_leg_seqs in test_loader:
            # 保存最后一个时间步的左腿数据作为输入特征
            all_last_inputs.append(left_leg_seqs[:, -1].cpu().numpy())
            
            # 保存目标（右腿最后一个时间步）
            all_targets.append(right_leg_seqs[:, -1].cpu().numpy())
            
            # 构建批量时序图
            batched_graphs, _ = graph_builder.build_batch_temporal_graphs(
                left_leg_seqs, right_leg_seqs, device
            )
            
            # 前向传播
            outputs = model(batched_graphs)
            all_preds.append(outputs.cpu().numpy())
    
    # 转换为numpy数组
    all_last_inputs = np.vstack(all_last_inputs)
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
        'last_inputs': all_last_inputs
    }

#-------------------------------------------------------------
# 5. 时序推理函数
#-------------------------------------------------------------

def temporal_gnn_inference(model, left_leg_sequence, graph_builder, device=None, right_leg_dummy=None):
    """
    使用训练好的时序GNN模型进行推理
    
    参数:
    - model: 训练好的时序GNN模型
    - left_leg_sequence: 左腿序列数据 [seq_len, 3]
    - graph_builder: 时序图构建器
    - device: 设备 (CPU/GPU)
    - right_leg_dummy: 占位符，用于构建图
    
    返回:
    - 预测的右腿数据 (3,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # 确保输入格式正确
    if isinstance(left_leg_sequence, np.ndarray):
        left_seq_tensor = torch.tensor(left_leg_sequence, dtype=torch.float32)
    elif isinstance(left_leg_sequence, torch.Tensor):
        left_seq_tensor = left_leg_sequence
    else:
        raise TypeError("输入数据类型必须是NumPy数组或PyTorch张量")
    
    # 为构建图，我们需要一个右腿占位符序列
    if right_leg_dummy is None:
        if isinstance(left_leg_sequence, np.ndarray):
            right_leg_dummy = np.zeros_like(left_leg_sequence)
        else:
            right_leg_dummy = torch.zeros_like(left_seq_tensor)
    
    # 构建时序图
    graph = graph_builder.build_temporal_graph_from_sequence(left_seq_tensor, right_leg_dummy)
    graph = graph.to(device)
    
    # 推理
    with torch.no_grad():
        output = model(graph)
            
    return output.cpu().numpy()

#-------------------------------------------------------------
# 6. 优化的时序训练主函数
#-------------------------------------------------------------

def train_temporal_walker_gnn(model, train_loader, val_loader, optimizer, scheduler, loss_fn, 
                             device, graph_builder, epochs=1000, early_stop_patience=100, 
                             grad_check_interval=200, save_dir="temporal_gnn_model",
                             validate_every=10):
    """
    训练时序图神经网络的主函数
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
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # 训练一个周期
        for left_leg_seqs, right_leg_seqs in train_loader:
            optimizer.zero_grad()
            
            # 构建批量时序图
            batched_graphs, stacked_targets = graph_builder.build_batch_temporal_graphs(
                left_leg_seqs, right_leg_seqs, device
            )
            
            # 前向传播
            outputs = model(batched_graphs)
            
            # 确保维度匹配
            if outputs.shape != stacked_targets.shape:
                if len(outputs.shape) > len(stacked_targets.shape):
                    outputs = outputs.squeeze()
                else:
                    stacked_targets = stacked_targets.view(outputs.shape)
            
            # 计算损失
            loss = loss_fn(outputs, stacked_targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # 定期在验证集上评估
        if epoch % validate_every == 0:
            val_loss = temporal_validate(model, val_loader, loss_fn, device, graph_builder)
            val_losses.append(val_loss)
        else:
            # 重复使用上一次的验证损失
            val_loss = val_losses[-1] if val_losses else float('inf')
            val_losses.append(val_loss)
        
        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau) and epoch % validate_every == 0:
            scheduler.step(val_loss)
        elif not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        
        # 检查是否有改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            
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
        
        # 打印训练进度
        if epoch % 10 == 0 or epoch == epochs-1:
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
    
    # 训练结束，绘制训练曲线
    plot_training_curves(train_losses, val_losses, save_dir)
    

    # 续接前一个代码块结尾
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
# 7. 时序模型的主函数
#-------------------------------------------------------------

def main_temporal(data, hidden_dim=16, sequence_length=5, stride=1, epochs=1000, 
                 batch_size=16, lr=0.001, weight_decay=1e-5, 
                 save_dir="temporal_walker_gnn_results", num_workers=2, 
                 validate_every=10):
    """
    时序GNN的主函数，集成所有步骤
    """
    print(f"{'='*20} 开始Temporal Walker GNN训练 {'='*20}")
    start_time = time.time()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 准备时序数据集
    print("\n[1/6] 准备时序数据集...")
    dataset = TemporalWalkerDataset(
        data, 
        sequence_length=sequence_length, 
        stride=stride,
        normalize=True, 
        add_noise=True, 
        noise_level=0.005
    )
    
    # 划分数据集
    from torch.utils.data import DataLoader, random_split
    
    # 计算分割大小
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # 随机划分
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    print(f"时序数据集大小: {len(dataset)}, 时序长度: {sequence_length}")
    print(f"训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
    
    # 2. 初始化时序模型和图构建器
    print("\n[2/6] 初始化时序模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建时序图构建器
    graph_builder = TemporalWalkerGraphBuilder(sequence_length=sequence_length)
    
    # 创建时序GNN模型
    model = TemporalWalkerGNN(
        input_dim=1, 
        hidden_dim=hidden_dim, 
        output_dim=3, 
        num_nodes=6, 
        sequence_length=sequence_length,
        num_relations=2,  # 空间关系和时间关系
        num_layers=2
    ).to(device)
    
    print(f"使用时序图神经网络模型，隐藏维度: {hidden_dim}, 序列长度: {sequence_length}")
    
    # 3. 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)
    
    loss_fn = nn.MSELoss()
    
    # 4. 训练模型
    print("\n[3/6] 开始训练时序模型...")
    training_results = train_temporal_walker_gnn(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, 
        graph_builder, epochs=epochs, save_dir=save_dir, validate_every=validate_every
    )
    
    # 5. 测试模型
    print("\n[4/6] 测试时序模型性能...")
    test_results = temporal_test(model, test_loader, device, graph_builder)
    
    # 6. 分析预测
    print("\n[5/6] 分析预测结果...")
    try:
        from original_code import analyze_predictions  # 尝试从原始代码导入
        analyze_predictions(test_results, save_dir)
    except ImportError:
        print("无法导入原始代码中的analyze_predictions函数，跳过预测结果分析")
    
    # 7. 比较与静态GNN模型
    print("\n[6/6] 比较时序GNN与静态GNN...")
    print("通过时序建模能够捕捉动态行走模式，考虑到了历史信息对未来预测的影响。")
    
    # 8. 打印总结
    print("\n" + "="*50)
    print(f"时序GNN训练完成! 总用时: {(time.time() - start_time) / 60:.2f} 分钟")
    print(f"时序长度: {sequence_length}")
    print(f"最佳验证损失: {training_results['best_val_loss']:.6f} (Epoch {training_results['best_epoch']})")
    print(f"测试MSE: {test_results['mse']:.6f}")
    print("="*50)
    
    return model, {
        'training_results': training_results,
        'test_results': test_results
    }

#-------------------------------------------------------------
# 8. 辅助函数 - 梯度健康检查和绘图
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
# 9. 示例数据生成
#-------------------------------------------------------------

def generate_temporal_synthetic_data(n_samples=1000, sequence_length=5, noise_level=0.1):
    """生成合成时序测试数据"""
    # 基础数据
    left_legs = []
    right_legs = []
    
    # 生成时序数据
    # 初始状态
    current_left = np.random.randn(3) * 0.5
    current_right = np.random.randn(3) * 0.5
    
    for _ in range(n_samples):
        # 更新左腿状态 - 添加一些随机变化，但保持连续性
        current_left = current_left + np.random.randn(3) * 0.1
        # 确保值在合理范围内
        current_left = np.clip(current_left, -2, 2)
        
        # 根据左腿的当前状态和历史确定右腿状态
        if len(left_legs) > 0:
            # 使用当前和前一时刻的左腿状态计算右腿
            prev_left = left_legs[-1]
            right_leg = np.zeros(3)
            # 关节0: 与左腿关节0有反向关系，加上前一状态影响
            right_leg[0] = -0.7 * current_left[0] + 0.3 * prev_left[0] + np.random.randn() * noise_level
            # 关节1: 与左腿关节0和1的非线性组合
            right_leg[1] = 0.6 * current_left[1] + 0.2 * np.sin(prev_left[0] * 2) + np.random.randn() * noise_level
            # 关节2: 与左腿关节2有强相关性，但有噪声和历史影响
            right_leg[2] = 0.8 * current_left[2] + 0.2 * prev_left[2] + np.random.randn() * noise_level
            current_right = right_leg
        else:
            # 第一个样本，简单的关系
            right_leg = np.zeros(3)
            right_leg[0] = -0.8 * current_left[0] + np.random.randn() * noise_level
            right_leg[1] = 0.7 * current_left[1] + np.random.randn() * noise_level  
            right_leg[2] = 0.9 * current_left[2] + np.random.randn() * noise_level
            current_right = right_leg
        
        # 确保右腿值在合理范围内
        current_right = np.clip(current_right, -2, 2)
        
        # 保存当前状态
        left_legs.append(current_left.copy())
        right_legs.append(current_right.copy())
    
    # 将数据格式化为[(left, right), ...]格式
    data = [(left, right) for left, right in zip(left_legs, right_legs)]
    
    return data

#-------------------------------------------------------------
# 10. 主脚本入口
#-------------------------------------------------------------

if __name__ == "__main__":
    # 生成或加载数据
    use_synthetic = False
    
    if use_synthetic:
        # 生成合成时序数据
        print("生成合成时序数据...")
        data = generate_temporal_synthetic_data(n_samples=2000, sequence_length=5, noise_level=0.1)
    else:
        # 加载真实数据
        print("加载真实数据...")
        import pickle
        with open('fixed_walker_data.pkl', 'rb') as f:
            data = pickle.load(f)
    
    # 训练时序GNN模型
    model, results = main_temporal(
        data, 
        hidden_dim=16,
        sequence_length=5,  # 使用5个时间步
        stride=1,           # 滑动窗口步长
        epochs=1000, 
        batch_size=16, 
        lr=0.001,
        validate_every=10
    )
    
    # 测试推理
    print("\n测试模型推理能力...")
    # 创建一个随机的左腿序列
    left_leg_sequence = np.random.randn(5, 3) * 0.5
    
    # 创建图构建器
    graph_builder = TemporalWalkerGraphBuilder(sequence_length=5)
    
    # 进行推理
    predicted_right_leg = temporal_gnn_inference(model, left_leg_sequence, graph_builder)
    print("预测的右腿角度:", predicted_right_leg)