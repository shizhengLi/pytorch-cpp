#!/usr/bin/env python3
"""
PyTorchCPP 线性回归示例 (Python版)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/lib'))

try:
    import pytorchcpp as ptc
except ImportError:
    print("错误: 无法导入pytorchcpp模块。请确保先构建Python绑定。")
    sys.exit(1)

def main():
    print("PyTorchCPP 线性回归示例 (Python版)")
    print("-------------------------------")
    
    # 生成合成数据: y = 2*x + 1 + noise
    np.random.seed(42)
    num_samples = 100
    x_np = np.linspace(0, 1, num_samples).reshape(-1, 1)
    y_np = 2 * x_np + 1 + 0.1 * np.random.randn(num_samples, 1)
    
    # 转换为PyTorchCPP张量
    x = ptc.from_numpy(x_np.astype(np.float32))
    y = ptc.from_numpy(y_np.astype(np.float32))
    
    # 创建变量
    x_var = ptc.Variable(x)
    y_var = ptc.Variable(y)
    
    print(f"数据生成完成: y = 2*x + 1 + noise")
    print(f"样本数量: {num_samples}")
    
    # 创建模型、损失函数和优化器
    model = ptc.nn.Linear(1, 1)
    criterion = ptc.nn.MSELoss()
    optimizer = ptc.optim.SGD(model.parameters(), learning_rate=0.1)
    
    # 训练模型
    num_epochs = 100
    losses = []
    
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x_var)
        
        # 计算损失
        loss = criterion(y_pred, y_var)
        
        # 记录损失
        loss_value = ptc.to_numpy(loss.data())[0]
        losses.append(loss_value)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value:.6f}")
    
    print("训练完成!\n")
    
    # 获取模型参数
    params = model.parameters()
    weight = ptc.to_numpy(params["weight"].data())[0, 0]
    bias = ptc.to_numpy(params["bias"].data())[0]
    
    print("模型参数:")
    print(f"weight: {weight:.4f} (真实值: 2.0)")
    print(f"bias: {bias:.4f} (真实值: 1.0)")
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点
    plt.scatter(x_np, y_np, color='blue', label='Data')
    
    # 绘制拟合线
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    y_plot = weight * x_plot + bias
    plt.plot(x_plot, y_plot, color='red', linewidth=2, label=f'Fit: y = {weight:.2f}x + {bias:.2f}')
    
    # 绘制真实线
    y_true = 2 * x_plot + 1
    plt.plot(x_plot, y_true, color='green', linewidth=2, linestyle='--', label='True: y = 2x + 1')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 