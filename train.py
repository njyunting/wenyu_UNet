from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from DataSet import MyDataset
from model import UNetPlusPlusWithMultiSpectralAttention
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import subprocess
import webbrowser

# 选取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定的权重文件路径，用于加载已有权重
initial_weight_path = "../Resources/Param/weights-agb-2023-up.pth"

# 数据集地址
data_path = "../Resources"


# 定义 Balanced MSE Loss（BMC 版本）
def bmc_loss(pred, target, noise_var):
    """
    计算 Balanced MSE Loss，同时忽略 target 中的 0 和 NaN 值
    """
    # 创建掩码，忽略 target 中的 0 和 NaN 值
    mask = ~torch.isnan(target) & (target != 0)

    # 应用掩码
    pred = pred[mask]
    target = target[mask]

    # 如果有效样本少于2个，返回0损失以避免计算问题
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 计算 BMC Loss
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    logits = -0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))
    loss = loss * (2 * noise_var)
    return loss


class BMCLoss(nn.Module):  # 继承 nn.Module
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)


def train_one_epoch(model, criterion, optimizer, data_loader, device, writer, epoch):
    model.train()
    epoch_loss = 0

    with tqdm(total=len(data_loader), desc="Training", unit="batch") as pbar:
        for image, height_image in data_loader:
            image, height_image = image.to(device), height_image.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, height_image)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            # Log the loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + pbar.n)

    return epoch_loss / len(data_loader)


def initialize_model(in_channels, out_channels, weight_path):
    model = UNetPlusPlusWithMultiSpectralAttention(in_channels=in_channels, out_channels=out_channels).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print("Weights loaded successfully.")
    else:
        print("No pre-trained weights found.")
    return model


# 自动启动 TensorBoard
def start_tensorboard(log_dir, port=6006):
    """
    启动 TensorBoard 并自动打开浏览器查看。
    :param log_dir: TensorBoard 日志文件的路径。
    :param port: TensorBoard 服务的端口，默认为 6006。
    """
    try:
        # 检查端口是否被占用
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            print(f"Port {port} is already in use. Trying another port...")
            port += 1  # 尝试下一个端口

        # 启动 TensorBoard
        tensorboard_command = f"tensorboard --logdir={log_dir} --port={port}"
        subprocess.Popen(tensorboard_command, shell=True)

        # 自动在默认浏览器中打开 TensorBoard 页面
        url = f"http://localhost:{port}"
        print(f"TensorBoard is running at {url}")
        webbrowser.open(url)
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")


if __name__ == '__main__':
    # 加载数据集
    dataset = MyDataset(data_path)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data path or preprocessing.")

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # 获取数据集中图像的通道数
    example_data, _ = next(iter(train_loader))
    input_channels = example_data.shape[1]

    # 初始化模型
    net = initialize_model(input_channels, 1, initial_weight_path)

    # 初始化优化器和 BMC Loss
    init_noise_sigma = 8.0
    sigma_lr = 1e-2
    criterion = BMCLoss(init_noise_sigma)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # 创建专属文件夹
    training_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    weights_dir = f"../Resources/Param/Weights/training_{training_time}"
    os.makedirs(weights_dir, exist_ok=True)

    # 日志文件路径
    loss_file_path = os.path.join(weights_dir, f"loss_log_{training_time}.txt")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=weights_dir)

    # 启动 TensorBoard
    start_tensorboard(log_dir=weights_dir, port=6008)

    # 训练参数
    max_epochs = 300
    best_loss = float('inf')
    epoch_list, loss_list = [], []

    for epoch in range(1, max_epochs + 1):
        avg_epoch_loss = train_one_epoch(net, criterion, optimizer, train_loader, device, writer, epoch)

        # 记录损失
        epoch_list.append(epoch)
        loss_list.append(avg_epoch_loss)
        print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss}")

        with open(loss_file_path, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch}: Loss = {avg_epoch_loss}\n")

        scheduler.step(avg_epoch_loss)

        # 保存最佳权重
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_weight_path = os.path.join(weights_dir, "best_model.pth")
            torch.save(net.state_dict(), best_weight_path)

        # 定期保存权重
        if epoch % 1 == 0:
            epoch_weight_path = os.path.join(weights_dir, f"epoch_{epoch}.pth")
            torch.save(net.state_dict(), epoch_weight_path)

    # 保存最终权重
    final_weight_path = os.path.join(weights_dir, "final_model.pth")
    torch.save(net.state_dict(), final_weight_path)

    # 绘制损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, loss_list, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Close TensorBoard writer
    writer.close()
