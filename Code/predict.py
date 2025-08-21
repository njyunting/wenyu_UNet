import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
from datetime import datetime
from tqdm import tqdm
from model import UNetPlusPlusWithMultiSpectralAttention

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重文件夹路径
weightsDir = "../Data/Param/Weights/training_"

# 数据集路径
dataPath = "../Data"

# 定义保存预测结果的函数
def save_predictions(predictions, filenames, outputPath):
    for prediction, filename in zip(predictions, filenames):
        prediction = prediction.cpu().detach().numpy().squeeze()
        save_path = os.path.join(outputPath, filename)
        sio.savemat(save_path, {'predicted_height': prediction})

class MyDataset(Dataset):
    def __init__(self, path, subfolder, mode='Infer'):
        self.path = path
        self.subfolder = subfolder
        self.mode = mode
        self.name = [f for f in os.listdir(os.path.join(path, mode, subfolder)) if f.endswith('.mat')]

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # 获取原图的名字
        imageName = self.name[index]
        # 获取原图的路径
        imagePath = os.path.join(self.path, self.mode, self.subfolder, imageName)
        # 读取原图
        image = sio.loadmat(imagePath)
        image = image['image_block']
        image = image.astype(np.float32)
        image = image.swapaxes(0, 2).swapaxes(1, 2)
        return image, imageName

if __name__ == '__main__':
    subfolder = "GRSM"  # 指定子文件夹的名称

    # 加载推理数据集并创建 DataLoader
    dataset = MyDataset(dataPath, subfolder, mode='Infer')
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)  # 推理时使用批量大小为1

    # 获取数据集中图像的通道数
    example_data, _ = next(iter(dataLoader))
    input_channels = example_data.shape[1]

    # 遍历指定文件夹中的所有权重文件
    for weightFile in os.listdir(weightsDir):
        if weightFile.endswith('.pth'):
            weightPath = os.path.join(weightsDir, weightFile)
            print(f"Processing weight file: {weightFile}")

            # 定义输出文件夹的名称
            weightFileName = os.path.basename(weightPath).split('.')[0]
            currentTime = datetime.now().strftime('%Y%m%d_%H%M%S')
            outputFolderName = f"{weightFileName}-{subfolder}-{currentTime}"
            outputPath = os.path.join("../Data/GRSM", outputFolderName)
            os.makedirs(outputPath, exist_ok=True)

            # 初始化网络并加载权重
            net = UNetPlusPlusWithMultiSpectralAttention(in_channels=12, out_channels=1).to(device)
            if os.path.exists(weightPath):
                net.load_state_dict(torch.load(weightPath, map_location=device))
                print("Load weights success")
            else:
                print(f"Weights file {weightPath} not found. Skipping.")
                continue

            net.eval()  # 将网络设置为评估模式

            filenames = dataset.name
            predictions = []

            with torch.no_grad():
                for image, filename in tqdm(dataLoader, desc=f"Inference with {weightFile}", unit="batch"):
                    image = image.to(device)
                    # 进行推理
                    outImage = net(image)
                    # 将预测结果添加到列表中
                    predictions.append(outImage)

            # 将预测结果保存为 .mat 文件
            save_predictions(predictions, filenames, outputPath)

            print(f"Inference with {weightFile} complete. Results saved to {outputPath}.")
