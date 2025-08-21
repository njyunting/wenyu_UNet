import os
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio


class MyDataset(Dataset):
    def __init__(self, path, mode='Train'):
        self.path = path
        self.mode = mode
        if self.mode == 'Train':
            # 提取并存储存放训练.mat文件的文件夹名称
            self.train_folder_name = next(
                f for f in os.listdir(os.path.join(path, 'Train')) if f.startswith('GRSM'))
            self.image_folder = os.path.join(path, f'Train\\{self.train_folder_name}\\s1s2')
            self.label_folder = os.path.join(path, f'Train\\{self.train_folder_name}\\GEDI')
        elif self.mode == 'Infer':
            self.image_folder = os.path.join(path, 'Infer')
        self.name = [f for f in os.listdir(self.image_folder) if f.endswith('.mat')]

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        imageName = self.name[index]  # xxx.mat
        imagePath = os.path.join(self.image_folder, imageName)
        image = sio.loadmat(imagePath)
        image = image['image_block']
        image = image.astype(np.float32)
        image = image.swapaxes(0, 2).swapaxes(1, 2)

        if self.mode == 'Train':
            heightPath = os.path.join(self.label_folder, imageName)
            heightImage = sio.loadmat(heightPath)
            heightImage = heightImage['label_block']
            heightImage = heightImage.reshape(1, 256, 256)
            heightImage = heightImage.astype(np.float32)
            return image, heightImage
        else:
            return image, imageName


if __name__ == '__main__':
    data = MyDataset(os.path.join('..', 'Resources'), mode='Train')
    temp = data[0][0]
    temp2 = data[0][1]

    print(temp.shape)
    print(temp2.shape)

    infer_data = MyDataset(os.path.join('..', 'Resources'), mode='Infer')
    infer_temp = infer_data[0][0]
    infer_name = infer_data[0][1]

    print(infer_temp.shape)
    print(infer_name)