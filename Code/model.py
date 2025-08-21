import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. 获取频率分量的索引
def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']

    num_freq = int(method[3:])

    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError

    return mapper_x, mapper_y


# 2. 构建DCT波滤器
class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # 初始化DCT滤波器
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))

        # DCT变换（点乘）
        x = x * self.weight

        # 消去H和W维度，进行压缩
        result = torch.sum(x, dim=[2, 3])
        return result


# 3. 多光谱通道注意力层
class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()

        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)

        # 调整频率索引
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        # 返回DCT特征层
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)

        # 通道注意力的全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape

        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        # 通过DCT层得到频域信息
        y = self.dct_layer(x_pooled)

        # 通道注意力计算
        y = self.fc(y).view(n, c, 1, 1)

        # 进行通道重加权
        return x * y.expand_as(x)


# 4. 卷积块（包含通道注意力）
class ConvBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, dct_h, dct_w):
        super(ConvBlockWithAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = MultiSpectralAttentionLayer(out_channels, dct_h, dct_w)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.attention(x)
        return x


# 5. U-Net++架构（包括多光谱通道注意力层）
class UNetPlusPlusWithMultiSpectralAttention(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, filters=[32, 64, 128, 256, 512], dct_h=7, dct_w=7):
        super(UNetPlusPlusWithMultiSpectralAttention, self).__init__()

        # 各层卷积块
        self.conv0_0 = ConvBlockWithAttention(in_channels, filters[0], dct_h, dct_w)
        self.conv1_0 = ConvBlockWithAttention(filters[0], filters[1], dct_h, dct_w)
        self.conv2_0 = ConvBlockWithAttention(filters[1], filters[2], dct_h, dct_w)
        self.conv3_0 = ConvBlockWithAttention(filters[2], filters[3], dct_h, dct_w)
        self.conv4_0 = ConvBlockWithAttention(filters[3], filters[4], dct_h, dct_w)

        self.conv0_1 = ConvBlockWithAttention(filters[0] + filters[1], filters[0], dct_h, dct_w)
        self.conv1_1 = ConvBlockWithAttention(filters[1] + filters[2], filters[1], dct_h, dct_w)
        self.conv2_1 = ConvBlockWithAttention(filters[2] + filters[3], filters[2], dct_h, dct_w)
        self.conv3_1 = ConvBlockWithAttention(filters[3] + filters[4], filters[3], dct_h, dct_w)

        self.conv0_2 = ConvBlockWithAttention(filters[0] * 2 + filters[1], filters[0], dct_h, dct_w)
        self.conv1_2 = ConvBlockWithAttention(filters[1] * 2 + filters[2], filters[1], dct_h, dct_w)
        self.conv2_2 = ConvBlockWithAttention(filters[2] * 2 + filters[3], filters[2], dct_h, dct_w)

        self.conv0_3 = ConvBlockWithAttention(filters[0] * 3 + filters[1], filters[0], dct_h, dct_w)
        self.conv1_3 = ConvBlockWithAttention(filters[1] * 3 + filters[2], filters[1], dct_h, dct_w)

        self.conv0_4 = ConvBlockWithAttention(filters[0] * 4 + filters[1], filters[0], dct_h, dct_w)

        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.conv0_1(
            torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.conv1_1(
            torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.conv2_1(
            torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))
        x3_1 = self.conv3_1(
            torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_4 = self.conv0_4(torch.cat(
            [x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        output = self.final(x0_4)
        return output


# 测试
if __name__ == '__main__':
    x = torch.randn(2, 12, 256, 256)  # 12通道输入
    net = UNetPlusPlusWithMultiSpectralAttention(in_channels=12, out_channels=1)
    y = net(x)
    print(y.shape)  # 输出应为 (2, 1, 256, 256)
