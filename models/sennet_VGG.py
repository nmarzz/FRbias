import torch
import torch.nn as nn


class Senet50_scratch_dag(nn.Module):

    def __init__(self):
        super(Senet50_scratch_dag, self).__init__()
        self.meta = {'mean': [123, 117, 104],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}

        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu_7x7_s2 = nn.ReLU()
        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv2_1_1x1_reduce = nn.Conv2d(64, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_1x1_reduce_relu = nn.ReLU()
        self.conv2_1_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_1_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_3x3_relu = nn.ReLU()
        self.conv2_1_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv2_1_1x1_down = nn.Conv2d(256, 16, kernel_size=[1, 1], stride=(1, 1))
        self.conv2_1_1x1_down_relu = nn.ReLU()
        self.conv2_1_1x1_up = nn.Conv2d(16, 256, kernel_size=[1, 1], stride=(1, 1))
        self.conv2_1_prob = nn.Sigmoid()
        self.conv2_1_1x1_proj = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_proj_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_relu = nn.ReLU()
        self.conv2_2_1x1_reduce = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_1x1_reduce_relu = nn.ReLU()
        self.conv2_2_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_2_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_3x3_relu = nn.ReLU()
        self.conv2_2_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv2_2_1x1_down = nn.Conv2d(256, 16, kernel_size=[1, 1], stride=(1, 1))
        self.conv2_2_1x1_down_relu = nn.ReLU()
        self.conv2_2_1x1_up = nn.Conv2d(16, 256, kernel_size=[1, 1], stride=(1, 1))
        self.conv2_2_prob = nn.Sigmoid()
        self.conv2_2_relu = nn.ReLU()
        self.conv2_3_1x1_reduce = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_1x1_reduce_relu = nn.ReLU()
        self.conv2_3_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_3_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_3x3_relu = nn.ReLU()
        self.conv2_3_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv2_3_1x1_down = nn.Conv2d(256, 16, kernel_size=[1, 1], stride=(1, 1))
        self.conv2_3_1x1_down_relu = nn.ReLU()
        self.conv2_3_1x1_up = nn.Conv2d(16, 256, kernel_size=[1, 1], stride=(1, 1))
        self.conv2_3_prob = nn.Sigmoid()
        self.conv2_3_relu = nn.ReLU()
        self.conv3_1_1x1_reduce = nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_1x1_reduce_relu = nn.ReLU()
        self.conv3_1_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_1_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_3x3_relu = nn.ReLU()
        self.conv3_1_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_1_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3_1_1x1_down = nn.Conv2d(512, 32, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_1_1x1_down_relu = nn.ReLU()
        self.conv3_1_1x1_up = nn.Conv2d(32, 512, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_1_prob = nn.Sigmoid()
        self.conv3_1_1x1_proj = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1_1x1_proj_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_relu = nn.ReLU()
        self.conv3_2_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_1x1_reduce_relu = nn.ReLU()
        self.conv3_2_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_2_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_3x3_relu = nn.ReLU()
        self.conv3_2_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3_2_1x1_down = nn.Conv2d(512, 32, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_2_1x1_down_relu = nn.ReLU()
        self.conv3_2_1x1_up = nn.Conv2d(32, 512, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_2_prob = nn.Sigmoid()
        self.conv3_2_relu = nn.ReLU()
        self.conv3_3_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_1x1_reduce_relu = nn.ReLU()
        self.conv3_3_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_3_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_3x3_relu = nn.ReLU()
        self.conv3_3_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3_3_1x1_down = nn.Conv2d(512, 32, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_3_1x1_down_relu = nn.ReLU()
        self.conv3_3_1x1_up = nn.Conv2d(32, 512, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_3_prob = nn.Sigmoid()
        self.conv3_3_relu = nn.ReLU()
        self.conv3_4_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_1x1_reduce_relu = nn.ReLU()
        self.conv3_4_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_4_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_3x3_relu = nn.ReLU()
        self.conv3_4_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3_4_1x1_down = nn.Conv2d(512, 32, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_4_1x1_down_relu = nn.ReLU()
        self.conv3_4_1x1_up = nn.Conv2d(32, 512, kernel_size=[1, 1], stride=(1, 1))
        self.conv3_4_prob = nn.Sigmoid()
        self.conv3_4_relu = nn.ReLU()
        self.conv4_1_1x1_reduce = nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_1x1_reduce_relu = nn.ReLU()
        self.conv4_1_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_1_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_3x3_relu = nn.ReLU()
        self.conv4_1_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_1_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv4_1_1x1_down = nn.Conv2d(1024, 64, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_1_1x1_down_relu = nn.ReLU()
        self.conv4_1_1x1_up = nn.Conv2d(64, 1024, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_1_prob = nn.Sigmoid()
        self.conv4_1_1x1_proj = nn.Conv2d(512, 1024, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1_1x1_proj_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_relu = nn.ReLU()
        self.conv4_2_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_1x1_reduce_relu = nn.ReLU()
        self.conv4_2_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_2_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_3x3_relu = nn.ReLU()
        self.conv4_2_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv4_2_1x1_down = nn.Conv2d(1024, 64, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_2_1x1_down_relu = nn.ReLU()
        self.conv4_2_1x1_up = nn.Conv2d(64, 1024, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_2_prob = nn.Sigmoid()
        self.conv4_2_relu = nn.ReLU()
        self.conv4_3_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_1x1_reduce_relu = nn.ReLU()
        self.conv4_3_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_3_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_3x3_relu = nn.ReLU()
        self.conv4_3_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv4_3_1x1_down = nn.Conv2d(1024, 64, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_3_1x1_down_relu = nn.ReLU()
        self.conv4_3_1x1_up = nn.Conv2d(64, 1024, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_3_prob = nn.Sigmoid()
        self.conv4_3_relu = nn.ReLU()
        self.conv4_4_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_1x1_reduce_relu = nn.ReLU()
        self.conv4_4_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_4_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_3x3_relu = nn.ReLU()
        self.conv4_4_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv4_4_1x1_down = nn.Conv2d(1024, 64, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_4_1x1_down_relu = nn.ReLU()
        self.conv4_4_1x1_up = nn.Conv2d(64, 1024, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_4_prob = nn.Sigmoid()
        self.conv4_4_relu = nn.ReLU()
        self.conv4_5_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_1x1_reduce_relu = nn.ReLU()
        self.conv4_5_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_5_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_3x3_relu = nn.ReLU()
        self.conv4_5_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv4_5_1x1_down = nn.Conv2d(1024, 64, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_5_1x1_down_relu = nn.ReLU()
        self.conv4_5_1x1_up = nn.Conv2d(64, 1024, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_5_prob = nn.Sigmoid()
        self.conv4_5_relu = nn.ReLU()
        self.conv4_6_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_1x1_reduce_relu = nn.ReLU()
        self.conv4_6_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_6_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_3x3_relu = nn.ReLU()
        self.conv4_6_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv4_6_1x1_down = nn.Conv2d(1024, 64, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_6_1x1_down_relu = nn.ReLU()
        self.conv4_6_1x1_up = nn.Conv2d(64, 1024, kernel_size=[1, 1], stride=(1, 1))
        self.conv4_6_prob = nn.Sigmoid()
        self.conv4_6_relu = nn.ReLU()
        self.conv5_1_1x1_reduce = nn.Conv2d(1024, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_1x1_reduce_relu = nn.ReLU()
        self.conv5_1_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_1_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_3x3_relu = nn.ReLU()
        self.conv5_1_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_1_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv5_1_1x1_down = nn.Conv2d(2048, 128, kernel_size=[1, 1], stride=(1, 1))
        self.conv5_1_1x1_down_relu = nn.ReLU()
        self.conv5_1_1x1_up = nn.Conv2d(128, 2048, kernel_size=[1, 1], stride=(1, 1))
        self.conv5_1_prob = nn.Sigmoid()
        self.conv5_1_1x1_proj = nn.Conv2d(1024, 2048, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1_1x1_proj_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_relu = nn.ReLU()
        self.conv5_2_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_1x1_reduce_relu = nn.ReLU()
        self.conv5_2_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_2_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_3x3_relu = nn.ReLU()
        self.conv5_2_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv5_2_1x1_down = nn.Conv2d(2048, 128, kernel_size=[1, 1], stride=(1, 1))
        self.conv5_2_1x1_down_relu = nn.ReLU()
        self.conv5_2_1x1_up = nn.Conv2d(128, 2048, kernel_size=[1, 1], stride=(1, 1))
        self.conv5_2_prob = nn.Sigmoid()
        self.conv5_2_relu = nn.ReLU()
        self.conv5_3_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_1x1_reduce_relu = nn.ReLU()
        self.conv5_3_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_3_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_3x3_relu = nn.ReLU()
        self.conv5_3_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv5_3_1x1_down = nn.Conv2d(2048, 128, kernel_size=[1, 1], stride=(1, 1))
        self.conv5_3_1x1_down_relu = nn.ReLU()
        self.conv5_3_1x1_up = nn.Conv2d(128, 2048, kernel_size=[1, 1], stride=(1, 1))
        self.conv5_3_prob = nn.Sigmoid()
        self.conv5_3_relu = nn.ReLU()
        self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.classifier = nn.Conv2d(2048, 8631, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, data):
        conv1_7x7_s2 = self.conv1_7x7_s2(data)
        conv1_7x7_s2_bn = self.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_7x7_s2_bnxx)
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1_global_pool = self.conv2_1_global_pool(conv2_1_1x1_increase_bn)
        conv2_1_1x1_down = self.conv2_1_1x1_down(conv2_1_global_pool)
        conv2_1_1x1_downx = self.conv2_1_1x1_down_relu(conv2_1_1x1_down)
        conv2_1_1x1_up = self.conv2_1_1x1_up(conv2_1_1x1_downx)
        conv2_1_1x1_upx = self.conv2_1_prob(conv2_1_1x1_up)
        conv2_1_prob_reshape = conv2_1_1x1_upx
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1 = conv2_1_prob_reshape.expand_as(conv2_1_1x1_increase_bn) * conv2_1_1x1_increase_bn + conv2_1_1x1_proj_bn
        conv2_1x = self.conv2_1_relu(conv2_1)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2_global_pool = self.conv2_2_global_pool(conv2_2_1x1_increase_bn)
        conv2_2_1x1_down = self.conv2_2_1x1_down(conv2_2_global_pool)
        conv2_2_1x1_downx = self.conv2_2_1x1_down_relu(conv2_2_1x1_down)
        conv2_2_1x1_up = self.conv2_2_1x1_up(conv2_2_1x1_downx)
        conv2_2_1x1_upx = self.conv2_2_prob(conv2_2_1x1_up)
        conv2_2_prob_reshape = conv2_2_1x1_upx
        conv2_2 = conv2_2_prob_reshape.expand_as(conv2_2_1x1_increase_bn) * conv2_2_1x1_increase_bn + conv2_1x
        conv2_2x = self.conv2_2_relu(conv2_2)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3_global_pool = self.conv2_3_global_pool(conv2_3_1x1_increase_bn)
        conv2_3_1x1_down = self.conv2_3_1x1_down(conv2_3_global_pool)
        conv2_3_1x1_downx = self.conv2_3_1x1_down_relu(conv2_3_1x1_down)
        conv2_3_1x1_up = self.conv2_3_1x1_up(conv2_3_1x1_downx)
        conv2_3_1x1_upx = self.conv2_3_prob(conv2_3_1x1_up)
        conv2_3_prob_reshape = conv2_3_1x1_upx
        conv2_3 = conv2_3_prob_reshape.expand_as(conv2_3_1x1_increase_bn) * conv2_3_1x1_increase_bn + conv2_2x
        conv2_3x = self.conv2_3_relu(conv2_3)
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1_global_pool = self.conv3_1_global_pool(conv3_1_1x1_increase_bn)
        conv3_1_1x1_down = self.conv3_1_1x1_down(conv3_1_global_pool)
        conv3_1_1x1_downx = self.conv3_1_1x1_down_relu(conv3_1_1x1_down)
        conv3_1_1x1_up = self.conv3_1_1x1_up(conv3_1_1x1_downx)
        conv3_1_1x1_upx = self.conv3_1_prob(conv3_1_1x1_up)
        conv3_1_prob_reshape = conv3_1_1x1_upx
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1 = conv3_1_prob_reshape.expand_as(conv3_1_1x1_increase_bn) * conv3_1_1x1_increase_bn + conv3_1_1x1_proj_bn
        conv3_1x = self.conv3_1_relu(conv3_1)
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2_global_pool = self.conv3_2_global_pool(conv3_2_1x1_increase_bn)
        conv3_2_1x1_down = self.conv3_2_1x1_down(conv3_2_global_pool)
        conv3_2_1x1_downx = self.conv3_2_1x1_down_relu(conv3_2_1x1_down)
        conv3_2_1x1_up = self.conv3_2_1x1_up(conv3_2_1x1_downx)
        conv3_2_1x1_upx = self.conv3_2_prob(conv3_2_1x1_up)
        conv3_2_prob_reshape = conv3_2_1x1_upx
        conv3_2 = conv3_2_prob_reshape.expand_as(conv3_2_1x1_increase_bn) * conv3_2_1x1_increase_bn + conv3_1x
        conv3_2x = self.conv3_2_relu(conv3_2)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3_global_pool = self.conv3_3_global_pool(conv3_3_1x1_increase_bn)
        conv3_3_1x1_down = self.conv3_3_1x1_down(conv3_3_global_pool)
        conv3_3_1x1_downx = self.conv3_3_1x1_down_relu(conv3_3_1x1_down)
        conv3_3_1x1_up = self.conv3_3_1x1_up(conv3_3_1x1_downx)
        conv3_3_1x1_upx = self.conv3_3_prob(conv3_3_1x1_up)
        conv3_3_prob_reshape = conv3_3_1x1_upx
        conv3_3 = conv3_3_prob_reshape.expand_as(conv3_3_1x1_increase_bn) * conv3_3_1x1_increase_bn + conv3_2x
        conv3_3x = self.conv3_3_relu(conv3_3)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4_global_pool = self.conv3_4_global_pool(conv3_4_1x1_increase_bn)
        conv3_4_1x1_down = self.conv3_4_1x1_down(conv3_4_global_pool)
        conv3_4_1x1_downx = self.conv3_4_1x1_down_relu(conv3_4_1x1_down)
        conv3_4_1x1_up = self.conv3_4_1x1_up(conv3_4_1x1_downx)
        conv3_4_1x1_upx = self.conv3_4_prob(conv3_4_1x1_up)
        conv3_4_prob_reshape = conv3_4_1x1_upx
        conv3_4 = conv3_4_prob_reshape.expand_as(conv3_4_1x1_increase_bn) * conv3_4_1x1_increase_bn + conv3_3x
        conv3_4x = self.conv3_4_relu(conv3_4)
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1_global_pool = self.conv4_1_global_pool(conv4_1_1x1_increase_bn)
        conv4_1_1x1_down = self.conv4_1_1x1_down(conv4_1_global_pool)
        conv4_1_1x1_downx = self.conv4_1_1x1_down_relu(conv4_1_1x1_down)
        conv4_1_1x1_up = self.conv4_1_1x1_up(conv4_1_1x1_downx)
        conv4_1_1x1_upx = self.conv4_1_prob(conv4_1_1x1_up)
        conv4_1_prob_reshape = conv4_1_1x1_upx
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1 = conv4_1_prob_reshape.expand_as(conv4_1_1x1_increase_bn) * conv4_1_1x1_increase_bn + conv4_1_1x1_proj_bn
        conv4_1x = self.conv4_1_relu(conv4_1)
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2_global_pool = self.conv4_2_global_pool(conv4_2_1x1_increase_bn)
        conv4_2_1x1_down = self.conv4_2_1x1_down(conv4_2_global_pool)
        conv4_2_1x1_downx = self.conv4_2_1x1_down_relu(conv4_2_1x1_down)
        conv4_2_1x1_up = self.conv4_2_1x1_up(conv4_2_1x1_downx)
        conv4_2_1x1_upx = self.conv4_2_prob(conv4_2_1x1_up)
        conv4_2_prob_reshape = conv4_2_1x1_upx
        conv4_2 = conv4_2_prob_reshape.expand_as(conv4_2_1x1_increase_bn) * conv4_2_1x1_increase_bn + conv4_1x
        conv4_2x = self.conv4_2_relu(conv4_2)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3_global_pool = self.conv4_3_global_pool(conv4_3_1x1_increase_bn)
        conv4_3_1x1_down = self.conv4_3_1x1_down(conv4_3_global_pool)
        conv4_3_1x1_downx = self.conv4_3_1x1_down_relu(conv4_3_1x1_down)
        conv4_3_1x1_up = self.conv4_3_1x1_up(conv4_3_1x1_downx)
        conv4_3_1x1_upx = self.conv4_3_prob(conv4_3_1x1_up)
        conv4_3_prob_reshape = conv4_3_1x1_upx
        conv4_3 = conv4_3_prob_reshape.expand_as(conv4_3_1x1_increase_bn) * conv4_3_1x1_increase_bn + conv4_2x
        conv4_3x = self.conv4_3_relu(conv4_3)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4_global_pool = self.conv4_4_global_pool(conv4_4_1x1_increase_bn)
        conv4_4_1x1_down = self.conv4_4_1x1_down(conv4_4_global_pool)
        conv4_4_1x1_downx = self.conv4_4_1x1_down_relu(conv4_4_1x1_down)
        conv4_4_1x1_up = self.conv4_4_1x1_up(conv4_4_1x1_downx)
        conv4_4_1x1_upx = self.conv4_4_prob(conv4_4_1x1_up)
        conv4_4_prob_reshape = conv4_4_1x1_upx
        conv4_4 = conv4_4_prob_reshape.expand_as(conv4_4_1x1_increase_bn) * conv4_4_1x1_increase_bn + conv4_3x
        conv4_4x = self.conv4_4_relu(conv4_4)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5_global_pool = self.conv4_5_global_pool(conv4_5_1x1_increase_bn)
        conv4_5_1x1_down = self.conv4_5_1x1_down(conv4_5_global_pool)
        conv4_5_1x1_downx = self.conv4_5_1x1_down_relu(conv4_5_1x1_down)
        conv4_5_1x1_up = self.conv4_5_1x1_up(conv4_5_1x1_downx)
        conv4_5_1x1_upx = self.conv4_5_prob(conv4_5_1x1_up)
        conv4_5_prob_reshape = conv4_5_1x1_upx
        conv4_5 = conv4_5_prob_reshape.expand_as(conv4_5_1x1_increase_bn) * conv4_5_1x1_increase_bn + conv4_4x
        conv4_5x = self.conv4_5_relu(conv4_5)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6_global_pool = self.conv4_6_global_pool(conv4_6_1x1_increase_bn)
        conv4_6_1x1_down = self.conv4_6_1x1_down(conv4_6_global_pool)
        conv4_6_1x1_downx = self.conv4_6_1x1_down_relu(conv4_6_1x1_down)
        conv4_6_1x1_up = self.conv4_6_1x1_up(conv4_6_1x1_downx)
        conv4_6_1x1_upx = self.conv4_6_prob(conv4_6_1x1_up)
        conv4_6_prob_reshape = conv4_6_1x1_upx
        conv4_6 = conv4_6_prob_reshape.expand_as(conv4_6_1x1_increase_bn) * conv4_6_1x1_increase_bn + conv4_5x
        conv4_6x = self.conv4_6_relu(conv4_6)
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1_global_pool = self.conv5_1_global_pool(conv5_1_1x1_increase_bn)
        conv5_1_1x1_down = self.conv5_1_1x1_down(conv5_1_global_pool)
        conv5_1_1x1_downx = self.conv5_1_1x1_down_relu(conv5_1_1x1_down)
        conv5_1_1x1_up = self.conv5_1_1x1_up(conv5_1_1x1_downx)
        conv5_1_1x1_upx = self.conv5_1_prob(conv5_1_1x1_up)
        conv5_1_prob_reshape = conv5_1_1x1_upx
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1 = conv5_1_prob_reshape.expand_as(conv5_1_1x1_increase_bn) * conv5_1_1x1_increase_bn + conv5_1_1x1_proj_bn
        conv5_1x = self.conv5_1_relu(conv5_1)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2_global_pool = self.conv5_2_global_pool(conv5_2_1x1_increase_bn)
        conv5_2_1x1_down = self.conv5_2_1x1_down(conv5_2_global_pool)
        conv5_2_1x1_downx = self.conv5_2_1x1_down_relu(conv5_2_1x1_down)
        conv5_2_1x1_up = self.conv5_2_1x1_up(conv5_2_1x1_downx)
        conv5_2_1x1_upx = self.conv5_2_prob(conv5_2_1x1_up)
        conv5_2_prob_reshape = conv5_2_1x1_upx
        conv5_2 = conv5_2_prob_reshape.expand_as(conv5_2_1x1_increase_bn) * conv5_2_1x1_increase_bn + conv5_1x
        conv5_2x = self.conv5_2_relu(conv5_2)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3_global_pool = self.conv5_3_global_pool(conv5_3_1x1_increase_bn)
        conv5_3_1x1_down = self.conv5_3_1x1_down(conv5_3_global_pool)
        conv5_3_1x1_downx = self.conv5_3_1x1_down_relu(conv5_3_1x1_down)
        conv5_3_1x1_up = self.conv5_3_1x1_up(conv5_3_1x1_downx)
        conv5_3_1x1_upx = self.conv5_3_prob(conv5_3_1x1_up)
        conv5_3_prob_reshape = conv5_3_1x1_upx
        conv5_3 = conv5_3_prob_reshape.expand_as(conv5_3_1x1_increase_bn) * conv5_3_1x1_increase_bn + conv5_2x
        conv5_3x = self.conv5_3_relu(conv5_3)
        pool5_7x7_s1 = self.pool5_7x7_s1(conv5_3x)
        classifier_preflatten = self.classifier(pool5_7x7_s1)
        classifier = classifier_preflatten.view(classifier_preflatten.size(0), -1)
        return classifier, pool5_7x7_s1

def senet50_scratch_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Senet50_scratch_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
