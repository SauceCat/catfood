# ResNets

## Reference

- https://erikgaas.medium.com/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096
- https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

## BasicBlock vs. Bottleneck

![1628040884098](/home/saucecat/.config/Typora/typora-user-images/1628040884098.png)

#### BasicBlock (Left): ResNet-18/34

```python
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        ...
    ) -> None:
        ...
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
		# self.conv1 = conv3x3(inplanes, planes, stride)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
		# self.conv2 = conv3x3(planes, planes)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

#### Bottleneck (Right): ResNet-50/101/152

```python
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        ...
    ) -> None:
        ...
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # support group convolution
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
		# self.conv1 = conv1x1(inplanes, width)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
		# self.conv2 = conv3x3(width, width, stride, groups, dilation)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
		# self.conv3 = conv1x1(width, planes * self.expansion)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

---

## Network Construction

![img](https://miro.medium.com/max/770/1*I2557MCaFdNUm4q9TfvOpw.png)

- **resnet18:** block = `BasicBlock`, layers = `[2, 2, 2, 2]`, output channels: `64, 64, [64, 64], [128, 128], [256, 256], [512, 512], 512`
- **resnet34:** block = `BasicBlock`, layers = `[3, 4, 6, 3]`, output channels: `64, 64, [64, 64], [128, 128], [256, 256], [512, 512], 512`
- **resnet50:** block = `Bottleneck`, layers = `[3, 4, 6, 3]`, output channels: `64, 64, [64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048], 2048`
- **resnet101:** block = `Bottleneck`, layers = `[3, 4, 23, 3]`, output channels: `64, 64, [64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048], 2048`
- **resnet152:** block = `Bottleneck`, layers = `[3, 8, 36, 3]`, output channels: `64, 64, [64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048], 2048`

```python
class ResNet(nn.Module):
	
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        # standard convolution
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # output width: [(W−K+2P)/S]+1, [(224−7+6)/2]+1=112
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # output width: 56
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        # output width: 56
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # output width: 28
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # output width: 14
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # output width: 7
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output width: 1
        
        # BasicBlock: 512 * block.expansion (1) = 512
        # Bottleneck: 512 * block.expansion (4) = 2048
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # modify the resolution for the add operation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # stride and downsample only for the first layer
        # within the sample layer group, the resolution won't change any more
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
```
