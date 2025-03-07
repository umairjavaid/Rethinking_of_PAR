def get_model():
  model = GoogleNet()
  return model

class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes: int = 51,
        aux_logits: bool = False,
        transform_input: bool = True,
        init_weights: Optional[bool] = False,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(blocks) == 2
        conv_block = blocks[0]
        inception_block = blocks[1]

        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        #add view block here
        self.convview = conv_block(480, 128, kernel_size=3)
        self.fc1 = nn.Linear(18432, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 3)
        self.softmax = nn.Softmax(dim=1)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        #attributes
        self.inception5cfront = inception_block(1024, 352, 192, 320, 192, 224, 128)
        self.maxpool5 = nn.MaxPool2d(7, stride=1, ceil_mode=True)
        self.fc3 = nn.Linear(1024, num_classes)
        self.inception5cside = inception_block(1024, 352, 192, 320, 192, 224, 128)
        self.maxpool6 = nn.MaxPool2d(7, stride=1, ceil_mode=True)
        self.fc4 = nn.Linear(1024, num_classes)
        self.inception5cback = inception_block(1024, 352, 192, 320, 192, 224, 128)
        self.maxpool7 = nn.MaxPool2d(7, stride=1, ceil_mode=True)
        self.fc5 = nn.Linear(1024, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        
        v = self.convview(x)
        v = self.fc1(v.reshape(v.size(0), -1))
        v = self.bn1(v)
        v = F.relu(v)
        v = self.fc2(v.reshape(v.size(0), -1))
        #v_ = self.softmax(v)
        
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)


        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        f = self.inception5cfront(x)
        f = self.maxpool5(f)
        f = self.fc3(f.reshape(f.size(0), -1))
        #print("f.shape: ",f.shape)
        #f = f*v[0]
        s = self.inception5cside(x)
        s = self.maxpool6(s)
        s = self.fc4(s.reshape(s.size(0), -1))
        #s = s*v[1]
        b = self.inception5cback(x)
        b = self.maxpool7(b)
        b = self.fc5(b.reshape(b.size(0), -1))
        #b = b*v[2]
        x = torch.cat((f.unsqueeze(0),s.unsqueeze(0),b.unsqueeze(0)),0)
        x = x.transpose(0,1)*v.unsqueeze(2)
        x = torch.sum(x,1)
        
        #x = f+s+b
        #x = self.sigmoid(x)
        return v, x

    def forward(self, x: Tensor):
        x = self._transform_input(x)
        v, x = self._forward(x)
        return v, x


class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
