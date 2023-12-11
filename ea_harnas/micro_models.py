from ea_harnas.micro_operations import *
from ea_harnas.ea_utils import drop_path


DEFAULT_PADDINGS = {
    'none': 1,
    'skip_connect': 1,
    'avg_pool_3x3': 1,
    'max_pool_3x3': 1,
    'sep_conv_3x3': 1,
    'sep_conv_5x5': 2,
    'sep_conv_7x7': 3,
    'dil_conv_3x3': 2,
    'dil_conv_5x5': 4,
    'conv_7x1_1x7': 3,
}


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        self.se_layer = None

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]

        return torch.cat([states[i] for i in self._concat], dim=1)

class Cell(nn.Module):

    def __init__(self, genotype, C_stem, C):
        super(Cell, self).__init__()
        print(C_stem, C)

        self.preprocess0 = ReLUConvBN(C_stem, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_stem, C, 1, 1, 0)

        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat 
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob=0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2 # add
            states += [s]

        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkHAR(nn.Module):

    def __init__(self, C, num_classes, layers, genotype, SE=False):
        super(NetworkHAR, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers

        stem_multiplier = 3
        C_stem = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C, C_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )

        C_curr = genotype.channels[0] # search the number of channels
        self.cell = Cell(genotype, C_stem, C_curr)

        mul_lstm = len(genotype.normal_concat)
        self.bilstm = nn.LSTM(C_curr*mul_lstm, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input):

        # Conv Stem
        s0 = s1 = self.stem(input)
        # Search Cell
        s1 = self.cell(s0, s1)
        # BILSTM + FC
        s1 = s1.view(s1.size(0), s1.size(1), -1) # b, c, (t, dim)
        s1 = s1.permute([0,2,1])
        lstm_out, _ = self.bilstm(s1)
        last_time_step = lstm_out[:, -1, :]
        logits = self.classifier(last_time_step)
        return logits



if __name__ == '__main__':
    import utils
    import micro_genotypes as genotypes

    genome = genotypes.NSGANet
    # model = AlterPyramidNetworkCIFAR(30, 10, 20, True, genome, 6, SE=False)
    # model = PyramidNetworkCIFAR(48, 10, 20, True, genome, 22, SE=True)
    model = NetworkHAR(34, 10, 20, True, genome, SE=True)
    # model = GradPyramidNetworkCIFAR(34, 10, 20, True, genome, 4)
    model.droprate = 0.0

    # calculate number of trainable parameters
    print("param size = {}MB".format(utils.count_parameters_in_MB(model)))
