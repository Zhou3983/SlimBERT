import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.autograd import Variable


class SlimLayer(Module):
    def __init__(self, dim, mask_thresh=0, kl_mult=1, divide_w=False, sample_in_training=True, sample_in_testing=True, masking=False):
        super(SlimLayer, self).__init__()
        self.alpha = Parameter(torch.Tensor(dim))

        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        # if masking=True, apply mask directly
        self.masking = masking

        # initialization
        self.alpha.data.fill_(1)

        self.mask_thresh = mask_thresh
        self.kl_mult = kl_mult
        self.divide_w = divide_w

    def adapt_shape(self, src_shape, x_shape):
        new_shape = src_shape if len(src_shape) == 2 else (1, src_shape[0])
        if len(x_shape) > 2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape) - 2)]
        return new_shape

    def get_logalpha(self):
        return self.alpha

    def get_mask_hard(self, threshold=0.001):
        logalpha = self.get_logalpha()
        logalpha = torch.abs(logalpha)
        hard_mask = (logalpha > threshold).float()
        return hard_mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        mask = (logalpha < threshold).float() * self.alpha.data
        return mask

    def forward(self, x):
        if self.masking:
            mask = self.get_mask_hard(self.mask_thresh)
            return x * mask

        if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
            z_scale = self.alpha.view(1, -1)
            if not self.training:
                z_scale *= Variable(self.get_mask_hard(self.mask_thresh))
        self.kld = self.kl_closed_form(x)
        return x * z_scale

    def kl_closed_form(self, x):
        new_shape = self.adapt_shape(self.alpha.size(), x.size())
        h_mu = self.alpha.view(new_shape)
        KLD = torch.sum(torch.log(1 + h_mu.pow(2)*8000))
        return KLD * 0.5 * self.kl_mult


