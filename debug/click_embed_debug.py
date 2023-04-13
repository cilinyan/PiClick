import torch
import torch.nn as nn
import math
import pdb
from mmcv.runner import BaseModule


class SinePositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None,
                 **kwargs):
        super(SinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                                                    'scale should be provided and in float or int type, ' \
                                                    f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        click_pos_enc_cfg: dict = dict(type='SinePositionalEncoding', num_feats=128, normalize=True)

        assert click_pos_enc_cfg['type'] == 'SinePositionalEncoding'
        self.click_positional_encoding = SinePositionalEncoding(**click_pos_enc_cfg)
        self.positive_positional_embed = nn.Embedding(1, click_pos_enc_cfg['num_feats'] * 2)  # [1, num_feats*2]
        self.negative_positional_embed = nn.Embedding(1, click_pos_enc_cfg['num_feats'] * 2)  # [1, num_feats*2]

        self.image_size = (448, 448)
        self.device = torch.device('cpu')

    def get_single_click_embedding(self, x):
        # x (Tensor): shape [24, 2]
        points_attn_mask = (x[:, 2] == -1)
        zero_mask = torch.zeros((1, *self.image_size), dtype=torch.bool, device=self.device, requires_grad=False)
        click_pos_embed = self.click_positional_encoding(zero_mask)  # [1, num_feats*2, h, w]

        n = x.shape[0] // 2
        x_pos, x_neg = x[:n], x[n:]

        x_pos_emb = torch.stack([
            click_pos_embed[0, :, x0, x1] + self.positive_positional_embed.weight[0]
            if flag.item() != -1 else torch.zeros_like(click_pos_embed[0, :, x0, x1])
            for x0, x1, flag in x_pos
        ])  # n, 1, num_feats*2
        x_neg_emb = torch.stack([
            click_pos_embed[0, :, x0, x1] + self.negative_positional_embed.weight[0]
            if flag.item() != -1 else torch.zeros_like(click_pos_embed[0, :, x0, x1])
            for x0, x1, flag in x_neg
        ])  # n, 1, num_feats*2
        x_emb = torch.cat([x_pos_emb, x_neg_emb], dim=0)
        return x_emb, points_attn_mask

    def get_single_click_embedding_huge(self, x):
        # x (Tensor): shape [24, 2]
        points_attn_mask = (x[:, 2] == -1)
        zero_mask = torch.zeros((1, 672, 672), dtype=torch.bool, device=self.device, requires_grad=False)
        click_pos_embed = self.click_positional_encoding(zero_mask)  # [1, num_feats*2, h, w]

        n = x.shape[0] // 2
        x_pos, x_neg = x[:n], x[n:]

        x_pos_emb = torch.stack([
            click_pos_embed[0, :, x0, x1] + self.positive_positional_embed.weight[0]
            if flag.item() != -1 else torch.zeros_like(click_pos_embed[0, :, x0, x1])
            for x0, x1, flag in x_pos
        ])  # n, 1, num_feats*2
        x_neg_emb = torch.stack([
            click_pos_embed[0, :, x0, x1] + self.negative_positional_embed.weight[0]
            if flag.item() != -1 else torch.zeros_like(click_pos_embed[0, :, x0, x1])
            for x0, x1, flag in x_neg
        ])  # n, 1, num_feats*2
        x_emb = torch.cat([x_pos_emb, x_neg_emb], dim=0)
        return x_emb, points_attn_mask


def main():
    net = Model()
    x = torch.tensor([
        [[1, 1, 0], [2, 20, 100], [2, 1, -1], [99, 212, -1], [100, 222, 1], [111, 333, 2], ],
        [[1, 1, 0], [2, 20, 100], [2, 1, 100], [99, 212, -1], [100, 222, 1], [111, 333, 2], ],
    ], dtype=torch.long)
    y1, a1 = net.get_single_click_embedding(x[0])
    y2, a2 = net.get_single_click_embedding_huge(x[0])
    pass


if __name__ == '__main__':
    main()
