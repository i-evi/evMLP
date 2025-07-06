import torch.nn as nn
from einops.layers.torch import Rearrange

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class ResidualBlock(nn.Module):
  def __init__(self, dim, expansion_factor=4, dropout_p=0.):
    super().__init__()
    inner_dim = int(dim * expansion_factor)
    self.fn = nn.Sequential(
      nn.Linear(dim, inner_dim, bias=True),
      nn.GELU(),
      nn.Dropout(dropout_p),
      nn.Linear(inner_dim, dim, bias=True),
    )
    self.norm = nn.LayerNorm(dim)
  def forward(self, x):
    return self.norm(self.fn(x) + x)

def evMLPBlock(image_size,
  channels, patch_size, dim, depth, expansion_factor, dropout_p=0.0):
  image_h, image_w = pair(image_size)
  num_patches = (image_h // patch_size) * (image_w // patch_size)
  num_patches_v = image_h // patch_size
  num_patches_h = image_w // patch_size
  blk = nn.Sequential(
    Rearrange('(b h p1 v p2) c -> (b h v) (p1 p2 c)',
      h=num_patches_h, v=num_patches_v, p1=patch_size, p2=patch_size),
    nn.Linear((patch_size ** 2) * channels, dim, bias=True),
  )
  for _ in range(depth):
    blk.append(ResidualBlock(
      dim, expansion_factor, dropout_p
    ))

  return blk

class evMLP(nn.Module):
  def __init__(self,
      classes = 1000,
      image_size = 224,
      image_channels = 3,
      config = [ # [dim, patch_size, depth, expansion_factor, dropout]
        [ 64, 7, 5, 4, 0  ],
        [128, 2, 5, 4, 0  ],
        [512, 2, 5, 4, 0  ],
        [512, 2, 5, 4, 0  ],
        [512, 2, 5, 4, 0  ],
        [512, 2, 5, 4, 0.2],
      ]
    ):
    super(evMLP, self).__init__()
    self.classes = classes
    self.image_size = image_size
    self.image_channels = image_channels
    self.blks = nn.Sequential()
    self.config = config
    curr_image_size = image_size
    curr_dim = image_channels
    self.rearrange_inp = Rearrange('b c h v -> (b h v) c')
    for i in config:
      dim, patch, depth, ef, drop = i
      self.blks.append(
        evMLPBlock(
          image_size = curr_image_size,
          channels = curr_dim,
          patch_size = patch,
          dim = dim,
          depth = depth,
          expansion_factor = ef,
          dropout_p = drop
        )
      )
      curr_dim = dim
      curr_image_size = int(curr_image_size / patch)

    self.linear = nn.Linear(curr_dim, self.classes)

  def forward(self, x):
    x = self.rearrange_inp(x)
    for i, blk in enumerate(self.blks):
      x = blk(x)
    x = self.linear(x)
    x = x.view(x.size(0), -1)
    return x

