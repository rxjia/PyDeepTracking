"""
use load_lua from pytorch 0.4
"""

import torch
from torch.utils.serialization import load_lua
import os

home_dir = os.environ['HOME']
data_dir = os.path.join(home_dir, 'data')
data_path = os.path.join(data_dir, 'DeepTracking_1_1.t7/data.t7')

a = load_lua(data_path)
torch.save(a, os.path.join(data_dir, 'DeepTracking_1_1.t7/data.t1'))
