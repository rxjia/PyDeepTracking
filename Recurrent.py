"""
DeepTracking: Seeing Beyond Seeing Using Recurrent Neural Networks.
Copyright (C) 2016  Peter Ondruska, Mobile Robotics Group, University of Oxford
email:   ondruska@robots.ox.ac.uk.
webpage: http://mrg.robots.ox.ac.uk/

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

# N steps of RNN
# {x1, x2, ..., xN, h0} -> {y1, y2, ..., yN}

# """
# # A classical implementation of the recurrent module; requires O(N * M(step)) memory.
# def Recurrent(step, N)
#    h = nn.Identity()
#    hx, y = { [params.N+1] = h }, {}
#    for i=1,N do
#       hx[i] = nn.Identity()
#       hy = step:clone('weight', 'bias', 'gradWeight', 'gradBias')({h, hx[i]})
#       h    = nn.SelectTable(1)(hy)
#       y[i] = nn.SelectTable(2)(hy)
#    end
#    return nn.gModule(hx, y)
# end
# """

import torch
from torch import nn


# One step of RNN --
# {h0,x1} -> {h1,y1}
class StepModule(nn.Module):
    def __init__(self):
        super(StepModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, (7, 7), (1, 1), (3, 3))
        self.conv2 = nn.Conv2d(48, 32, (7, 7), (1, 1), (3, 3))
        self.conv3 = nn.Conv2d(32, 1, (7, 7), (1, 1), (3, 3))

    def forward(self, h0, x1):
        e = torch.sigmoid(self.conv1(x1))
        j = torch.cat([e, h0], 1)
        h1 = torch.sigmoid(self.conv2(j))
        y1 = torch.sigmoid(self.conv3(h1))
        return h1, y1


class Recurrent(torch.nn.Module):
    def __init__(self, step, N, height, width):
        super(Recurrent, self).__init__()
        self.N = N
        self.step = step
        self.height = height
        self.width = width

    def forward(self, hidden, input):
        output = torch.Tensor(self.N, 1, self.height, self.width)
        for i in range(0, self.N):
            stepOutput = self.step.forward(hidden, input[i].unsqueeze(0))
            hidden = stepOutput[0]
            output[i] = stepOutput[1]
        return output

    def get_initial_state(self):
        return torch.zeros(1, 32, self.height, self.width)
