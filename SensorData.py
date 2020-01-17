import math
import torch


class SensorData:
    def __init__(self):
        self.index = None
        self.dist = None
        self.params = None
        self.width = None
        self.height = None
        self.data = None

    def __getitem__(self, i):
        dist = self.data[i].index_select(0, self.index - 1).reshape(self.height, self.width)
        input = torch.FloatTensor(2, self.height, self.width)
        input[0] = torch.lt(torch.abs(dist - self.dist), self.params.grid_step * 0.7071)
        input[1] = torch.gt(dist + self.params.grid_step * 0.7071, self.dist)
        return input


def LoadSensorData(file, params):
    self = SensorData()
    self.params = params
    # -- load raw 1D depth sensor data
    self.data = torch.load(file)
    self.width = int((params.grid_maxX - params.grid_minX) / params.grid_step + 1)
    self.height = int((params.grid_maxY - params.grid_minY) / params.grid_step + 1)
    # -- pre-compute lookup arrays
    self.dist = torch.FloatTensor(self.height, self.width)
    self.index = torch.LongTensor(self.height, self.width)
    for y in range(self.height):
        for x in range(self.width):
            px = x * params.grid_step + params.grid_minX
            py = y * params.grid_step + params.grid_minY
            angle = math.degrees(math.atan2(px, py))
            self.dist[y][x] = math.sqrt(px * px + py * py)
            self.index[y][x] = math.floor((angle - params.sensor_start) / params.sensor_step + 1.5)
    self.index = self.index.reshape(self.width * self.height)

    return self
