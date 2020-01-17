import argparse
import math
import torch
import os
from torch import optim

from CustomLoss import CustomLoss
from Recurrent import Recurrent, StepModule
from DataShow import save_tensor_img

home_dir = os.environ['HOME']
data_dir = os.path.join(home_dir, 'data/DeepTracking_1_1.t7')
data_file = os.path.join(data_dir, 'data.t1')

from SensorData import LoadSensorData

# DEFAULT_TENSOR_TYPE = torch.FloatTensor
DEFAULT_TENSOR_TYPE = torch.cuda.FloatTensor
torch.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


class NC:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-gpu', type=int, default=0, help='use GPU')
    parser.add_argument('-iter', type=int, default=100000, help='the number of training iterations')
    parser.add_argument('-N', type=int, default=100, help='training sequence length')
    parser.add_argument('-model', type=str, default='model', help='neural network model')
    parser.add_argument('-data', type=str, default=data_file, help='training data')
    parser.add_argument('-learningRate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-initweights', type=float, default=0, help='initial weights')

    parser.add_argument('-grid_minX', type=float, default=-25, help='occupancy grid bounds [m]')
    parser.add_argument('-grid_maxX', type=float, default=25, help='occupancy grid bounds [m]')
    parser.add_argument('-grid_minY', type=float, default=-45, help='occupancy grid bounds [m]')
    parser.add_argument('-grid_maxY', type=float, default=5, help='occupancy grid bounds [m]')
    parser.add_argument('-grid_step', type=float, default=1, help='resolution of the occupancy grid [m]')
    parser.add_argument('-sensor_start', type=float, default=-180, help='first depth measurement [degrees]')
    parser.add_argument('-sensor_step', type=float, default=0.5, help='resolution of depth measurements [degrees]')
    params = parser.parse_args()

    params = NC()
    params.gpu = 0
    params.iter = 100000
    params.N = 100
    params.model = 'model'
    params.data = data_file
    params.learningRate = 0.01
    params.initweights = 0
    params.grid_minX = -25
    params.grid_maxX = 25
    params.grid_minY = -45
    params.grid_maxY = 5
    params.grid_step = 1
    params.sensor_start = -180
    params.sensor_step = 0.5

    data = LoadSensorData(params.data, params)
    width = data.width  # occupancy 2D grid width
    height = data.height  # occupancy 2D grid height
    print('Occupancy grid has size ' + str(width) + 'x' + str(height))
    M = math.floor(data.data.size(0) / params.N)  # total number of training sequences
    print('Number of sequences ' + str(M))

    # one step of RNN
    step = StepModule()
    # network weights + gradients
    # w = step.parameters()
    # print('Model has ' + str(len(list(w))) + ' parameters')
    #
    # if params.initweights > 0:
    #     print('Loading weights ' + params.initweights)
    #     w.copy(torch.load(params.initweights))

    # chain N steps into a recurrent neural network
    model = Recurrent(step, params.N, width, height)
    print(model)
    # initial hidden state
    h0 = model.get_initial_state()

    # cost function
    # {y1, y2, ..., yN},{t1, t2, ..., tN} -> cost
    criterion = CustomLoss()


    # return i-th training sequence
    def getSequence(i):
        input = torch.Tensor(params.N, 2, height, width)
        for j in range(params.N):
            input[j] = data[i * params.N + j].type(DEFAULT_TENSOR_TYPE)
        return input


    input = getSequence(0)
    save_tensor_img(input[1][1] / 2, '/home/do/data/a.png')
    save_tensor_img(input[1][0], '/home/do/data/b.png')
    save_tensor_img(input[1][1], '/home/do/data/c.png')
    save_tensor_img(input[1][1] / 2 + input[1][0], '/home/do/data/d.png')


    # filter and save model performance on a sample sequence
    def evalModel():
        with torch.no_grad():
            input = getSequence(0)
            output = model(h0, input)

            # temporarily switch to FloatTensor as image does not work otherwise.
            torch.set_default_tensor_type(torch.FloatTensor)
            for i in range(len(input)):
                save_tensor_img(input[i][1] / 2 + input[i][0],
                                data_dir + '/video_' + params.model + '/ainput/' + str(i) + '.png')
                save_tensor_img(input[i][1] / 2 + output[i],
                                data_dir + '/video_' + params.model + '/aoutput/' + str(i) + '.png')
            torch.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


    # blanks part of the sequence for predictive training
    def dropoutInput(target):
        input = torch.empty_like(target)
        for i in range(len(target)):
            input[i] = target[i].clone()
            if i % 20 >= 10:
                input[i].zero_()
        return input


    # create directory to save weights and videos
    dir = os.path.join(data_dir, 'weights_' + params.model)
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = os.path.join(data_dir, 'video_' + params.model)
    if not os.path.exists(dir):
        os.mkdir(dir)

    optimizer = optim.Adagrad(model.parameters(), lr=params.learningRate)
    running_loss = 0.0
    p_size = 100
    s_size = 2000
    for k in range(params.iter):

        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs
        target = getSequence(torch.randint(low=0, high=M, size=[]))
        input = dropoutInput(target)

        # forward pass
        output = model(h0, input)

        loss = criterion(output, target)

        # backward pass
        loss.backward()

        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (k + 1) % p_size == 0:  # print every 2000 mini-batches
            print('Iteration' + str(k) + ', lost: ' + str(running_loss / p_size))
            running_loss = 0.0

            if (k + 1) % s_size == 0:
                for i in range(len(input)):
                    save_tensor_img(output[i],
                                    data_dir + '/video_' + params.model + '/output/' + str(i) + '.png')
                # save weights
                state = {'net': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'k': k}
                torch.save(state, data_dir + '/' + 'weights_' + params.model + '/' + str(k) + '.dat')
                # visualise performance
                evalModel()
