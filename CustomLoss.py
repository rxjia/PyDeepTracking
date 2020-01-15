import torch


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, otarget):
        # -log(input) * target - log(1 - input) * (1 - target)
        eps = 1e-12
        target, weights = torch.chunk(otarget, 2, 1)
        buf = torch.mul(torch.log(torch.add(input, eps)), weights)
        out = -torch.mul(target, buf)
        buf = torch.mul(torch.log(torch.add(torch.add(torch.mul(input, -1), 1), eps)), weights)
        out = torch.mean(out - buf + torch.mul(target, buf))
        return out

    # def forward(self, oinput, otarget):
    #     eps = 1e-12
    #     loss = torch.empty(len(otarget))
    #     for i in range(len(otarget)):
    #         t0 = otarget[i]
    #         input = oinput[i].squeeze(0)
    #         target, weights = t0
    #         # logxn = torch.mul(torch.log(torch.add(input, eps)), weights)
    #         # ynlogxn = torch.mul(target, logxn)
    #         #
    #         # log1xn = torch.mul(torch.log(torch.add(torch.add(torch.mul(input, -1), 1), eps)), weights)
    #         # ynlog1xn = torch.mul(target, log1xn)
    #         # out = -(ynlogxn + log1xn - ynlog1xn)
    #         # outsum = torch.mean(out)
    #         # loss[i] = outsum
    #
    #         # buf = torch.empty_like(input, requires_grad=True)
    #         buf = torch.mul(torch.log(torch.add(input, eps)), weights)
    #         # out = - torch.dot(target, buf)
    #         out = -torch.sum(torch.mul(target, buf))
    #         buf = torch.mul(torch.log(torch.add(torch.add(torch.mul(input, -1), 1), eps)), weights)
    #         out = (out - torch.sum(buf) + torch.sum(torch.mul(target, buf))) / input.nelement()
    #         loss[i] = out
    #     return torch.mean(loss)
    #     # - log(input) * target - log(1 - input) * (1 - target)
