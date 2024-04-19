import torch
from torch import nn

class BN_Model(object):
    def __init__(self,model, rank):
        self.rank = rank
        self.model=model
        self.MSE_loss = nn.MSELoss().to(rank)
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.save_BN_mean = []
        self.save_BN_var = []
        self.model.eval()
        self.register_bn()

    def register_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)
        
    def get_BN(self,x):
        self.mean_list.clear()
        self.var_list.clear()
        BNS_loss = torch.zeros(1).to(self.rank)
        feature=self.model(x)

        for num in range(len(self.mean_list)):
            BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
                self.var_list[num], self.teacher_running_var[num])

        BNS_loss = BNS_loss / len(self.mean_list)
        return BNS_loss, feature