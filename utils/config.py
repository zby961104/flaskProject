import torch.nn as nn


class Configs(object):
    def __init__(self, ParamList):
        # initiale a Config object with given paramlist
        super(Configs, self).__init__()
        self.args = {}
        for param in ParamList.keys():
            self.set_args(param, ParamList.get(param))

    def set_args(self, argname, value):
        if argname in self.args:
            print("Arg", argname, "is updated.")
            self.args[argname] = value
        else:
            print('Arg', argname, 'is added.')
            self.args.update({argname: value})

    def get_args(self, argname, value):
        if argname in self.args:
            return self.args[argname]
        else:
            return value


def build_criterion(classNum, taskNum, device):
    if classNum == 2:
        criterion = [nn.CrossEntropyLoss().to(device) for i in range(taskNum)]
    else:
        criterion = [nn.MSELoss().to(device) for i in range(taskNum)]

    return criterion
