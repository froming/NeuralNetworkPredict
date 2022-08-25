#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import torch
from torch import nn as nn

class MamlModel(nn.Module):
    def __init__(self, input_dim, out_dim, x_train, hidden_units):
        super(MamlModel, self).__init__()
        self.X_train = x_train
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, hidden_units)
        self.linear4 = nn.Linear(hidden_units, hidden_units)
        self.linear5 = nn.Linear(hidden_units, out_dim)
    def forward(self, input, input_label):
        # 单层网络的训练
        x = self.linear1(input)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.tanh(x)
        x = self.linear4(x)
        x = torch.relu(x)
        Y_predict = self.linear5(x)
        return input_label, Y_predict

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'EnergyPredict.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()