import torch
from torch import nn
from models import resnet

def generate_model():
    model = resnet.generate_model(model_depth = 50,
                                      n_classes = 1139,
                                      n_input_channels = 3,
                                      shortcut_type = 'B',
                                      conv1_t_size = 7,
                                      conv1_t_stride = 1,
                                      no_max_pool = False,
                                      widen_factor = 1.0)
    return model


def load_pretrained_model(model, pretrain_path):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
    return model