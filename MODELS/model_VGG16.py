import torch
import torch.nn as nn
import torchvision.models as models
from iterative_normalization import IterNormRotation as cw_layer

class VGGBN(nn.Module):
    def __init__(self, num_classes, args, arch = 'vgg16_bn', model_file = None):
        super(VGGBN, self).__init__()
        self.model = models.vgg16_bn(num_classes = num_classes)
        if model_file != None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k,'module.model.',''): v for k,v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
    
    
class VGGBNTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch = 'vgg16_bn', model_file = None):
        super(VGGBNTransfer, self).__init__()
        self.model = models.vgg16_bn(num_classes = num_classes)
        if model_file != None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k,'module.model.',''): v for k,v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

        self.whitened_layers = whitened_layers
        self.layers = [1,4,8,11,15,18,21,25,28,31,35,38,41]
        for whitened_layer in whitened_layers:
            whitened_layer -= 1
            if whitened_layer in range(0,2):
                channel = 64
            elif whitened_layer in range(2,4):
                channel = 128
            elif whitened_layer in range(4,7):
                channel = 256
            else:
                channel = 512
            self.model.features[self.layers[whitened_layer]] = cw_layer(channel, activation_mode = 'pool_max')

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[layers[whitened_layer-1]].mode = mode
    
    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[layers[whitened_layer-1]].update_rotation_matrix()

    def forward(self, x):
        return self.model(x)