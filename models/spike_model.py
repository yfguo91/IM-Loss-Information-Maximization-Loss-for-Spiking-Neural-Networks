import torch.nn as nn
from models.spike_layer import SpikeConv, LIFAct, tdBatchNorm2d, SpikePool, SpikeModule, myBatchNorm3d
from models.spike_block import specials, SpikeBasicBlock
from .distrloss_layer import Distrloss_layer
from IPython import embed

class SpikeModel(SpikeModule):

    def __init__(self, model: nn.Module, step=2, distribution = False):
        super().__init__()
        self.model = model
        self.step = step
        self.distribution = distribution
        self.distrloss_layers = []
        self.spike_module_refactor(self.model, step=step)
        self.loss = []
    def spike_module_refactor(self, module: nn.Module, step=2):
        """
        Recursively replace the normal conv2d and Linear layer to SpikeLayer
        """
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, step=step, distribution = self.distribution))
                self.distrloss_layers.append(Distrloss_layer())
                #print("hello")
            elif isinstance(child_module, nn.Sequential):
                self.spike_module_refactor(child_module, step=step)

            elif isinstance(child_module, nn.Conv2d):
                setattr(module, name, SpikePool(child_module, step=step))
                #self.models.append(getattr(module, name))

            elif isinstance(child_module, nn.Linear):
                setattr(module, name, SpikeConv(child_module, step=step))
                #self.models.append(getattr(module, name))

            elif isinstance(child_module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                setattr(module, name, SpikePool(child_module, step=step))
                #self.models.append(getattr(module, name))

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                setattr(module, name, LIFAct(step=step))
                #self.models.append(getattr(module, name))
                self.distrloss_layers.append(Distrloss_layer())
                #print("hello")
            #elif isinstance(child_module, nn.BatchNorm2d):
            #    setattr(module, name, SpikeConv(child_module, step=step))
            #elif isinstance(child_module, nn.BatchNorm2d):
            #    setattr(module, name, tdBatchNorm2d(bn=child_module, alpha=1))
            elif isinstance(child_module, nn.BatchNorm2d):
                setattr(module, name, myBatchNorm3d(child_module, step=step))
                #self.models.append(getattr(module, name))
            
            else:
                self.spike_module_refactor(child_module, step=step)
                
            

    def spike_module_forward(self, module: nn.Module, x):
        """
        Recursively replace the normal conv2d and Linear layer to SpikeLayer
        """
        i = 0
        for name, child_module in module.named_children():
        
            if isinstance(child_module, SpikeBasicBlock):
                out, x = child_module(x)
                self.loss.append(self.distrloss_layers[i](out))
                i = i + 1
                #print("SpikeBasicBlock")
                #print(x.shape)
            elif isinstance(child_module, nn.Sequential):
                x = self.spike_module_forward(child_module, x)
                #print("nn.Sequential")
                #print(x.shape)
            elif isinstance(child_module, SpikePool):
                x = child_module(x)
                #print("SpikePool")
                #print(x.shape)
            elif isinstance(child_module, SpikeConv):
                #print("SpikeConv")
                #print(x.shape)
                if len(x.shape) == 5:
                    x = x.view(x.size(0), x.size(1), -1)
                x = child_module(x)

            elif isinstance(child_module, LIFAct):
                x = child_module(x)
                self.loss.append(self.distrloss_layers[i](x))
                i = i + 1
                #print("LIFAct")
                #print(x.shape)  
            elif isinstance(child_module, myBatchNorm3d):
                x = child_module(x)
                #print("myBatchNorm3d")
                #print(x.shape)
            else:
                x = self.spike_module_forward(child_module, x)
                
        return x
            
    def forward(self, input):
        #print(self.model)
        #embed()
        if self.distribution:
            self.loss = []
            if len(input.shape) == 4:
                x = input.repeat(self.step, 1, 1, 1, 1)
            else:
                x = input.permute([1, 0, 2, 3, 4])
            
            i = 0
            
            x = self.spike_module_forward(self.model,x)
            
            #print(x.shape)
      
            if len(x.shape) == 3:
                out = x.mean([0])
            #print(len(self.loss))
            distrloss = (sum([ele for ele in self.loss])-self.loss[-1]-self.loss[0]) / (len(self.loss)-2)
            
            return out, distrloss
            
        else:
            if len(input.shape) == 4:
                out = input.repeat(self.step, 1, 1, 1, 1)
            else:
                out = input.permute([1, 0, 2, 3, 4])
            
            out = self.model(out)
            
            if len(out.shape) == 3:
                out = out.mean([0])
                
            return out, out
            
    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(use_spike)

    def set_spike_before(self, name):
        self.set_spike_state(False)
        for n, m in self.model.named_modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(True)
            if name == n:
                break


# from models.resnet import resnet20_cifar_modified
# model = SpikeModel(resnet20_cifar_modified())
# model.set_spike_before('layer1')
# for n, m in model.named_modules():
#     if isinstance(m, SpikeModule):
#         if m._spiking is True:
#             print(n)
# import torch
# model(torch.randn(1,3,32,32))
