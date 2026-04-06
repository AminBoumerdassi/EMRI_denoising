import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
import visualtorch
from torch import nn
from torchview import draw_graph
import graphviz


#Load model
from src.tools.model_architecture import *

#Create instance of model
model= ConvAE_2D()

batch_size=32
no_chans=2
height=512
width=512

#Define input shape
input_shape = (batch_size, no_chans, height, width)


'''A model visualisation with visualtorch'''
#Layered view with 
img = visualtorch.layered_view(model, input_shape=input_shape, legend=True, draw_volume=True)

#LeNet style view
# img = visualtorch.lenet_view(model, input_shape=input_shape)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.savefig("test_model_visualtorch.png")


'''A model visualisation with torchview'''
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_size=input_shape, device='meta',
                          save_graph=True, filename="test_model_torchview",
                          expand_nested=True, graph_dir='LR')
# model_graph.resize_graph()

