import torch

class Embedding(object):
    def __init__(self, x_min = 0, x_max = 12, batch_num = 1, input_dim= 1):
        super().__init__()
        self.x_min = torch.tensor(x_min, dtype = torch.float32)
        self.x_max = torch.tensor(x_max, dtype = torch.float32)
        self.batch_num = batch_num
        self.input_dim = input_dim
    
    def __call__(self, x):
        output_list = []
        for data in x:
            data = torch.tensor(self.norm(data), dtype = torch.float32, requires_grad = True).view(1, 1).contiguous()
            output_list.append(data)
        
        for i in range(1, len(x)):
            output_list[0] = torch.cat((output_list[0], output_list[i]), dim = 0)
        output = output_list[0]
        output = output.view(len(x), self.batch_num, self.input_dim)
        return output
    
    def _norm(self, x):
        x = torch.tensor(x, dtype = torch.float32)
        norm_x = (x - self.xmin)/(self.xmax - self.xmin)
        return norm_x

def linear_interpolation():
    """
    This is interpolating our dataset
        Find missing parts in the dataset and add values
    """
    # TODO

    return 0

def make_graph():
    """
    This is a method to draw a graph by using our model's output. 
    This would be used to check our model's performance before we connect with our website.
    The period must be twice of the input sequence
        X axis (Period)         : Input sequence + output sequence
        Y axis (Blood sugar)    : Input values, output values
        
    """
    # TODO
    
    return 0
    