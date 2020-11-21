from torch.utils.tensorboard import SummaryWriter
writer= SummaryWriter

_MIN = 0
_MAX = 300
def make_graph(loss,val, epoch):
    """
    This is a method to draw a graph by using our model's output. 
    This would be used to check our model's performance before we connect with our website.
    The period must be twice of the input sequence
        X axis (Period)         : Input sequence + output sequence
        Y axis (Blood sugar)    : Input values, output values
        
    """
    # TODO
    writer.add_scaler('train_likelihood', loss, epoch)
    writer.add_scaler('validation_mse', val, epoch)
    writer.close()
    return 0

class Normalization(object):
    def __init__(self, min = _MIN, max = _MAX):
        super().__init__()
        self.min = min
        self.max = max

    def normalize(self,data):
        # x- min/ max-min
        norm_data = (data-self.min) / (self.max - self.min)
        return norm_data

    def de_normalize(self, data):
        de_norm_data = data*(self.max-self.min)+self.min
        return de_norm_data
    
