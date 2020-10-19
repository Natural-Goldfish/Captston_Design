_MIN = 0
_MAX = 300
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
    
class Normalization():
    def __init__(self, min= _MIN, max = _MAX):
        self.min = min
        self.max = max

    def normalize(self, data):
        norm_data = (data - self.min)/(self.max - self.min)
        return norm_data

    def de_normalize(self, data):
        denorm_data = data*(self.max - self.min) + self.min
        return denorm_data