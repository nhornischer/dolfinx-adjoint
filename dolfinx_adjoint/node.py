class AbstractNode:
    def __init__(self, object, **kwargs):
        self.id = id(object)
        self.gradFuncs = []
        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = str(object.__module__ + "."+ object.__class__.__name__)

    def set_gradFuncs(self, list):
        self.gradFuncs = list

    def append_gradFuncs(self, _list : list or object):
        if isinstance(_list, list):
            self.gradFuncs.extend(_list)
        else:
            self.gradFuncs.append(_list)

    def get_gradFuncs(self):
        return self.gradFuncs

    def __str__(self):
        return str(self.name)

class Node(AbstractNode): 
    """
    These nodes of the graph are used as context and operation variables. 
    Since we do not want"""
    def __init__(self, object, **kwargs):
        super().__init__(object, **kwargs)
        self.grad = None

    def set_grad(self, value):
        self.grad = value

    def get_grad(self):
        return self.grad
    
    def reset_grad(self):
        self.grad = None

    def accumulate_grad(self, value):
        import numpy as np
        if self.grad is None:
            print(f"Accumulating gradient: {np.shape(value)}")
            self.grad = value
        else:
            print(f"Accumulating gradient: {np.shape(self.grad)} + {np.shape(value)}")
            self.grad += value