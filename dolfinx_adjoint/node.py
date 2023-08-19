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

    def remove_gradFunc(self, gradFunc):
        self.gradFuncs.remove(gradFunc)

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
        self.version = 0
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
            if type(value) == np.ndarray:
                print(f"Accumulating gradient: {np.shape(value)}")
            else:
                print(f"Accumulating gradient: {value}")
            self.grad = value
        else:
            if type(value) == np.ndarray:
                print(f"Accumulating gradient: {np.shape(self.grad)} + {np.shape(value)}")
            else:
                print(f"Accumulating gradient: {self.grad} + {value}")
            self.grad += value