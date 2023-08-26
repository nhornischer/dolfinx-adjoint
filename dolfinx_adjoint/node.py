class AbstractNode:
    def __init__(self, object, version = 0, **kwargs):
        self.id = id(object)
        self.version = version
        if version != 0:
            object.version = version
        self.gradFuncs = []
        if "name" in kwargs:
            self._name = kwargs["name"]
        else:
            self._name = str(object.__class__.__name__)

    @property
    def name(self):
        if self.version != 0:
            return f"{self._name} [{str(self.version)}]"
        else:
            return str(self._name)

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
    """
    def __init__(self, object, **kwargs):
        super().__init__(object, **kwargs)
        import copy
        self.data = copy.copy(object)
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
                print(f"Accumulating gradient for {self.name}: {np.shape(value)}")
            else:
                print(f"Accumulating gradient for {self.name}: {value}")
            self.grad = value
        else:
            if type(value) == np.ndarray:
                print(f"Accumulating gradient for {self.name}: {np.shape(self.grad)} + {np.shape(value)}")
            else:
                print(f"Accumulating gradient for {self.name}: {self.grad} + {value}")
            self.grad += value