import ctypes
class AbstractNode:
    def __init__(self, object, version = 0, **kwargs):
        self.id = id(object)
        self.version = version
        self.object = object
        if version != 0:
            object.version = version
        self.gradFuncs = []
        if "name" in kwargs:
            self._name = kwargs["name"]
        else:
            self._name = str(object.__class__.__name__)

    def set_object(self, object):
        self.object = object

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
    
    def __call__(self, *args, **kwargs):
        # This function recomputes the node with the given arguments
        pass

    def __str__(self):
        return str(self.name)
    
    def __del__(self):
        del self

class Node(AbstractNode): 
    """
    These nodes of the graph are used as context and operation variables. 
    """
    def __init__(self, object, **kwargs):
        super().__init__(object, **kwargs)
        import copy
        self.data = copy.copy(object)
        self.grad = None

    def get_object(self):
        if hasattr(self, "object"):
            return self.object
        return ctypes.cast(self.id, ctypes.py_object).value

    def set_grad(self, value):
        self.grad = value

    def get_grad(self):
        return self.grad
    
    def reset_grad(self):
        self.grad = None

    def accumulate_grad(self, value):
        import numpy as np
        if self.grad is None:
            self.grad = value
        else:
            self.grad += value

    def __del__(self):
        del self.data
        del self.grad
        return super().__del__()