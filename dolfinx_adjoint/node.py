class AbstractNode:
    def __init__(self, object, **kwargs):
        self.id = id(object)
        self.object = object
        self.gradFuncs = []
        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = str(self.object.__module__ + "."+ self.object.__class__.__name__)

    def set_gradFuncs(self, list):
        self.gradFuncs = list

    def append_gradFuncs(self, _list : list or object):
        if isinstance(_list, list):
            self.gradFuncs.extend(_list)
        else:
            self.gradFuncs.append(_list)

    def get_gradFuncs(self):
        return self.gradFuncs

    def set_adjoint_value(self, value):
        self.adjoint_value = value

    def get_adjoint_value(self):
        return self.adjoint_value
    
    def reset_adjoint_value(self):
        self.adjoint_value = None

    def __str__(self):
        return str(self.name)

class Node(AbstractNode): 
    """
    These nodes of the graph are used as context and operation variables. 
    Since we do not want"""
    def __init__(self, object, **kwargs):
        super().__init__(object, **kwargs)
        self.grad = None

    def set_adjoint_value(self, value):
        self.adjoint_value = value

    def get_adjoint_value(self):
        return self.adjoint_value
    
    def reset_adjoint_value(self):
        self.adjoint_value = None