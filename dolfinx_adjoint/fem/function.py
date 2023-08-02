from dolfinx import fem
import dolfinx_adjoint.graph as graph


class Function(fem.Function):
    def __init__(self, *args, **kwargs):
        if not "name" in kwargs:
            kwargs["name"] = "Function"
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            function_node = graph.Node(self, name=self.name)
            _graph.add_node(function_node)

    
class Constant(object):
    def __new__(cls, *args, **kwargs):
        if "graph" in kwargs:
            domain, c = args
            if not "name" in kwargs:
                kwargs["name"] = "Constant"
            DG0 = fem.FunctionSpace(domain, ("DG", 0))
            instance = Function(DG0, **kwargs)
            instance.vector.array[:] = c
            instance.c = c

        else:
            if "name" in kwargs:
                del kwargs["name"]
            instance = fem.Constant(*args, **kwargs)
        return instance
        
    def __init__(self, *args, **kwargs):
        if "graph" in kwargs:
            _graph = kwargs["graph"]

            Constant_node = graph.Node(self)
            _graph.add_node(Constant_node)



        
