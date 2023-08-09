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
    
class Constant(fem.Constant):
    def __init__(self, *args, **kwargs):
        if not "graph" in kwargs:
            if "name" in kwargs:
                del kwargs["name"]
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            if "name" in kwargs:
                name = kwargs["name"]
                del kwargs["name"]
            super().__init__(*args, **kwargs)

            kwargs = {"name": name}
            Constant_node = graph.Node(self, **kwargs)
            _graph.add_node(Constant_node)
            self.domain = args[0]
            self.c = args[1]



        
