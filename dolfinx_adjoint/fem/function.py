from dolfinx import fem
import dolfinx_adjoint.graph as graph


class Function(fem.Function):
    def __init__(self, *args, **kwargs):
        if not "name" in kwargs:
            kwargs["name"] = "f"
        if "map" in kwargs:
            map = kwargs["map"]
            del kwargs["map"]
        else: 
            map = None
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)
            self.map = map
            function_node = graph.Node(self, name=self.name)
            _graph.add_node(function_node)

    def copy(self, *args, **kwargs):
        if not "name" in kwargs:
            kwargs["name"] = "f_copy"
        if not "graph" in kwargs:
            return Function(self.function_space, type(self.x)(self.x), **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            output = Function(self.function_space, type(self.x)(self.x), **kwargs)
            copy_node = graph.Node(output, name=output.name)
            _graph.add_node(copy_node)

            # Create a standard edge between the function at its copy since they are both identical
            original_node = _graph.get_node(id(self))
            copy_edge = graph.Edge(original_node, copy_node)
            copy_edge.set_next_functions(original_node.get_gradFuncs())
            copy_node.append_gradFuncs(copy_edge)

            _graph.add_edge(copy_edge)
            return output
        
    def assign(self, vector, graph = None):
        self.vector[:] = vector
        if graph is not None:
            function_node = graph.get_node(id(self))
            function_node.version += 1

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