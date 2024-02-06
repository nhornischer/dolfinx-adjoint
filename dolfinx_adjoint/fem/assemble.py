import dolfinx_adjoint.graph as graph
from dolfinx import fem

def assemble_scalar(*args, **kwargs):
    """OVERLOADS: :py:func:dolfinx.fem.assemble_scalar.
    Assemble functional. The returned value is local and not accumulated across processes.
    
    The overloaded function adds the functionality to keep track of the dependencies 
    in the computational graph. The original functionality is kept.
    
    Args: 
        args: Arguments to :py:func:dolfinx.fem.assemble_scalar. 
        kwargs: Keyword arguments to :py:func:dolfinx.fem.assemble_scalar.
        graph: An additional keyword argument to specifier wheter the assemble
            operation should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    Returns:
        The computed scalar on the calling rank

    Note:
        When a form is assembled into a scalar value, the information about its dependencies is lost, 
        and the resulting scalar does not support automatic differentiation. To this end
        the graph is used to keep track of the dependencies between the resulting scalar value and the
        used form to obtain the scalar value.

    """
    if not "graph" in kwargs:
        output = fem.assemble_scalar(*args, **kwargs)
    else:
        _graph = kwargs["graph"]
        del kwargs["graph"]
        output = fem.assemble_scalar(*args, **kwargs)

        # Creating and adding node to graph
        assemble_node = AssembleScalarNode(output, args[0])
        _graph.add_node(assemble_node)

        # Create edge between form and assemble
        form_node = _graph.get_node(id(args[0]))

        assemble_edge = graph.Edge(form_node, assemble_node)
        assemble_node.set_gradFuncs([assemble_edge])

        # Create connectivity to previous edges
        assemble_edge.set_next_functions(form_node.get_gradFuncs())
        _graph.add_edge(assemble_edge)

    return output

class AssembleScalarNode(graph.Node):
    def __init__(self, object, M : fem.Form):
        super().__init__(object)
        self.M = M
        self._name = "AssembleScalar"

    def __call__(self):
        output = fem.assemble_scalar(self.M)
        self.object = output
        return output
