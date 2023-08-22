from .node import Node

class Edge:
    def __init__(self, predecessor : Node, successor : Node, input_value = None, ctx = None):
        self.predecessor = predecessor
        self.successor = successor
        self.input_value = None
        self.next_functions = []
        self.ctx = ctx

    def set_next_functions(self, funcList):
        self.next_functions  = funcList

    def remove_next_function(self, func):
        self.next_functions.remove(func)

    def calculate_tlm(self):
        """
        This method calculates the default tangent linear model (TLM) for
        the edge, which corresponds to the derivative
            ∂successor/∂predecessor = 1.0
        
        This operator is stored in the edge and applied to the input. The adjoint value of the predeccessor
        node is generally calculated as follows:
            adjoint(predecessor) += adjoint(successor) * ∂successor/∂predecessor
        """
        self.tlm = 1.0

        return self.input_value

    def calculate_adjoint(self):
        """This method is a placeholder for variants of one part of the chain rule,
        using an adjoint method.

        Most of the times, the adjoint method is the same as the TLM method,
        but it can be different for some cases, e.g. when the edge represents
        a projection.

        Per default we use the TLM method.
        """
        return self.calculate_tlm()
        
    def __call__(self, value):
        self.input_value = value
        print(f"Call to {self.__class__.__name__} from {str(self.successor)} to {str(self.predecessor)}")
        grad_value = self.calculate_adjoint()
        for function in self.next_functions:
            function(grad_value)
        if self.next_functions == [] and type(self.predecessor) == Node:
            self.predecessor.accumulate_grad(grad_value)
    def __str__(self):
        return f"{str(self.predecessor)} -> {str(self.successor)}"