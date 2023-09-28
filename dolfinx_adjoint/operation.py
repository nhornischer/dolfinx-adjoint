from . import Node 

class Operation:
    def __init__(self, inputs : list or Node, output : Node):
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.inputs = inputs
        self.output = output

    def __str__(self):
        return f"Operation: {[str(input) for input in self.inputs]} -> {str(self.output)}"
    
    def __call__(self):
        pass
