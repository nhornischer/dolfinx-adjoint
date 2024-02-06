import ctypes
import copy

import petsc4py.PETSc as PETSc

class AbstractNode:
    """
    The base class for nodes in the graph.
    
    AbstractNodes provide the functionality to represent the object and operations in the graph.
    To create a new node, the user should inherit from this class and implement the __call__ method or 
    use the this AbstractNode class to directly create a new node without any additional functionalities.

    The AbstractNode class is used to represent objects without any numerical value.

    Attributes:
        id (int): The python id of the object
        version (int, optional): The version of the object, indicates if an object is an updated version of an
            already existing object. Defaults to 0. 
        object (object): The object that the node represents
        gradFuncs (list): A list of the gradient functions that are connected to the node
        _name (str): The name of the node

    Methods:
        name: Returns the name of the node
        set_object: Sets the object of the node
        set_gradFuncs: Sets the gradient functions of the node
        append_gradFuncs: Appends a gradient function to the list of gradient functions
        get_gradFuncs: Returns the gradient functions of the node

    """
    def __init__(self, object : object, version = 0, **kwargs):
        """
        Constructor for the AbstractNode class.
        
        Args:
            object (object): The object that the node represents
            version (int, optional): The version of the object, indicates if an object is an updated version of an
                already existing object. Defaults to 0. 
            **kwargs: Additional keyword arguments
        
        """
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

    @property
    def name(self):
        """
        Returns the name of the node.
        
        Returns:
            str: The name of the node
        """
        if self.version != 0:
            return f"{self._name} [{str(self.version)}]"
        else:
            return str(self._name)

    def set_object(self, object : object):
        """
        Sets the object of the node.
        
        Args:
            object (object): The object that the node represents
        """
        self.object = object

    def set_gradFuncs(self, list : list):
        """
        Sets the gradient functions of the node.

        Args:
            list (list): A list of the gradient functions that are connected to the node
        """
        self.gradFuncs = list

    def append_gradFuncs(self, Funcs : list or object):
        """
        Appends a gradient function to the list of gradient functions.

        Args:
            _list (list or object): A list of the gradient functions that are connected to the node.
                If a single function is given, it is appended to the list.
        """

        if isinstance(Funcs, list):
            self.gradFuncs.extend(Funcs)
        else:
            self.gradFuncs.append(Funcs)

    def get_gradFuncs(self):
        """
        Returns the gradient functions of the node.

        Returns:
            list: A list of the gradient functions that are connected to the node
        """
        return self.gradFuncs
    
    
    def __call__(self, *args, **kwargs):
        """
        This method is a placeholder for the computation of the node.

        The __call__ method is used to compute the node with the given arguments.
        The method should be overwritten by the user to implement the computation of the node.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        
        """
        pass

    def __str__(self):
        """
        Returns the string representation of the node.
        
        Returns:
            str: The string representation of the node
        """
        return str(self.name)
    
    def __del__(self):
        """
        Destructor for the AbstractNode class.
        
        """
        del self.object
        del self

class Node(AbstractNode): 
    """
    The Node class is used to represent objects with a numerical value.
    
    The Node class inherits from the AbstractNode class and provides the functionality to represent the object and operations in the graph.

    Attributes:
        data (object): The object that the node represents
        grad (float or PETSc.Vec): The gradient of the object

    Methods:
        get_object: Returns the object of the node
        set_grad: Sets the gradient of the node
        get_grad: Returns the gradient of the node
        reset_grad: Resets the gradient of the node
        accumulate_grad: Accumulates the gradient of the node

    """
    def __init__(self, object : object, **kwargs):
        super().__init__(object, **kwargs)
        self.data = copy.copy(object)
        self.grad = None

    def get_object(self):
        if hasattr(self, "object"):
            return self.object
        return ctypes.cast(self.id, ctypes.py_object).value

    def set_grad(self, value : float or PETSc.Vec):
        self.grad = value

    def get_grad(self):
        return self.grad
    
    def reset_grad(self):
        self.grad = None

    def accumulate_grad(self, value : float or PETSc.Vec):
        if self.grad is None:
            self.grad = value
        else:
            self.grad += value

    def __del__(self):
        del self.data
        del self.grad
        return super().__del__()