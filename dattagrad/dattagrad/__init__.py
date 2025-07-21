from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import pickle
from functools import partial


def make_array(x):
    return x if isinstance(x, np.ndarray) else np.array(x)


class ModuleList(list):
    pass


class Module(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def find_parameters(self, prefix, parameter_dict):
        # TODO use a visited set to prevent circular reference infinite loop
        for var_name, value in vars(self).items():
            if isinstance(value, Parameter):
                name = f"{prefix}{'.' if prefix else ''}{var_name}"
                parameter_dict.setdefault(name, []).append(value)
            elif isinstance(value, Module):
                new_prefix = f"{prefix}{'.' if prefix else ''}{var_name}"
                value.find_parameters(new_prefix, parameter_dict)
            elif isinstance(value, ModuleList):
                for i, module in enumerate(value):
                    new_prefix = f"{prefix}{'.' if prefix else ''}{var_name}[{i}]"
                    module.find_parameters(new_prefix, parameter_dict)

    def parameters(self):
        parameter_dict = {}
        self.find_parameters(type(self).__name__, parameter_dict)
        return parameter_dict

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class PrimitiveModule(Module, ABC):
    def __init__(self):
        self.arg_input_nodes = None
        self.kwarg_input_nodes = None
        self.out: Tensor | None = None

    def _raw_prediction_with_hooks(self, *args, **kwargs):
        self.arg_input_nodes = args
        self.kwarg_input_nodes = kwargs

        input_args, input_kwargs = self.get_args_kwargs_with_values()
        return self.forward(
            *input_args, **input_kwargs
        )

    def _get_output_requires_grad(self):
        return any(getattr(arg, "requires_grad", False) for arg in self.arg_input_nodes) or any(
            getattr(v, "requires_grad", False) for v in self.kwarg_input_nodes.values())

    def get_args_kwargs_with_values(self):
        return [getattr(arg, "value", arg) for arg in self.arg_input_nodes], {k: getattr(v, "value", v) for k, v in
                                                                              self.kwarg_input_nodes.items()}

    def make_backward(self):
        data_args, data_kwargs = self.get_args_kwargs_with_values()
        args, kwargs = self.arg_input_nodes, self.kwarg_input_nodes
        out = self.out

        return partial(backward_func, self, data_args, data_kwargs, args, kwargs, out)

    def __call__(self, *args, **kwargs):
        out_value = self._raw_prediction_with_hooks(*args, **kwargs)
        output_requires_grad = self._get_output_requires_grad()
        self.out = Tensor(out_value,
                          producer=self if output_requires_grad else None,
                          requires_grad=output_requires_grad)
        self.out.grad_fn = self.make_backward() if output_requires_grad else None

        return self.out

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError


def backward_func(self, data_args, data_kwargs, args, kwargs, out):
    individual_gradients = self.backward(*data_args, **data_kwargs)
    if not kwargs:
        individual_gradients = individual_gradients, {}
    if len(args) == 1:
        individual_gradients = (individual_gradients[0],), individual_gradients[1]

    for node, individual_gradient in chain(zip(args, individual_gradients[0]),
                                           zip(kwargs.values(), individual_gradients[1])):
        if not hasattr(node, "grad") or not node.requires_grad:
            continue

        out_shape = make_array(out.value).shape
        out_grad_shape = make_array(out.grad).shape
        if node.grad is None:
            node.grad = np.zeros(out_grad_shape[:len(out_grad_shape) - len(out_shape)] + node.value.shape)
        individual_gradient_dim = make_array(individual_gradient).ndim
        node_value_dim = make_array(node.value).ndim
        node.grad = (node.grad + np.tensordot(out.grad, individual_gradient,
                                              axes=individual_gradient_dim - node_value_dim))


from .primitive_operations import self_gradient, Add, Multiply, MatrixMultiply, Negate, Subtract, SumAxes, MeanAxes, \
    Power, Divide, Exp, Log, Reshape, Abs


class Tensor:
    def __init__(self, value, producer: PrimitiveModule | None = None, requires_grad: bool = False, grad_fn=None):
        self.value = make_array(value)
        self.producer = producer
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn

    def set_value(self, value):
        self.value = value
        self.grad = None

    @staticmethod
    def _build_topo(topo, visited, v):
        if v in visited or not hasattr(v, "producer"):
            return
        visited.add(v)
        if v.producer is not None:
            for child_node in chain(v.producer.arg_input_nodes, v.producer.kwarg_input_nodes.values()):
                if getattr(child_node, "grad_fn", None) is not None:
                    Tensor._build_topo(topo, visited, child_node)

        topo.append(v)

    def build_topo(self):
        topo = []
        visited = set()
        Tensor._build_topo(topo, visited, self)
        return topo

    def backward(self):
        if not self.requires_grad:
            raise ValueError("Can't backward on not required grad")
        self.grad = self_gradient(make_array(self.value))
        traversal_order = reversed(self.build_topo())
        for node in traversal_order:
            node.grad_fn()

    def __add__(self, other):
        return Add()(self, other)

    def sum(self, axis: None | int | tuple[int] = None):
        return SumAxes()(self, axis=axis)

    def mean(self, axis: None | int | tuple[int] = None):
        return MeanAxes()(self, axis=axis)

    def __pow__(self, other):
        return Power()(self, other)

    def __rpow__(self, other):
        return Power()(other, self)

    def __radd__(self, other):
        return Add()(other, self)

    def __mul__(self, other):
        return Multiply()(self, other)

    def __rmul__(self, other):
        return Multiply()(other, self)

    def __matmul__(self, other):
        return MatrixMultiply()(self, other)

    def __rmatmul__(self, other):
        return MatrixMultiply()(other, self)

    def __neg__(self):
        return Negate()(self)

    def __sub__(self, other):
        return Subtract()(self, other)

    def __rsub__(self, other):
        return Subtract()(other, self)

    def __repr__(self):
        return f"Tensor({self.value}, {self.producer})"

    def __truediv__(self, other):
        return Divide()(self, other)

    def __rtruediv__(self, other):
        return Divide()(other, self)

    @property
    def shape(self):
        return self.value.shape

    def exp(self):
        return Exp()(self)

    def log(self):
        return Log()(self)

    def reshape(self, *shape):
        return Reshape()(self, shape)

    def abs(self):
        return Abs()(self)


class Parameter(Tensor):
    def __init__(self, initial_value, frozen: bool = False):
        super().__init__(initial_value, requires_grad=not frozen)


class Pipeline(Module):
    def __init__(self, *modules):
        self.modules = ModuleList(modules)

    def forward(self, *args, **kwargs):
        module_out = self.modules[0](*args, **kwargs)
        for module in self.modules[1:]:
            module_out = module(module_out)

        return module_out


class Optimizer(ABC):
    def __init__(self, parameters: dict[str, list[Parameter]]):
        self.parameters = parameters

    def zero_grad(self):
        for param_list in self.parameters.values():
            for param in param_list:
                param.grad = None

    def set_train(self, train: bool):
        for param_list in self.parameters.values():
            for param in param_list:
                param.requires_grad = train

    @abstractmethod
    def optimizer_step(self):
        raise NotImplementedError
