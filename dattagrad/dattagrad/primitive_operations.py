import numpy as np

from . import PrimitiveModule, make_array


def self_gradient(tensor):
    # Get the shape of the input tensor
    shape = make_array(tensor).shape

    # Create a new tensor with double the dimensions
    new_shape = shape + shape  # e.g., (2, 2) -> (2, 2, 2, 2)
    tensor_of_tensors = np.zeros(new_shape, dtype=int)

    # Create indices for the original tensor positions
    indices = np.indices(shape)

    # Set the corresponding elements in the new tensor to 1
    tensor_of_tensors[tuple(indices) + tuple(indices)] = 1

    return tensor_of_tensors


class Add(PrimitiveModule):

    def forward(self, x, y):
        return x + y

    def backward(self, x, y):
        x = make_array(x)
        y = make_array(y)

        grad_x = np.broadcast_to(self_gradient(x), make_array(self.out.value).shape + x.shape)

        grad_y = np.broadcast_to(self_gradient(y), make_array(self.out.value).shape + y.shape)
        return grad_x, grad_y

    def __repr__(self):
        return f"Add({self.arg_input_nodes[0]}, {self.arg_input_nodes[1]})"


class Negate(PrimitiveModule):
    def forward(self, x):
        return -x

    def backward(self, x):
        return - self_gradient(x)


class Subtract(PrimitiveModule):
    def forward(self, x, y):
        return x - y

    def backward(self, x, y):
        x = make_array(x)
        y = make_array(y)

        grad_x = np.broadcast_to(self_gradient(x), make_array(self.out.value).shape + x.shape)

        grad_y = np.broadcast_to(self_gradient(y), make_array(self.out.value).shape + y.shape)
        return grad_x, -grad_y

    def __repr__(self):
        return f"Subtract({self.arg_input_nodes[0]}, {self.arg_input_nodes[1]})"


class SumAxes(PrimitiveModule):
    def forward(self, x, axis=None):
        return x.sum(axis=axis)

    def backward(self, x, axis=None):
        x = make_array(x)
        if axis is not None:
            if isinstance(axis, int) and axis < 0:
                axis = x.ndim + axis
            elif isinstance(axis, tuple):
                axis = tuple(i if i >= 0 else x.ndim + i for i in axis)
        else:
            axis = tuple(range(x.ndim))
        return self_gradient(x).sum(axis=axis), {}


class Power(PrimitiveModule):
    def forward(self, x, n):
        return x ** n

    def backward(self, x, n):
        # TODO might err
        x = make_array(x)
        n = make_array(n)

        prev = x ** (n - 1)

        return (n * prev * self_gradient(prev)).sum(axis=tuple(range(prev.ndim, 2 * prev.ndim - x.ndim))).reshape(
            *self.out.value.shape, *x.shape) if (*self.out.value.shape, *x.shape) else (
                    n * prev * self_gradient(prev)).sum(axis=tuple(range(prev.ndim, 2 * prev.ndim - x.ndim))), (
                    self.out.value * np.log(n) * self_gradient(n)).transpose(
            *range(n.ndim, n.ndim + prev.ndim), *range(n.ndim))


class MeanAxes(PrimitiveModule):
    def forward(self, x, axis=None):
        return x.mean(axis=axis)

    def backward(self, x, axis=None):
        x = make_array(x)
        if axis is not None:
            if isinstance(axis, int) and axis < 0:
                axis = x.ndim + axis
            elif isinstance(axis, tuple):
                axis = tuple(i if i >= 0 else x.ndim + axis for i in axis)
        else:
            axis = tuple(range(x.ndim))
        return self_gradient(x).mean(axis=axis), {}


class Multiply(PrimitiveModule):
    def forward(self, x, y):
        return x * y

    def backward(self, x, y):
        x = make_array(x)
        y = make_array(y)

        x_reshaped = x.reshape(*x.shape, *((1,) * y.ndim)) if x.ndim + y.ndim != 0 else x
        y_reshaped = y.reshape(*y.shape, *((1,) * x.ndim)) if x.ndim + y.ndim != 0 else y

        grad_x = np.broadcast_to(self_gradient(x), make_array(self.out.value).shape + x.shape) * y_reshaped

        grad_y = np.broadcast_to(self_gradient(y), make_array(self.out.value).shape + y.shape) * x_reshaped
        return grad_x, grad_y

    def __repr__(self):
        return f"Multiply({self.arg_input_nodes[0]}, {self.arg_input_nodes[1]}"


class Divide(PrimitiveModule):
    def forward(self, x, y):
        return x / y

    def backward(self, x, y):
        x = make_array(x)
        y = make_array(y)

        x_reshaped = x.reshape(*x.shape, *((1,) * y.ndim)) if x.ndim + y.ndim != 0 else x
        y_reshaped = y.reshape(*y.shape, *((1,) * x.ndim)) if x.ndim + y.ndim != 0 else y

        grad_x = np.broadcast_to(self_gradient(x), make_array(self.out.value).shape + x.shape) / y_reshaped

        grad_y = np.broadcast_to(self_gradient(y), make_array(self.out.value).shape + y.shape) * (
                -x_reshaped / (y ** 2))
        return grad_x, grad_y


class MatrixMultiply(PrimitiveModule):
    def forward(self, x, y):
        return x @ y

    def backward(self, x, y):
        x = make_array(x)
        y = make_array(y)

        out_shape = make_array(self.out.value).shape
        out_dims = len(out_shape)

        grad_x = (self_gradient(x) @ y).transpose(*range(x.ndim, x.ndim + out_dims), *range(x.ndim))
        if y.ndim == 1:
            y_added = y[:, None]
            grad_y = (x @ self_gradient(y_added)).transpose(*range(y_added.ndim, y_added.ndim + out_dims + 1),
                                                            *range(y_added.ndim)).reshape(*out_shape, *y.shape)
        else:
            grad_y = (x @ self_gradient(y)).transpose(*range(y.ndim, y.ndim + out_dims), *range(y.ndim))
        return grad_x, grad_y

    def __repr__(self):
        return f"MatrixMultiply({self.arg_input_nodes[0]}, {self.arg_input_nodes[1]}"


class Exp(PrimitiveModule):
    def forward(self, x):
        return np.exp(x)

    def backward(self, x):
        return self_gradient(x) * self.out.value


class Log(PrimitiveModule):
    def forward(self, x):
        return np.log(x)

    def backward(self, x):
        return self_gradient(x) / x


class Reshape(PrimitiveModule):
    def forward(self, x, new_shape: tuple[int]):
        return x.reshape(*new_shape)

    def backward(self, x, new_shape: tuple[int]):
        return self_gradient(x).reshape(*(new_shape + x.shape)), None


class Abs(PrimitiveModule):
    def forward(self, x):
        return np.abs(x)

    def backward(self, x):
        return self_gradient(x) * (-1 * (x < 0) + 1 * (x >= 0))
