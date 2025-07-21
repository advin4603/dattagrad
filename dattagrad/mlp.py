import math
from abc import ABC, abstractmethod
from statistics import mean

import wandb

from models.mlp.dattagrad import Module, Parameter, Tensor, PrimitiveModule, Pipeline, Optimizer
from models.mlp.dattagrad.primitive_operations import self_gradient
import numpy as np
import random
from typing import Iterable, TypeVar, Generic, Callable, TypedDict, Sequence


class DataTransform(ABC):

    @abstractmethod
    def transform_point(self, point):
        raise NotImplementedError


class MinSubtract(DataTransform):
    def __init__(self, data):
        self.min = data.min(axis=0)

    def transform_point(self, point):
        return point - self.min


class Standardize(DataTransform):
    def __init__(self, data):
        # shape (rows, columns)
        self.means = data.mean(axis=0)
        self.stds = data.std(axis=0)

    def transform_point(self, point):
        # shape (rows, columns)
        return (point - self.means) / self.stds


class MinMaxScale(DataTransform):
    def __init__(self, data):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform_point(self, point):
        return (point - self.min) / (self.max - self.min)


class MakeOneHot(DataTransform):
    def __init__(self, classes: list):
        self.classes = classes
        self.class_to_id = dict(zip(self.classes, range(len(self.classes))))
        self.id_to_class = dict(zip(range(len(self.classes)), self.classes))

    def transform_point(self, point):
        class_id = self.class_to_id[point]
        one_hot = np.zeros(len(self.classes))
        one_hot[class_id] = 1
        return one_hot


class MakeMultiLabelOneHot(DataTransform):
    def __init__(self, classes: list):
        self.classes = classes
        self.class_to_id = dict(zip(self.classes, range(len(self.classes))))
        self.id_to_class = dict(zip(range(len(self.classes)), self.classes))

    def transform_point(self, points):
        class_ids = [self.class_to_id[point] for point in points]
        one_hot = np.zeros(len(self.classes))
        one_hot[class_ids] = 1
        return one_hot


T = TypeVar('T')


class Dataset(ABC, Generic[T]):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        raise NotImplementedError

    def get_index_list(self, indices: Iterable[int]):
        return [self[i] for i in indices]

    @abstractmethod
    def collate(self, points: list[T]):
        raise NotImplementedError

    def get_batches(self, batch_size: int, shuffle: bool = True):
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(self), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = self.get_index_list(batch_indices)
            yield self.collate(batch_data)


class Linear(Module):
    def __init__(self, in_dimensions, out_dimensions):
        super().__init__()
        self.weights = Parameter(np.random.randn(in_dimensions, out_dimensions))
        self.bias = Parameter(np.random.randn(out_dimensions))

    def forward(self, x):
        return x @ self.weights + self.bias


class ReLU(PrimitiveModule):
    def forward(self, x):
        return x * (x > 0)

    def backward(self, x):
        return self_gradient(x) * (x > 0)


class Sigmoid(PrimitiveModule):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self_gradient(x) * self.out.value * (1 - self.out.value)


class Tanh(PrimitiveModule):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return self_gradient(x) * (1 - self.out.value ** 2)


class StochasticGradientDescent(Optimizer):
    def __init__(self, parameters: dict[str, list[Parameter]], learning_rate: float):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def optimizer_step(self):
        for param_list in self.parameters.values():
            for param in param_list:
                param.value = param.value - self.learning_rate * param.grad


MetricDictValue = TypedDict("MetricDictValue", {"epoch": int, "step": int, "value": float})
MetricDict = dict[str, list[MetricDictValue]]
Callback = Callable[[MetricDict], None]


class MSELoss(Module):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return ((x - y) ** 2).mean()


class CrossEntropyLoss(Module):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """x and y are (batch_size, num_classes)"""
        return -(y * x.log()).sum(axis=-1).mean()


class CrossEntropyClassificationLoss(CrossEntropyLoss):
    def forward(self, prediction, target):
        """prediction is (batch_size, n_classes) and target is (batch_size)"""
        one_hot_target = np.zeros_like(prediction.value)
        one_hot_target[np.arange(prediction.shape[0]), target.value] = 1
        return super().forward(prediction, Tensor(one_hot_target))


class MSEClassificationLoss(MSELoss):
    def forward(self, prediction, target):
        """prediction is (batch_size, n_classes) and target is (batch_size)"""
        one_hot_target = np.zeros_like(prediction.value)
        one_hot_target[np.arange(prediction.shape[0]), target.value] = 1
        return super().forward(prediction, Tensor(one_hot_target))


class Softmax(Module):
    def __init__(self, axis: int | tuple[int] = None):
        self.axis = axis

    def forward(self, x: Tensor, axis: int | tuple[int] = None) -> Tensor:
        if axis is None:
            axis = self.axis
        exp = x.exp()
        sums = exp.sum(axis=axis)
        if axis is not None:
            broadcast_sums_shape = list(exp.shape)
            if isinstance(axis, int):
                axis = (axis,)
            for a in axis:
                broadcast_sums_shape[a] = 1
            sums = sums.reshape(*broadcast_sums_shape)
        return exp / sums


class WandbCallback:
    def __init__(self, metric_aggregate: Callable[[Sequence[float]], float] = mean, **wandb_init_kwargs):
        wandb.init(**wandb_init_kwargs)
        wandb.define_metric("*", step_metric="train_step")
        self.metric_aggregate = metric_aggregate

    def __call__(self, metric_dict: MetricDict):
        train_step = 0
        last_epoch = 0
        for metric_name, metric_value in metric_dict.items():
            if not metric_name.startswith("train/"):
                continue

            train_step = metric_value[-1]["step"]
            last_epoch = metric_value[-1]["epoch"]
            break
        log_dict = {"train_step": train_step}
        for metric_name, metric_value in metric_dict.items():
            metrics = [i["value"] for i in metric_value if i["epoch"] == last_epoch]
            if not metrics:
                continue
            log_value = self.metric_aggregate(metrics)
            log_dict[metric_name] = log_value

        wandb.log(log_dict)


class MLP(Pipeline):
    def fit(self, train_dataset: Dataset, optimizer: Optimizer, batch_size: int, epochs: int, loss_function: Module,
            validation_dataset: Dataset | None = None, validation_batch_size: int = 32,
            extra_metrics: list[Module] | None = None, train_callbacks: list[Callback] | None = None,
            validation_callbacks: list[Callback] | None = None) -> MetricDict:
        metrics_dict = {}
        optimizer.set_train(True)
        all_metrics = [loss_function] + extra_metrics if extra_metrics is not None else [loss_function]
        total_steps = math.ceil(len(train_dataset) / batch_size)
        total_val_steps = math.ceil(
            len(validation_dataset) / (validation_batch_size or batch_size)) if validation_dataset and (
                validation_batch_size or batch_size) else None
        for epoch in range(epochs):
            optimizer.zero_grad()
            for step, (X, Y) in enumerate(train_dataset.get_batches(batch_size, shuffle=True)):
                X = Tensor(X)
                Y = Tensor(Y)
                prediction = self(X)
                train_loss = loss_function(prediction, Y)
                metrics_dict.setdefault(f"train/{type(loss_function).__name__}", []).append(
                    {"epoch": epoch, "step": epoch + step / total_steps, "value": float(train_loss.value)})
                for metric in all_metrics[1:]:
                    metrics_dict.setdefault(f"train/{type(metric).__name__}", []).append(
                        {"epoch": epoch, "step": epoch + step / total_steps,
                         "value": float(metric(prediction, Y).value)})
                train_loss.backward()
                optimizer.optimizer_step()
                if train_callbacks is not None:
                    for callback in train_callbacks:
                        callback(metrics_dict)

            if validation_dataset is None:
                continue
            optimizer.set_train(False)
            for step, (X, Y) in enumerate(validation_dataset.get_batches(validation_batch_size or batch_size, False)):
                X = Tensor(X)
                Y = Tensor(Y)
                prediction = self(X)
                for metric in all_metrics:
                    metrics_dict.setdefault(f"validation/{type(metric).__name__}", []).append(
                        {"epoch": epoch, "step": epoch + step / total_val_steps,
                         "value": float(metric(prediction, Y).value)})

                if validation_callbacks is not None:
                    for callback in validation_callbacks:
                        callback(metrics_dict)

            optimizer.set_train(True)
        return metrics_dict

    def gradient_check(self, test_point, test_label, loss_function, epsilon=1e-3):
        test_point = Tensor(test_point)
        test_label = Tensor(test_label)

        test_out = self(test_point)
        test_loss = loss_function(test_out, test_label)
        test_loss.backward()
        test_parameter: Parameter | None = None
        for module in self.modules:
            parameters = module.parameters()
            test_parameter: Parameter = next(iter(parameters.values()))[0]
            break

        if test_parameter is None:
            raise ValueError("No parameters in model")

        flat_index = 0
        index = np.unravel_index(flat_index, test_parameter.shape)
        test_parameter.value[index] += epsilon
        test_out_2 = self(test_point)
        test_loss_2 = loss_function(test_out_2, test_label)

        test_parameter.value[index] -= 2 * epsilon
        test_out_3 = self(test_point)
        test_loss_3 = loss_function(test_out_3, test_label)

        estimated_grad = (test_loss_2.value - test_loss_3.value) / (2 * epsilon)
        relative_error = abs(estimated_grad - test_parameter.grad[index]) / max(abs(estimated_grad),
                                                                                abs(test_parameter.grad[index]))
        return relative_error < epsilon


def to_classification_metric_module(metric_function):
    class ClassificationMetricModule(Module):
        def forward(self, prediction, target):
            """prediction is (batch_size, n_classes) and target is (batch_size)"""
            predicted_classes = prediction.value.argmax(axis=-1)
            return Tensor(metric_function(target.value, predicted_classes))

    name = metric_function.__name__.replace("_", " ").title().replace(" ", "")
    ClassificationMetricModule.__name__ = name

    return ClassificationMetricModule


def to_multilabel_classification_metric_module(metric_function):
    class ClassificationMetricModule(Module):
        def forward(self, prediction, target):
            """prediction is (batch_size, n_classes) and target is (batch_size, n_classes)"""
            predicted_classes = (prediction.value > 0.5).round()
            return Tensor(metric_function(target.value, predicted_classes))

    name = metric_function.__name__.replace("_", " ").title().replace(" ", "")
    ClassificationMetricModule.__name__ = name

    return ClassificationMetricModule


class MAELoss(Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean()


class RMSELoss(MSELoss):
    def forward(self, prediction, target):
        return ((prediction - target) ** 2).mean() ** 0.5


class RSquared(Module):
    def forward(self, prediction, target):
        """prediction is (batch_size, dim) and target is (batch_size, dim)"""
        ss_total = ((prediction - prediction.mean(axis=-1)) ** 2).sum()
        ss_res = ((prediction - target) ** 2).sum()

        return 1 - ss_res / ss_total


class RSquaredLoss(Module):
    def forward(self, prediction, target):
        """prediction is (batch_size, dim) and target is (batch_size, dim)"""
        ss_total = ((prediction - prediction.mean(axis=-1)) ** 2).sum()
        ss_res = ((prediction - target) ** 2).sum()

        return ss_res / ss_total


l = MAELoss()
a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)
b = Tensor([[1, 2, 3], [1, 2, 3]])
l(a, b).backward()
