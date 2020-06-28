import os
import pathlib
import pprint
from abc import abstractmethod, ABCMeta
from typing import Dict

import numpy as np
import torch
import torch.utils.data
from torch import nn


class Component(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def loss(*args, **kwargs):
        """
        This is a contract that all the concrete subclasses should implement a loss function on their own. The loss
        function is supposed to be static since the loss should not depend on the state of an instance.

        The purpose is deferring the implementation of loss to the concrete components of the model, i.e. generative,
        dyanmics and recognition, that are supposed to know how the partial losses are computed, for example, analytic
        or sampling.
        """
        pass


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = {}

        path = pathlib.Path(config.setdefault("path", os.getcwd()))
        if config.get("resume", False):
            config_file = path / "config.npy"
            if config_file.exists() and config_file.is_file():
                loaded_config = np.load(config_file.as_posix()).tolist()
                config.update(loaded_config)

        self.path = config["path"]
        self.config.update(config)
        self.set_props()

        self.random_seed = self.config.setdefault("random_seed", None)
        self.max_iter = self.config.get("max_iter")
        self.lr = self.config["lr"]

        if self.config["debug"]:
            pprint.pprint(self.config)

    @abstractmethod
    def set_props(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    @classmethod
    def loadfrom(cls, path):
        """Load saved model"""
        path = pathlib.Path(path)
        config_file = path / "config.npy"
        config = np.load(config_file.as_posix())[()]
        config["resume"] = True

        mdl = cls(config)
        mdl.load()

        # TODO: use torch.save/load

        return mdl


class Trainer(metaclass=ABCMeta):
    def __init__(self, model: Model):
        self.model = model

    def fit(self, y, u):
        train_set = torch.utils.data.TensorDataset(y, u)
        loader = torch.utils.data.DataLoader(train_set)

        batch, _, _ = y.shape

        mu0 = torch.zeros(batch, self.model.xdim)
        logvar0 = torch.zeros(batch, self.model.xdim)
        q0 = (mu0, logvar0)

        qs = []
        for yi, ui in loader:
            q1 = self.model.filter(yi, ui, q0)
            self.step(q1, yi, ui, q0)
            qi = (p.detach() for p in q1)
            qs.append(qi)
            q0 = q1
        return qs

    @abstractmethod
    def step(self, output, y, u, q):
        """
        Update parameters
        Named after torch optimizer's step function
        """
        pass


class Noise(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, xhat, q, sample=True):
        """Loss function
        The likelihood is determined by the state noise.
        """
        pass
