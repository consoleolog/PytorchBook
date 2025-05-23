from abc import abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model Output
        """
        raise NotImplementedError

    @abstractmethod
    def shared_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_collate_fn(self, dataset_name: str):
        return getattr(self, f"collate_fn_{dataset_name}")

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
