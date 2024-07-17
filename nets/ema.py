from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA(nn.Module):
    """ 
    Based on Ross Wightmans ModelEMAV2: https://github.com/rwightman/pytorch-image-models/blob/02aaa785b97af5cbf22295033b4d3cc0137d8553/timm/utils/model_ema.py#L82
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class JointModelEMA(nn.Module):
    """
    Based on Ross Wightmans ModelEMAV2: https://github.com/rwightman/pytorch-image-models/blob/02aaa785b97af5cbf22295033b4d3cc0137d8553/timm/utils/model_ema.py#L82
    """
    def __init__(self, rgb_model, opt_model, decay=0.9999, device=None):
        super(JointModelEMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.rgb_module = deepcopy(rgb_model)
        self.rgb_module.eval()

        self.opti_module = deepcopy(opt_model)
        self.opti_module.eval()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.rgb_module.to(device=device)
            self.opti_module.to(device=device)

    def _update(self, rgb_model, opti_model, update_fn):
        def internal_update(model, module):
            with torch.no_grad():
                for ema_v, model_v in zip(module.state_dict().values(), model.state_dict().values()):
                    if self.device is not None:
                        model_v = model_v.to(device=self.device)
                    ema_v.copy_(update_fn(ema_v, model_v))
        internal_update(rgb_model, self.rgb_module)
        internal_update(opti_model, self.opti_module)

    def update(self, rgb_model, opti_model):
        self._update(rgb_model, opti_model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, rgb_model, opti_model):
        self._update(rgb_model, opti_model, update_fn=lambda e, m: m)