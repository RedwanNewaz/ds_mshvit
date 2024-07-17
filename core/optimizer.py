import torch

def adjusted_parameter_setting(model, lr, weight_decay=1e-5):

    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    else:
        skip_list = set()

    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or any(skip_list_name in name for skip_list_name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, "lr": lr, 'weight_decay': 0.},
        {'params': decay, "lr": lr, 'weight_decay': weight_decay}]


def get_optimizer(params, args):

    if args["optim"] == "Adam":
        optim = torch.optim.Adam(params, lr = args["lr"], weight_decay = args["weight_decay"])
    elif args["optim"] == "AdamW":
        optim = torch.optim.AdamW(params, lr = args["lr"], weight_decay = args["weight_decay"])
    elif args["optim"] == "SGD":
        optim = torch.optim.SGD(params, lr = args["lr"], weight_decay = args["weight_decay"], momentum = args["momentum"], nesterov = args["nesterov"])
    return optim