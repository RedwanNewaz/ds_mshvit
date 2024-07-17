import torch

def get_lr_scheduler(optim, args):
    if args["lr_schedule"] == "Step":
        if args["schedule_int"] == "epoch":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=args["lr_steps"], gamma=args["lr_gamma"])
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[x*args["epoch_steps"] for x in args["lr_steps"]], gamma=args["lr_gamma"])
    return scheduler