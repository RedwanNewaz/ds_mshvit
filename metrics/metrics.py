from torchmetrics import Metric
import torch


class F2CIW(Metric):
    def __init__(self, dist_sync_on_step=False, threshold = 0.5):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        LabelWeightDict = {"RB":5.55,"OB":3.0625,"PF":1.6075,"DE":0.9,"FS":3.5625,"IS":1.025,"RO":1.975,"IN":1.7375,"AF":0.45,"BE":1.2625,"FO":1.375,"GR":0.5,"PH":2.3125,"PB":2.3125,"OS":5.0,"OP":2.125,"OK":2.44}
        self.weights = torch.tensor([LabelWeightDict[key] for key in LabelWeightDict.keys()])
        self.weights_sum = torch.sum(self.weights)

        self.add_state("tp", default=torch.zeros(len(LabelWeightDict)), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(len(LabelWeightDict)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(LabelWeightDict)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(LabelWeightDict)), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        
        # B x C
        assert preds.shape == target.shape, "The input and targets do not have the same size: Input: {} - Targets: {}".format(preds.shape, target.shape)

        preds = (preds >= self.threshold).int()
        
        true_pred = target == preds
        false_pred = target != preds
        pos_pred = preds == 1
        neg_pred = preds == 0

        self.tp += (true_pred * pos_pred).sum(dim=0).long()
        self.fp += (false_pred * pos_pred).sum(dim=0).long()

        self.tn += (true_pred * neg_pred).sum(dim=0).long()
        self.fn += (false_pred * neg_pred).sum(dim=0).long()

    def compute(self):

        precision_k = self.tp / (self.tp+self.fp)
        recall_k = self.tp / (self.tp+self.fn)
        F2_k = (5 * precision_k * recall_k)/(4*precision_k + recall_k)

        F2_k[torch.isnan(F2_k)] = 0

        ciwF2 = F2_k * self.weights.to(self.tp.device)
        ciwF2 = torch.sum(ciwF2) / torch.sum(self.weights_sum.to(self.tp.device))

        return ciwF2


class F1Normal(Metric):
    def __init__(self, dist_sync_on_step=False, threshold = 0.5):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(1), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):        
        # B x C
        assert preds.shape == target.shape, "The input and targets do not have the same size: Input: {} - Targets: {}".format(preds.shape, target.shape)

        preds = preds >= self.threshold
        target = 1 - target.sum(1).bool().long()
        preds = 1 - preds.sum(1).bool().long()
        
        true_pred = target == preds
        false_pred = target != preds
        pos_pred = preds == 1
        neg_pred = preds == 0

        self.tp += (true_pred * pos_pred).sum(dim=0).long()
        self.fp += (false_pred * pos_pred).sum(dim=0).long()

        self.tn += (true_pred * neg_pred).sum(dim=0).long()
        self.fn += (false_pred * neg_pred).sum(dim=0).long()

    def compute(self):

        precision = self.tp / (self.tp+self.fp)
        recall = self.tp / (self.tp+self.fn)
        f1_normal = (2 * precision * recall)/(precision + recall)

        return f1_normal