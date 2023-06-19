import os
import re
import sys
import json
import logging
import torch
from torch import Tensor
from typing import List, Dict, Any, Union, Optional, Callable
from copy import deepcopy


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)

def dim_zero_sum(x: Tensor) -> Tensor:
    """Summation along the zero dimension."""
    return torch.sum(x, dim=0)


def dim_zero_mean(x: Tensor) -> Tensor:
    """Average along the zero dimension."""
    return torch.mean(x, dim=0)


def dim_zero_max(x: Tensor) -> Tensor:
    """Max along the zero dimension."""
    return torch.max(x, dim=0).values


def dim_zero_min(x: Tensor) -> Tensor:
    """Min along the zero dimension."""
    return torch.min(x, dim=0).values

class SegMetric():
    """
    ref: torchmetrics
    """

    def __init__(self):

        # initialize state
        self._defaults: Dict[str, Union[List, Tensor]] = {}
        self._persistent: Dict[str, bool] = {}
        self._reductions: Dict[str, Union[str, Callable[..., Any], None]] = {}
        
        self.eps = 1e-5
        self.threshold = 0.5
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

        

    def update(self, pred, labels):

        assert isinstance(pred, torch.LongTensor) or isinstance(
            pred, torch.cuda.LongTensor
        )
        assert isinstance(labels, torch.LongTensor) or isinstance(
            labels, torch.cuda.LongTensor
        )

        gt_one = labels == 1
        gt_zero = labels == 0
        pred_one = pred == 1
        pred_zero = pred == 0

        self.tp += (gt_one * pred_one).sum()
        self.fp += (gt_zero * pred_one).sum()
        self.tn += (gt_zero * pred_zero).sum()
        self.fn += (gt_one * pred_zero).sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        # compute final result
        tp = self.tp
        fp = self.fp
        tn = self.tn
        fn = self.fn

        assert (tp + fn) > 0
        assert (fp + tn) > 0

        output = {}
        output["acc1"] = 100.0 * tp / (tp + fn)
        output["acc0"] = 100.0 * tn / (fp + tn)
        output["acc"] = 100.0 * (tp + tn) / (tp + fn + fp + tn)
        output["prec"] = 100.0 * tp / (tp + fn)
        output["rec"] = 100.0 * tp / (tp + fp)
        output["f1"] = (2 * output["prec"] * output["rec"]) / (output["prec"] + output["rec"])

        return output

    def add_state(
        self,
        name: str,
        default: Union[list, Tensor],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:

        if not isinstance(default, (Tensor, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "max":
            dist_reduce_fx = dim_zero_max
        elif dist_reduce_fx == "min":
            dist_reduce_fx = dim_zero_min
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not callable(dist_reduce_fx):
            raise ValueError("`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', None]")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fx


def eval_task1(eval_path, ref_path, evalstrategy=0):
    with open(ref_path) as jh:
        scene_ref = json.load(jh)['scene']
    with open(eval_path) as jh:
        scene_pred = json.load(jh)['scene']
    seg_metric = SegMetric()
    for vid, scene in scene_ref.items():
        seg_metric.update(torch.Tensor(scene_pred[vid]).long()[:len(scene)], torch.Tensor(scene).long())
    output = seg_metric.compute()    
    acc, f1 = output['acc'], output['f1']
    # print("acc:{} f1:{}".format(acc, f1))
    return f1.item()

def eval_task2(eval_path, ref_path, evalstrategy=0):
    with open(ref_path) as jh:
        session_ref = json.load(jh)['session']
    with open(eval_path) as jh:
        session_pred = json.load(jh)['session']
    seg_metric = SegMetric()
    for vid, session in session_ref.items():
        seg_metric.update(torch.Tensor(session_pred[vid]).long()[:len(session)], torch.Tensor(session).long())
    output = seg_metric.compute()    
    acc, f1 = output['acc'], output['f1']
    # print("acc:{} f1:{}".format(acc, f1))
    return f1.item()

def judge(standResultFile, UserCommitFile, evalstrategy):
    try:
        task1_f1 = eval_task1(UserCommitFile, standResultFile, evalstrategy=0)
        task2_f1 = eval_task2(UserCommitFile, standResultFile, evalstrategy=0)
        result = {'err-code':0, 'err-type':0, 'err-info':'', 'score':round(0.5*task1_f1+0.5*task2_f1, 2), 'score_1':round(task1_f1, 2), 'score_2':round(task2_f1, 2)}
    except Exception as e:
        err_info = e.args[0]
        result = {'err-code':8, 'err-type':8, 'err-info':err_info, 'score':0.00, 'score_1':0.00, 'score_2':0.00}
    return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    """ 
    input type:
    {"vid": answer, ...}
    metrics:
    segmentation: f1
    """
    args = sys.argv[1:]
    if len(args) != 3:
        print('argument has error,' + str(len(args)) + ' not equal 3')
        print(args)
        logging.info("judge <main>: %s", args)
    else:
        result = judge(args[0], args[1], int(args[2]))
        print(result)
    # print(judge('evaluation/test_ref_seg.json', 'evaluation/submission.json', 0))

    
    