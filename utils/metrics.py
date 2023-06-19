import torchmetrics

__DEBUG__ = False


class F1ScoreMetric(torchmetrics.classification.F1):
    def __init__(self, average, num_classes, multiclass, threshold, **metric_args):

        metrics_args = {"average":average, "num_classes":num_classes, "multiclass":multiclass, "threshold":threshold, "compute_on_step": False, "dist_sync_on_step": True}

        super().__init__(**metrics_args)
