import math

import numpy as np
import paddle

def psnr(pred, gt):
    pred = paddle.clip(pred, min=0, max=1)
    gt = paddle.clip(gt, min=0, max=1)
    imdff = np.asarray(pred - gt)
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)