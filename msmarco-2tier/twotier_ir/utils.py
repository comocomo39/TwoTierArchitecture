import random, warnings, os
import numpy as np
import pyterrier as pt

def bootstrap(seed=42):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed); np.random.seed(seed)
    warnings.filterwarnings("ignore", category=FutureWarning, module="colbert.utils.amp")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.grad_scaler")
    if not pt.started():
        pt.init()
