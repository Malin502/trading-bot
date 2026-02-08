
import sys
from pathlib import Path
import json

# eval_model1_folds.pyをインポートして実行
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# TOPKを書き換え
import src.model1.eval_model1_folds as eval_module
eval_module.TOPK = 3

# 実行
if __name__ == "__main__":
    eval_module.main()
