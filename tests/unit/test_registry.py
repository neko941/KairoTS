from __future__ import annotations

import subprocess
import sys


def test_registry_listing_is_lazy_and_excludes_families() -> None:
    script = """
import sys
from kairots.registry import names
assert {'ETTh1', 'ETTh2', 'ETDatasetHour'} <= set(names(category='data/sources'))
assert 'Traditional' not in names(category='strategies')
assert 'ETDataset' not in names(category='data/sources')
assert 'kairots.strategies.FedAvg' not in sys.modules
assert 'kairots.data.sources.ETDataset.ETTh1' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_category_import_returns_class() -> None:
    import kairots.registry
    from kairots.models import Linear

    assert isinstance(Linear, type)
    assert kairots.registry.names(category="models") == (
        "CSDI",
        "DLinear",
        "Denoiser",
        "GRUD",
        "Informer",
        "Linear",
        "NLinear",
        "Naive",
        "Policy",
        "SplitNN",
        "TiDE",
        "TimeGrad",
        "TimeKAN",
        "TimesNet",
        "XGBoost",
        "iFlashformer",
        "iFlowformer",
        "iInformer",
        "iReformer",
        "iTransformer",
    )
    kairots.registry.get(category="models", name="Linear")
    from kairots.models import Linear as LinearAgain

    assert LinearAgain is Linear
