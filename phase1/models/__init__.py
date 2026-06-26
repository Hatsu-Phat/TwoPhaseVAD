from .evit import EViTAnomalyModel
from .lstm import LSTMAnomalyModel
from .tcn import TCNAnomalyModel
from .transformer import TransformerAnomalyModel
from .stgnn import STGNNAnomalyModel


def build_phase1_model(name: str, **kwargs):
    name = name.lower()
    if name == 'evit':
        return EViTAnomalyModel(**kwargs)
    if name == 'lstm':
        return LSTMAnomalyModel(**kwargs)
    if name == 'tcn':
        return TCNAnomalyModel(**kwargs)
    if name == 'transformer':
        return TransformerAnomalyModel(**kwargs)
    if name == 'stgnn':
        return STGNNAnomalyModel(**kwargs)
    raise ValueError(f'Unknown phase1 model: {name}')
