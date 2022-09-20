from .random import RandomSampling, LabeledRandomSampling
from .batchbald import EnsembleBatchBALD
from .vr import EnsembleVariationRatio
from .entropy import EnsembleEntropy
from .bald import EnsembleBALD
from .batchbald import EnsembleBatchBALD
from .greedyvr import EnsembleGreedyVR
from .stconal import EnsembleStConal
from .maxentropy import EnsembleMaxEntropy

NAME_TO_CLS = {
    "random": RandomSampling,
    "vr": EnsembleVariationRatio,
    "entropy": EnsembleEntropy,
    "bald": EnsembleBALD,
    "batchbald": EnsembleBatchBALD,
    "greedyvr": EnsembleGreedyVR,
    "stconal": EnsembleStConal,
    "maxentropy": EnsembleMaxEntropy,
}

ALL_METHODS = list(NAME_TO_CLS.keys())