from .random import RandomSampling, LabeledRandomSampling
from .lc import LeastConfidentSampling
from .entropy import EntropySampling

from .batchbald import EnsembleBatchBALD
from .vr import EnsembleVariationRatio
from .ensvoteentropy import EnsembleEntropy
from .ensbald import EnsembleBALD
from .batchbald import EnsembleBatchBALD
from .greedyvr import EnsembleGreedyVR
from .stconal import EnsembleStConal
from .maxentropy import EnsembleMaxEntropy

NAME_TO_CLS = {
    "random": RandomSampling,
    "ensvr": EnsembleVariationRatio,
    "ensentropy": EnsembleEntropy,
    "ensbald": EnsembleBALD,
    "ensbatchbald": EnsembleBatchBALD,
    "ensgreedyvr": EnsembleGreedyVR,
    "ensstconal": EnsembleStConal,
    "ensmaxentropy": EnsembleMaxEntropy,
    "lc": LeastConfidentSampling,
    "entropy": EntropySampling
}

ALL_METHODS = list(NAME_TO_CLS.keys())