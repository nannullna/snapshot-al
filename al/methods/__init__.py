from .random import RandomSampling, LabeledRandomSampling
from .lc import LeastConfidentSampling
from .entropy import EntropySampling

from .batchbald import EnsembleBatchBALD
from .ensvr import EnsembleVariationRatio
from .ensvoteentropy import EnsembleEntropy
from .ensbald import EnsembleBALD
from .batchbald import EnsembleBatchBALD
from .greedyvr import EnsembleGreedyVR
from .stconal import EnsembleStConal
from .ensmaxentropy import EnsembleMaxEntropy
from .ensmar import EnsembleMarginSampling

from .mcbald import MCBald
from .mcentropy import MCEntropy
from .mcvr import MCVariationRatio
from .mcmar import MCMarginSampling

NAME_TO_CLS = {
    "random": RandomSampling,
    "ensvr": EnsembleVariationRatio,
    "ensentropy": EnsembleEntropy,
    "ensbald": EnsembleBALD,
    "ensbatchbald": EnsembleBatchBALD,
    "ensgreedyvr": EnsembleGreedyVR,
    "ensstconal": EnsembleStConal,
    "ensmaxentropy": EnsembleMaxEntropy,
    "ensmar": EnsembleMarginSampling,
    "lc": LeastConfidentSampling,
    "entropy": EntropySampling,
    "mcbald": MCBald,
    "mcmaxentropy": MCEntropy,
    "mcvr": MCVariationRatio,
    "mcmar": MCMarginSampling
}

ALL_METHODS = list(NAME_TO_CLS.keys())