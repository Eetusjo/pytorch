"""RCCL tuning tables for the analytical collective-runtime estimator.

Data only; consumed by comm_analysis._rccl_algo_time. Transcribed from
https://github.com/ROCm/rccl/blob/develop/src/graph/tuning.cc (rcclTuningModel).

RCCL's tables are indexed [algo][proto] over all of NCCL's algorithms; only the
[TREE, RING] x [LL, LL128, SIMPLE] sub-block is kept here, matching the
algo/proto space that comm_analysis models.
"""

from dataclasses import dataclass


RCCL_FACTOR_TABLE_SIZE = 27  # tuning.cc:148


@dataclass(frozen=True)
class RcclTuningModel:
    hwLat: tuple[tuple[tuple[float, ...], ...], ...]  # [hw:3][algo:2][proto:3]
    bwRatio: tuple[tuple[tuple[float, ...], ...], ...]  # [nNodes<=2][algo:2][proto:3]
    treeCorrectionFactor: tuple[tuple[float, ...], ...]  # [proto:3][logSize:27]
    ringCorrectionFactor: tuple[tuple[float, ...], ...]  # [proto:3][logSize:27]


# Correction-factor rows are indexed by
# index = clamp(log2i(nBytes >> 6), 0, 26): 0=64B, 9=32K, 18=16M, 26=4G (tuning.cc:1127-1136)

# fmt: off

# tuning.cc:166-197. Default model: gfx906/908/90a/1030/1100/1101/1102.
TUNING_MODEL_0 = RcclTuningModel(
    hwLat=(
        # NVLINK: Tree(LL/LL128/Simple), Ring(LL/LL128/Simple)
        ((0.8, 1.4, 2.5), (0.8, 2.2, 3.6)),
        # PCI
        ((2.2, 2.2, 5.7), (2.2, 2.2, 5.7)),
        # NET
        ((11.8, 18.2, 20.8), (9.5, 19.8, 15.1)),
    ),
    bwRatio=(
        # <= 2 nodes
        ((0.04, 0.22, 0.91), (0.04, 0.34, 1.00)),
        # > 2 nodes
        ((0.04, 0.22, 0.95), (0.04, 0.34, 1.00)),
    ),
    treeCorrectionFactor=(
        # LL
        (0.1, 0.2, 0.1, 0.1, 0.9, 0.3, 0.4, 0.1, 0.2,
         0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.3, 1.0, 0.1, 0.5, 1.0, 0.9, 1.0, 1.0,
         1.0, 0.3, 0.1, 0.4, 0.5, 0.5, 0.4, 0.4, 0.3,
         0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        # SIMPLE
        (0.2, 1.0, 0.1, 0.1, 0.7, 0.2, 0.4, 0.1, 0.1,
         0.3, 0.4, 0.3, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0,
         0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9),
    ),
    ringCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.2, 0.3,
         0.5, 0.3, 0.1, 0.5, 0.5, 0.3, 0.2, 0.2, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7,
         0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
        # SIMPLE
        (1.0, 0.8, 0.2, 1.0, 1.0, 0.3, 1.0, 0.1, 0.1,
         0.2, 0.2, 0.1, 0.5, 1.0, 0.8, 0.8, 1.0, 0.9,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ),
)

# tuning.cc:331-375. gfx942 (MI300X).
# hwLat[NET][RING][SIMPLE] = 320.0 is verbatim from source, not a typo: it steers
# multi-node AllReduce away from Ring/Simple towards Tree/Simple, which bwRatio
# also favors for > 2 nodes (0.59 vs 1.00).
TUNING_MODEL_5 = RcclTuningModel(
    hwLat=(
        # NVLINK (XGMI): Tree(LL/LL128/Simple), Ring(LL/LL128/Simple)
        ((0.9, 0.9, 2.3), (0.8, 0.8, 2.1)),
        # PCI
        ((2.2, 2.2, 5.7), (2.2, 2.2, 5.7)),
        # NET
        ((10.5, 10.5, 25.0), (9.5, 9.5, 320.0)),
    ),
    bwRatio=(
        # <= 2 nodes
        ((0.06, 0.06, 0.11), (0.08, 0.08, 1.00)),
        # > 2 nodes
        ((0.06, 0.06, 0.59), (0.08, 0.08, 1.00)),
    ),
    treeCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.0, 1.0,
         1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.6, 0.6, 0.6),
    ),
    ringCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.7, 0.2, 1.0, 1.0, 1.0),
    ),
)

# tuning.cc:377-425. gfx950 (MI355X).
# hwLat and bwRatio are byte-identical to model 5 in the source; only the
# correction tables differ. Not a copy-paste error.
# The correction tables also correct upward in places: 2.0 at tree/LL128 indices
# 19-20 (tuning.cc:396) and 5.5 at ring/LL128 index 22 (tuning.cc:402).
TUNING_MODEL_6 = RcclTuningModel(
    hwLat=TUNING_MODEL_5.hwLat,
    bwRatio=TUNING_MODEL_5.bwRatio,
    treeCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0,
         1.0, 2.0, 2.0, 1.0, 0.1, 0.9, 0.9, 0.1, 0.1),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.0, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.7, 0.15, 0.6, 0.7, 0.6),
    ),
    ringCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.1, 0.1, 0.1, 0.1, 5.5, 0.1, 0.1, 1.0, 1.0),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.9, 0.1, 1.0, 0.1, 0.6, 1.0, 1.0),
    ),
)

# fmt: on

# tuning.cc:1157-1163 (rcclGetTuningIndexForArch). Archs mapping to index 0 are
# covered by RCCL_DEFAULT_TUNING_MODEL. Index 7/8/9 models are not ported:
# gfx1200/1201 and gfx1151 are not datacenter parts, and index 8 (gfx942 with an
# AI-NIC) has no Python-visible NIC-type signal.
RCCL_TUNING_MODELS: dict[str, RcclTuningModel] = {
    "gfx942": TUNING_MODEL_5,
    "gfx950": TUNING_MODEL_6,
}

RCCL_DEFAULT_TUNING_MODEL = TUNING_MODEL_0  # tuning.cc:1180

# tuning.cc:998-1001. gfx1250 is listed there too but has no tuning model here.
RCCL_LL128_ARCHS = frozenset({"gfx90a", "gfx942", "gfx950"})

# tuning.cc:781: Tree/Simple is skipped entirely on single-node gfx942/gfx950.
RCCL_TREE_SIMPLE_DISABLED_ARCHS = frozenset({"gfx942", "gfx950"})

# Aggregate XGMI bus bandwidth per GPU (GB/s): per-link width from topo.h:37-41
# times the peer count of the standard 8-GPU platform. Folds in RCCL's
# graphs[a]->nChannels factor (tuning.cc:805), which is exact because the AMD
# path has no per-channel bandwidth caps.
# Provisional: physically plausible, not measured against hardware.
RCCL_XGMI_BW: dict[str, float] = {
    "gfx90a": 144.0,
    "gfx942": 336.0,
    "gfx950": 336.0,
}
