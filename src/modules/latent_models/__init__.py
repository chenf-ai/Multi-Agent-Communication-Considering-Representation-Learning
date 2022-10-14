REGISTRY = {}

from .old_mlp import TransitionModel as OldMLPModel
from .mlp import TransitionModel as MLPModel

REGISTRY["old_mlp"] = OldMLPModel
REGISTRY["mlp"] = MLPModel
