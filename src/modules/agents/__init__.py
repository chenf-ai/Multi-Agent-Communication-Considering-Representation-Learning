REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .full_comm_agent import FullCommAgent
from .masia_agent import MASIAAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["full_comm"] = FullCommAgent
REGISTRY["masia"] = MASIAAgent