REGISTRY = {}

from .vae import VAEEnc
from .ob_vae import ObVAEEnc
from .ob_ae import ObAEEnc
from .ob_ind_ae import ObIndAEEnc
from .ob_attn_ae import ObAttnAEEnc
from .ob_attn_skipcat_ae import ObAttnSkipCatAEEnc
from .ob_attn_skipsum_ae import ObAttnSkipSumAEEnc

REGISTRY["vae"] = VAEEnc
REGISTRY["ob_vae"] = ObVAEEnc
REGISTRY["ob_ae"] = ObAEEnc
REGISTRY["ob_ind_ae"] = ObIndAEEnc
REGISTRY["ob_attn_ae"] = ObAttnAEEnc
REGISTRY["ob_attn_skipcat_ae"] = ObAttnSkipCatAEEnc
REGISTRY["ob_attn_skipsum_ae"] = ObAttnSkipSumAEEnc