from ..LlamaConfig import LlamaConfig

class Galore:
    def __init__(self, config: LlamaConfig):
        self._config = config
        self._config.use_galore = True
        self.set_galore_layerwise()
        self.set_galore_rank()
        self.set_galore_target()
        self.set_galore_scale()
        # dataset config
        self.set_overwrite_cache()
        # train config
        self.set_pure_bf16()

    def set_galore_layerwise(self, flag: bool = True):
        self._config.galore_layerwise = flag

    def set_galore_target(self, target: str = 'all'):
        self._config.galore_target = target

    def set_galore_rank(self, rank: int = 128):
        self._config.galore_rank = rank
        
    def set_galore_scale(self, scale: float = 2.0):
        self._config.galore_scale = scale

    def set_overwrite_cache(self, flag: bool = True):
        self._config.overwrite_cache = flag

    def set_pure_bf16(self, flag: bool = True):
        """
        this function will disable 'bf16'
        """
        self._config.bf16 = None
        self._config.pure_bf16 = flag
