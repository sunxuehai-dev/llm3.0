from .ExtrasEnable import Galore
from ..LlamaConfig import LlamaConfig

class Extras:
    def __init__(self, config: LlamaConfig) -> None:
        self._config = config

    def enable_galore(self) -> Galore:
        return Galore(self._config)
