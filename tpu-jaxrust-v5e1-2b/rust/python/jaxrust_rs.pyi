"""Type stub for jaxrust_rs, the PyO3 module built from rust/src. Keep in step with bindings.rs."""

from collections.abc import Callable
from os import PathLike
from typing import Any

__version__: str
BUILD: dict[str, str]  # rustc, pjrt_header, crate
BUCKET: int  # prompts are padded to a multiple of this

def init(plugin: str) -> dict[str, Any]:
    """Opens the PJRT plugin once per process: plugin, header_api, plugin_api, devices."""

def selftest() -> list[float]:
    """y = x * 2 + 1 on [0, 1, 2, 3] through the emitter and PJRT. Needs init()."""

def emit(model_dir: str | PathLike[str], kind: str, s: int, t: int) -> str:
    """StableHLO text of the "prefill" (bucket s, context t) or "decode" (context t) program."""

class DecodeResult:
    tokens: list[int]
    finish: str  # "stop", "length" or "cancelled"
    prefill_s: float
    decode_s: float

class Engine:
    def __init__(self, model_dir: str | PathLike[str], context: int) -> None: ...
    @property
    def context(self) -> int: ...
    @property
    def weight_bytes(self) -> int: ...
    @property
    def cache_bytes(self) -> int: ...
    def timings(self) -> dict[str, float]: ...
    def logits(self, ids: list[int]) -> tuple[int, bytes]:
        """(greedy next token, last-position logits as little-endian f32 bytes)."""
    def decode(
        self,
        ids: list[int],
        max_new: int,
        stop: list[int] = ...,
        on_tokens: Callable[[list[int]], object] | None = None,
    ) -> DecodeResult: ...
