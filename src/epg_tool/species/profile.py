"""Species profile: waveform label set, rule-based thresholds, and model
registry for one insect/species, loaded from a YAML config file.

This is the extensibility point requested for the tool -- adding a new
insect (aphid, leafhopper, ...) means writing a new YAML file under
``species/profiles/``, not touching any parsing/feature/model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PROFILES_DIR = Path(__file__).parent / "profiles"


@dataclass(frozen=True)
class WaveformDef:
    code: int
    label: str
    description: str = ""
    color: str = "#888888"


@dataclass
class SpeciesProfile:
    name: str
    common_name: str
    reference: str
    waveforms: list[WaveformDef]
    sentinel_codes: frozenset[int]
    rule_based_thresholds: dict = field(default_factory=dict)
    model_registry: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    trim_start_s: float = 0.0
    # Per-recording min-max normalization to [0,1] before feature
    # extraction (DiscoEPG / Dinh et al. 2026, Eq. 1). Removes per-insect
    # acquisition-gain differences so amplitude/shape features generalize
    # across individuals; only affects the ML feature path, never the
    # rule-based classifier (which reasons in absolute volts).
    normalize: bool = False
    # Fixed analysis-window length (seconds) for feature extraction. The
    # trained model is tied to this value, so training and inference must
    # use the same one -- hence it lives in the profile, not just a CLI
    # default. A longer window buys frequency resolution (helps separate
    # low-frequency waveforms like D from higher ones) at the cost of time
    # resolution near transitions.
    window_s: float = 1.0
    # Extra per-label multiplier on top of inverse-frequency ("balanced")
    # sample weights during training -- for waveforms that are both rare
    # and easily confused with a much more common one (plain balanced
    # weighting alone under-corrects for that combination). Tuned per
    # species/dataset, not a fixed constant.
    class_weight_multipliers: dict = field(default_factory=dict)
    # --- sequence post-processing (Viterbi decoding + confidence gate) ---
    # decode_sequence: run Viterbi over per-window class probabilities using
    # a transition matrix learned at train time, to enforce the waveform
    # grammar and smooth spurious single-window flips. allowed_transitions
    # (label -> permitted successor labels; self-transitions always allowed)
    # is the *biological* grammar, used only to hard-zero transitions that
    # are both listed-impossible here AND essentially absent from training
    # data -- everything else stays data-driven. confidence_threshold: if a
    # window's top posterior falls below it, the window is relabeled
    # `unclassified` for the user to review instead of forcing a guess.
    decode_sequence: bool = False
    allowed_transitions: dict = field(default_factory=dict)
    confidence_threshold: float = 0.0
    unclassified_code: int = 0
    unclassified_label: str = "unclassified"
    unclassified_color: str = "#d9d9d9"

    def __post_init__(self) -> None:
        codes = [w.code for w in self.waveforms]
        if len(codes) != len(set(codes)):
            raise ValueError(f"{self.name}: duplicate waveform codes in {codes}")
        overlap = set(codes) & set(self.sentinel_codes)
        if overlap:
            raise ValueError(f"{self.name}: codes {overlap} are both waveforms and sentinels")

    @property
    def code_to_label(self) -> dict[int, str]:
        return {w.code: w.label for w in self.waveforms}

    @property
    def label_to_code(self) -> dict[str, int]:
        return {w.label: w.code for w in self.waveforms}

    def label_for_code(self, code: int) -> str | None:
        return self.code_to_label.get(code)

    def display_label_for_code(self, code: int) -> str:
        """Like ``label_for_code`` but also resolves the ``unclassified``
        sentinel and never returns None -- for plotting/tables that may
        encounter post-processed predictions."""
        if code == self.unclassified_code:
            return self.unclassified_label
        return self.code_to_label.get(code, f"code {code}")

    def display_color_for_code(self, code: int) -> str:
        """Waveform color for a code, the neutral unclassified color for
        the sentinel, or a light-grey fallback for anything unknown."""
        if code == self.unclassified_code:
            return self.unclassified_color
        return next((w.color for w in self.waveforms if w.code == code), "#cccccc")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpeciesProfile":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        waveforms = [
            WaveformDef(
                code=w["code"],
                label=w["label"],
                description=w.get("description", ""),
                color=w.get("color", "#888888"),
            )
            for w in raw.get("waveforms", [])
        ]
        return cls(
            name=raw["name"],
            common_name=raw.get("common_name", raw["name"]),
            reference=raw.get("reference", ""),
            waveforms=waveforms,
            sentinel_codes=frozenset(raw.get("sentinel_codes", [99])),
            rule_based_thresholds=raw.get("rule_based_thresholds", {}),
            model_registry=raw.get("model_registry", {}),
            parameters=raw.get("parameters", {}),
            trim_start_s=raw.get("preprocessing", {}).get("trim_start_s", 0.0),
            normalize=raw.get("preprocessing", {}).get("normalize", False),
            window_s=raw.get("preprocessing", {}).get("window_s", 1.0),
            class_weight_multipliers=raw.get("training", {}).get("class_weight_multipliers", {}),
            decode_sequence=raw.get("sequence", {}).get("decode", False),
            allowed_transitions=raw.get("sequence", {}).get("allowed_transitions", {}),
            confidence_threshold=raw.get("sequence", {}).get("confidence_threshold", 0.0),
            unclassified_code=raw.get("sequence", {}).get("unclassified", {}).get("code", 0),
            unclassified_label=raw.get("sequence", {}).get("unclassified", {}).get("label", "unclassified"),
            unclassified_color=raw.get("sequence", {}).get("unclassified", {}).get("color", "#d9d9d9"),
        )


def list_profiles() -> list[str]:
    """Names of every built-in species profile (YAML filename stems)."""
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str) -> SpeciesProfile:
    """Load a built-in profile by name (matches the YAML filename stem)."""
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No species profile named {name!r}. Available: {available}")
    return SpeciesProfile.from_yaml(path)
