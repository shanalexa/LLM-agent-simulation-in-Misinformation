"""Load experiment settings from config.yaml (see config.yaml in repo root)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_CONFIG: Optional["AppConfig"] = None
_CONFIG_PATH: Optional[Path] = None

_EXPERIMENT_REQUIRED = (
    "population_n",
    "agents_per_ideology",
    "simulation_days",
    "checkpoint_day",
    "intervention_day",
    "contacts_per_day",
    "belief_threshold",
    "belief_min",
    "belief_max",
    "ltm_max_words",
    "branches",
    "liberal_level_max",
    "moderate_level",
    "conservative_level_min",
    "ibg_liberal_max",
    "ibg_conservative_min",
    "control_branch",
    "random_seed",
)

_PROMPT_FIELDS = (
    "day0_cognitive",
    "day0_ltm_init",
    "daily_stm",
    "daily_cognitive",
    "daily_ltm_compress",
    "intervention_cognitive",
    "intervention_ltm_compress",
)


@dataclass
class ExperimentConfig:
    population_n: int
    agents_per_ideology: int
    simulation_days: int
    checkpoint_day: int
    intervention_day: int
    contacts_per_day: int
    belief_threshold: int
    belief_min: int
    belief_max: int
    ltm_max_words: int
    branches: Tuple[str, ...]
    liberal_level_max: int
    moderate_level: int
    conservative_level_min: int
    ibg_liberal_max: int
    ibg_conservative_min: int
    control_branch: str
    random_seed: int


@dataclass
class PathsConfig:
    agents_config: str
    personas: str
    prompts: str
    output_dir: str
    checkpoint_dir: str
    checkpoint_day7: str


@dataclass
class PersonasConfig:
    archetypes: Tuple[dict, ...]
    ideology_profiles: Tuple[dict, ...]
    system_prompt_template: str


@dataclass
class LLMConfig:
    model: str
    max_retries: int = 30
    max_workers: int = 8
    max_tokens: int = 2048
    enable_thinking: bool = False
    use_json_mode: bool = True
    retry_sleep_base: float = 0.5
    retry_sleep_cap: float = 2.0
    temperature_day0_cognitive: float = 0.65
    temperature_stm: float = 0.5
    temperature_ltm: float = 0.5
    temperature_daily_cognitive: float = 1.0


@dataclass
class RunConfig:
    default_name: str = "MGS"
    default_phase: str = "full"
    default_branch: str = "NYT"
    default_population: int = 35
    default_belief_threshold: int = 5


@dataclass
class WorldConfig:
    start_date: str = "2020-03-03"
    save_daily_pickle: bool = True


@dataclass
class PromptsConfig:
    day0_cognitive: str
    day0_ltm_init: str
    daily_stm: str
    daily_cognitive: str
    daily_ltm_compress: str
    intervention_cognitive: str
    intervention_ltm_compress: str


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    paths: PathsConfig
    personas: PersonasConfig
    llm: LLMConfig
    run: RunConfig
    world: WorldConfig
    prompts: PromptsConfig
    rumor_text: str
    refutation_texts: Dict[str, str]


def _require_section(raw: dict, name: str) -> dict:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"config.yaml must define a non-empty '{name}' section")
    return section


def _require_keys(data: dict, keys: tuple[str, ...], section: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"config.yaml '{section}' missing required keys: {', '.join(missing)}")


def _require_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.yaml '{field}' must be a non-empty string")
    return value.strip()


def _load_experiment(data: dict) -> ExperimentConfig:
    _require_keys(data, _EXPERIMENT_REQUIRED, "experiment")
    branches = data["branches"]
    if not isinstance(branches, list) or not branches:
        raise ValueError("config.yaml 'experiment.branches' must be a non-empty list")
    return ExperimentConfig(
        population_n=int(data["population_n"]),
        agents_per_ideology=int(data["agents_per_ideology"]),
        simulation_days=int(data["simulation_days"]),
        checkpoint_day=int(data["checkpoint_day"]),
        intervention_day=int(data["intervention_day"]),
        contacts_per_day=int(data["contacts_per_day"]),
        belief_threshold=int(data["belief_threshold"]),
        belief_min=int(data["belief_min"]),
        belief_max=int(data["belief_max"]),
        ltm_max_words=int(data["ltm_max_words"]),
        branches=tuple(str(b) for b in branches),
        liberal_level_max=int(data["liberal_level_max"]),
        moderate_level=int(data["moderate_level"]),
        conservative_level_min=int(data["conservative_level_min"]),
        ibg_liberal_max=int(data["ibg_liberal_max"]),
        ibg_conservative_min=int(data["ibg_conservative_min"]),
        control_branch=str(data["control_branch"]),
        random_seed=int(data["random_seed"]),
    )


def _load_llm(data: dict) -> LLMConfig:
    model = _require_non_empty_str(data.get("model"), "llm.model")
    rest = {k: v for k, v in data.items() if k in LLMConfig.__dataclass_fields__ and k != "model"}
    return LLMConfig(model=model, **rest)


def _load_paths(data: dict) -> PathsConfig:
    _require_keys(
        data,
        ("agents_config", "personas", "prompts", "output_dir", "checkpoint_dir", "checkpoint_day7"),
        "paths",
    )
    return PathsConfig(
        agents_config=str(data["agents_config"]),
        personas=str(data["personas"]),
        prompts=str(data["prompts"]),
        output_dir=str(data["output_dir"]),
        checkpoint_dir=str(data["checkpoint_dir"]),
        checkpoint_day7=str(data["checkpoint_day7"]),
    )


def _load_prompts(data: dict) -> PromptsConfig:
    _require_keys(data, _PROMPT_FIELDS, "prompts")
    return PromptsConfig(
        **{k: _require_non_empty_str(data[k], f"prompts.{k}") for k in _PROMPT_FIELDS}
    )


def _load_prompts_yaml(prompts_path: Path) -> PromptsConfig:
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    with open(prompts_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "prompts" in raw and isinstance(raw["prompts"], dict):
        raw = raw["prompts"]
    return _load_prompts(raw)


def _validate_config(cfg: AppConfig) -> None:
    e = cfg.experiment
    if e.checkpoint_day >= e.intervention_day:
        raise ValueError(
            "experiment.checkpoint_day must be less than experiment.intervention_day"
        )
    if e.intervention_day > e.simulation_days:
        raise ValueError("experiment.intervention_day must not exceed experiment.simulation_days")
    if e.population_n != e.agents_per_ideology * len(cfg.personas.ideology_profiles):
        raise ValueError(
            "experiment.population_n must equal agents_per_ideology × len(ideology_profiles) "
            f"(got {e.population_n}, expected {e.agents_per_ideology * len(cfg.personas.ideology_profiles)})"
        )
    for branch in e.branches:
        if branch not in cfg.refutation_texts:
            raise ValueError(f"texts.refutation must define branch '{branch}'")
    if e.control_branch not in e.branches:
        raise ValueError("experiment.control_branch must appear in experiment.branches")


def daily_stm_prompt(cfg: Optional[AppConfig] = None) -> str:
    c = cfg or get_config()
    return c.prompts.daily_stm.replace(
        "{contacts_per_day}", str(c.experiment.contacts_per_day)
    )


def _resolve_config_path(config_path: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def _load_personas_yaml(personas_path: Path) -> PersonasConfig:
    if not personas_path.is_file():
        raise FileNotFoundError(f"Personas file not found: {personas_path}")
    with open(personas_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    archetypes = tuple(raw.get("archetypes") or ())
    profiles = tuple(raw.get("ideology_profiles") or ())
    template = (raw.get("system_prompt_template") or "").strip()
    if not archetypes or not profiles or not template:
        raise ValueError(
            f"personas.yaml must define archetypes, ideology_profiles, and system_prompt_template: {personas_path}"
        )
    return PersonasConfig(
        archetypes=archetypes,
        ideology_profiles=profiles,
        system_prompt_template=template,
    )


def _load_yaml(path: Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    exp = _load_experiment(_require_section(raw, "experiment"))
    paths = _load_paths(_require_section(raw, "paths"))

    llm = _load_llm(_require_section(raw, "llm"))

    run_raw = raw.get("run") or {}
    run = RunConfig(**{k: v for k, v in run_raw.items() if k in RunConfig.__dataclass_fields__})

    world_raw = raw.get("world") or {}
    world = WorldConfig(**{k: v for k, v in world_raw.items() if k in WorldConfig.__dataclass_fields__})

    texts = _require_section(raw, "texts")
    rumor = _require_non_empty_str(texts.get("misinformation"), "texts.misinformation")
    refutation = texts.get("refutation")
    if not isinstance(refutation, dict) or not refutation:
        raise ValueError("config.yaml 'texts.refutation' must be a non-empty mapping")

    personas_path = _resolve_config_path(path, paths.personas)
    personas = _load_personas_yaml(personas_path)
    prompts_path = _resolve_config_path(path, paths.prompts)
    prompts = _load_prompts_yaml(prompts_path)

    cfg = AppConfig(
        experiment=exp,
        paths=paths,
        personas=personas,
        llm=llm,
        run=run,
        world=world,
        prompts=prompts,
        rumor_text=rumor,
        refutation_texts={k: (v if v is not None else "") for k, v in refutation.items()},
    )
    _validate_config(cfg)
    return cfg


def init_config(path: str | Path = "config.yaml") -> AppConfig:
    global _CONFIG, _CONFIG_PATH
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    _CONFIG_PATH = path.resolve()
    _CONFIG = _load_yaml(path)
    return _CONFIG


def get_config() -> AppConfig:
    if _CONFIG is None:
        raise RuntimeError("Config not loaded. Call init_config(path) before running the simulation.")
    return _CONFIG


def get_config_path() -> Optional[Path]:
    return _CONFIG_PATH


def _resolve_path(relative: str) -> Path:
    cfg = get_config()
    base = get_config_path() or Path("config.yaml")
    p = Path(relative)
    if p.is_absolute():
        return p
    return (base.parent / p).resolve()


def get_personas_path() -> Path:
    return _resolve_path(get_config().paths.personas)


def get_prompts_path() -> Path:
    return _resolve_path(get_config().paths.prompts)
