"""Load experiment settings from config.yaml (see config.yaml in repo root)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_CONFIG: Optional["AppConfig"] = None
_CONFIG_PATH: Optional[Path] = None


@dataclass
class ExperimentConfig:
    population_n: int = 35
    agents_per_ideology: int = 5
    simulation_days: int = 15
    checkpoint_day: int = 7
    intervention_day: int = 8
    contacts_per_day: int = 3
    belief_threshold: int = 5
    belief_min: int = 1
    belief_max: int = 7
    ltm_max_words: int = 300
    branches: Tuple[str, ...] = ("NYT", "WSJ", "AI", "Neutral", "Control")
    liberal_level_max: int = 3
    moderate_level: int = 4
    conservative_level_min: int = 5
    ibg_liberal_max: int = 3
    ibg_conservative_min: int = 5
    control_branch: str = "Control"


@dataclass
class PathsConfig:
    agents_config: str = "agents_config.json"
    personas: str = "personas.yaml"
    output_dir: str = "output"
    checkpoint_dir: str = "checkpoint"
    checkpoint_day7: str = "output/checkpoint_day7.json"


@dataclass
class PersonasConfig:
    archetypes: Tuple[dict, ...] = ()
    ideology_profiles: Tuple[dict, ...] = ()
    system_prompt_template: str = ""


@dataclass
class LLMConfig:
    model: str = "gpt-3.5-turbo"
    api_key_fallback: str = "ollama"
    max_retries: int = 30
    max_workers: int = 8
    max_tokens: int = 2048
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
    day0_cognitive: str = ""
    day0_ltm_init: str = ""
    daily_stm: str = ""
    daily_cognitive: str = ""
    daily_ltm_compress: str = ""
    intervention_cognitive: str = ""
    intervention_ltm_compress: str = ""


@dataclass
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    personas: PersonasConfig = field(default_factory=PersonasConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    run: RunConfig = field(default_factory=RunConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    rumor_text: str = ""
    refutation_texts: Dict[str, str] = field(default_factory=dict)


def _dc(cls, data: Optional[dict]):
    if not data:
        return cls()
    valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    if cls is ExperimentConfig and "branches" in valid:
        valid["branches"] = tuple(valid["branches"])
    return cls(**valid)


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

    exp = _dc(ExperimentConfig, raw.get("experiment"))
    paths = _dc(PathsConfig, raw.get("paths"))
    llm = _dc(LLMConfig, raw.get("llm"))
    run = _dc(RunConfig, raw.get("run"))
    world = _dc(WorldConfig, raw.get("world"))
    prompts_raw = raw.get("prompts") or {}
    prompts = PromptsConfig(
        day0_cognitive=prompts_raw.get("day0_cognitive", ""),
        day0_ltm_init=prompts_raw.get("day0_ltm_init", ""),
        daily_stm=prompts_raw.get("daily_stm", ""),
        daily_cognitive=prompts_raw.get("daily_cognitive", ""),
        daily_ltm_compress=prompts_raw.get("daily_ltm_compress", ""),
        intervention_cognitive=prompts_raw.get("intervention_cognitive", ""),
        intervention_ltm_compress=prompts_raw.get("intervention_ltm_compress", ""),
    )

    texts = raw.get("texts") or {}
    rumor = (texts.get("misinformation") or "").strip()
    refutation = texts.get("refutation") or {}
    if not isinstance(refutation, dict):
        refutation = {}

    personas_path = _resolve_config_path(path, paths.personas)
    personas = _load_personas_yaml(personas_path)

    return AppConfig(
        experiment=exp,
        paths=paths,
        personas=personas,
        llm=llm,
        run=run,
        world=world,
        prompts=prompts,
        rumor_text=rumor,
        refutation_texts={k: (v or "") for k, v in refutation.items()},
    )


def _sync_utils_module(cfg: AppConfig) -> None:
    import utils as u

    e = cfg.experiment
    p = cfg.personas
    u.AGENTS_CONFIG_PATH = cfg.paths.agents_config
    u.DESIGN_POPULATION_N = e.population_n
    u.DESIGN_AGENTS_PER_IDEOLOGY_LEVEL = e.agents_per_ideology
    u.EXPERIMENT_SIMULATION_DAYS = e.simulation_days
    u.EXPERIMENT_RANDOM_CONTACTS_PER_DAY = e.contacts_per_day
    u.DEMOGRAPHIC_ARCHETYPES = p.archetypes
    u.IDEOLOGY_PROFILES = p.ideology_profiles
    u.AGENT_SYSTEM_PROMPT_TEMPLATE = p.system_prompt_template


def _sync_prompt_module(cfg: AppConfig) -> None:
    import prompt as p

    contacts = cfg.experiment.contacts_per_day
    p.LTM_MAX_WORDS = cfg.experiment.ltm_max_words
    p.CHECKPOINT_DAY = cfg.experiment.checkpoint_day
    p.INTERVENTION_DAY = cfg.experiment.intervention_day
    p.EXPERIMENT_BRANCHES = tuple(cfg.experiment.branches)
    p.DEFAULT_EXPERIMENT_RUMOR_TOPIC = cfg.rumor_text
    p.REFUTATION_TEXTS = dict(cfg.refutation_texts)

    p.day0_cognitive_prompt = cfg.prompts.day0_cognitive
    p.day0_ltm_init_prompt = cfg.prompts.day0_ltm_init
    p.daily_stm_prompt = cfg.prompts.daily_stm.replace("{contacts_per_day}", str(contacts))
    p.daily_cognitive_prompt = cfg.prompts.daily_cognitive
    p.daily_ltm_compress_prompt = cfg.prompts.daily_ltm_compress
    p.intervention_cognitive_prompt = cfg.prompts.intervention_cognitive
    p.intervention_ltm_compress_prompt = cfg.prompts.intervention_ltm_compress

    p.day0_broadcast_initial_prompt = p.day0_cognitive_prompt
    p.reflecting_prompt = p.daily_stm_prompt
    p.long_memory_prompt = p.daily_ltm_compress_prompt
    p.update_opinion_prompt = p.daily_cognitive_prompt


def init_config(path: str | Path = "config.yaml") -> AppConfig:
    global _CONFIG, _CONFIG_PATH
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    _CONFIG_PATH = path.resolve()
    _CONFIG = _load_yaml(path)
    _sync_utils_module(_CONFIG)
    _sync_prompt_module(_CONFIG)
    return _CONFIG


def get_config() -> AppConfig:
    if _CONFIG is None:
        init_config()
    return _CONFIG


def get_config_path() -> Optional[Path]:
    return _CONFIG_PATH
