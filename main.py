import argparse
import os

import pandas as pd

from config import (
    get_config,
    get_config_path,
    get_personas_path,
    get_prompts_path,
    init_config,
    resolve_checkpoint_day7_path,
)
from run_manifest import (
    apply_random_seed,
    create_run_manifest,
    finalize_run_manifest,
    write_run_manifest,
)
from utils import (
    configure_llm,
    export_agents_config,
    get_llm_config,
)
from world import World


def _build_parser(cfg):
    parser = argparse.ArgumentParser(
        description="AIS/DOA misinformation simulation: Day 0–7 baseline, Day 8 five branches, Day 9–15 diffusion."
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("MGS_CONFIG", "config.yaml"),
        help="Path to config.yaml (experiment, texts, prompts, LLM defaults).",
    )
    parser.add_argument("--name", default=cfg.run.default_name, help="Run label for checkpoints and CSV exports.")
    parser.add_argument(
        "--no_init_healthy",
        default=cfg.run.default_population,
        type=int,
        help="Agent count (must match experiment.population_n in config).",
    )
    parser.add_argument("--no_of_runs", default=1, type=int, help="Number of independent run folders.")
    parser.add_argument("--offset", default=0, type=int, help="Resume daily steps from this day index (1–15).")
    parser.add_argument("--load_from_run", default=0, type=int, help="Run folder index for pickle resume.")
    parser.add_argument(
        "--belief_threshold",
        default=cfg.experiment.belief_threshold,
        type=int,
        help="Belief >= T => Infected (overrides config experiment.belief_threshold).",
    )
    parser.add_argument("--rumor_topic", default=None, type=str, help="Misinformation text (overrides config texts).")
    parser.add_argument(
        "--phase",
        choices=["full", "phase1", "phase2", "branch"],
        default=cfg.run.default_phase,
        help="full=Day0–7 baseline then 5 branches (Day8–15); phase1=Day0–7 only; phase2/branches from checkpoint.",
    )
    parser.add_argument(
        "--branch",
        choices=list(cfg.experiment.branches),
        default=cfg.run.default_branch,
        help="Intervention branch for phase2/branch.",
    )
    parser.add_argument(
        "--checkpoint_day7",
        default=None,
        help=(
            "Day 7 JSON checkpoint (default: <output_dir>/run-N/"
            f"{cfg.paths.checkpoint_day7})."
        ),
    )
    parser.add_argument("--llm_model", default=None, help="Override config llm.model / env LLM_MODEL.")
    parser.add_argument("--vllm_api_base", default=None, help="Override env VLLM_API_BASE.")
    parser.add_argument("--llm_api_key", default=None, help="Override config / env OPENAI_API_KEY.")
    parser.add_argument(
        "--no_json_mode",
        action="store_true",
        help="Force disable JSON response_format (overrides config llm.use_json_mode).",
    )
    parser.add_argument("--llm_workers", type=int, default=None, help="Override config / env LLM_MAX_WORKERS.")
    parser.add_argument("--llm_max_tokens", type=int, default=None, help="Override config / env LLM_MAX_TOKENS.")
    parser.add_argument(
        "--no_daily_pickle",
        action="store_true",
        help="Skip daily .pkl checkpoints (overrides config world.save_daily_pickle).",
    )
    return parser


def _run_single_iteration(args, cfg, checkpoint_path, output_path, run_branch):
    ckpt_day7 = str(resolve_checkpoint_day7_path(args.checkpoint_day7, output_path, cfg))
    args.checkpoint_day7 = ckpt_day7

    if args.phase in ("phase2", "branch"):
        if not os.path.isfile(ckpt_day7):
            raise FileNotFoundError(
                f"Day 7 checkpoint not found: {ckpt_day7}. Run --phase phase1 first."
            )
        branches = [args.branch] if args.phase == "branch" else list(cfg.experiment.branches)
        for branch in branches:
            print(f"=== Branch {branch} (Day 8–15) ===")
            model = World.from_checkpoint_day7(args, ckpt_day7, experiment_branch=branch)
            model.name = f"{args.name}_{branch}"
            run_branch(
                model,
                checkpoint_path,
                output_path,
                offset=cfg.experiment.checkpoint_day,
                end_day=cfg.experiment.simulation_days,
            )
        return

    if args.load_from_run != 0:
        checkpoint_file = f"{cfg.paths.checkpoint_dir}/run-{args.load_from_run + 1}/{args.name}-{args.offset}.pkl"
        if os.path.exists(checkpoint_file):
            model = World.load_checkpoint(checkpoint_file)
        else:
            model = World(args, population=args.no_init_healthy)
    else:
        model = World(args, population=args.no_init_healthy)

    exp = cfg.experiment
    phase1_end = exp.checkpoint_day if args.phase in ("phase1", "full") else exp.simulation_days
    run_branch(
        model,
        checkpoint_path,
        output_path,
        offset=args.offset,
        end_day=phase1_end,
    )

    if args.phase == "phase1":
        print(f"Phase 1 complete. Checkpoint: {ckpt_day7}")
        return

    if args.phase == "full":
        if not os.path.isfile(ckpt_day7):
            model.export_checkpoint_day7(ckpt_day7)
        print(f"Phase 1 complete. Checkpoint: {ckpt_day7}")
        for branch in exp.branches:
            print(f"=== Branch {branch} (Day 8–15) ===")
            branch_model = World.from_checkpoint_day7(args, ckpt_day7, experiment_branch=branch)
            branch_model.name = f"{args.name}_{branch}"
            run_branch(
                branch_model,
                checkpoint_path,
                output_path,
                offset=exp.checkpoint_day,
                end_day=exp.simulation_days,
            )

    if args.phase != "phase1":
        model.save_checkpoint(file_path=f"{checkpoint_path}/{args.name}-completed.pkl")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.environ.get("MGS_CONFIG", "config.yaml"))
    pre_args, _ = pre.parse_known_args()
    init_config(pre_args.config)
    cfg = get_config()

    parser = _build_parser(cfg)
    args = parser.parse_args()

    if args.no_daily_pickle:
        cfg.world.save_daily_pickle = False

    configure_llm(
        base_url=args.vllm_api_base or os.getenv("VLLM_API_BASE"),
        api_key=args.llm_api_key,
        model=args.llm_model,
        use_json_mode=False if args.no_json_mode else cfg.llm.use_json_mode,
        max_workers=args.llm_workers,
        max_tokens=args.llm_max_tokens,
    )
    apply_random_seed(cfg.experiment.random_seed)
    config_path = get_config_path()
    if config_path is None:
        raise RuntimeError("Config path unavailable after init_config()")
    print(f"Config: {config_path}")
    print(f"Parameters: {args}")
    print(f"LLM: {get_llm_config()}")

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    export_agents_config(cfg.paths.agents_config)

    def run_branch(world, checkpoint_path, output_path, offset=0, end_day=None):
        world.run_model(
            checkpoint_path, offset=offset, export_dir=output_path, end_day=end_day
        )
        agent_df = world.datacollector.get_agent_vars_dataframe()
        if len(agent_df):
            tag = world.experiment_branch or "Phase1"
            out = os.path.join(output_path, f"{args.name}-{tag}-agents-panel.csv")
            agent_df.to_csv(out)
            print(f"  Saved {out}")

    for i in range(args.load_from_run, args.no_of_runs):
        print(f"--------Run - {i + 1}---------")
        checkpoint_path = f"{cfg.paths.checkpoint_dir}/run-{i + 1}"
        output_path = f"{cfg.paths.output_dir}/run-{i + 1}"
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        manifest_path = os.path.join(output_path, "run_manifest.json")
        manifest = create_run_manifest(
            run_name=args.name,
            run_index=i + 1,
            phase=args.phase,
            args=args,
            config_path=config_path,
            personas_path=get_personas_path(),
            prompts_path=get_prompts_path(),
            random_seed=cfg.experiment.random_seed,
            llm_config=get_llm_config(),
        )
        write_run_manifest(manifest_path, manifest)
        print(f"  Run manifest: {manifest_path}")

        try:
            _run_single_iteration(
                args=args,
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                run_branch=run_branch,
            )
            finalize_run_manifest(manifest, status="completed")
        except Exception as exc:
            finalize_run_manifest(manifest, status="failed", error=str(exc))
            raise
        finally:
            write_run_manifest(manifest_path, manifest)
