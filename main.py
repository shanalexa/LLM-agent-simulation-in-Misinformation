import argparse
import json
import os

import pandas as pd
from matplotlib import pyplot as plt

from prompt import CHECKPOINT_DAY, EXPERIMENT_BRANCHES
from utils import (
    AGENTS_CONFIG_PATH,
    EXPERIMENT_SIMULATION_DAYS,
    configure_llm,
    export_agents_config,
    get_llm_config,
)
from world import World

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AIS/DOA misinformation simulation: Day 0–7 baseline, Day 8 five branches, Day 9–15 diffusion."
    )
    parser.add_argument("--name", default="GABM", help="Run label for checkpoints and CSV exports.")
    parser.add_argument("--no_init_healthy", default=35, type=int, help="Agent count (must be 35).")
    parser.add_argument("--no_init_infect", default=0, type=int, help="Deprecated; Day 0 uses LLM + threshold.")
    parser.add_argument("--no_of_runs", default=1, type=int, help="Number of independent run folders.")
    parser.add_argument("--offset", default=0, type=int, help="Resume daily steps from this day index (1–15).")
    parser.add_argument("--load_from_run", default=0, type=int, help="Run folder index for pickle resume.")
    parser.add_argument("--belief_threshold", default=5, type=int, help="Belief >= T => Infected.")
    parser.add_argument("--rumor_topic", default=None, type=str, help="Misinformation text (Day 0 broadcast).")
    parser.add_argument(
        "--phase",
        choices=["full", "phase1", "phase2", "branch"],
        default="full",
        help="full=Day0–7 then 5 branches; phase1=Day0–7 only; phase2=all branches from checkpoint; branch=one branch.",
    )
    parser.add_argument(
        "--branch",
        choices=list(EXPERIMENT_BRANCHES),
        default="NYT",
        help="Intervention branch for phase2/branch (NYT, WSJ, AI, Neutral, Control).",
    )
    parser.add_argument(
        "--checkpoint_day7",
        default="output/checkpoint_day7.json",
        help="Path to Day 7 JSON checkpoint for phase2.",
    )
    parser.add_argument(
        "--llm_model",
        default=None,
        help="Chat model name (default: env LLM_MODEL or gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--llm_base_url",
        default=None,
        help="OpenAI-compatible API base URL (default: env OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--llm_api_key",
        default=None,
        help="API key (default: env OPENAI_API_KEY; local Ollama often uses 'ollama').",
    )
    parser.add_argument(
        "--no_json_mode",
        action="store_true",
        help="Disable response_format json_object; parse JSON from plain text instead.",
    )
    parser.add_argument(
        "--llm_workers",
        type=int,
        default=None,
        help="Parallel LLM threads per day step (default: env LLM_MAX_WORKERS or 8; use 1 for serial).",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=None,
        help="Max tokens per completion (default: env LLM_MAX_TOKENS or 2048).",
    )
    args = parser.parse_args()

    configure_llm(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        use_json_mode=not args.no_json_mode,
        max_workers=args.llm_workers,
        max_tokens=args.llm_max_tokens,
    )
    print(f"Parameters: {args}")
    print(f"LLM: {get_llm_config()}")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    export_agents_config(AGENTS_CONFIG_PATH)

    def run_branch(world, checkpoint_path, output_path, offset=0, end_day=None):
        world.run_model(
            checkpoint_path, offset=offset, export_dir=output_path, end_day=end_day
        )
        data = world.datacollector.get_model_vars_dataframe()
        df = pd.DataFrame(data)
        if len(df):
            df.insert(0, "Step", range(len(df)))
            tag = world.experiment_branch or "Phase1"
            df.to_csv(os.path.join(output_path, f"{args.name}-{tag}-data.csv"))
            agent_df = world.datacollector.get_agent_vars_dataframe()
            agent_df.to_csv(os.path.join(output_path, f"{args.name}-{tag}-agents-panel.csv"))

    for i in range(args.load_from_run, args.no_of_runs):
        print(f"--------Run - {i + 1}---------")
        checkpoint_path = f"checkpoint/run-{i + 1}"
        output_path = f"output/run-{i + 1}"
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        ckpt_day7 = os.path.join(output_path, "checkpoint_day7.json")
        if args.checkpoint_day7 and os.path.isabs(args.checkpoint_day7) is False:
            if not os.path.exists(args.checkpoint_day7) and os.path.exists(ckpt_day7):
                args.checkpoint_day7 = ckpt_day7

        if args.phase in ("phase2", "branch"):
            if not os.path.exists(args.checkpoint_day7):
                raise FileNotFoundError(
                    f"Day 7 checkpoint not found: {args.checkpoint_day7}. Run --phase phase1 first."
                )
            branches = [args.branch] if args.phase == "branch" else list(EXPERIMENT_BRANCHES)
            for branch in branches:
                print(f"=== Branch {branch} (Day 8–15) ===")
                model = World.from_checkpoint_day7(args, args.checkpoint_day7, experiment_branch=branch)
                model.name = f"{args.name}_{branch}"
                run_branch(
                    model,
                    checkpoint_path,
                    output_path,
                    offset=CHECKPOINT_DAY,
                    end_day=EXPERIMENT_SIMULATION_DAYS,
                )
            continue

        if args.load_from_run != 0:
            checkpoint_file = f"checkpoint/run-{args.load_from_run + 1}/{args.name}-{args.offset}.pkl"
            if os.path.exists(checkpoint_file):
                model = World.load_checkpoint(checkpoint_file)
            else:
                model = World(
                    args,
                    initial_healthy=args.no_init_healthy,
                    initial_infected=args.no_init_infect,
                )
        else:
            model = World(
                args,
                initial_healthy=args.no_init_healthy,
                initial_infected=args.no_init_infect,
            )

        phase1_end = CHECKPOINT_DAY if args.phase == "phase1" else EXPERIMENT_SIMULATION_DAYS
        run_branch(
            model,
            checkpoint_path,
            output_path,
            offset=args.offset,
            end_day=phase1_end,
        )

        if args.phase == "phase1":
            model.export_checkpoint_day7(ckpt_day7)
            print(f"Phase 1 complete. Checkpoint: {ckpt_day7}")
            continue

        if args.phase == "full":
            if not os.path.exists(ckpt_day7):
                src = os.path.join("output", "checkpoint_day7.json")
                if os.path.exists(src):
                    ckpt_day7 = src
                elif os.path.exists(os.path.join(output_path, "checkpoint_day7.json")):
                    ckpt_day7 = os.path.join(output_path, "checkpoint_day7.json")
                else:
                    model.export_checkpoint_day7(ckpt_day7)
            for branch in EXPERIMENT_BRANCHES:
                print(f"=== Branch {branch} (Day 8–15) ===")
                branch_model = World.from_checkpoint_day7(args, ckpt_day7, experiment_branch=branch)
                branch_model.name = f"{args.name}_{branch}"
                run_branch(
                    branch_model,
                    checkpoint_path,
                    output_path,
                    offset=CHECKPOINT_DAY,
                    end_day=EXPERIMENT_SIMULATION_DAYS,
                )

        if args.phase != "phase1":
            model.save_checkpoint(file_path=f"{checkpoint_path}/{args.name}-completed.pkl")

            data = model.datacollector.get_model_vars_dataframe()
            df = pd.DataFrame(data)
            if len(df):
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df["Susceptible"], label="Susceptible")
                plt.plot(df.index, df["Infected"], label="Infected")
                plt.plot(df.index, df["Recovered"], label="Recovered")
                plt.xlabel("Step")
                plt.ylabel("# of People")
                plt.title("SIR (Phase 1)")
                plt.legend()
                plt.savefig(f"{output_path}/{args.name}-SIR.png", bbox_inches="tight")
