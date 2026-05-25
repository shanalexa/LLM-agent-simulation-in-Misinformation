import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import mesa
import numpy as np
from tqdm import tqdm

from citizen import Citizen
from prompt import (
    CHECKPOINT_DAY,
    DEFAULT_EXPERIMENT_RUMOR_TOPIC,
    EXPERIMENT_BRANCHES,
    INTERVENTION_DAY,
    REFUTATION_TEXTS,
)
from utils import (
    DESIGN_POPULATION_N,
    EXPERIMENT_RANDOM_CONTACTS_PER_DAY,
    EXPERIMENT_SIMULATION_DAYS,
    build_agents_config,
    clear_cache,
    get_llm_config,
    ideology_level_from_name,
)

# functions for mesa.DataCollector in World class
def compute_num_susceptible(model):
    return sum([1 for a in model.schedule.agents if a.health_condition == "Susceptible"])


def compute_num_infected(model):
    return sum([1 for a in model.schedule.agents if a.health_condition == "Infected"])


def compute_num_recovered(model):
    return sum([1 for a in model.schedule.agents if a.health_condition == "Recovered"])


def _agents(model):
    return list(model.schedule.agents)


def compute_mean_belief_intensity(model):
    agents = _agents(model)
    if not agents:
        return float("nan")
    return float(np.mean([a.beliefs[-1] for a in agents]))


def compute_polarization_index(model):
    agents = _agents(model)
    vals = [a.beliefs[-1] for a in agents]
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals))


def compute_inter_group_belief_gap(model):
    liberal = [a.beliefs[-1] for a in _agents(model) if a.political_ideology <= 3]
    conservative = [a.beliefs[-1] for a in _agents(model) if a.political_ideology >= 5]
    if not liberal or not conservative:
        return float("nan")
    return abs(float(np.mean(liberal)) - float(np.mean(conservative)))


def compute_average_attitude_shift(model):
    agents = _agents(model)
    if not agents:
        return float("nan")
    return float(np.mean([abs(a.beliefs[-1] - a.beliefs[0]) for a in agents]))


def compute_mean_daily_abs_belief_change(model):
    deltas = []
    for a in _agents(model):
        if len(a.beliefs) >= 2:
            deltas.append(abs(a.beliefs[-1] - a.beliefs[-2]))
    return float(np.mean(deltas)) if deltas else 0.0


def compute_rsr(model):
    T = model.belief_infected_threshold
    infected0 = [a for a in _agents(model) if a.beliefs and a.beliefs[0] >= T]
    if not infected0:
        return float("nan")
    recovered = [a for a in infected0 if a.beliefs[-1] < T]
    return len(recovered) / len(infected0)


def _agent_belief_intensity(agent):
    return agent.beliefs[-1] if agent.beliefs else None


def _agent_political_ideology(agent):
    return agent.political_ideology


def _agent_rumor_belief_state(agent):
    return agent.health_condition


def _agent_health_label(agent):
    return agent.health_condition


def _agent_age(agent):
    return agent.age


def _agent_education(agent):
    return agent.qualification


def _agent_occupation(agent):
    return agent.occupation


def _agent_config_id(agent):
    return agent.agent_config_id


class World(mesa.Model):
    """AIS: environment controller for 35 DOA agents, 16 days (0–15)."""

    def __init__(
        self,
        args,
        initial_healthy=35,
        initial_infected=0,
        experiment_branch=None,
        checkpoint_agents=None,
        run_day0=True,
    ):
        self.initial_healthy = initial_healthy
        self.initial_infected = initial_infected
        self.population = initial_healthy + initial_infected
        self.step_count = EXPERIMENT_SIMULATION_DAYS
        self.offset = 0
        self.name = args.name
        self.experiment_branch = experiment_branch
        self.simulation_day = 0
        self.topic = getattr(args, "rumor_topic", None) or DEFAULT_EXPERIMENT_RUMOR_TOPIC

        self.total_contact_rates = 0
        self.track_contact_rate = [0]
        self.list_new_infected_cases = [0]
        self.list_new_susceptible_cases = [0]
        self.daily_new_infected_cases = 0
        self.daily_new_susceptible_cases = 0
        self.infected = 0
        self.susceptible = 0
        self.current_date = datetime(2020, 3, 3)
        self.contact_rate = EXPERIMENT_RANDOM_CONTACTS_PER_DAY
        self.belief_infected_threshold = int(getattr(args, "belief_threshold", 5))

        self.schedule = mesa.time.RandomActivation(self)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Susceptible": compute_num_susceptible,
                "Infected": compute_num_infected,
                "Recovered": compute_num_recovered,
                "MeanBeliefIntensity": compute_mean_belief_intensity,
                "PolarizationIndex": compute_polarization_index,
                "InterGroupBeliefGap": compute_inter_group_belief_gap,
                "AverageAttitudeShift": compute_average_attitude_shift,
                "MeanDailyAbsBeliefChange": compute_mean_daily_abs_belief_change,
                "RefutationSuccessRate": compute_rsr,
            },
            agent_reporters={
                "BeliefIntensity": _agent_belief_intensity,
                "PoliticalIdeology": _agent_political_ideology,
                "RumorBeliefState": _agent_rumor_belief_state,
                "HealthState": _agent_health_label,
                "Age": _agent_age,
                "Education": _agent_education,
                "Occupation": _agent_occupation,
                "AgentConfigID": _agent_config_id,
            },
        )

        if self.population != DESIGN_POPULATION_N:
            raise ValueError(
                f"This experiment uses {DESIGN_POPULATION_N} agents; got {self.population}."
            )

        if checkpoint_agents is not None:
            self._init_agents_from_checkpoint(checkpoint_agents)
            self.simulation_day = CHECKPOINT_DAY
            self._sync_population_counts()
        else:
            self._init_agents_fresh()
            if run_day0:
                self._run_day0_broadcast()
                self.datacollector.collect(self)
                self._sync_population_counts()
                self.list_new_infected_cases[0] = self.daily_new_infected_cases
                self.list_new_susceptible_cases[0] = self.daily_new_susceptible_cases

    def _init_agents_fresh(self):
        agent_id = 0
        personas = build_agents_config()
        for i in range(self.population):
            p = personas[i]
            ideology_level = ideology_level_from_name(p["ideology"])
            citizen = Citizen(
                model=self,
                unique_id=agent_id,
                name=p["agent_id"],
                age=p["age"],
                qualification=p["education"],
                occupation=p["occupation"],
                archetype_id=p["archetype_id"],
                agent_config_id=p["agent_id"],
                system_prompt=p["system_prompt"],
                health_condition="Susceptible",
                topic=self.topic,
                political_ideology=ideology_level,
            )
            self.schedule.add(citizen)
            agent_id += 1

    def _init_agents_from_checkpoint(self, checkpoint_agents):
        for i, data in enumerate(checkpoint_agents):
            citizen = Citizen.from_checkpoint(self, unique_id=i, data=data)
            self.schedule.add(citizen)

    def _llm_parallel_workers(self) -> int:
        return max(1, int(get_llm_config().get("max_workers", 1)))

    def _parallel_for_agents(self, agents, fn, label: str = ""):
        """
        Run fn(agent) over agents with a thread pool (LLM calls are I/O-bound).
        Snapshot / apply barrier is unchanged — only per-agent compute is parallel.
        """
        agents = list(agents)
        workers = min(self._llm_parallel_workers(), len(agents))
        if workers <= 1:
            for a in agents:
                fn(a)
            return

        if label:
            print(f"  LLM parallel: {label} ({len(agents)} agents, {workers} workers)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fn, a): a for a in agents}
            for fut in as_completed(futures):
                fut.result()

    def _run_day0_broadcast(self):
        agents = sorted(self.schedule.agents, key=lambda a: a.unique_id)
        topic = self.topic

        def _run(a):
            a.run_day0(topic)

        self._parallel_for_agents(agents, _run, label="Day 0 broadcast")

    def _sync_population_counts(self):
        self.susceptible = sum(
            1 for a in self.schedule.agents if a.health_condition == "Susceptible"
        )
        self.infected = sum(1 for a in self.schedule.agents if a.health_condition == "Infected")
        self.initial_healthy = self.susceptible
        self.initial_infected = self.infected

    def decide_agent_interactions(self):
        k = min(self.contact_rate, max(0, len(self.schedule.agents) - 1))
        for agent in self.schedule.agents:
            agent.agent_interaction.clear()
        for agent in self.schedule.agents:
            others = [a for a in self.schedule.agents if a is not agent]
            if not others:
                continue
            take = min(k, len(others))
            agent.agent_interaction.extend(random.sample(others, k=take))

    def _take_eod_snapshot(self):
        """Snapshot public state at start of day t (= end of day t-1)."""
        snapshot = {}
        for a in self.schedule.agents:
            if not a.opinions or not a.beliefs:
                raise ValueError(f"Agent {a.unique_id} missing opinions/beliefs for snapshot.")
            snapshot[a.unique_id] = {
                "opinion": a.opinions[-1],
                "belief": a.beliefs[-1],
                "long_memory": a.long_term_memory or "",
            }
        return snapshot

    def _is_intervention_day(self, day: int) -> bool:
        return (
            day == INTERVENTION_DAY
            and self.experiment_branch
            and self.experiment_branch != "Control"
        )

    def step_day(self, day: int):
        self.simulation_day = day
        self.decide_agent_interactions()

        for agent in self.schedule.agents:
            self.total_contact_rates += len(agent.agent_interaction)
        self.track_contact_rate.append(self.total_contact_rates)
        self.total_contact_rates = 0

        snapshot = self._take_eod_snapshot()
        refutation = None
        if self._is_intervention_day(day):
            refutation = REFUTATION_TEXTS.get(self.experiment_branch, "")

        agents = list(self.schedule.agents)

        def _compute(a):
            contact_ids = [p.unique_id for p in a.agent_interaction]
            peer_opinion_beliefs = [
                (snapshot[p.unique_id]["opinion"], snapshot[p.unique_id]["belief"])
                for p in a.agent_interaction
            ]
            s = snapshot[a.unique_id]
            payload = a.compute_daily_update(
                day=day,
                start_opinion=s["opinion"],
                start_long_memory=s["long_memory"],
                peer_opinion_beliefs=peer_opinion_beliefs,
                contact_ids=contact_ids,
                refutation_text=refutation,
            )
            return a, payload

        workers = min(self._llm_parallel_workers(), len(agents))
        pending = []
        if workers <= 1:
            for a in agents:
                pending.append(_compute(a))
        else:
            print(f"  LLM parallel: Day {day} ({len(agents)} agents, {workers} workers)")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_compute, a) for a in agents]
                for fut in as_completed(futures):
                    pending.append(fut.result())

        for a, payload in pending:
            a.apply_daily_update(payload)

    def export_checkpoint_day7(self, path: str):
        payload = {
            "checkpoint_day": CHECKPOINT_DAY,
            "misinformation_text": self.topic,
            "agents": [
                a.to_checkpoint_dict()
                for a in sorted(self.schedule.agents, key=lambda x: x.unique_id)
            ],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_checkpoint_day7(cls, args, json_path: str, experiment_branch: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        topic = data.get("misinformation_text") or getattr(args, "rumor_topic", None) or DEFAULT_EXPERIMENT_RUMOR_TOPIC
        if not getattr(args, "rumor_topic", None):
            args.rumor_topic = topic
        world = cls(
            args,
            initial_healthy=len(data["agents"]),
            initial_infected=0,
            experiment_branch=experiment_branch,
            checkpoint_agents=data["agents"],
            run_day0=False,
        )
        return world

    def export_results_day15(self, path: str):
        branch = self.experiment_branch or "Phase1"
        payload = {
            "experiment_branch": branch,
            "simulation_day": self.simulation_day,
            "misinformation_text": self.topic,
            "belief_threshold": self.belief_infected_threshold,
            "agents": [
                a.to_export_dict()
                for a in sorted(self.schedule.agents, key=lambda x: x.unique_id)
            ],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    def run_model(self, checkpoint_path, offset=0, export_dir="output", end_day=None):
        self.offset = offset
        start_day = max(1, offset + 1)
        end_day = end_day if end_day is not None else EXPERIMENT_SIMULATION_DAYS

        for day in tqdm(range(start_day, end_day + 1)):
            self.time = float(day)
            self.daily_new_infected_cases = 0
            self.daily_new_susceptible_cases = 0

            self.step_day(day)
            self.datacollector.collect(self)

            self.list_new_infected_cases.append(self.daily_new_infected_cases)
            self.list_new_susceptible_cases.append(self.daily_new_susceptible_cases)

            print(f"Day {day} ({self.current_date.date()}) branch={self.experiment_branch}")
            print(f"  Infected={self.infected} Susceptible={self.susceptible}")

            if day == CHECKPOINT_DAY and not self.experiment_branch:
                ckpt = os.path.join(export_dir, "checkpoint_day7.json")
                self.export_checkpoint_day7(ckpt)
                print(f"  Saved {ckpt}")

            if day == EXPERIMENT_SIMULATION_DAYS and self.experiment_branch:
                fname = f"results_{self.experiment_branch}_Day15.json"
                out = os.path.join(export_dir, fname)
                self.export_results_day15(out)
                print(f"  Saved {out}")

            self.current_date += timedelta(days=1)
            if checkpoint_path:
                path = f"{checkpoint_path}/{self.name}-{day}.pkl"
                self.save_checkpoint(file_path=path)
            clear_cache()

    def save_checkpoint(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_checkpoint(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
