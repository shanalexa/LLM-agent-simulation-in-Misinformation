# -*- coding: utf-8 -*-
import json

import mesa

from config import daily_stm_prompt, get_config
from utils import (
    get_completion_from_messages,
    get_completion_from_messages_json,
    ideology_group_from_level,
    truncate_words,
)


def _peer_records_text(peer_opinion_beliefs):
    if not peer_opinion_beliefs:
        return "(no peer records)"
    lines = []
    for tweet, belief in peer_opinion_beliefs:
        lines.append(f'{{"tweet": {json.dumps(tweet, ensure_ascii=False)}, "belief_score": {int(belief)}}}')
    return "\n".join(lines)


class Citizen(mesa.Agent):
    """DOA: LLM-driven opinion agent with STM/LTM and spec SIR rules."""

    def __init__(
        self,
        model,
        unique_id,
        name,
        age,
        qualification,
        occupation,
        archetype_id,
        agent_config_id,
        system_prompt,
        health_condition,
        topic,
        political_ideology: int,
    ):
        # Mesa scheduler/Agent APIs changed across major versions.
        # - Mesa <3: Agent.__init__(unique_id, model)
        # - Mesa >=3: Agent.__init__(model, *args, **kwargs)
        try:
            super().__init__(model)
            self.unique_id = unique_id
        except TypeError:
            super().__init__(unique_id, model)
        self.agent_config_id = agent_config_id
        self.name = name
        self.age = age
        self.qualification = qualification
        self.occupation = occupation
        self.archetype_id = archetype_id
        self.persona_prompt = system_prompt
        self.system_prompt = system_prompt
        self.political_ideology = int(political_ideology)
        self.ideology_level = self.political_ideology
        self.ideology_group = ideology_group_from_level(self.ideology_level)
        self.topic = topic
        self.health_condition = health_condition
        self.has_been_infected = False

        self.opinion = ""
        self.belief = None
        self.reasoning = ""
        self.long_term_memory = ""
        self.short_term_memory = ""
        self.history_log = []

        self.opinions = []
        self.beliefs = []
        self.reasonings = []
        self.contact_ids = []
        self.agent_interaction = []

    @property
    def current_belief_score(self):
        return self.belief

    @property
    def current_tweet(self):
        return self.opinion

    @property
    def sir_state(self):
        return self.health_condition

    def _messages(self, user_msg: str):
        msg = []
        if self.system_prompt:
            msg.append({"role": "system", "content": self.system_prompt})
        msg.append({"role": "user", "content": user_msg})
        return msg

    def _parse_cognitive_json(self, response_json: str):
        output = json.loads(response_json)
        return (
            output["opinion"],
            int(output["belief_score"]),
            output["reasoning"],
        )

    def _llm_cognitive_json(self, user_msg: str, temperature: float = 1.0):
        response_json = get_completion_from_messages_json(
            self._messages(user_msg), temperature=temperature
        )
        return self._parse_cognitive_json(response_json)

    def _llm_text(self, user_msg: str, temperature: float = 0.5):
        return get_completion_from_messages(self._messages(user_msg), temperature=temperature)

    def apply_sir_from_belief(self, belief_score: int) -> None:
        """Spec SIR: score>=T => Infected; score<T & ever infected => Recovered; else Susceptible."""
        T = int(getattr(self.model, "belief_infected_threshold", 5))
        prev = self.health_condition
        if belief_score >= T:
            self.health_condition = "Infected"
            self.has_been_infected = True
        elif self.has_been_infected:
            self.health_condition = "Recovered"
        else:
            self.health_condition = "Susceptible"

        if prev != self.health_condition:
            m = self.model
            if self.health_condition == "Infected" and prev != "Infected":
                m.daily_new_infected_cases += 1
            elif self.health_condition == "Recovered" and prev == "Infected":
                m.daily_new_recovered_cases += 1

    def _append_history(self, day: int, belief_score: int, tweet: str, reasoning: str, ltm_snapshot: str):
        entry = {
            "day": day,
            "belief_score": int(belief_score),
            "sir_state": self.health_condition,
            "tweet": tweet,
            "internal_reasoning": reasoning,
            "ltm_snapshot": ltm_snapshot,
        }
        self.history_log.append(entry)

    def run_day0(self, misinformation_text: str) -> None:
        cfg = get_config()
        prompts = cfg.prompts
        ltm_words = cfg.experiment.ltm_max_words
        user_cog = prompts.day0_cognitive.format(
            persona_prompt=self.persona_prompt,
            misinformation_text=misinformation_text,
            belief_scale_definition=cfg.experiment.belief_scale_definition,
        )
        llm = cfg.llm
        tweet, belief, reasoning = self._llm_cognitive_json(
            user_cog, temperature=llm.temperature_day0_cognitive
        )

        user_ltm = prompts.day0_ltm_init.format(
            misinformation_text=misinformation_text,
            day0_reasoning=reasoning,
            max_words=ltm_words,
        )
        ltm = truncate_words(
            self._llm_text(user_ltm, temperature=llm.temperature_ltm),
            ltm_words,
        )

        self.opinion = tweet
        self.belief = belief
        self.reasoning = reasoning
        self.long_term_memory = ltm
        self.opinions = [tweet]
        self.beliefs = [belief]
        self.reasonings = [reasoning]

        self.apply_sir_from_belief(belief)
        self._append_history(0, belief, tweet, reasoning, ltm)

    def compute_daily_update(
        self,
        day: int,
        start_opinion: str,
        start_belief: int,
        start_long_memory: str,
        peer_opinion_beliefs: list,
        contact_ids: list,
        refutation_text: str = None,
    ):
        cfg = get_config()
        prompts = cfg.prompts
        ltm_words = cfg.experiment.ltm_max_words
        llm = cfg.llm
        peer_text = _peer_records_text(peer_opinion_beliefs)
        stm = self._llm_text(
            daily_stm_prompt(cfg).format(peer_records=peer_text),
            temperature=llm.temperature_stm,
        )

        prior_ltm = start_long_memory or "(empty)"
        cog_kwargs = dict(
            persona_prompt=self.persona_prompt,
            long_term_memory=prior_ltm,
            previous_day_tweet=start_opinion or "(none)",
            previous_day_belief_score=int(start_belief),
            short_term_memory=stm,
            belief_scale_definition=cfg.experiment.belief_scale_definition,
        )
        if refutation_text:
            user_cog = prompts.intervention_cognitive.format(
                refutation_text=refutation_text,
                **cog_kwargs,
            )
            user_ltm = prompts.intervention_ltm_compress.format(max_words=ltm_words)
            ltm_user = (
                f"Existing long-term memory:\n{prior_ltm}\n\n"
                f"Short-term peer observations:\n{stm}\n\n"
                f"Refutation:\n{refutation_text}\n\n"
            )
        else:
            user_cog = prompts.daily_cognitive.format(**cog_kwargs)
            user_ltm = prompts.daily_ltm_compress.format(max_words=ltm_words)
            ltm_user = (
                f"Existing long-term memory:\n{prior_ltm}\n\n"
                f"Short-term peer observations:\n{stm}\n\n"
            )

        tweet, belief, reasoning = self._llm_cognitive_json(
            user_cog, temperature=llm.temperature_daily_cognitive
        )
        ltm_user += f"New internal reasoning:\n{reasoning}\n"
        ltm = truncate_words(
            self._llm_text(user_ltm + ltm_user, temperature=llm.temperature_ltm),
            ltm_words,
        )

        return {
            "day": day,
            "tweet": tweet,
            "belief": belief,
            "reasoning": reasoning,
            "long_mem": ltm,
            "stm": stm,
            "contact_ids": contact_ids,
        }

    def apply_daily_update(self, payload: dict) -> None:
        day = payload["day"]
        tweet = payload["tweet"]
        belief = payload["belief"]
        reasoning = payload["reasoning"]
        ltm = payload["long_mem"]

        self.opinion = tweet
        self.belief = belief
        self.reasoning = reasoning
        self.opinions.append(tweet)
        self.beliefs.append(belief)
        self.reasonings.append(reasoning)
        self.long_term_memory = ltm

        self.apply_sir_from_belief(belief)
        self._append_history(day, belief, tweet, reasoning, ltm)

        self.short_term_memory = ""
        self.contact_ids = payload.get("contact_ids", [])
        self.agent_interaction.clear()

    def to_export_dict(self):
        return {
            "agent_id": self.agent_config_id,
            "persona_prompt": self.persona_prompt,
            "ideology_level": self.ideology_level,
            "ideology_group": self.ideology_group,
            "current_belief_score": self.belief,
            "sir_state": self.health_condition,
            "has_been_infected": self.has_been_infected,
            "current_tweet": self.opinion,
            "long_term_memory": self.long_term_memory,
            "final_sir_state": self.health_condition,
            "history_log": self.history_log,
        }

    def to_checkpoint_dict(self):
        d = self.to_export_dict()
        d["unique_id"] = self.unique_id
        d["name"] = self.name
        d["age"] = self.age
        d["qualification"] = self.qualification
        d["occupation"] = self.occupation
        d["archetype_id"] = self.archetype_id
        d["political_ideology"] = self.political_ideology
        d["opinions"] = self.opinions
        d["beliefs"] = self.beliefs
        d["reasonings"] = self.reasonings
        return d

    @classmethod
    def from_checkpoint(cls, model, unique_id: int, data: dict):
        agent = cls(
            model=model,
            unique_id=unique_id,
            name=data.get("name", data["agent_id"]),
            age=data["age"],
            qualification=data["qualification"],
            occupation=data["occupation"],
            archetype_id=data["archetype_id"],
            agent_config_id=data["agent_id"],
            system_prompt=data["persona_prompt"],
            health_condition=data["sir_state"],
            topic=model.topic,
            political_ideology=data["ideology_level"],
        )
        agent.has_been_infected = bool(data.get("has_been_infected", False))
        agent.history_log = list(data.get("history_log", []))
        agent.opinions = list(data.get("opinions", []))
        agent.beliefs = list(data.get("beliefs", []))
        agent.reasonings = list(data.get("reasonings", []))
        agent.long_term_memory = data.get("long_term_memory", "")
        if agent.opinions:
            agent.opinion = agent.opinions[-1]
        if agent.beliefs:
            agent.belief = agent.beliefs[-1]
        if agent.reasonings:
            agent.reasoning = agent.reasonings[-1]
        return agent
