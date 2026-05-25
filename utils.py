import random
import re
import time
import os
import shutil
import json
import importlib

AGENTS_CONFIG_PATH = "agents_config.json"

DEMOGRAPHIC_ARCHETYPES = (
    {
        "archetype_id": "A1",
        "age": 25,
        "education": "Bachelor's degree",
        "occupation": "Junior corporate analyst",
    },
    {
        "archetype_id": "A2",
        "age": 36,
        "education": "Vocational training",
        "occupation": "Blue collar tradesperson",
    },
    {
        "archetype_id": "A3",
        "age": 48,
        "education": "Master's degree",
        "occupation": "Senior corporate executive",
    },
    {
        "archetype_id": "A4",
        "age": 55,
        "education": "High school diploma",
        "occupation": "Service industry worker",
    },
    {
        "archetype_id": "A5",
        "age": 68,
        "education": "PhD",
        "occupation": "Professor in university",
    },
)

IDEOLOGY_PROFILES = (
    {
        "ideology": "Very Liberal",
        "background": "You deeply value systemic equality, social justice, and the dismantling of structural oppression. You prioritize marginalized voices and radical inclusivity. For information, you rely on progressive independent journalism, grassroots activist networks, and academic publications, maintaining a strong skepticism toward corporate-owned mainstream media which you believe often protects the status quo.",
    },
    {
        "ideology": "Liberal",
        "background": "You deeply value equal opportunity, robust social safety nets, diversity, and environmental protection. You believe in the positive role of government regulation for the public good. You primarily consume established mainstream progressive journalism, high-quality investigative reporting, and fact-checked national dailies, generally trusting institutional expertise and scientific consensus.",
    },
    {
        "ideology": "Slightly Liberal",
        "background": "You value social tolerance and pragmatic reform, seeking to balance economic growth with social responsibility. You favor evidence-based policies over ideological extremes. You typically stay informed through mainstream centrist-left corporate media, local news outlets, and bipartisan analysis platforms, preferring balanced reporting without highly polarized commentary.",
    },
    {
        "ideology": "Moderate",
        "background": "You value pragmatism, compromise, and institutional stability, focusing on practical solutions and daily economic realities rather than ideological purity. You find extreme political polarization exhausting. Your media diet consists mostly of non-partisan wire services, financial news, and local community newspapers, actively avoiding heavily slanted political commentary.",
    },
    {
        "ideology": "Slightly Conservative",
        "background": "You value free markets, individual responsibility, traditional family structures, and fiscal discipline. You prefer gradual societal change over rapid disruption. You get your news mainly from the mainstream business press, centrist-right editorial boards, and local news broadcasts, with a particular focus on economic stability and community safety.",
    },
    {
        "ideology": "Conservative",
        "background": "You deeply value law and order, constitutional originalism, limited government, and strong border security. You believe in preserving traditional American foundational principles. You heavily consume mainstream conservative broadcast networks, established right-leaning digital publications, and community-based conservative radio, tending to view liberal mainstream media with considerable suspicion.",
    },
    {
        "ideology": "Very Conservative",
        "background": "You deeply value national sovereignty, strict law enforcement, and the protection of core traditional foundations. You are highly vigilant against bureaucratic overreach and globalist policies. You exclusively trust independent conservative alternative media, populist digital forums, and anti-establishment podcasts. You are profoundly distrustful of mainstream corporate journalism and institutional fact-checkers, viewing them as inherently biased and manipulative.",
    },
)

AGENT_SYSTEM_PROMPT_TEMPLATE = """You are a participant in an academic sociological simulation. You must completely immerse yourself in the assigned persona.
STRICT RULES:

Act explicitly as the human persona described. Speak in the first person ("I").

NEVER break character. NEVER reference that you are an AI, a language model, or that you are "simulating" a perspective.

Do not provide meta-commentary, objective fact-checking, or AI safety disclaimers. Your output must ONLY contain the subjective thoughts that strictly reflect your personal worldview and perspective of your persona.

--- YOUR PERSONA ---
You are a {age}-year-old {education} graduate currently working as a {occupation}. Politically, you identify as {ideology}. {background}"""


def _agent_id(ideology: str, archetype_id: str) -> str:
    return f"{ideology.replace(' ', '_')}_{archetype_id}"


def build_agents_config():
    """Build the deterministic 5 demographic archetypes × 7 ideology profiles agent config list."""
    agents = []
    for ideology_profile in IDEOLOGY_PROFILES:
        for archetype in DEMOGRAPHIC_ARCHETYPES:
            system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
                age=archetype["age"],
                education=archetype["education"],
                occupation=archetype["occupation"],
                ideology=ideology_profile["ideology"],
                background=ideology_profile["background"],
            )
            agents.append(
                {
                    "agent_id": _agent_id(
                        ideology_profile["ideology"],
                        archetype["archetype_id"],
                    ),
                    "ideology": ideology_profile["ideology"],
                    "archetype_id": archetype["archetype_id"],
                    "age": archetype["age"],
                    "education": archetype["education"],
                    "occupation": archetype["occupation"],
                    "system_prompt": system_prompt,
                }
            )
    return agents


def ideology_level_from_name(ideology: str) -> int:
    """Return the 1..7 ideology level for a config ideology label."""
    for idx, profile in enumerate(IDEOLOGY_PROFILES, start=1):
        if profile["ideology"] == ideology:
            return idx
    raise ValueError(f"Unknown ideology label: {ideology}")


def ideology_group_from_level(level: int) -> str:
    """Map 1–7 ideology level to Liberal / Moderate / Conservative."""
    level = int(level)
    if level <= 3:
        return "Liberal"
    if level == 4:
        return "Moderate"
    return "Conservative"


def truncate_words(text: str, max_words: int = 300) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def export_agents_config(output_path: str = AGENTS_CONFIG_PATH):
    """Export the 35-agent configuration JSON and return the generated list."""
    agents = build_agents_config()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)
    return agents

# Persona: 5 education levels × 5 age brackets, orthogonal shuffle within each ideology stratum.
PERSONA_EDUCATION_LEVELS = (
    "High School",
    "Associate's",
    "Bachelor's",
    "Master's",
    "PhD",
)

# (label, min_age, max_age) — one slot per stratum after shuffling brackets.
PERSONA_AGE_BRACKETS = (
    ("Young adult (18–25)", 18, 25),
    ("Young midlife (26–35)", 26, 35),
    ("Midlife (36–45)", 36, 45),
    ("Older midlife (46–55)", 46, 55),
    ("Senior (56–65)", 56, 65),
)

DESIGN_POPULATION_N = 35
DESIGN_AGENTS_PER_IDEOLOGY_LEVEL = 5
EXPERIMENT_SIMULATION_DAYS = 15
# Each simulated day, every agent independently draws this many distinct random peers (reshuffled daily).
# Interaction rules: see World.decide_agent_interactions (单向独立 / 自我剔除 / 日内不放回 / 跨日有放回).
EXPERIMENT_RANDOM_CONTACTS_PER_DAY = 3


def allocate_equal_ideology_levels(n: int):
    """
    Return a shuffled list of length n with political ideology levels 1..7,
    as evenly distributed as possible (for IBG / moderation analyses).
    """
    if n <= 0:
        return []
    base = n // 7
    rem = n % 7
    levels = []
    for k in range(1, 8):
        count = base + (1 if k <= rem else 0)
        levels.extend([k] * count)
    random.shuffle(levels)
    return levels


def build_personas_stratified_orthogonal(rng=None):
    """
    Stratified orthogonal assignment (35 agents, 5 per ideology 1..7).

    Within each ideology stratum: all 5 education levels and all 5 age brackets appear
    exactly once. Education order and age-bracket order are shuffled independently,
    then paired by index so age and education are marginally balanced but not collinear
    (e.g. PhD is not tied to a fixed age band).
    """
    rnd = rng or random
    personas = []
    edu_template = list(PERSONA_EDUCATION_LEVELS)
    bracket_template = list(PERSONA_AGE_BRACKETS)
    if len(edu_template) != DESIGN_AGENTS_PER_IDEOLOGY_LEVEL:
        raise ValueError("PERSONA_EDUCATION_LEVELS must have length 5.")
    if len(bracket_template) != DESIGN_AGENTS_PER_IDEOLOGY_LEVEL:
        raise ValueError("PERSONA_AGE_BRACKETS must have length 5.")

    for ideology in range(1, 8):
        edu_order = edu_template[:]
        rnd.shuffle(edu_order)
        bracket_order = bracket_template[:]
        rnd.shuffle(bracket_order)
        for j in range(DESIGN_AGENTS_PER_IDEOLOGY_LEVEL):
            _label, lo, hi = bracket_order[j]
            personas.append(
                {
                    "political_ideology": ideology,
                    "age": rnd.randint(lo, hi),
                    "qualification": edu_order[j],
                }
            )
    rnd.shuffle(personas)
    return personas


def probability_threshold(threshold):
    '''
    Used in self.infect_interaction()
    '''
    import numpy as np

    #Generates random number from 0 to 1
    
    return (np.random.rand()<threshold)

def generate_qualifications(n: int):
    '''
    Returns a list of random educational qualifications.

    Parameters:
    n (int): The number of qualifications to generate.
    '''

    generated_qualifications = random.choices(list(PERSONA_EDUCATION_LEVELS), k=n)

    return generated_qualifications


def generate_names(n: int, s: int, country_alpha2='US'):
    '''
    Returns random names as names for agents from top names in the USA
    Used in World.init to initialize agents
    '''
    try:
        NameDataset = importlib.import_module("names_dataset").NameDataset
    except ModuleNotFoundError:
        fallback_names = [f"Agent_{i + 1}" for i in range(n)]
        random.shuffle(fallback_names)
        return fallback_names

    # This function will randomly selct n names (n/2 male and n/2 female) without
    # replacement from the s most popular names in the country defined by country_alpha2
    if n % 2 == 1:
        n += 1
    if s % 2 == 1:
        s += 1

    nd = NameDataset()
    male_names = nd.get_top_names(s//2, 'Male', country_alpha2)[country_alpha2]['M']
    female_names = nd.get_top_names(s//2, 'Female', country_alpha2)[country_alpha2]['F']
    if s < n:
        raise ValueError(f"Cannot generate {n} unique names from a list of {s} names.")
    # generate names without repetition
    names = random.sample(male_names, k=n//2) + random.sample(female_names, k=n//2)
    del male_names
    del female_names
    random.shuffle(names)
    return names


def update_day(agent):
    '''
    Update day funtion to update day_sick
    Used in World.step()
    '''
    # print("Agent ID: {} Day infected: {}".format(agent.unique_id,agent.day_infected))


    #if person is healthy, no reason to update health status
    if agent.health_condition=="Susceptible" or agent.health_condition=="Infected":
        return

    if agent.health_condition=="to_be_infected":
        agent.health_condition="Infected"
        agent.model.daily_new_infected_cases +=1 #every time new infection occurs in a day, counter is updated
        agent.model.infected += 1 #Update amount infected at any given time
        agent.model.susceptible -= 1

    if agent.health_condition=="to_be_recover":
        agent.health_condition="Recovered"
        agent.model.daily_new_susceptible_cases +=1 #every time new infection occurs in a day, counter is updated
        agent.model.infected -= 1 #Update amount infected at any given time
        agent.model.susceptible += 1


def factorize(n):
    '''
    Factorize number for ideal grid dimensions for # of agents
    Used in World.init
    '''
    for i in range(int(n**0.5), 1, -1):
        if n % i == 0:
            return (i, n // i)
    return (n, 1)

# LLM settings (override via configure_llm() or env: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL)
_LLM = {
    "base_url": None,
    "api_key": None,
    "model": "gpt-3.5-turbo",
    "use_json_mode": True,
    "max_retries": 30,
    "max_workers": 8,
    "max_tokens": 2048,
}


def configure_llm(
    base_url=None,
    api_key=None,
    model=None,
    use_json_mode=None,
    max_retries=None,
    max_workers=None,
    max_tokens=None,
):
    """Apply CLI/env LLM settings before simulation starts."""
    if base_url is not None:
        _LLM["base_url"] = base_url or None
    if api_key is not None:
        _LLM["api_key"] = api_key
    if model is not None:
        _LLM["model"] = model
    if use_json_mode is not None:
        _LLM["use_json_mode"] = use_json_mode
    if max_retries is not None:
        _LLM["max_retries"] = max_retries
    if max_workers is not None:
        _LLM["max_workers"] = max(1, int(max_workers))
    elif os.environ.get("LLM_MAX_WORKERS"):
        _LLM["max_workers"] = max(1, int(os.environ["LLM_MAX_WORKERS"]))
    if max_tokens is not None:
        _LLM["max_tokens"] = max(64, int(max_tokens))
    elif os.environ.get("LLM_MAX_TOKENS"):
        _LLM["max_tokens"] = max(64, int(os.environ["LLM_MAX_TOKENS"]))

    if _LLM["base_url"] is None:
        _LLM["base_url"] = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if _LLM["api_key"] is None:
        _LLM["api_key"] = os.environ.get("OPENAI_API_KEY", "ollama")
    if _LLM["model"] is None:
        _LLM["model"] = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")


def get_llm_config():
    return dict(_LLM)


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from model output: {text[:500]}...")


def _chat_create(messages, model, temperature, json_mode=False):
    """Call OpenAI-compatible chat API (v1 client, with legacy fallback)."""
    import openai

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": _LLM.get("max_tokens", 2048),
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # Prefer OpenAI Python SDK v1+ (Ollama / vLLM / LM Studio / cloud)
    try:
        from openai import OpenAI

        client_kwargs = {"api_key": _LLM["api_key"] or "ollama"}
        if _LLM["base_url"]:
            client_kwargs["base_url"] = _LLM["base_url"]
        client = OpenAI(**client_kwargs)
        return client.chat.completions.create(**kwargs)
    except Exception:
        pass

    # Legacy openai.ChatCompletion (some pinned installs)
    if _LLM["base_url"]:
        openai.api_base = _LLM["base_url"]
    openai.api_key = _LLM["api_key"] or "ollama"
    return openai.ChatCompletion.create(**kwargs)


def _message_content(response) -> str:
    choice = response.choices[0]
    msg = choice.message
    if isinstance(msg, dict):
        return msg["content"]
    return msg.content


def _chat_with_retries(messages, model, temperature, json_mode=False) -> str:
    last_err = None
    for attempt in range(_LLM["max_retries"]):
        try:
            response = _chat_create(messages, model, temperature, json_mode=json_mode)
            return _message_content(response)
        except Exception as e:
            last_err = e
            if json_mode:
                try:
                    response = _chat_create(messages, model, temperature, json_mode=False)
                    return _message_content(response)
                except Exception as e2:
                    last_err = e2
            print(f"LLM error (attempt {attempt + 1}/{_LLM['max_retries']}): {last_err}")
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    raise RuntimeError(f"LLM request failed after {_LLM['max_retries']} retries: {last_err}")


def get_completion_from_messages(messages, model=None, temperature=0):
    model = model or _LLM["model"]
    return _chat_with_retries(messages, model, temperature, json_mode=False)


def get_completion_from_messages_json(messages, model=None, temperature=0):
    model = model or _LLM["model"]
    use_json = _LLM["use_json_mode"]
    content = _chat_with_retries(messages, model, temperature, json_mode=use_json)
    parsed = _extract_json_object(content)
    return json.dumps(parsed, ensure_ascii=False)



def clear_cache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")