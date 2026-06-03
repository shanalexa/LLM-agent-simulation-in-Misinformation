import random
import re
import time
import os
import json

def _agent_id(ideology: str, archetype_id: str) -> str:
    return f"{ideology.replace(' ', '_')}_{archetype_id}"


def build_agents_config():
    """Build the deterministic 5 demographic archetypes × 7 ideology profiles agent config list."""
    from config import get_config

    cfg = get_config()
    agents = []
    for ideology_profile in cfg.personas.ideology_profiles:
        for archetype in cfg.personas.archetypes:
            system_prompt = cfg.personas.system_prompt_template.format(
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
    from config import get_config

    for idx, profile in enumerate(get_config().personas.ideology_profiles, start=1):
        if profile["ideology"] == ideology:
            return idx
    raise ValueError(f"Unknown ideology label: {ideology}")


def ideology_group_from_level(level: int) -> str:
    """Map ideology level to Liberal / Moderate / Conservative (bounds from config.yaml)."""
    from config import get_config

    e = get_config().experiment
    level = int(level)
    if level <= e.liberal_level_max:
        return "Liberal"
    if level == e.moderate_level:
        return "Moderate"
    return "Conservative"


def truncate_words(text: str, max_words: int | None = None) -> str:
    if max_words is None:
        from config import get_config

        max_words = get_config().experiment.ltm_max_words
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def export_agents_config(output_path: str | None = None):
    """Export the 35-agent configuration JSON and return the generated list."""
    from config import get_config

    if output_path is None:
        output_path = get_config().paths.agents_config
    agents = build_agents_config()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)
    return agents


# Runtime LLM client state (filled by configure_llm from config + env/CLI overrides).
_LLM: dict = {}


def configure_llm(
    base_url=None,
    api_key=None,
    model=None,
    use_json_mode=None,
    max_retries=None,
    max_workers=None,
    max_tokens=None,
):
    """Apply config.yaml defaults, then CLI/env overrides."""
    from config import get_config

    c = get_config().llm

    _LLM.clear()

    if base_url is not None and str(base_url).strip():
        resolved_base = str(base_url).strip()
    elif os.environ.get("VLLM_API_BASE", "").strip():
        resolved_base = os.environ["VLLM_API_BASE"].strip()
    else:
        raise ValueError(
            "VLLM_API_BASE is required: export VLLM_API_BASE or pass --vllm_api_base"
        )
    _LLM["base_url"] = resolved_base

    if api_key is not None:
        resolved_key = str(api_key)
    elif "OPENAI_API_KEY" in os.environ:
        resolved_key = os.environ["OPENAI_API_KEY"]
    else:
        raise ValueError(
            "OPENAI_API_KEY is required: export OPENAI_API_KEY or pass --llm_api_key"
        )
    _LLM["api_key"] = resolved_key
    if model is not None and str(model).strip():
        resolved_model = str(model).strip()
    elif os.environ.get("LLM_MODEL", "").strip():
        resolved_model = os.environ["LLM_MODEL"].strip()
    elif c.model.strip():
        resolved_model = c.model.strip()
    else:
        raise ValueError(
            "LLM model is required: set llm.model in config.yaml, export LLM_MODEL, or pass --llm_model"
        )
    _LLM["model"] = resolved_model
    _LLM["enable_thinking"] = c.enable_thinking
    _LLM["use_json_mode"] = use_json_mode if use_json_mode is not None else c.use_json_mode
    _LLM["max_retries"] = max_retries if max_retries is not None else c.max_retries
    _LLM["max_cognitive_retries"] = c.max_cognitive_retries
    if max_workers is not None:
        _LLM["max_workers"] = max(1, int(max_workers))
    elif os.environ.get("LLM_MAX_WORKERS"):
        _LLM["max_workers"] = max(1, int(os.environ["LLM_MAX_WORKERS"]))
    else:
        _LLM["max_workers"] = c.max_workers
    if max_tokens is not None:
        _LLM["max_tokens"] = max(64, int(max_tokens))
    elif os.environ.get("LLM_MAX_TOKENS"):
        _LLM["max_tokens"] = max(64, int(os.environ["LLM_MAX_TOKENS"]))
    else:
        _LLM["max_tokens"] = c.max_tokens


def get_llm_config():
    return dict(_LLM)


def _normalize_model_text(model_output: str) -> str:
    """Collapse API fields and strip common non-JSON wrappers before parsing."""
    text = (model_output or "").strip()
    if not text:
        return ""
    # Qwen / reasoning models may still emit thinking tags when misconfigured.
    text = re.sub(
        r"<(?:redacted_thinking|think)\b[^>]*>[\s\S]*?</(?:redacted_thinking|think)>",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"<(?:redacted_thinking|think)\b[^>]*>",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return text


def parse_json_safely(model_output: str) -> dict:
    """
    Parse JSON from LLM text that may include markdown fences or leading prose.
    Tries json.loads on the full string first; falls back to fenced or {...} extraction.
    """
    text = _normalize_model_text(model_output)
    if not text:
        raise ValueError("Empty model output (no content from API)")

    def _loads(candidate: str) -> dict | None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    parsed = _loads(text)
    if parsed is not None:
        return parsed

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        parsed = _loads(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        parsed = _loads(match.group(0))
        if parsed is not None:
            return parsed

    preview = text[:800]
    print(f"JSON解析彻底失败！原始输出是: {preview!r}")
    raise ValueError("文本中未找到任何大括号结构")


def validate_cognitive_payload(data: object) -> dict:
    """
    Validate LLM cognitive JSON (belief_score, opinion/tweet, reasoning).
    Raises ValueError so get_completion_from_messages_json can retry the API call.
    """
    if not isinstance(data, dict):
        raise ValueError("Cognitive output must be a JSON object")

    belief_raw = data.get("belief_score", data.get("belief"))
    if belief_raw is None:
        raise ValueError("Missing belief_score or belief")
    try:
        belief = int(round(float(belief_raw)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid belief_score: {belief_raw!r}") from exc

    from config import get_config

    exp = get_config().experiment
    belief = max(exp.belief_min, min(exp.belief_max, belief))

    opinion = data.get("opinion")
    tweet = data.get("tweet")
    public_text = ""
    if isinstance(opinion, str) and opinion.strip():
        public_text = opinion.strip()
    elif isinstance(tweet, str) and tweet.strip():
        public_text = tweet.strip()
    elif opinion is not None and not isinstance(opinion, str):
        public_text = str(opinion).strip()
    elif tweet is not None and not isinstance(tweet, str):
        public_text = str(tweet).strip()
    if not public_text:
        raise ValueError("Missing non-empty opinion or tweet")

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning) if reasoning is not None else ""
    reasoning = reasoning.strip()
    if not reasoning:
        raise ValueError("Missing reasoning")

    return {
        "reasoning": reasoning,
        "belief_score": belief,
        "opinion": public_text,
    }


def _qwen_thinking_extra_body(model: str) -> dict | None:
    """Qwen3 reasoning kwargs only; Qwen2.5-Instruct does not use enable_thinking."""
    name = (model or "").lower()
    if "qwen3" not in name:
        return None
    enable_thinking = bool(_LLM.get("enable_thinking", False))
    return {
        "enable_thinking": enable_thinking,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


def _chat_create(messages, model, temperature, json_mode=False):
    """Call OpenAI-compatible chat API (v1 client, with legacy fallback)."""
    import openai

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": _LLM.get("max_tokens") or 2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    extra_body = _qwen_thinking_extra_body(model)

    # Prefer OpenAI Python SDK v1+ (vLLM / other OpenAI-compatible servers)
    try:
        from openai import OpenAI

        client_kwargs = {"api_key": _LLM["api_key"], "base_url": _LLM["base_url"]}
        client = OpenAI(**client_kwargs)
        if extra_body:
            return client.chat.completions.create(**kwargs, extra_body=extra_body)
        return client.chat.completions.create(**kwargs)
    except Exception:
        pass

    # Legacy openai.ChatCompletion (some pinned installs)
    if _LLM["base_url"]:
        openai.api_base = _LLM["base_url"]
    openai.api_key = _LLM["api_key"]
    if extra_body:
        kwargs.update(extra_body)
    return openai.ChatCompletion.create(**kwargs)


def _message_content(response) -> str:
    """Merge content + reasoning fields (Qwen/vLLM may put text in reasoning_content)."""
    choice = response.choices[0]
    msg = choice.message

    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    parts: list[str] = []
    for key in ("content", "reasoning_content", "reasoning"):
        val = _get(msg, key)
        if not val:
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))

    if not parts:
        return ""
    # Prefer the segment that actually contains JSON braces.
    for part in reversed(parts):
        if "{" in part:
            return part.strip()
    return "\n".join(parts).strip()


def _chat_with_retries(messages, model, temperature, json_mode=False) -> str:
    last_err = None
    max_retries = _LLM.get("max_retries") or 30
    for attempt in range(max_retries):
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
            from config import get_config

            llm = get_config().llm
            print(f"LLM error (attempt {attempt + 1}/{max_retries}): {last_err}")
            time.sleep(
                min(
                    llm.retry_sleep_cap,
                    llm.retry_sleep_base * (attempt + 1),
                )
            )
    raise RuntimeError(f"LLM request failed after {max_retries} retries: {last_err}")


def _require_llm_model(explicit=None) -> str:
    if not _LLM:
        raise RuntimeError("LLM not configured. Call configure_llm() after init_config().")
    resolved = (explicit or _LLM.get("model") or "").strip()
    if not resolved:
        raise RuntimeError(
            "LLM model is not set. Configure via llm.model, LLM_MODEL, or --llm_model before calling the API."
        )
    return resolved


def get_completion_from_messages(messages, model=None, temperature=0):
    return _chat_with_retries(
        messages, _require_llm_model(model), temperature, json_mode=False
    )


def _fetch_chat_content_once(messages, model, temperature, json_mode=False) -> str:
    """One API round-trip; retry json_mode=False once if the server rejects JSON mode."""
    try:
        response = _chat_create(messages, model, temperature, json_mode=json_mode)
        return _message_content(response)
    except Exception:
        if json_mode:
            response = _chat_create(messages, model, temperature, json_mode=False)
            return _message_content(response)
        raise


def get_completion_from_messages_json(messages, model=None, temperature=0):
    """
    Cognitive JSON: single retry loop (API + parse + validate share one budget).
    """
    from config import get_config

    model = _require_llm_model(model)
    use_json = _LLM["use_json_mode"]
    llm = get_config().llm
    max_attempts = _LLM.get("max_cognitive_retries") or llm.max_cognitive_retries or 8
    last_err = None
    last_content = ""

    for attempt in range(max_attempts):
        try:
            content = _fetch_chat_content_once(
                messages, model, temperature, json_mode=use_json
            )
        except Exception as exc:
            last_err = exc
            print(
                f"LLM API error (cognitive attempt {attempt + 1}/{max_attempts}): {exc}"
            )
            time.sleep(
                min(llm.retry_sleep_cap, llm.retry_sleep_base * (attempt + 1))
            )
            continue

        last_content = content or ""
        try:
            parsed = parse_json_safely(content)
            validated = validate_cognitive_payload(parsed)
            return json.dumps(validated, ensure_ascii=False)
        except ValueError as exc:
            last_err = exc
            preview = (last_content or "(empty)")[:500]
            print(
                f"JSON/cognitive validation error (attempt {attempt + 1}/{max_attempts}): {exc}; "
                f"output preview: {preview!r}"
            )
            time.sleep(
                min(llm.retry_sleep_cap, llm.retry_sleep_base * (attempt + 1))
            )

    raise RuntimeError(
        f"Could not obtain valid cognitive JSON after {max_attempts} attempts: {last_err}; "
        f"last output: {(last_content or '(empty)')[:500]!r}"
    )