"""
Provider abstraction for the two places an LLM is called: llm_classifier.py
(live TRIM/EXIT/ADD/NOISE routing, async) and llm_trim_evaluator.py (offline
batch eval, sync). Both share one system prompt + one forced-tool-call
schema per call site; this module is the only place that knows how to turn
that (system_prompt, tool, text) triple into a provider-specific request and
back into a plain dict of the tool's arguments. Neither caller needs to
branch on provider — they get a dict back or an exception, same as if there
were only ever one vendor.

`tool` is always an Anthropic-shaped tool definition — {"name", "description",
"input_schema"} with input_schema as JSON Schema. That's the schema both
call sites already write today (see ROUTE_TOOL / RECORD_LABELS_TOOL). For
Gemini, that same JSON Schema dict is passed straight through via
FunctionDeclaration.parameters_json_schema — no separate schema to maintain
per provider.

Adding a third provider: add it to SUPPORTED_PROVIDERS, teach make_client
about its client type, and add a branch in classify_async/classify_sync.
Nothing in llm_classifier.py or llm_trim_evaluator.py needs to change.
"""

import anthropic

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # google-genai not installed — fine unless provider="gemini" is actually used
    genai = None
    genai_types = None

SUPPORTED_PROVIDERS = ("anthropic", "gemini")


def _check_provider(provider):
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unknown llm provider {provider!r} — expected one of {SUPPORTED_PROVIDERS}")


def _require_gemini_sdk():
    if genai is None:
        raise RuntimeError(
            "provider is 'gemini' but the google-genai package isn't installed — "
            "run `pip install google-genai` (see requirements.txt)"
        )


def make_client(provider, api_key, *, is_async):
    """is_async=True for the live bot (bot.py awaits llm_classifier.classify());
    is_async=False for llm_trim_evaluator.py's synchronous batch loop.
    Gemini's genai.Client exposes both client.models (sync) and
    client.aio.models (async) off the same object, so is_async only matters
    for Anthropic, which has separate client classes."""
    _check_provider(provider)
    if provider == "anthropic":
        return anthropic.AsyncAnthropic(api_key=api_key) if is_async else anthropic.Anthropic(api_key=api_key)
    _require_gemini_sdk()
    return genai.Client(api_key=api_key)


def _extract_anthropic_tool_input(resp, tool_name):
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError(f"no tool_use block in response: {resp.content!r}")


def _extract_gemini_function_args(resp, tool_name):
    for candidate in resp.candidates or []:
        parts = candidate.content.parts if candidate.content else None
        for part in parts or []:
            fc = getattr(part, "function_call", None)
            if fc is not None and fc.name == tool_name:
                return dict(fc.args or {})
    raise RuntimeError(f"no function_call part for {tool_name!r} in response: {resp!r}")


def _gemini_config(system_prompt, tool, *, timeout_secs=None):
    function_declaration = genai_types.FunctionDeclaration(
        name=tool["name"],
        description=tool["description"],
        parameters_json_schema=tool["input_schema"],
    )
    kwargs = dict(
        system_instruction=system_prompt,
        tools=[genai_types.Tool(function_declarations=[function_declaration])],
        # mode="ANY" + allowed_function_names is Gemini's equivalent of
        # Anthropic's tool_choice={"type": "tool", "name": ...} — forces
        # this exact function call rather than letting the model reply
        # with plain text or pick a different tool.
        tool_config=genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=[tool["name"]],
            )
        ),
    )
    if timeout_secs is not None:
        kwargs["http_options"] = genai_types.HttpOptions(timeout=int(timeout_secs * 1000))
    return genai_types.GenerateContentConfig(**kwargs)


async def classify_async(provider, client, model, system_prompt, tool, text, *, max_tokens=256, timeout_secs=None):
    """Used by llm_classifier.classify() (live path). Returns the forced
    tool call's arguments as a plain dict. Raises on any failure — the
    caller is responsible for the fail-safe-to-NOISE catch, same as before
    this abstraction existed."""
    _check_provider(provider)
    if provider == "anthropic":
        kwargs = {"timeout": timeout_secs} if timeout_secs is not None else {}
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": text}],
            **kwargs,
        )
        return _extract_anthropic_tool_input(resp, tool["name"])

    _require_gemini_sdk()
    resp = await client.aio.models.generate_content(
        model=model,
        contents=text,
        config=_gemini_config(system_prompt, tool, timeout_secs=timeout_secs),
    )
    return _extract_gemini_function_args(resp, tool["name"])


def classify_sync(provider, client, model, system_prompt, tool, text, *, max_tokens=4096):
    """Used by llm_trim_evaluator.py's offline batch loop — same contract as
    classify_async, just synchronous (the evaluator doesn't run inside an
    event loop and batches calls, so there's no latency reason to await)."""
    _check_provider(provider)
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": text}],
        )
        return _extract_anthropic_tool_input(resp, tool["name"])

    _require_gemini_sdk()
    resp = client.models.generate_content(
        model=model,
        contents=text,
        config=_gemini_config(system_prompt, tool),
    )
    return _extract_gemini_function_args(resp, tool["name"])
