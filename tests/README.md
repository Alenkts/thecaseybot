# tests — LLM integration test package

Tests for the project's LLM integration surface: `llm_providers.py` (the
Anthropic/Gemini provider abstraction), `llm_classifier.py` (live
TRIM/EXIT/ADD/NOISE routing), and `llm_trim_evaluator.py`'s LLM-facing
pieces (batch classification, provider/model config resolution).

## Isolation

This package is deliberately kept separate from the rest of the project and
never touches anything live — with exactly one opt-in exception, see
`test_live_smoke.py` below.

- **No network calls, ever, by default.** Every Anthropic/Gemini response is a
  hand-built stand-in from `fakes.py` (`FakeAsyncAnthropicClient`,
  `FakeSyncGeminiClient`, etc.) that mimics just enough of each SDK's
  response shape (`resp.content[i].type/.input` for Anthropic,
  `resp.candidates[i].content.parts[i].function_call` for Gemini) for
  `llm_providers.py` to parse. No test constructs a real
  `anthropic.Anthropic()` / `genai.Client()` against the real API.
- **No real API keys required.** Fakes are handed placeholder strings like
  `"fake-key"` — a real `llm.api_key` is never read from `config.yaml`.
- **No dependency on `config.yaml`, `db.py`, Discord, or IBKR.** Tests that
  need a config file write a throwaway one to pytest's `tmp_path`. The
  modules under test here don't import `discord_listener.py`,
  `trade_executor.py`, `web/`, or `db.py` at all — this package doesn't
  either.
- **Own test-only dependency file** (`requirements-test.txt`, just
  `pytest`) — kept out of the project's `requirements.txt` so running the
  bot never needs a test runner installed, and running tests never needs
  the full bot stack (`ib_async`, `discord.py-self`, `flask`, ...).
- **`signal_classifier.classify()` (the regex path) runs for real** inside
  `test_llm_trim_evaluator.py` — it's pure, deterministic, and has zero
  external dependencies, so mocking it would only make those tests less
  meaningful without buying back any real isolation.

## Running

From the repo root, with the project's venv active:

```bash
pip install -r tests/requirements-test.txt
pytest tests/
```

No `config.yaml`, running IBKR Gateway, or live Discord/API credentials
required.

## Layout

| File | Covers |
|---|---|
| `fakes.py` | Fake Anthropic/Gemini SDK response objects and clients — shared by every test module here |
| `conftest.py` | Makes sure the project's root-level modules import correctly regardless of invocation directory |
| `test_llm_providers.py` | `llm_providers.py` itself: `make_client`, response extraction, Gemini's forced-function-call config, `classify_async`/`classify_sync` dispatch |
| `test_llm_classifier.py` | `llm_classifier.classify()`: label→`Signal` mapping, ticker normalization, fail-safe-to-NOISE on provider errors/malformed responses, both providers |
| `test_llm_trim_evaluator.py` | `load_llm_config`'s provider/model defaulting, `classify_batch`, and `run()`'s provider/model resolution (including the "override provider without overriding model" edge case) |
| `test_live_smoke.py` | **The one exception to this package's isolation rule.** One test, one real question ("What is 1 + 1?"), through the exact `llm_providers.classify_sync` tool-calling path the bot uses — proves your real `config.yaml` credentials and model name actually work. Skipped by default; opt in with `RUN_LIVE_LLM_TEST=1 pytest tests/test_live_smoke.py -v -s`. Makes a real, billed API call against whatever `llm.provider`/`llm.model` your `config.yaml` names. |

## Adding a third provider

If `llm_providers.py` grows a third provider, add matching fakes to
`fakes.py` (a `Fake*Client` pair, sync and async, plus a response builder)
and parametrize the existing `@pytest.mark.parametrize("provider", ...)`
tests over it — the test structure doesn't otherwise change.
