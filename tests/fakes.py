"""
Hand-rolled stand-ins for the Anthropic and Gemini SDK response shapes that
llm_providers.py reads from — resp.content[i].type/.input for Anthropic,
resp.candidates[i].content.parts[i].function_call for Gemini. These exist so
every test in this package can build a Fake*Client and hand it to the code
under test exactly as bot.py / llm_trim_evaluator.py would hand it a real
one, without ever constructing a real anthropic.Anthropic() / genai.Client()
or touching the network — see tests_llm/README.md's Isolation section.

Each fake's create()/generate_content() records the kwargs it was called
with (readable via `.last_call` on the client) so tests can assert on what
llm_providers.py actually sent, and can be configured to raise instead of
returning a response, to exercise the fail-safe-to-NOISE / error paths in
llm_classifier.classify().
"""

# ── Anthropic ──


class FakeToolUseBlock:
    """Mirrors an Anthropic tool_use content block."""

    def __init__(self, input_):
        self.type = "tool_use"
        self.input = input_


class FakeTextBlock:
    """Mirrors an Anthropic text content block — used to simulate the model
    ignoring the forced tool_choice and replying with plain text instead."""

    def __init__(self, text=""):
        self.type = "text"
        self.text = text


class FakeAnthropicResponse:
    def __init__(self, content):
        self.content = content


class _FakeAnthropicMessagesBase:
    """Shared call-recording / raise-on-call behavior for the sync and
    async Anthropic messages-endpoint fakes below."""

    def __init__(self, response=None, exception=None):
        self.response = response
        self.exception = exception
        self.last_call = None

    def _record_and_resolve(self, kwargs):
        self.last_call = kwargs
        if self.exception is not None:
            raise self.exception
        return self.response


class FakeAnthropicMessagesAsync(_FakeAnthropicMessagesBase):
    async def create(self, **kwargs):
        return self._record_and_resolve(kwargs)


class FakeAnthropicMessagesSync(_FakeAnthropicMessagesBase):
    def create(self, **kwargs):
        return self._record_and_resolve(kwargs)


class FakeAsyncAnthropicClient:
    """Stands in for anthropic.AsyncAnthropic — the client bot.py builds
    (via llm_providers.make_client) and hands to llm_classifier.classify()."""

    def __init__(self, response=None, exception=None):
        self.messages = FakeAnthropicMessagesAsync(response=response, exception=exception)


class FakeSyncAnthropicClient:
    """Stands in for anthropic.Anthropic — the client
    llm_trim_evaluator.py's batch loop uses."""

    def __init__(self, response=None, exception=None):
        self.messages = FakeAnthropicMessagesSync(response=response, exception=exception)


def anthropic_tool_response(**tool_input):
    """The common case: the model called the forced tool with exactly these
    arguments (e.g. anthropic_tool_response(label="TRIM", ticker="SPY"))."""
    return FakeAnthropicResponse([FakeToolUseBlock(tool_input)])


def anthropic_text_only_response(text="I cannot help with that."):
    """The model replied with plain text instead of the forced tool call —
    llm_providers._extract_anthropic_tool_input (and classify_*) should
    raise on this, and llm_classifier.classify() should turn that into a
    fail-safe NOISE rather than propagating it."""
    return FakeAnthropicResponse([FakeTextBlock(text)])


# ── Gemini ──


class FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakePart:
    def __init__(self, function_call=None, text=None):
        self.function_call = function_call
        self.text = text


class FakeContent:
    def __init__(self, parts):
        self.parts = parts


class FakeCandidate:
    def __init__(self, content):
        self.content = content


class FakeGeminiResponse:
    def __init__(self, candidates):
        self.candidates = candidates


class _FakeGeminiModelsBase:
    """Shared call-recording / raise-on-call behavior for the sync and
    async Gemini models-endpoint fakes below."""

    def __init__(self, response=None, exception=None):
        self.response = response
        self.exception = exception
        self.last_call = None

    def _record_and_resolve(self, kwargs):
        self.last_call = kwargs
        if self.exception is not None:
            raise self.exception
        return self.response


class FakeGeminiModelsAsync(_FakeGeminiModelsBase):
    async def generate_content(self, **kwargs):
        return self._record_and_resolve(kwargs)


class FakeGeminiModelsSync(_FakeGeminiModelsBase):
    def generate_content(self, **kwargs):
        return self._record_and_resolve(kwargs)


class _FakeGeminiAio:
    def __init__(self, models):
        self.models = models


class FakeAsyncGeminiClient:
    """Stands in for genai.Client() used via its .aio.models surface — the
    async path llm_classifier.classify() drives."""

    def __init__(self, response=None, exception=None):
        self._models = FakeGeminiModelsAsync(response=response, exception=exception)
        self.aio = _FakeGeminiAio(self._models)

    @property
    def last_call(self):
        return self._models.last_call


class FakeSyncGeminiClient:
    """Stands in for genai.Client() used via its synchronous .models
    surface — the path llm_trim_evaluator.py's batch loop drives."""

    def __init__(self, response=None, exception=None):
        self.models = FakeGeminiModelsSync(response=response, exception=exception)

    @property
    def last_call(self):
        return self.models.last_call


def gemini_function_call_response(tool_name, **args):
    """The common case: the model called the forced function with exactly
    these arguments (e.g. gemini_function_call_response("route_signal",
    label="ADD", ticker="QQQ"))."""
    return FakeGeminiResponse(
        [FakeCandidate(FakeContent([FakePart(function_call=FakeFunctionCall(tool_name, args))]))]
    )


def gemini_text_only_response(text="I cannot help with that."):
    """The model replied with plain text instead of the forced function
    call — should raise, same as anthropic_text_only_response above."""
    return FakeGeminiResponse([FakeCandidate(FakeContent([FakePart(text=text)]))])


def gemini_empty_response():
    """No candidates at all — e.g. a safety-filtered response."""
    return FakeGeminiResponse([])
