"""
Groq LLM client with retry-with-backoff and safe-default fallback.

The rest of the agent code calls invoke() and gets back either a real
LLM response or a structured safe-default. It never sees network errors,
rate limits, or malformed responses.

This is the reliability boundary between the deterministic simulation
loop and the non-deterministic LLM. If Groq goes down for a minute, the
simulation continues on last-known-good setpoints instead of crashing.
"""

import time
import json
from typing import Optional, Any
from dataclasses import dataclass, field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool

import config


@dataclass
class LLMResponse:
    """Structured result from an LLM invocation."""
    text: str = ""
    tool_calls: list = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None
    used_fallback: bool = False


class LLMClient:
    """
    Groq-backed LLM with hardening around it.

    Features:
      - Retry with exponential backoff on transient failures
      - Optional tool binding (LangChain-compatible tool schemas)
      - Timeout enforcement
      - Structured LLMResponse (never raises to caller)
    """

    def __init__(
        self,
        model: str = config.GROQ_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
        max_retries: int = config.LLM_MAX_RETRIES,
        timeout: int = config.LLM_TIMEOUT_SECONDS,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=0,  # we do our own retries
            api_key=api_key or config.GROQ_API_KEY,
        )
        self._tools_bound = None

    def bind_tools(self, tools: list):
        """
        Bind LangChain-style tools to the model.
        Returns self for chaining.
        """
        self._tools_bound = self._client.bind_tools(tools)
        return self

    def invoke(self, messages: list) -> LLMResponse:
        """
        Call the LLM with retry-backoff. Never raises.

        messages: list of LangChain BaseMessage objects
        Returns: LLMResponse with text, tool_calls, latency, and (on failure)
                 error string + used_fallback=True.
        """
        client = self._tools_bound if self._tools_bound is not None else self._client

        last_error = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                result: AIMessage = client.invoke(messages)
                elapsed = (time.perf_counter() - t0) * 1000.0

                # Extract token counts (Groq returns them in usage_metadata)
                tokens_in = 0
                tokens_out = 0
                if hasattr(result, "usage_metadata") and result.usage_metadata:
                    tokens_in = result.usage_metadata.get("input_tokens", 0)
                    tokens_out = result.usage_metadata.get("output_tokens", 0)

                return LLMResponse(
                    text=result.content or "",
                    tool_calls=getattr(result, "tool_calls", []) or [],
                    latency_ms=elapsed,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)  # 0.5, 1.0, 2.0 seconds
                    time.sleep(backoff)
                    continue

        # All retries exhausted -> return safe fallback
        elapsed = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(
            text="",
            tool_calls=[],
            latency_ms=elapsed,
            error=last_error,
            used_fallback=True,
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Sanity test: send a trivial message to Groq and print the response.
    Also tests tool binding with a dummy tool.
    """
    from langchain_core.tools import tool

    print(f"[TEST] Model: {config.GROQ_MODEL}")
    print(f"[TEST] Temperature: {config.LLM_TEMPERATURE}")
    print()

    # Test 1: plain invoke
    print("[TEST 1] Plain text response")
    llm = LLMClient()
    resp = llm.invoke([
        SystemMessage(content="You are a terse building control assistant."),
        HumanMessage(content="If a zone is empty and outdoor is 22°C, "
                             "should the cooling setpoint go up or down? "
                             "Answer in one word."),
    ])
    print(f"  Response: {resp.text!r}")
    print(f"  Latency:  {resp.latency_ms:.0f} ms")
    print(f"  Tokens:   in={resp.tokens_in}, out={resp.tokens_out}")
    print(f"  Error:    {resp.error}")
    print(f"  Fallback: {resp.used_fallback}")
    print()

    # Test 2: tool-bound invoke
    print("[TEST 2] Tool-calling response")

    @tool
    def set_cooling_setpoint(temperature_c: float, reasoning: str) -> str:
        """Set the building cooling setpoint to a specific temperature in Celsius."""
        return f"Set to {temperature_c}C"

    @tool
    def get_outdoor_temperature() -> float:
        """Get the current outdoor temperature in Celsius."""
        return 28.5

    llm2 = LLMClient().bind_tools([set_cooling_setpoint, get_outdoor_temperature])
    resp2 = llm2.invoke([
        SystemMessage(content="You are a building control agent. Use tools to inspect "
                              "state and adjust setpoints. Reason briefly."),
        HumanMessage(content="It's noon. Check the outdoor temperature and set the "
                             "cooling setpoint appropriately for an occupied office."),
    ])
    print(f"  Text:       {resp2.text!r}")
    print(f"  Tool calls: {len(resp2.tool_calls)}")
    for tc in resp2.tool_calls:
        print(f"    -> {tc.get('name')}({tc.get('args')})")
    print(f"  Latency:    {resp2.latency_ms:.0f} ms")
    print(f"  Tokens:     in={resp2.tokens_in}, out={resp2.tokens_out}")
    print(f"  Fallback:   {resp2.used_fallback}")

    if resp.text and resp2.tool_calls and not resp.used_fallback and not resp2.used_fallback:
        print()
        print("[SUCCESS] Groq client working, tool binding working.")
    else:
        print()
        print("[FAIL] Something is off. Review above.")