"""LLM provider configuration with multi-provider support."""

from __future__ import annotations

import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {"model": "gpt-4o", "env_key": "OPENAI_API_KEY"},
    "bedrock": {"model": "arn:aws:bedrock:us-east-1:149935825740:inference-profile/global.anthropic.claude-sonnet-4-6", "env_key": None},
    "anthropic": {"model": "claude-sonnet-4-5-20250929", "env_key": "ANTHROPIC_API_KEY"},
    "gemini": {"model": "gemini-2.0-flash", "env_key": "GOOGLE_API_KEY"},
}

# Aggressive Bedrock retry configuration
BEDROCK_CONFIG = Config(
    retries={
        "max_attempts": 15,
        "mode": "adaptive",
    },
    connect_timeout=120,
    read_timeout=120,
)

# Module-level Bedrock client (reused across all LLM calls)
_bedrock_client = None


def _get_bedrock_client():
    """Get or create a singleton Bedrock client with retry configuration."""
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client

    region = os.getenv("AWS_REGION", "us-east-1")
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

    if bearer_token:
        # Use bearer token with unsigned requests
        config = BEDROCK_CONFIG.merge(Config(signature_version=UNSIGNED))
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=config,
        )
        # Inject bearer token into HTTP adapter
        from botocore.httpsession import URLLib3Session

        original_send = URLLib3Session.send

        def patched_send(self, request):
            request.headers["Authorization"] = f"Bearer {bearer_token}"
            return original_send(self, request)

        URLLib3Session.send = patched_send
    else:
        # Use standard AWS credential chain
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BEDROCK_CONFIG,
        )

    return _bedrock_client


def get_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
):
    """Return a chat model instance based on the configured provider.

    Resolution order for *provider*:
        1. Explicit ``provider`` argument.
        2. ``LLM_PROVIDER`` environment variable.
        3. Falls back to ``"openai"``.

    Parameters
    ----------
    provider:
        ``"openai"``, ``"anthropic"``, ``"gemini"``, or ``"bedrock"``.
    model:
        Override the default model name for the chosen provider.
    temperature:
        Sampling temperature. Defaults to ``0.0`` for deterministic output.

    Returns
    -------
    ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatBedrock
        A ready-to-use chat model.

    Raises
    ------
    ValueError
        If the provider string is not recognised.
    RuntimeError
        If the required API key environment variable is not set.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower().strip()

    if provider not in _DEFAULTS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Choose from: {', '.join(_DEFAULTS)}"
        )

    cfg = _DEFAULTS[provider]
    model = model or cfg["model"]

    if cfg.get("env_key"):
        api_key = os.getenv(cfg["env_key"])
        if not api_key:
            raise RuntimeError(
                f"{cfg['env_key']} environment variable is required for provider '{provider}'"
            )
    else:
        api_key = None

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    elif provider == "bedrock":
        region = os.getenv("AWS_REGION", "us-east-1")
        bedrock_client = _get_bedrock_client()

        # ChatBedrock with explicit provider hint for inference-profile ARN
        return ChatBedrock(
            client=bedrock_client,
            model=model,
            provider="anthropic",
            temperature=temperature,
        )
    elif provider == "anthropic":
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)

    raise ValueError(f"Unhandled provider: {provider}")
