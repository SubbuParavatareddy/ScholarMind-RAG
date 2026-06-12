from __future__ import annotations
import tomllib
from pathlib import Path


class APIKeyLoader:
    """Resolves the Gemini API key for both deployment and local environments.

    Deployment (Streamlit Cloud / GitHub Secrets):
        Add GOOGLE_API_KEY as a repository secret; Streamlit reads it via st.secrets.

    Local development:
        Add GOOGLE_API_KEY to .streamlit/config.toml under [secrets]:
            [secrets]
            GOOGLE_API_KEY = "AIzaSy..."
    """

    KEY_NAME = "GOOGLE_API_KEY"

    @classmethod
    def load(cls) -> str | None:
        # 1. Streamlit Cloud / GitHub Secrets — imported lazily to stay testable
        try:
            import streamlit as st
            val = st.secrets.get(cls.KEY_NAME, "")
            if val:
                return val
        except Exception:
            pass

        # 2. Local: .streamlit/secrets.toml (Streamlit's dedicated secrets file)
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            with open(secrets_path, "rb") as f:
                cfg = tomllib.load(f)
            val = cfg.get(cls.KEY_NAME, "")
            if val:
                return val

        return None
