import json
import os
from dotenv import load_dotenv

# Try importing Streamlit for secrets & session_state support if available
try:
    import streamlit as st
except ImportError:
    st = None

# Try importing Groq gracefully
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()


class AutoDSAgent:
    """
    LLM-powered orchestrator for AutoDS.
    Uses Groq (fast LLM inference with Qwen 3.8 / LLaMA) to observe, decide, analyze, and act on datasets.
    Supports dynamic resolution from UI sidebar, Streamlit secrets, or .env.
    """

    def __init__(self):
        self.history = []
        self.client = None
        self.api_key = None
        self.model = "qwen/qwen3.8-27b"
        self._get_client()

    def _get_client(self):
        """Multi-layer dynamic client resolver (UI Session State -> Env -> Streamlit Secrets)."""
        api_key = None

        # 1. Check Streamlit session_state (entered directly via UI sidebar)
        if st is not None:
            try:
                api_key = st.session_state.get("groq_api_key")
            except Exception:
                pass

        # 2. Check environment variable (.env or exported container env)
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")

        # 3. Check Streamlit Cloud secrets (secrets.toml or Cloud Secrets tab)
        if not api_key and st is not None:
            try:
                if hasattr(st, "secrets"):
                    if "GROQ_API_KEY" in st.secrets:
                        api_key = st.secrets["GROQ_API_KEY"]
                    elif hasattr(st.secrets, "get"):
                        api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass

        if GROQ_AVAILABLE and api_key:
            clean_key = str(api_key).strip()
            if clean_key and (self.client is None or self.api_key != clean_key):
                try:
                    self.client = Groq(api_key=clean_key)
                    self.api_key = clean_key
                except Exception:
                    self.client = None

        return self.client

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Internal helper to call the Groq LLM with heuristic fallback."""
        if not GROQ_AVAILABLE:
            return "💡 AI intelligence is in offline fallback mode (groq package not installed)."

        client = self._get_client()
        if not client:
            return "💡 **Groq API Key Required**: Enter your key in the **Sidebar (🔑 Groq AI)** or add `GROQ_API_KEY` to **Streamlit Secrets**."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Notice from AI Agent: Unable to reach Groq API ({str(e)}). Pipeline continues with deterministic analysis."

    def observe(self, profile):
        self.history.append({"step": "observe", "profile": profile})
        return profile

    def decide(self, profile):
        system_prompt = (
            "You are an expert Data Scientist assistant operating an Automated Data Science (AutoDS) pipeline. "
            "Based on the dataset profile provided, decide on the most critical next step for data preprocessing or modeling. "
            "Reply with a short, concise sentence explaining what to do next."
        )
        user_prompt = f"Here is the dataset profile: {json.dumps(profile)}. What should we do next?"

        decision = self._call_llm(system_prompt, user_prompt)
        self.history.append({"step": "decide", "decision": decision})
        return decision

    def analyze(self, profile: dict, validation: dict) -> str:
        """
        Generate a comprehensive AI-powered analysis of the dataset.
        Takes the full profile + validation report and returns a written analysis.
        """
        system_prompt = (
            "You are a senior Data Scientist writing a professional dataset analysis report. "
            "Analyze the dataset profile and validation results provided. "
            "Write a concise but insightful analysis covering:\n"
            "1. Dataset overview (size, types, memory)\n"
            "2. Data quality issues found\n"
            "3. Key patterns or concerns (skewness, outliers, imbalance)\n"
            "4. Recommendations for preprocessing\n\n"
            "Use bullet points and keep it under 200 words. Be specific with numbers."
        )
        user_prompt = (
            f"Dataset Profile:\n{json.dumps(profile, default=str)}\n\n"
            f"Validation Report:\n{json.dumps(validation, default=str)}"
        )

        analysis = self._call_llm(system_prompt, user_prompt)
        self.history.append({"step": "analyze", "analysis": analysis})
        return analysis

    def suggest_filters(self, profile: dict, validation: dict) -> str:
        """
        Recommend which cleaning/filtering steps to apply based on the data profile.
        """
        system_prompt = (
            "You are a Data Engineer recommending data cleaning steps. "
            "Based on the dataset profile and validation report, suggest which of these "
            "cleaning operations should be applied and why:\n"
            "1. Drop duplicate rows\n"
            "2. Drop fully-null rows\n"
            "3. Drop constant columns\n"
            "4. Drop fully-null columns\n"
            "5. Drop ID-like columns\n"
            "6. Impute missing values (median for numeric, mode for categorical)\n"
            "7. Remove outliers (IQR method)\n\n"
            "For each, say YES or NO and give a brief reason. Keep it concise."
        )
        user_prompt = (
            f"Dataset Profile:\n{json.dumps(profile, default=str)}\n\n"
            f"Validation Report:\n{json.dumps(validation, default=str)}"
        )

        suggestions = self._call_llm(system_prompt, user_prompt)
        self.history.append({"step": "suggest_filters", "suggestions": suggestions})
        return suggestions

    def act(self, action):
        self.history.append({"step": "act", "action": action})
        print(f"[ACTION EXECUTED]: {action}")

    def state(self):
        return self.history
