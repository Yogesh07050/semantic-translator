"""
Semantic Translation Streamlit App (Fixed for OpenAI v1.x)
----------------------------------------------------------
- Loads OPENAI_API_KEY from Streamlit secrets (fallback to .env)
- Works around proxy injection issue on Streamlit Cloud
- Model picker includes GPT-5 / GPT-4.1 / GPT-4o families
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Fix: Import directly from the top-level OpenAI module
from openai import OpenAI

# -------------------------------------------------------------------------
# Load API Key from secrets (fallback to .env for local)
# -------------------------------------------------------------------------
load_dotenv()
API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Semantic Translator", layout="centered")

if not API_KEY:
    st.error("Missing API key. Please set OPENAI_API_KEY in Streamlit secrets or .env.")
    st.stop()

# -------------------------------------------------------------------------
# Patch: Disable proxy settings Streamlit may inject
# -------------------------------------------------------------------------
# Streamlit Cloud injects proxy-related env vars that confuse httpx/OpenAI
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if var in os.environ:
        del os.environ[var]

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# -------------------------------------------------------------------------
# Model list
# -------------------------------------------------------------------------
COMMON_MODELS = [
    "gpt-5", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4o", "gpt-4o-mini",
    "gpt-4", "gpt-3.5-turbo"
]

# -------------------------------------------------------------------------
# Translation Function
# -------------------------------------------------------------------------
def semantic_translate(text, source_lang, target_lang, model="gpt-4.1"):
    """Performs semantic translation."""
    prompt = f"""You are a professional translator. Translate the following text from {source_lang} to {target_lang}.
Ensure the translation is SEMANTIC — preserving meaning, tone, and emotion, not just words.

Text:
{text}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a semantic translator preserving meaning and context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------------------------------
# Semantic Validation
# -------------------------------------------------------------------------
def validate_semantics(original, translated, source_lang, target_lang, model="gpt-4.1"):
    """Returns semantic similarity score (0.0 - 1.0)."""
    prompt = f"""
Compare the semantic meaning of these two sentences.
Original ({source_lang}): {original}
Translated ({target_lang}): {translated}

Rate similarity from 0 (completely different) to 1 (identical in meaning).
Respond only with the number.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except Exception:
        return None

# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------
st.title("Semantic Translator")
st.markdown("Translate text semantically — choose a model or enter a custom model name.")

col1, col2 = st.columns(2)
with col1:
    source_lang = st.text_input("Source Language", value="English")
with col2:
    target_lang = st.text_input("Target Language", value="Tamil")

text_input = st.text_area("Text to translate", height=160)

model_picker = st.selectbox("Choose a model (or select Custom):", ["-- Select model --"] + COMMON_MODELS + ["Custom model name"])
custom_model = ""
if model_picker == "Custom model name":
    custom_model = st.text_input("Enter custom model name (exact):", value="gpt-4.1")

selected_model = (
    custom_model if model_picker == "Custom model name"
    else (model_picker if model_picker != "-- Select model --" else "gpt-4.1")
)

# -------------------------------------------------------------------------
# Translate Button
# -------------------------------------------------------------------------
if st.button("Translate"):
    if not text_input.strip():
        st.error("Please enter text to translate.")
    elif not source_lang.strip() or not target_lang.strip():
        st.error("Please specify both source and target languages.")
    else:
        try:
            with st.spinner(f"Translating using model `{selected_model}`..."):
                translated = semantic_translate(text_input, source_lang, target_lang, model=selected_model)
            st.subheader("Translated Text")
            st.write(translated)

            with st.spinner("Checking semantic similarity..."):
                score = validate_semantics(text_input, translated, source_lang, target_lang, model=selected_model)
            if score is not None:
                st.metric("Semantic Similarity Score (0.0–1.0)", f"{score:.2f}")
            else:
                st.info("Could not parse numeric similarity score.")
        except Exception as e:
            st.error(f"Model error: {e}")
