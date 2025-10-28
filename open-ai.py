"""
Semantic Translation Streamlit App (models updated)
--------------------------------------------------
- Loads OPENAI_API_KEY from Streamlit secrets
- Model picker includes GPT-5 family, GPT-4.1 family, GPT-4o family, and a Custom option
- Generic language input (free text for source/target)
- Semantic translation + similarity check
"""

import streamlit as st
from openai import OpenAI

# -------------------------------------------------------------------------
# Load API Key from Streamlit Secrets
# -------------------------------------------------------------------------
API_KEY = st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="Semantic Translator", layout="centered")

if not API_KEY:
    st.error("Missing API key. Please add `OPENAI_API_KEY` in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# -------------------------------------------------------------------------
# Pre-populated model list (includes GPT-5 and GPT-4.1 families)
# -------------------------------------------------------------------------
COMMON_MODELS = [
    # GPT-5 family
    "gpt-5",
    "gpt-5-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    # GPT-4.1 family
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # GPT-4o family
    "gpt-4o",
    "gpt-4o-mini",
    # Legacy / fallback
    "gpt-4",
    "gpt-3.5-turbo"
]

# -------------------------------------------------------------------------
# Translation Function
# -------------------------------------------------------------------------
def semantic_translate(text, source_lang, target_lang, model="gpt-4.1"):
    """Performs a semantic translation using the specified model."""
    prompt = f"""You are a professional translator. Translate the following text from {source_lang} to {target_lang}.
Ensure the translation is SEMANTIC — preserving meaning, tone, and emotion, not just words.

Text:
{text}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a semantic translator preserving meaning and context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
    except Exception as e:
        if "temperature" in str(e).lower():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a semantic translator preserving meaning and context."},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            raise e

    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------------
# Semantic Validation Function
# -------------------------------------------------------------------------
def validate_semantics(original, translated, source_lang, target_lang, model="gpt-4.1"):
    """Returns a semantic similarity score (0.0 - 1.0) or None if unavailable."""
    prompt = f"""
Compare the semantic meaning of these two sentences.
Original ({source_lang}): {original}
Translated ({target_lang}): {translated}

Rate similarity from 0 (completely different) to 1 (identical in meaning).
Respond only with the number.
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    except Exception as e:
        if "temperature" in str(e).lower():
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise e

    try:
        return float(response.choices[0].message.content.strip())
    except Exception:
        return None


# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------
st.title("Semantic Translator")
st.markdown("Translate text semantically — choose a model or enter a custom model name if yours isn't listed.")

col1, col2 = st.columns(2)
with col1:
    source_lang = st.text_input("Source Language", value="English", help="e.g., English or `auto` (for detection)")
with col2:
    target_lang = st.text_input("Target Language", value="Tamil", help="e.g., Tamil")

text_input = st.text_area("Text to translate", height=160, placeholder="Type or paste the text to translate...")

model_picker = st.selectbox("Choose a model (or select Custom):", ["-- Select model --"] + COMMON_MODELS + ["Custom model name"])
custom_model = ""
if model_picker == "Custom model name":
    custom_model = st.text_input("Enter custom model name (exact):", value="gpt-4.1")

selected_model = custom_model if model_picker == "Custom model name" else (model_picker if model_picker != "-- Select model --" else "gpt-4.1")

# -------------------------------------------------------------------------
# Translation Trigger
# -------------------------------------------------------------------------
if st.button("Translate"):
    if not text_input.strip():
        st.error("Please enter the text to translate.")
    elif not source_lang.strip() or not target_lang.strip():
        st.error("Please provide both source and target languages.")
    elif source_lang.strip().lower() == target_lang.strip().lower():
        st.warning("Source and target languages are the same.")
    else:
        try:
            with st.spinner(f"Translating using model `{selected_model}`..."):
                translated = semantic_translate(text_input, source_lang, target_lang, model=selected_model)
            st.subheader("Translated Text")
            st.write(translated)

            with st.spinner("Checking semantic similarity..."):
                score = validate_semantics(text_input, translated, source_lang, target_lang, model=selected_model)
            if score is not None:
                st.metric(label="Semantic Similarity Score (0.0–1.0)", value=f"{score:.2f}")
            else:
                st.info("Could not parse a numeric similarity score from the model response.")
        except Exception as e:
            st.error(f"Error calling model `{selected_model}`: {e}")
