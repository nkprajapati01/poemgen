import os
os.environ["STREAMLIT_WATCH_INSTALLER"] = "false"

import streamlit as st
from transformers import pipeline
import torch

torch.set_grad_enabled(False)

# Load a lightweight, safe model for poem generation
@st.cache_resource
def load_poem_generator():
    """
    Load GPT-2 pipeline for poem generation.
    """
    return pipeline("text-generation", model="gpt2", device=-1)

# UI
st.title("📜 Poem Generator: Unleash Your Inner Bard!")
st.write("#### Create beautiful verses in moments! 🖋️🎶")
st.markdown(
    """
    Welcome to **Poem Generator**, your AI muse for crafting lyrical expressions!
    Provide a theme or some keywords, and watch as poetry unfolds. 🌠
    Let your creativity flow and allow AI to weave your thoughts into verse! 🚀
    """
)

st.subheader("✍️ Enter Your Poem Prompt")
poem_topic = st.text_input(
    "Describe a theme or some keywords for your poem:",
    placeholder="E.g., Write a short poem about the beauty of a rainy day."
)

if st.button("✨ Generate Poem"):
    if poem_topic.strip():
        poem_generator = load_poem_generator()
        try:
            poem = poem_generator(
                poem_topic,
                max_length=150, # Increased max length slightly
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=50256  # GPT-2 uses this as eos_token_id
            )
            generated_poem_raw = poem[0]["generated_text"]
            # Add line breaks for better readability
            formatted_poem = ""
            for char in generated_poem_raw:
                formatted_poem += char
                if char in ['.', ',', '!', '?']:
                    formatted_poem += '\n'
                elif len(formatted_poem.split('\n')[-1].split()) > 10: # Add line break if a line gets too long (adjust as needed)
                    formatted_poem += '\n'

            st.success("✅ Poem generated! Here's your poetic creation:")
            st.text_area("📜 Your Poem", value=formatted_poem.strip(), height=250) # Use formatted_poem and strip leading/trailing whitespace
        except Exception as e:
            st.error(f"⚠️ An error occurred while generating the poem: {e}")
    else:
        st.warning("⚠️ Please enter a theme or keywords to generate a poem.")

st.markdown("---")

st.write("🌟 **Made by Adarsh Kumar** 🌟")
