import streamlit as st
import os
import subprocess

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center;'>🎵 AI Music Generator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>"
    "Create unique music using an AI-powered LSTM model."
    "</p>",
    unsafe_allow_html=True
)
st.divider()
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
    url("https://images.unsplash.com/photo-1511379938547-c1f69419868d");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    max-width: 850px;
    padding-top: 3rem;
}

h1, h2, h3, p, label {
    color: white !important;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.8rem;
    font-size: 20px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);

}
</style>
""", unsafe_allow_html=True)

st.divider()
st.info(
    "🎼 This AI music generator uses an LSTM deep learning model "
    "trained on MIDI music sequences to create new musical patterns."
)

st.markdown(
    "<h2 style='text-align: center;'>🎶 Generate Your Music</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>"
    "Choose the length of your AI-generated music and create a unique tune."
    "</p>",
    unsafe_allow_html=True
)

length = st.slider(
    "Select music length",
    min_value=50,
    max_value=200,
    value=100,
    step=10
)


if st.button("🎶 Generate AI Music", use_container_width=True):

    with st.spinner("Generating music..."):
        result = subprocess.run(
            ["python", "generate_music.py", str(length)],
            capture_output=True,
            text=True
        )

    if result.returncode == 0:
        st.success("Music generated successfully! 🎉")

        output_file = "generated_music.mid"

        if os.path.exists(output_file):
            st.write("Your generated MIDI file is ready:")

            with open(output_file, "rb") as file:
                st.download_button(
                    label="⬇️ Download Generated MIDI",
                    data=file,
                    file_name="generated_music.mid",
                    mime="audio/midi"
                )

    else:
        st.error("Something went wrong while generating music.")
        st.code(result.stderr)

        st.divider()

st.markdown(
    "<h3 style='text-align: center;'>🤖 About This Project</h3>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center;'>
    This project demonstrates AI-based music generation using an
    <b>LSTM neural network</b>. MIDI data is processed using
    <b>Music21</b>, musical patterns are learned by the model,
    and new sequences are generated automatically.
    </p>
    """,
    unsafe_allow_html=True
)

st.caption("Built with Python • TensorFlow • Music21 • Streamlit")