🎵 AI Music Generation with LSTM

An AI-powered music generation project that uses a Long Short-Term Memory (LSTM) neural network to learn patterns from MIDI music and generate new musical sequences.

🚀 Features

- 🎼 MIDI music dataset processing using Music21
- 🤖 LSTM-based deep learning model
- 🧠 Automatic learning of musical note patterns
- 🎵 AI-generated music sequences
- 🎚️ Adjustable music generation length
- 💾 Generated MIDI file output
- 🌐 Interactive Streamlit web interface
- ⬇️ Download generated MIDI music

🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Music21
- NumPy
- Streamlit

📂 Project Structure

CodeAlpha_Music_Generation_AI/
│
├── dataset/
│   └── MIDI music files
│
├── app.py
├── train_model.py
├── generate_music.py
├── requirements.txt
├── music_model.keras
├── notes.pkl
└── generated_music.mid

🧠 How It Works

The project follows these steps:

MIDI Dataset
     ↓
Music21 Preprocessing
     ↓
Note Sequence Extraction
     ↓
LSTM Model Training
     ↓
Learned Musical Patterns
     ↓
AI Music Generation
     ↓
Generated MIDI File
     ↓
Streamlit Web Application

⚙️ Installation

Clone the repository and install the required dependencies:

git clone  https://github.com/kunalmishra2005/CodeAlpha_Music_Generation_AI.git
cd CodeAlpha_Music_Generation_AI
pip install -r requirements.txt

▶️ Run the Web Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

🎶 Generate Music

1. Open the Streamlit application.
2. Select the desired music length using the slider.
3. Click Generate AI Music.
4. Wait for the model to generate a new sequence.
5. Download the generated MIDI file.

🏋️ Training the Model

To retrain the model using the MIDI dataset:

python train_model.py

The trained model is saved as:

music_model.keras

The extracted note information is saved as:

notes.pkl

🎵 Generate Music from the Trained Model

You can also generate music directly from the terminal:

python generate_music.py 100

The generated MIDI file will be saved as:

generated_music.mid

📌 Project Purpose

This project was developed as part of the CodeAlpha Artificial Intelligence Internship to demonstrate practical implementation of deep learning for music generation.

## 👨‍💻 Author

**Kunal Mishra**

GitHub: https://github.com/kunalmishra2005

## 📌 Internship Project

This project was developed as part of my **CodeAlpha AI Internship**.


