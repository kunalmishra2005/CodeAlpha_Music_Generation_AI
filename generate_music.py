import pickle
import numpy as np
from music21 import stream, note, chord
from tensorflow.keras.models import load_model


# Load the trained model
MODEL_PATH = "music_model.keras"

# Load note information
NOTES_PATH = "notes.pkl"

# Number of notes to generate
GENERATE_LENGTH = 100


def generate_music(generate_length=100):
    print("Loading trained model...")

    model = load_model(MODEL_PATH)

    with open(NOTES_PATH, "rb") as file:
        data = pickle.load(file)

    notes = data["notes"]
    pitchnames = data["pitchnames"]

    note_to_int = {
        note_name: number
        for number, note_name in enumerate(pitchnames)
    }

    int_to_note = {
        number: note_name
        for number, note_name in enumerate(pitchnames)
    }

    # Choose a random starting point
    start = np.random.randint(0, len(notes) - 40)

    pattern = [
        note_to_int[notes[i]]
        for i in range(start, start + 40)
    ]

    output_notes = []

    print("Generating new music...")

    for _ in range(generate_length):

        # Prepare input for the model
        prediction_input = np.reshape(
            pattern,
            (1, len(pattern), 1)
        )

        prediction_input = prediction_input / float(len(pitchnames))

        # Predict the next note
        prediction = model.predict(
            prediction_input,
            verbose=0
        )

        index = np.argmax(prediction)

        result = int_to_note[index]

        output_notes.append(result)

        # Add the predicted note to the pattern
        pattern.append(index)

        # Keep only the last 40 notes
        pattern = pattern[-40:]

    # Create a MIDI stream
    midi_stream = stream.Stream()

    for element in output_notes:

        if "." in element:

            chord_notes = element.split(".")

            chord_notes = [
                int(n) for n in chord_notes
            ]

            new_chord = chord.Chord(chord_notes)

            midi_stream.append(new_chord)

        else:

            new_note = note.Note(element)

            midi_stream.append(new_note)

    output_file = "generated_music.mid"

    midi_stream.write(
        "midi",
        fp=output_file
    )

    print("\nMusic generated successfully!")
    print(f"Saved as: {output_file}")


if __name__ == "__main__":
    import sys

    length = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    generate_music(length)