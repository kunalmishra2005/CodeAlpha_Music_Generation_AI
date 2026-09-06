import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Folder containing MIDI files
DATASET_PATH = "dataset"

# File where the trained model will be saved
MODEL_PATH = "music_model.keras"

# Sequence length used for training
SEQUENCE_LENGTH = 40


def get_notes():
    """Read MIDI files and extract notes and chords."""

    notes = []

    for file_name in os.listdir(DATASET_PATH):

        if not file_name.lower().endswith((".mid", ".midi")):
            continue

        file_path = os.path.join(DATASET_PATH, file_name)

        try:
            midi = converter.parse(file_path)

            print(f"Processing: {file_name}")

            parts = instrument.partitionByInstrument(midi)

            if parts:
                music_stream = parts.parts[0].recurse()
            else:
                music_stream = midi.flat.notes

            for element in music_stream:

                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

        except Exception as error:
            print(f"Could not process {file_name}: {error}")

    return notes


def create_model(notes):
    """Create and train the LSTM model."""

    pitchnames = sorted(set(notes))

    note_to_int = {
        note_name: number
        for number, note_name in enumerate(pitchnames)
    }

    network_input = []
    network_output = []

    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]

        network_input.append(
            [note_to_int[character] for character in sequence_in]
        )

        network_output.append(note_to_int[sequence_out])

    if len(network_input) == 0:
        raise ValueError(
            "Not enough MIDI notes. Add more MIDI files to the dataset folder."
        )

    n_patterns = len(network_input)
    n_vocab = len(pitchnames)

    network_input = np.reshape(
        network_input,
        (n_patterns, SEQUENCE_LENGTH, 1)
    )

    network_input = network_input / float(n_vocab)

    network_output = to_categorical(
        network_output,
        num_classes=n_vocab
    )

    model = Sequential()

    model.add(
        LSTM(
            128,
            input_shape=(SEQUENCE_LENGTH, 1),
            return_sequences=True
        )
    )

    model.add(Dropout(0.3))

    model.add(
        LSTM(128)
    )

    model.add(Dropout(0.3))

    model.add(
        Dense(n_vocab, activation="softmax")
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    model.summary()

    model.fit(
        network_input,
        network_output,
        epochs=30,
        batch_size=64
    )

    model.save(MODEL_PATH)

    with open("notes.pkl", "wb") as file:
        pickle.dump(
            {
                "notes": notes,
                "pitchnames": pitchnames
            },
            file
        )

    print("\nTraining completed successfully!")
    print(f"Model saved as: {MODEL_PATH}")
    print("Note information saved as: notes.pkl")


if __name__ == "__main__":

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    notes = get_notes()

    print(f"\nTotal extracted notes: {len(notes)}")

    if len(notes) == 0:
        print(
            "\nNo MIDI files found."
            "\nPlease put .mid or .midi files inside the dataset folder."
        )
    else:
        create_model(notes)