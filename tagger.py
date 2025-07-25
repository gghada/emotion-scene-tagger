# tagger.py

import os
from transformers import pipeline

# Load Hugging Face zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Classification labels
emotion_labels = ["angry", "happy", "neutral", "confused", "grateful", "impatient", "calm"]
scene_labels = ["support", "sales", "casual", "medical", "product inquiry"]
completeness_labels = ["complete", "incomplete"]

# Folder where call transcripts are stored
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "transcripts")

def classify_transcript(text):
    emotion = classifier(text, emotion_labels)
    scene = classifier(text, scene_labels)
    completeness = classifier(text, completeness_labels)

    # Return top predictions
    return {
        "emotion": emotion["labels"][0],
        "scene": scene["labels"][0],
        "completeness": completeness["labels"][0]
    }

def main():
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(TRANSCRIPTS_DIR, filename)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            tags = classify_transcript(text)

            print(f"\n {filename}")
            print(f"  Emotion: {tags['emotion']}")
            print(f"  Scene: {tags['scene']}")
            print(f"  Completeness: {tags['completeness']}")
            print("-" * 40)

if __name__ == "__main__":
    main()
