# Emotion & Scene Tagger for AI-Generated Calls

## Project Overview
This Python tool tags call transcripts with labels for emotion, scene context, and completeness using a zero-shot classification model (`facebook/bart-large-mnli`). 

## Features
- Classifies emotions: happy, angry, neutral
- Detects scene context: support, sales, casual
- Checks completeness of call sections: intro, main, closing
- Uses Hugging Face transformers for zero-shot text classification
- Works on mocked sample call transcripts (5–10 examples)


## Getting Started

### Prerequisites
- Python 3.7+
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage
Run the tagger script to classify sample transcripts:
```bash
python tagger.py
```

You can add or modify sample transcripts directly in the script or place them in a designated `transcripts/` folder (adjust code accordingly).

### Dependencies

- `transformers`
- `torch`

These are listed in `requirements.txt` for easy installation.
