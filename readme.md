Here's the complete README.md content you can copy directly into your file:

```
# Audio Similarity Search API

A Flask API that finds audio files semantically similar to text queries using the CLAP model and Qdrant vector database.

## Features
- Text-to-audio similarity search
- Returns top 5 matching audio files
- Metadata includes filename, genre, and similarity score
- GPU acceleration support (if available)

## Prerequisites
- Python 3.8+
- pip
- (Optional) NVIDIA GPU with CUDA for faster inference

## Installation

1. **Clone the repository**
   Clone this repo with `git clone`

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
   pip install flask qdrant-client transformers librosa tqdm numpy
   ```

   For CPU-only:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install flask qdrant-client transformers librosa tqdm numpy
   ```

4. **Download the dataset**
Run this in the root folder
   ```
   python gtzan_dataset.py
   ```

5. **Create Audio Vector Database**
Run this in the root folder
   ```
   python audio_embedder.py
   ```

## API Documentation

### Endpoint
`POST /find_similar_audio`

### Request
```json
{
  "text": "search query"
}
```

### Response
```json
{
  "query": "search query",
  "results": [
    {
      "filename": "song.wav",
      "genre": "genre_name",
      "full_path": "path/to/song.wav",
      "original_id": "song_id",
      "similarity_score": 0.95
    }
  ]
}
```

### Example Usage

**Using curl:**
```bash
curl -X POST http://localhost:5000/find_similar_audio \
  -H "Content-Type: application/json" \
  -d '{"text":"party dance song"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:5000/find_similar_audio",
    json={"text": "heavy metal guitar"}
)

if response.status_code == 200:
    for result in response.json()["results"]:
        print(f"Found: {result['filename']} (Score: {result['similarity_score']:.2f})")
else:
    print("Error:", response.json())
```

## Running the API

1. **Start the server**
   ```bash
   python api.py
   ```

2. **The API will be available at**
   ```
   http://localhost:5000
   ```