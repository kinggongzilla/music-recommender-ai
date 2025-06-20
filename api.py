from flask import Flask, request, jsonify
from transformers import AutoProcessor, ClapModel
from qdrant_client import QdrantClient
import torch

app = Flask(__name__)

# Initialize CLAP model and Qdrant client
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
qdrant_client = QdrantClient(path="./qdrant_data")  # Same path as before

@app.route('/find_similar_audio', methods=['POST'])
def find_similar_audio():
    # Get text input from request
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Generate text embedding
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embedding = model.get_text_features(**inputs).cpu().numpy().squeeze()
        
        # Search Qdrant for top 5 similar audio files
        results = qdrant_client.search(
            collection_name="gtzan_clap_embeddings",
            query_vector=text_embedding.tolist(),
            limit=5  # Get top 5 matches
        )
        
        if not results:
            return jsonify({"error": "No matching audio found"}), 404
        
        # Prepare response with all matches
        response = {
            "query": text,
            "results": []
        }
        
        for match in results:
            response["results"].append({
                "filename": match.payload['filename'],
                "genre": match.payload['genre'],
                "full_path": match.payload['full_path'],
                "original_id": match.payload['original_id'],
                "similarity_score": float(match.score)  # Convert to native Python float
            })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)