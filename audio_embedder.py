import os
import librosa
import numpy as np
from tqdm import tqdm
import torch
import uuid
from transformers import AutoProcessor, ClapModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

class CLAPEmbeddingPipeline:
    def __init__(self, model_name="laion/clap-htsat-unfused"):
        """
        Initialize CLAP model and Qdrant vector database
        
        Args:
            model_name (str): CLAP model name from HuggingFace
        """
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLAP model
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Initialize Qdrant client with local storage
        self.qdrant_client = QdrantClient(path="./qdrant_data")
        
        # Get or create collection
        try:
            collection_info = self.qdrant_client.get_collection("gtzan_clap_embeddings")
            print(f"Using existing collection with {collection_info.vectors_count} vectors")
        except:
            self.qdrant_client.recreate_collection(
                collection_name="gtzan_clap_embeddings",
                vectors_config=VectorParams(
                    size=512,  # CLAP embedding size
                    distance=Distance.COSINE
                )
            )
            print("Created new collection")
    
    def load_audio(self, audio_path, target_sr=48000):
        """Audio loading and preprocessing"""
        try:
            waveform, sr = librosa.load(audio_path, sr=None)
            if sr != target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            return waveform
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            return None
    
    def get_embedding(self, audio_path):
        """Generate CLAP embedding for an audio file"""
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
            
        inputs = self.processor(
            audios=waveform, 
            return_tensors="pt", 
            sampling_rate=48000
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self.model.get_audio_features(**inputs)
        
        return embedding.cpu().numpy().squeeze()

    def process_dataset(self, root_dir="gtzan_dataset/genres_original"):
        """Process all audio files in the GTZAN dataset"""
        # First pass to count files
        total_files = 0
        for genre in os.listdir(root_dir):
            genre_path = os.path.join(root_dir, genre)
            if os.path.isdir(genre_path):
                total_files += len([f for f in os.listdir(genre_path) 
                                if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))])
        
        # Initialize progress bar
        pbar = tqdm(total=total_files, desc="Processing audio files")
        
        # Process files
        points = []
        for genre in os.listdir(root_dir):
            genre_path = os.path.join(root_dir, genre)
            
            if not os.path.isdir(genre_path):
                continue
                
            for audio_file in os.listdir(genre_path):
                if not audio_file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    continue
                    
                audio_path = os.path.join(genre_path, audio_file)
                
                # Generate consistent UUID from filename
                file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, audio_path))
                
                embedding = self.get_embedding(audio_path)
                if embedding is None:
                    pbar.update(1)
                    continue
                
                # Create proper PointStruct object
                point = PointStruct(
                    id=file_id,  # Now using proper UUID
                    vector=embedding.tolist(),
                    payload={
                        "genre": genre,
                        "filename": audio_file,
                        "full_path": audio_path,
                        "original_id": os.path.splitext(audio_file)[0]  # Keep original ID in payload
                    }
                )
                points.append(point)
                
                pbar.update(1)
                
                # Upload in batches of 100
                if len(points) >= 100:
                    try:
                        self.qdrant_client.upsert(
                            collection_name="gtzan_clap_embeddings",
                            points=points
                        )
                        points = []
                    except Exception as e:
                        print(f"Error uploading batch: {str(e)}")
                        continue
        
        # Upload any remaining points
        if points:
            try:
                self.qdrant_client.upsert(
                    collection_name="gtzan_clap_embeddings",
                    points=points
                )
            except Exception as e:
                print(f"Error uploading final batch: {str(e)}")
        
        pbar.close()
        print(f"Successfully processed {total_files} files")

if __name__ == "__main__":
    try:
        pipeline = CLAPEmbeddingPipeline()
        pipeline.process_dataset("gtzan_dataset/1/Data/genres_original")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure Qdrant is properly installed and you have disk space available")