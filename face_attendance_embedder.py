"""
Face Embedding Module
ArcFace (InsightFace) based embedding extractor.
"""

import cv2
import numpy as np
from typing import Optional, List
from pathlib import Path

import insightface


class FaceEmbedder:
    """Extract face embeddings using ArcFace models from InsightFace."""

    def __init__(self, model_name: str = "arcface_r100_v1", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.input_size = (112, 112)
        self.model = self._load_arcface_model()
        self._embedding_dim = getattr(self.model, "feature_dim", 512)

    def _load_arcface_model(self):
        """Load ArcFace model via InsightFace."""
        try:
            model_path = Path(self.model_name).expanduser()
            if model_path.exists():
                model = insightface.model_zoo.get_model(str(model_path))
            else:
                model = insightface.model_zoo.get_model(self.model_name)
            if model is None:
                raise RuntimeError("InsightFace returned None for this model name.")
            ctx_id = -1 if self.device == "cpu" else 0
            model.prepare(ctx_id=ctx_id)
            print(f"✓ ArcFace model loaded ({self.model_name})")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ArcFace model '{self.model_name}': {exc}"
            ) from exc

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for embedding extraction.
        Args:
            face: Face image (H, W, 3) in BGR.
        Returns:
            Preprocessed face ready for ArcFace.
        """
        face = cv2.resize(face, self.input_size)
        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        if face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_BGRA2BGR)
        face = face.astype("float32")
        return face

    def get_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a single embedding.
        """
        try:
            processed = self.preprocess_face(face)
            embedding = self.model.get_feat(processed)
            return embedding.astype("float32")
        except Exception as exc:
            print(f"Error extracting embedding: {exc}")
            return None

    def get_embeddings_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a batch of faces.
        """
        if not faces:
            return np.array([])
        processed = [self.preprocess_face(face) for face in faces]
        batch = np.stack(processed, axis=0)
        embeddings = self.model.get_feats(batch)
        return embeddings.astype("float32")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
