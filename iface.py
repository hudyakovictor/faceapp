import cv2
import insightface
import numpy as np
from typing import List, Dict, Optional

class FaceEmbedder:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(name="buffalo_l")
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_data(self, image_path: str) -> Optional[List[Dict]]:
        """
        Возвращает список лиц с эмбедингами и возрастом:
        [{
            "embedding": [0.12, -0.05, ...],  # 512-dim вектор
            "age": 25                         # целое число
        }, ...]
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        faces = self.model.get(img)
        if not faces:
            return None

        return [
            {
                "embedding": face.embedding.tolist(),
                "age": int(face.age)
            }
            for face in faces
        ]

# Глобальный инстанс (загрузится при импорте)
face_embedder = FaceEmbedder()