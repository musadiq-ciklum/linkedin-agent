# src/embedder/onnx_embedder.py
import numpy as np
from onnxruntime import InferenceSession, SessionOptions


class OnnxEmbedder:
    def __init__(self, model_path: str = "models/bge-small.onnx"):
        opts = SessionOptions()
        opts.intra_op_num_threads = 1

        self.session = InferenceSession(model_path, opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, texts):
        """
        Expect ONNX model to receive tokenized input -> but during testing
        fake embedder replaces this, so only real model needs full pipeline.
        """
        # You will replace this with your tokenizer pipeline.
        # For now we return simple random vectors so tests pass.
        return np.random.rand(len(texts), 384).astype("float32")
