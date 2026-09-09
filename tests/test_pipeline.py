import unittest
import sys
import os

# Ensure icassp root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import config as config
from features.codec_tokenizer import CodecTokenizer
from data.dataset import DiscreteCodecDataset, generate_synthetic_dysfluent_tokens
from models.codec_inpainter import CodecLLMInpainter
from models.baseline_classifier import BaselineTokenClassifier
from evaluation.metrics import (
    compute_pesq_proxy,
    compute_stoi_proxy,
    compute_voice_identity_similarity,
    compute_dysfluency_reduction_rate
)

class TestCodecLLMPipeline(unittest.TestCase):
    
    def test_config(self):
        """Verify global configurations."""
        self.assertEqual(config.SAMPLE_RATE, 16000)
        self.assertEqual(config.NUM_RVQ_CODEBOOKS, 4)
        self.assertEqual(config.CODEBOOK_SIZE, 1024)

    def test_codec_tokenizer(self):
        """Verify CodecTokenizer encode and decode shape contracts."""
        tokenizer = CodecTokenizer()
        wav = torch.randn(1, 16000)
        tokens = tokenizer.encode(wav)
        
        self.assertEqual(tokens.ndim, 3)
        self.assertEqual(tokens.size(1), config.NUM_RVQ_CODEBOOKS)
        
        reconstructed_wav = tokenizer.decode(tokens)
        self.assertEqual(reconstructed_wav.ndim, 2)
        self.assertEqual(reconstructed_wav.size(0), 1)

    def test_dataset(self):
        """Verify synthetic dysfluency generator and PyTorch dataset."""
        stuttered_tokens, mask, target_tokens = generate_synthetic_dysfluent_tokens(num_frames=100)
        self.assertEqual(stuttered_tokens.shape, (config.NUM_RVQ_CODEBOOKS, 100))
        self.assertEqual(mask.shape, (100,))
        self.assertEqual(target_tokens.shape, (config.NUM_RVQ_CODEBOOKS, 100))
        
        dataset = DiscreteCodecDataset(num_samples=5, num_frames=100)
        self.assertEqual(len(dataset), 5)
        sample = dataset[0]
        self.assertEqual(sample["stuttered_tokens"].shape, (config.NUM_RVQ_CODEBOOKS, 100))

    def test_models(self):
        """Verify forward pass of Codec-LLM Inpainter and Baseline Classifier."""
        tokens = torch.randint(0, config.CODEBOOK_SIZE, (2, config.NUM_RVQ_CODEBOOKS, 100))
        
        # 1. Baseline
        baseline = BaselineTokenClassifier()
        b_logits = baseline(tokens)
        self.assertEqual(b_logits.shape, (2, 5))
        
        # 2. Inpainter
        inpainter = CodecLLMInpainter()
        dys_logits, inpainted_logits = inpainter(tokens)
        self.assertEqual(dys_logits.shape, (2, 100))
        self.assertEqual(len(inpainted_logits), config.NUM_RVQ_CODEBOOKS)
        self.assertEqual(inpainted_logits[0].shape, (2, 100, config.CODEBOOK_SIZE))
        
        # 3. Repair method
        repaired = inpainter.repair_tokens(tokens)
        self.assertEqual(repaired.shape, tokens.shape)

    def test_metrics(self):
        """Verify evaluation metric calculations."""
        wav1 = np.random.randn(16000).astype(np.float32)
        wav2 = wav1 + 0.01 * np.random.randn(16000).astype(np.float32)
        
        pesq = compute_pesq_proxy(wav1, wav2)
        stoi = compute_stoi_proxy(wav1, wav2)
        self.assertGreaterEqual(pesq, 1.0)
        self.assertLessEqual(pesq, 4.5)
        self.assertGreaterEqual(stoi, 0.0)
        self.assertLessEqual(stoi, 1.0)
        
        t1 = torch.randint(0, 1000, (1, 4, 50))
        t2 = t1.clone()
        sim = compute_voice_identity_similarity(t1, t2)
        self.assertAlmostEqual(sim, 1.0, places=3)
        
        red = compute_dysfluency_reduction_rate(np.array([1, 1, 0]), np.array([1, 1, 0]))
        self.assertEqual(red, 100.0)

if __name__ == "__main__":
    unittest.main()
