import os

# Signal & Codec Parameters
SAMPLE_RATE = 16000
ENCODEC_BANDWIDTH = 6.0  # kbps
NUM_RVQ_CODEBOOKS = 4   # 4 codebook layers (Layer 1=Semantic, Layers 2-4=Acoustics)
CODEBOOK_SIZE = 1024     # Number of discrete tokens per codebook (0 to 1023)
MASK_TOKEN_ID = 1024     # Reserved token ID for masked token inpainting

# Model Hyperparameters (Lightweight 1.5M Parameter Transformer)
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_EPOCHS = 10

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
