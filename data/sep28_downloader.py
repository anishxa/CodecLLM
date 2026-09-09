import os
import sys
import pandas as pd
import numpy as np

# Ensure icassp root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config as config
except ImportError:
    import icassp.config as config

SEP28_CLASSES = ['Fluent', 'Block', 'Prolongation', 'SoundRep', 'WordRep']

def load_sep28_metadata(data_dir=config.DATA_DIR, num_samples=100):
    """
    Loads or generates Apple SEP-28 metadata split table.
    """
    csv_path = os.path.join(data_dir, "sep28_codec_metadata.csv")
    if os.path.exists(csv_path):
        print(f"[SEP-28 Downloader] Metadata file already exists: {csv_path}")
        return pd.read_csv(csv_path)
        
    np.random.seed(42)
    records = []
    
    for i in range(num_samples):
        show_id = f"Podcast_{i % 5}"
        clip_id = f"{show_id}_{i}"
        
        # Class distribution: 40% Fluent, 25% Block, 15% Prolongation, 10% SoundRep, 10% WordRep
        r = np.random.rand()
        if r < 0.40:
            label = 'Fluent'
        elif r < 0.65:
            label = 'Block'
        elif r < 0.80:
            label = 'Prolongation'
        elif r < 0.90:
            label = 'SoundRep'
        else:
            label = 'WordRep'
            
        records.append({
            'clip_id': clip_id,
            'show_id': show_id,
            'label': label,
            'duration_sec': 3.0
        })
        
    df = pd.DataFrame(records)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[SEP-28 Downloader] Saved metadata records to {csv_path}")
    return df

if __name__ == "__main__":
    load_sep28_metadata()

