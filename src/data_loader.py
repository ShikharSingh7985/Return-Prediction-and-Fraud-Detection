import pandas as pd
import numpy as np
import os

def load_and_preprocess_raw(trans_path, ident_path, sample_fraction=0.05, random_state=42):
    """
    Loads transaction and identity datasets in a memory-efficient chunked manner,
    merges them, selects a subset of features, and synthesizes the isReturned label.
    """
    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"Transaction file not found: {trans_path}")
    
    print(f"Reading transaction file in chunks: {trans_path}")
    chunks = []
    # Use chunksize to read transaction file without loading it all in RAM
    for chunk in pd.read_csv(trans_path, chunksize=50000):
        # Sample a subset of each chunk to maintain class representation and keep memory usage low
        sampled_chunk = chunk.sample(frac=sample_fraction, random_state=random_state)
        chunks.append(sampled_chunk)
        
    df_trans = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df_trans)} transactions.")
    
    # Load identity dataset (smaller, can be read directly or chunked)
    if os.path.exists(ident_path):
        print(f"Reading identity file: {ident_path}")
        df_ident = pd.read_csv(ident_path)
        print(f"Loaded {len(df_ident)} identity records.")
        # Left join on TransactionID
        df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')
        print(f"Merged shape: {df.shape}")
    else:
        print("Identity file not found or path not provided. Proceeding with transaction data only.")
        df = df_trans
        
    # Feature selection
    # Selected a subset of predictive features that have reasonable density and representation
    num_cols = ['TransactionAmt', 'dist1', 'C1', 'C2', 'C11', 'C13', 'D1', 'D2']
    cat_cols = ['ProductCD', 'card4', 'card6']
    
    if 'DeviceType' in df.columns:
        cat_cols.append('DeviceType')
        
    target_col = 'isFraud'
    
    # Ensure selected features exist, if not, print warning and use what's available
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    keep_cols = [target_col] + num_cols + cat_cols
    df_subset = df[keep_cols].copy()
    
    # Synthesize return label
    print("Synthesizing return label (isReturned)...")
    rng = np.random.default_rng(random_state)
    
    w_high_amt = (df_subset['ProductCD'] == 'W') & (df_subset['TransactionAmt'] > 120)
    # Handle missing ProductCD or other values safely
    w_high_amt = w_high_amt.fillna(False)
    
    sr_high_amt = df_subset['ProductCD'].isin(['S', 'R']) & (df_subset['TransactionAmt'] > 200)
    sr_high_amt = sr_high_amt.fillna(False)
    
    fraud = df_subset[target_col] == 1
    
    # Base probability for returns
    probs = np.zeros(len(df_subset))
    probs[w_high_amt] += 0.25
    probs[sr_high_amt] += 0.35
    probs[fraud] += 0.15
    probs += 0.02  # baseline return rate
    
    probs = np.clip(probs, 0.0, 1.0)
    df_subset['isReturned'] = (rng.random(len(df_subset)) < probs).astype(int)
    
    # Print class distributions
    fraud_dist = df_subset['isFraud'].value_counts(normalize=True) * 100
    return_dist = df_subset['isReturned'].value_counts(normalize=True) * 100
    print(f"isFraud distribution:\n{df_subset['isFraud'].value_counts()} ({fraud_dist.get(1, 0):.2f}% fraud)")
    print(f"isReturned distribution:\n{df_subset['isReturned'].value_counts()} ({return_dist.get(1, 0):.2f}% returned)")
    
    return df_subset, num_cols, cat_cols
