#!/usr/bin/env python3
"""
Train and Save ML Models for Binding Affinity Prediction

This script trains the best-performing models identified in the analysis
and saves them as pkl files for use in the prediction notebook.

Models saved:
- xgboost_model.pkl: XGBoost (best performer, R² = 0.52)
- random_forest_model.pkl: Random Forest (simpler baseline, R² = 0.48)
- lightgbm_model.pkl: LightGBM (R² = 0.50)

Usage:
    python 02_train_and_save_models.py
"""

import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Check for optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Skipping LightGBM model.")

import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Feature columns (molecular descriptors)
FEATURE_COLS = [
    'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
    'NumHeavyAtoms', 'RingCount'
]

def calculate_descriptors(smiles):
    """Calculate molecular descriptors for a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
            'RingCount': Descriptors.RingCount(mol),
        }
        return descriptors
    except:
        return None

def load_and_prepare_data():
    """Load PDBbind data and calculate descriptors"""
    print("Loading training data...")

    # Try multiple possible locations
    training_file = DATA_DIR / 'raw' / 'pdbbind_with_smiles.tsv'
    if not training_file.exists():
        training_file = DATA_DIR / 'processed' / 'pdbbind_classified.tsv'
    if not training_file.exists():
        training_file = Path(__file__).parent.parent.parent / 'data' / 'pdbbind_with_smiles.tsv'

    if not training_file.exists():
        print(f"ERROR: Training data not found. Looked for:")
        print(f"  - {DATA_DIR / 'raw' / 'pdbbind_with_smiles.tsv'}")
        print(f"  - {DATA_DIR / 'processed' / 'pdbbind_classified.tsv'}")
        return None, None

    print(f"Loading from: {training_file}")
    df = pd.read_csv(training_file, sep='\t')
    print(f"Loaded {len(df):,} rows")

    # Calculate descriptors
    print("Calculating molecular descriptors...")
    descriptors_list = []
    skipped = 0

    for idx, row in df.iterrows():
        if idx % 2000 == 0:
            print(f"  Progress: {idx:,}/{len(df):,}")

        smiles = row.get('canonical_smiles')
        pkd = row.get('pKd')

        if pd.isna(smiles) or pd.isna(pkd):
            skipped += 1
            continue

        desc = calculate_descriptors(smiles)
        if desc:
            desc['pKd'] = pkd
            desc['smiles'] = smiles
            descriptors_list.append(desc)
        else:
            skipped += 1

    print(f"Calculated descriptors for {len(descriptors_list):,} compounds")
    print(f"Skipped {skipped:,} invalid entries")

    train_df = pd.DataFrame(descriptors_list)

    X = train_df[FEATURE_COLS].values
    y = train_df['pKd'].values

    return X, y

def train_random_forest(X, y):
    """Train Random Forest model"""
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  CV R² scores: {scores}")
    print(f"  Mean R² = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    # Train on full data
    model.fit(X, y)

    # Training performance
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    print(f"  Full training: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    return model, {'r2': r2, 'rmse': rmse, 'mae': mae, 'cv_r2_mean': scores.mean()}

def train_xgboost(X, y):
    """Train XGBoost model"""
    if not HAS_XGBOOST:
        return None, None

    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  CV R² scores: {scores}")
    print(f"  Mean R² = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    # Train on full data
    model.fit(X, y)

    # Training performance
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    print(f"  Full training: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    return model, {'r2': r2, 'rmse': rmse, 'mae': mae, 'cv_r2_mean': scores.mean()}

def train_lightgbm(X, y):
    """Train LightGBM model"""
    if not HAS_LIGHTGBM:
        return None, None

    print("\n" + "="*60)
    print("Training LightGBM...")
    print("="*60)

    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  CV R² scores: {scores}")
    print(f"  Mean R² = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    # Train on full data
    model.fit(X, y)

    # Training performance
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    print(f"  Full training: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    return model, {'r2': r2, 'rmse': rmse, 'mae': mae, 'cv_r2_mean': scores.mean()}

def save_model(model, name, metrics, models_dir):
    """Save model and metadata to pkl file"""
    if model is None:
        return

    model_path = models_dir / f'{name}_model.pkl'

    # Save model with metadata
    model_data = {
        'model': model,
        'feature_cols': FEATURE_COLS,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
        'description': f'{name.replace("_", " ").title()} model for binding affinity prediction'
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"  Saved: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")

def main():
    print("="*60)
    print("Training ML Models for Binding Affinity Prediction")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    X, y = load_and_prepare_data()
    if X is None:
        print("ERROR: Could not load training data")
        return 1

    print(f"\nTraining data: {len(y):,} samples, {len(FEATURE_COLS)} features")
    print(f"Target range: pKd = {y.min():.2f} to {y.max():.2f}")

    # Train models
    rf_model, rf_metrics = train_random_forest(X, y)
    xgb_model, xgb_metrics = train_xgboost(X, y)
    lgb_model, lgb_metrics = train_lightgbm(X, y)

    # Save models
    print("\n" + "="*60)
    print("Saving models...")
    print("="*60)

    save_model(rf_model, 'random_forest', rf_metrics, MODELS_DIR)
    if xgb_model:
        save_model(xgb_model, 'xgboost', xgb_metrics, MODELS_DIR)
    if lgb_model:
        save_model(lgb_model, 'lightgbm', lgb_metrics, MODELS_DIR)

    # Save summary
    summary = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': len(y),
        'feature_cols': FEATURE_COLS,
        'models': {}
    }

    if rf_metrics:
        summary['models']['random_forest'] = rf_metrics
    if xgb_metrics:
        summary['models']['xgboost'] = xgb_metrics
    if lgb_metrics:
        summary['models']['lightgbm'] = lgb_metrics

    summary_path = MODELS_DIR / 'model_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Total files: {len(list(MODELS_DIR.glob('*.pkl')))} pkl, 1 json")

    return 0

if __name__ == '__main__':
    sys.exit(main())
