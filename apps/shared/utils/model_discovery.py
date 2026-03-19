"""
Shared utility for discovering trained models across both apps.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add parent project to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from config import settings


def list_available_models(base_dir: Optional[str] = None) -> List[Dict]:
    """
    Scan artifacts directory for trained models.
    
    Returns:
        List of dicts with model info:
        {
            'name': folder name,
            'path': full path,
            'timestamp': datetime,
            'has_model': bool,
            'has_ledger': bool,
            'metadata': dict or None,
        }
    """
    base_dir = base_dir or settings.ARTIFACTS_BASE_DIR
    
    if not os.path.exists(base_dir):
        return []
    
    models = []
    run_pattern = re.compile(
        r"^ppo_(?P<symbol>[A-Z]+)_(?P<timeframe>\w+)_(?P<action>discrete_[35])_"
        r"(?P<news>news|nonews)_(?P<macro>macro|nomacro)_(?P<time>time|notime)_(?P<ts>\d{8}_\d{4})$"
    )
    
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
            
        match = run_pattern.match(name)
        if not match:
            continue
        
        # Parse timestamp
        try:
            ts = datetime.strptime(match.group("ts"), "%Y%m%d_%H%M")
        except ValueError:
            ts = None
        
        # Check for required files
        has_model = os.path.exists(os.path.join(path, "model.zip"))
        has_scaler = os.path.exists(os.path.join(path, "scaler.pkl"))
        has_ledger = os.path.exists(os.path.join(path, "backtest_ledger.csv"))
        
        # Load metadata if available
        metadata_path = os.path.join(path, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                pass
        
        models.append({
            "name": name,
            "path": path,
            "timestamp": ts,
            "has_model": has_model,
            "has_scaler": has_scaler,
            "has_ledger": has_ledger,
            "metadata": metadata,
            "parsed": match.groupdict(),
        })
    
    # Sort by timestamp, newest first
    models.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)
    return models


def get_model_by_name(name: str, base_dir: Optional[str] = None) -> Optional[Dict]:
    """Get specific model info by folder name."""
    models = list_available_models(base_dir)
    for model in models:
        if model["name"] == name:
            return model
    return None


def get_latest_model(symbol: Optional[str] = None, 
                     timeframe: Optional[str] = None,
                     base_dir: Optional[str] = None) -> Optional[Dict]:
    """Get the most recent compatible model."""
    models = list_available_models(base_dir)
    
    for model in models:
        if not model["has_model"]:
            continue
        if symbol and model["parsed"]["symbol"] != symbol:
            continue
        if timeframe and model["parsed"]["timeframe"] != timeframe:
            continue
        return model
    
    return None


def format_model_display_name(model: Dict) -> str:
    """Create human-readable name for dropdown."""
    name = model["name"]
    ts = model["timestamp"]
    
    if ts:
        date_str = ts.strftime("%b %d, %H:%M")
    else:
        date_str = "Unknown date"
    
    # Extract key info
    parsed = model["parsed"]
    action = parsed.get("action", "unknown")
    symbol = parsed.get("symbol", "?")
    
    # Add return if available
    return_info = ""
    if model["has_ledger"]:
        # Could compute actual return here
        return_info = " 📊"
    
    return f"{symbol} {action} ({date_str}){return_info}"


def get_model_action_space(model_path: str) -> Optional[str]:
    """Read action space from model metadata."""
    metadata_path = os.path.join(model_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("action_space")
    except (json.JSONDecodeError, IOError):
        return None


def validate_model_compatibility(model_path: str, 
                                  current_settings: Dict) -> tuple[bool, List[str]]:
    """
    Check if a model is compatible with current settings.
    
    Returns:
        (is_compatible, list_of_mismatches)
    """
    metadata_path = os.path.join(model_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return False, ["No metadata.json found"]
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        return False, ["Invalid metadata.json"]
    
    mismatches = []
    
    # Key fields to compare
    checks = [
        ("symbol", "SYMBOL"),
        ("timeframe", "TIMEFRAME"),
        ("action_space", "ACTION_SPACE_TYPE"),
        ("reward_strategy", "REWARD_STRATEGY"),
    ]
    
    for meta_key, settings_key in checks:
        meta_val = metadata.get(meta_key)
        curr_val = current_settings.get(settings_key)
        if meta_val != curr_val:
            mismatches.append(f"{meta_key}: model={meta_val}, current={curr_val}")
    
    # Check feature count
    meta_features = metadata.get("features_used", [])
    curr_features = current_settings.get("FEATURES_LIST", [])
    if set(meta_features) != set(curr_features):
        mismatches.append(f"Feature mismatch: model has {len(meta_features)} features, "
                         f"current config has {len(curr_features)}")
    
    return len(mismatches) == 0, mismatches
