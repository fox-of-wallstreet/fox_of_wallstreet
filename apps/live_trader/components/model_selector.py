"""
Model selector component for Live Trader app.
Allows users to browse and select trained models.
"""

import os
import sys
import streamlit as st

# Add shared and parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from shared.utils.model_discovery import (
    list_available_models,
    format_model_display_name,
    get_model_action_space,
    validate_model_compatibility,
)
from config import settings


def render_model_selector():
    """
    Render the model selection dropdown with metadata preview.
    Returns the selected model dict or None.
    """
    st.subheader("🧠 Select AI Model")
    
    # Get available models
    models = list_available_models()
    
    if not models:
        st.warning("No trained models found in artifacts/ directory. Run training first.")
        return None
    
    # Filter to models with both model.zip and scaler.pkl
    valid_models = [m for m in models if m["has_model"] and m["has_scaler"]]
    
    if not valid_models:
        st.warning("Found artifact folders but no complete models (need model.zip + scaler.pkl)")
        return None
    
    # Create display names
    model_options = {format_model_display_name(m): m for m in valid_models}
    
    # Dropdown
    selected_name = st.selectbox(
        "Choose a trained model:",
        options=list(model_options.keys()),
        index=0,
        help="Select from available trained models in artifacts/"
    )
    
    selected_model = model_options[selected_name]
    
    # Show model details
    with st.expander("📋 Model Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Info:**")
            st.write(f"- Name: `{selected_model['name']}`")
            st.write(f"- Created: {selected_model['timestamp'].strftime('%Y-%m-%d %H:%M') if selected_model['timestamp'] else 'Unknown'}")
            st.write(f"- Has Backtest: {'✅' if selected_model['has_ledger'] else '❌'}")
        
        with col2:
            st.write("**Configuration:**")
            parsed = selected_model["parsed"]
            st.write(f"- Symbol: {parsed.get('symbol', 'N/A')}")
            st.write(f"- Timeframe: {parsed.get('timeframe', 'N/A')}")
            st.write(f"- Action Space: {parsed.get('action', 'N/A')}")
            st.write(f"- Features: News={parsed.get('news')}, Macro={parsed.get('macro')}, Time={parsed.get('time')}")
        
        # Metadata section
        if selected_model["metadata"]:
            st.write("**Training Metadata:**")
            meta = selected_model["metadata"]
            
            col3, col4 = st.columns(2)
            with col3:
                st.write(f"- Timesteps: {meta.get('total_timesteps', 'N/A'):,}")
                st.write(f"- Learning Rate: {meta.get('learning_rate', 'N/A')}")
                st.write(f"- Gamma: {meta.get('gamma', 'N/A')}")
            with col4:
                st.write(f"- Batch Size: {meta.get('batch_size', 'N/A')}")
                st.write(f"- Ent Coef: {meta.get('ent_coef', 'N/A')}")
                st.write(f"- N Stack: {meta.get('n_stack', 'N/A')}")
            
            # Feature list
            features = meta.get("features_used", [])
            if features:
                st.write(f"**Features ({len(features)}):**")
                st.write(", ".join(features[:10]) + ("..." if len(features) > 10 else ""))
    
    # Compatibility check
    is_compatible, mismatches = validate_model_compatibility(
        selected_model["path"],
        {
            "SYMBOL": settings.SYMBOL,
            "TIMEFRAME": settings.TIMEFRAME,
            "ACTION_SPACE_TYPE": settings.ACTION_SPACE_TYPE,
            "REWARD_STRATEGY": settings.REWARD_STRATEGY,
            "FEATURES_LIST": settings.FEATURES_LIST,
        }
    )
    
    if not is_compatible:
        st.warning("⚠️ Model may not be compatible with current settings:")
        for m in mismatches:
            st.write(f"  - {m}")
        st.info("You can still load the model, but behavior may be unpredictable.")
    
    # Load button
    if st.button("🔄 Load Selected Model", type="primary"):
        return selected_model
    
    return None


def render_loaded_model_status():
    """Show status of currently loaded model."""
    if "loaded_model" not in st.session_state or st.session_state["loaded_model"] is None:
        st.info("No model loaded. Select and load a model to begin.")
        return False
    
    model_info = st.session_state.get("model_info", {})
    
    st.success(f"✅ Model loaded: {model_info.get('name', 'Unknown')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Action Space:** {model_info.get('action_space', 'Unknown')}")
    with col2:
        st.write(f"**Features:** {model_info.get('feature_count', 'Unknown')}")
    with col3:
        st.write(f"**Trained:** {model_info.get('train_dates', 'Unknown')}")
    
    return True


def load_model_to_session(model_dict: dict):
    """Load model and store in session state."""
    import joblib
    from stable_baselines3 import PPO
    
    model_path = os.path.join(model_dict["path"], "model")
    scaler_path = os.path.join(model_dict["path"], "scaler.pkl")
    
    with st.spinner("Loading model... This may take a moment."):
        try:
            # Load model
            model = PPO.load(model_path)
            
            # Load scaler
            scaler = joblib.load(scaler_path)
            
            # Store in session
            st.session_state["loaded_model"] = model
            st.session_state["scaler"] = scaler
            # Extract metadata
            metadata = model_dict.get("metadata", {})
            
            st.session_state["model_info"] = {
                "name": model_dict["name"],
                "path": model_dict["path"],
                "action_space": metadata.get("action_space", "unknown"),
                "symbol": metadata.get("symbol", settings.SYMBOL),  # ← Store symbol!
                "timeframe": metadata.get("timeframe", settings.TIMEFRAME),
                "feature_count": len(metadata.get("features_used", [])),
                "train_dates": metadata.get("train_dates", "Unknown"),
            }
            
            # Reset portfolio for new model
            st.session_state["portfolio"] = {
                "cash": settings.INITIAL_BALANCE,
                "position": 0.0,
                "entry_price": 0.0,
                "last_action": 0,
            }
            st.session_state["last_ai_decision"] = None
            
            st.success(f"✅ Model '{model_dict['name']}' loaded successfully!")
            st.balloons()
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
