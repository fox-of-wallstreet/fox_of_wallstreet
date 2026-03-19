import streamlit as st
import requests

# Set this to your mapped Docker port (e.g., 8081 or 8080)
BACKEND_URL = "http://localhost:8081" 

st.set_page_config(page_title="Fox of Wallstreet Control", layout="wide")
st.title("🦊 Fox of Wallstreet: Task Control")

# --- Initialize Session State for persistent logs ---
for task in ["data", "train", "trade"]:
    if f"{task}_logs" not in st.session_state:
        st.session_state[f"{task}_logs"] = ""

def run_task(endpoint, display_name, params):
    """Calls FastAPI, streams logs, and saves them to session state."""
    url = f"{BACKEND_URL}/{endpoint}"
    
    # Clear previous logs for this specific task before starting
    st.session_state[f"{endpoint}_logs"] = ""
    
    with st.status(f"Running {display_name}...", expanded=True) as status:
        try:
            with requests.get(url, params={"params": params}, stream=True, timeout=3600) as r:
                if r.status_code != 200:
                    st.error(f"Backend Error: {r.status_code}")
                    return

                # Create a placeholder to show live updates
                log_placeholder = st.empty()
                
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8") + "\n"
                        # Append to session state so it persists
                        st.session_state[f"{endpoint}_logs"] += decoded_line
                        # Update the live UI
                        log_placeholder.code(st.session_state[f"{endpoint}_logs"])
                
            status.update(label=f"{display_name} Finished", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Connection failed: {e}")
            status.update(label="Failed", state="error")

# --- UI Layout ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("1. Data Engine")
    data_params = st.text_input("Data Args", "", key="data_input")
    if st.button("Start Data Sync", use_container_width=True):
        run_task("data", "Data Engine", data_params)
    # Display persistent logs from session state
    st.code(st.session_state["data_logs"] if st.session_state["data_logs"] else "No logs yet.")

with col2:
    st.header("2. Training")
    train_params = st.text_input("Train Args", "", key="train_input")
    if st.button("Start Training", use_container_width=True):
        run_task("train", "Model Trainer", train_params)
    st.code(st.session_state["train_logs"] if st.session_state["train_logs"] else "No logs yet.")

with col3:
    st.header("3. Live Trader")
    trade_params = st.text_input("Trade Args", "--bot", key="trade_input")
    if st.button("Start Live Trading", use_container_width=True):
        run_task("trade", "Live Trader", trade_params)
    st.code(st.session_state["trade_logs"] if st.session_state["trade_logs"] else "No logs yet.")
