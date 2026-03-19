import streamlit as st
from google.oauth2 import service_account
from google.cloud import run_v2
from google.protobuf import field_mask_pb2

# --- CONFIGURATION ---
PROJECT_ID = "your-project-id"
REGION = "us-central1"
SERVICE_NAME = "live-trader-bot"

st.set_page_config(page_title="Trader Command Center", page_icon="📈")

# --- AUTHENTICATION ---
@st.cache_resource
def get_run_client():
    # Streamlit automatically parses the 'gcp_service_account' section from Secrets
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    return run_v2.ServicesClient(credentials=credentials)

client = get_run_client()
service_path = f"projects/{PROJECT_ID}/locations/{REGION}/services/{SERVICE_NAME}"

# --- APP LOGIC ---
st.title("🐍 Live Trader Control")

def get_current_status():
    service = client.get_service(name=service_path)
    return service.template.scaling.min_instance_count

def update_bot(min_instances):
    service = client.get_service(name=service_path)
    service.template.scaling.min_instance_count = min_instances
    
    # We must tell GCP exactly which field we are changing
    update_mask = field_mask_pb2.FieldMask(paths=["template.scaling.min_instance_count"])
    
    operation = client.update_service(service=service, update_mask=update_mask)
    with st.spinner(f"Communicating with Google Cloud..."):
        operation.result()
    st.success(f"Bot successfully set to {min_instances} instances!")

# --- UI ELEMENTS ---
current_min = get_current_status()

# Status Indicator
if current_min >= 1:
    st.success("🟢 BOT STATUS: LIVE & TRADING")
else:
    st.error("🔴 BOT STATUS: ASLEEP (STOCKED)")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Start Trader", use_container_width=True):
        update_bot(1)
        st.rerun()

with col2:
    if st.button("🛑 Stop Trader", use_container_width=True, type="primary"):
        update_bot(0)
        st.rerun()

st.divider()
st.info(f"Connected to Service: `{SERVICE_NAME}` in `{REGION}`")
