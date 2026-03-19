"""
Simple password protection for demo deployments.

Usage:
    from utils.auth import require_auth
    require_auth()  # Add at top of each page
"""

import os
import streamlit as st
from typing import Optional

# You can set this in Streamlit Secrets or .env
# If not set, no password protection (for local dev)
DEFAULT_DEMO_PASSWORD = "fox2024"  # Change this for your demo!


def get_password() -> str:
    """Get password from environment or use default."""
    # Priority: DEMO_PASSWORD env var > Streamlit Secret > default
    return os.getenv(
        "DEMO_PASSWORD",
        os.getenv("STREAMLIT_DEMO_PASSWORD", DEFAULT_DEMO_PASSWORD)
    )


def require_auth() -> bool:
    """
    Require password authentication.
    
    Returns True if authenticated, False otherwise.
    Stops execution if not authenticated.
    """
    # Skip auth if no password configured (local dev)
    password = get_password()
    if not password:
        return True
    
    # Check if already authenticated
    if st.session_state.get("demo_authenticated", False):
        return True
    
    # Show login form
    st.set_page_config(
        page_title="Fox of Wallstreet - Login",
        page_icon="🔒",
        layout="centered",
    )
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🦊 Fox of Wallstreet")
        st.subheader("Live AI Trader")
        
        st.divider()
        
        st.markdown("### 🔒 Demo Access")
        st.write("Please enter the demo password to continue.")
        
        # Password input
        entered_password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter demo password...",
            key="demo_password_input",
        )
        
        # Login button
        if st.button("🔓 Login", type="primary", use_container_width=True):
            if entered_password == password:
                st.session_state["demo_authenticated"] = True
                st.success("✅ Access granted! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
        
        st.divider()
        
        # Help section
        with st.expander("💡 Need access?"):
            st.write("""
            This is a password-protected demo.
            
            Contact the administrator for access:
            - Email: admin@example.com
            - Or request the demo password
            """)
    
    # Stop execution if not authenticated
    st.stop()
    return False


def logout():
    """Log out the current user."""
    if "demo_authenticated" in st.session_state:
        del st.session_state["demo_authenticated"]
    st.rerun()


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    password = get_password()
    if not password:
        return True
    return st.session_state.get("demo_authenticated", False)


def show_logout_button():
    """Show logout button in sidebar."""
    password = get_password()
    if password and st.session_state.get("demo_authenticated", False):
        st.sidebar.divider()
        if st.sidebar.button("🔒 Logout", use_container_width=True):
            logout()
