import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from auth import AuthManager
from database import DatabaseManager
from dashboard import DashboardManager
from ml_models import MLManager
from utils import Utils
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Supply Chain Optimization Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize managers
@st.cache_resource
def initialize_managers():
    """Initialize all manager classes"""
    auth_manager = AuthManager()
    db_manager = DatabaseManager()
    dashboard_manager = DashboardManager(db_manager)
    ml_manager = MLManager(db_manager)
    utils = Utils()
    return auth_manager, db_manager, dashboard_manager, ml_manager, utils

def show_user_management(auth_manager, user_role, username):
    """Enhanced user management interface with counts and removal"""
    st.header("👥 User Management")
    
    # Get user statistics
    user_counts = auth_manager.get_user_count_by_role()
    
    # Display user statistics
    st.subheader("📊 User Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", user_counts['total'])
    with col2:
        st.metric("Admins", user_counts['admin'])
    with col3:
        st.metric("Managers", user_counts['manager'])
    with col4:
        st.metric("Viewers", user_counts['viewer'])
    
    st.markdown("---")
    
    # Tabs for different management functions
    tab1, tab2 = st.tabs(["👥 Manage Existing Users", "➕ Register New User"])
    
    with tab1:
        st.subheader("Current Users")
        
        # Get users based on role permissions
        users_df = auth_manager.get_all_users(user_role)
        
        if not users_df.empty:
            # Display users table
            st.dataframe(
                users_df.style.format({
                    'created_at': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'N/A',
                    'last_login': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'Never'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            st.subheader("🗑️ Remove User")
            
            # User removal section
            if user_role == 'admin':
                removable_users = users_df[users_df['username'] != username]['username'].tolist()
                help_text = "Admins can remove any user except themselves"
            else:  # manager
                removable_users = users_df[
                    (users_df['role'] == 'viewer') & (users_df['username'] != username)
                ]['username'].tolist()
                help_text = "Managers can only remove viewers"
            
            if removable_users:
                selected_user = st.selectbox(
                    "Select user to remove:",
                    removable_users,
                    help=help_text
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🗑️ Remove User", type="secondary"):
                        success, message = auth_manager.remove_user(
                            selected_user, user_role, username
                        )
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                with col2:
                    st.warning(f"⚠️ This will permanently remove user '{selected_user}' from the system.")
            else:
                st.info("No users available for removal based on your permissions.")
        else:
            st.info("No users to display based on your permission level.")
    
    with tab2:
        # Registration form
        auth_manager.show_register_form()

def main():
    """Main application function"""
    try:
        # Initialize managers
        auth_manager, db_manager, dashboard_manager, ml_manager, utils = initialize_managers()
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'username' not in st.session_state:
            st.session_state.username = None

        # Authentication check
        if not st.session_state.authenticated:
            st.title("🔐 Supply Chain Optimization Platform")
            st.markdown("### Please login to access the platform")
            
            auth_manager.show_login_form()
            
            return

        # Main application interface
        st.title("📦 Supply Chain Optimization Platform")
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown(f"**Welcome, {st.session_state.username}!**")
            st.markdown(f"*Role: {st.session_state.user_role.title()}*")
            
            st.markdown("---")
            
            # Navigation menu
            page = st.selectbox(
                "Navigate to:",
                ["Dashboard Overview", "Inventory Management", "Order Analytics", 
                 "Supplier Analytics", "Demand Forecasting", "ML Model Training", 
                 "User Management", "Reports & Export"],
                key="navigation"
            )
            
            st.markdown("---")
            
            # User actions
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
            
            if st.button("🚪 Logout"):
                auth_manager.logout()

        # Role-based access control
        user_role = st.session_state.user_role
        
        # Page routing
        if page == "Dashboard Overview":
            dashboard_manager.show_overview_dashboard()
        
        elif page == "Inventory Management":
            if user_role in ['admin', 'manager']:
                dashboard_manager.show_inventory_dashboard()
            else:
                st.error("❌ Access denied. Manager or Admin role required.")
        
        elif page == "Order Analytics":
            dashboard_manager.show_order_analytics()
        
        elif page == "Supplier Analytics":
            if user_role in ['admin', 'manager']:
                dashboard_manager.show_supplier_analytics()
            else:
                st.error("❌ Access denied. Manager or Admin role required.")
        
        elif page == "Demand Forecasting":
            dashboard_manager.show_demand_forecasting(ml_manager)
        
        elif page == "ML Model Training":
            if user_role == 'admin':
                ml_manager.show_model_training_interface()
            else:
                st.error("❌ Access denied. Admin role required.")
        
        elif page == "User Management":
            if user_role in ['admin', 'manager']:
                show_user_management(auth_manager, user_role, st.session_state.username)
            else:
                st.error("❌ Access denied. Manager or Admin role required.")
        
        elif page == "Reports & Export":
            if user_role in ['admin', 'manager']:
                dashboard_manager.show_reports_export()
            else:
                st.error("❌ Access denied. Manager or Admin role required.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
