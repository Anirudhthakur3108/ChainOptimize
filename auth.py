import streamlit as st
import bcrypt
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    """Handles user authentication and session management"""
    
    def __init__(self):
        """Initialize authentication manager with mock users"""
        self.users_data = self._create_mock_users()
    
    def _create_mock_users(self):
        """Create mock user data matching the users table schema"""
        users = [
            {
                'user_id': 1,
                'username': 'admin',
                'password_hash': self._hash_password('admin123'),
                'email': 'admin@company.com',
                'role': 'admin',
                'created_at': datetime(2024, 1, 1),
                'last_login': None
            },
            {
                'user_id': 2,
                'username': 'manager1',
                'password_hash': self._hash_password('manager123'),
                'email': 'manager1@company.com',
                'role': 'manager',
                'created_at': datetime(2024, 1, 2),
                'last_login': None
            },
            {
                'user_id': 3,
                'username': 'viewer1',
                'password_hash': self._hash_password('viewer123'),
                'email': 'viewer1@company.com',
                'role': 'viewer',
                'created_at': datetime(2024, 1, 3),
                'last_login': None
            }
        ]
        return pd.DataFrame(users)
    
    def _hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password, hashed):
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials"""
        try:
            user = self.users_data[self.users_data['username'] == username]
            if user.empty:
                return False, "User not found"
            
            user_row = user.iloc[0]
            if self._verify_password(password, user_row['password_hash']):
                # Update last login
                self.users_data.loc[self.users_data['username'] == username, 'last_login'] = datetime.now()
                return True, user_row['role']
            else:
                return False, "Invalid password"
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, "Authentication failed"
    
    def register_user(self, username, password, email, role='viewer'):
        """Register new user"""
        try:
            # Check if username already exists
            if not self.users_data[self.users_data['username'] == username].empty:
                return False, "Username already exists"
            
            # Check if email already exists
            if not self.users_data[self.users_data['email'] == email].empty:
                return False, "Email already exists"
            
            # Create new user
            new_user = {
                'user_id': self.users_data['user_id'].max() + 1,
                'username': username,
                'password_hash': self._hash_password(password),
                'email': email,
                'role': role,
                'created_at': datetime.now(),
                'last_login': None
            }
            
            # Add to users data
            self.users_data = pd.concat([self.users_data, pd.DataFrame([new_user])], ignore_index=True)
            return True, "User registered successfully"
        
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False, "Registration failed"
    
    def remove_user(self, username, requesting_user_role, requesting_username):
        """Remove user with hierarchy permission checks"""
        try:
            # Cannot remove yourself
            if username == requesting_username:
                return False, "Cannot remove your own account"
            
            # Find user to remove
            user_to_remove = self.users_data[self.users_data['username'] == username]
            if user_to_remove.empty:
                return False, "User not found"
            
            user_role = user_to_remove.iloc[0]['role']
            
            # Hierarchy permission checks
            if requesting_user_role == 'manager':
                # Managers can only remove viewers
                if user_role != 'viewer':
                    return False, "Managers can only remove viewers"
            elif requesting_user_role == 'admin':
                # Admins can remove anyone except themselves
                pass
            else:
                return False, "Insufficient permissions to remove users"
            
            # Remove user
            self.users_data = self.users_data[self.users_data['username'] != username].reset_index(drop=True)
            return True, f"User '{username}' removed successfully"
            
        except Exception as e:
            logger.error(f"User removal error: {str(e)}")
            return False, "Failed to remove user"
    
    def get_user_count_by_role(self):
        """Get count of users by role"""
        try:
            role_counts = self.users_data['role'].value_counts().to_dict()
            total_users = len(self.users_data)
            return {
                'total': total_users,
                'admin': role_counts.get('admin', 0),
                'manager': role_counts.get('manager', 0),
                'viewer': role_counts.get('viewer', 0)
            }
        except Exception as e:
            logger.error(f"Error getting user counts: {str(e)}")
            return {'total': 0, 'admin': 0, 'manager': 0, 'viewer': 0}
    
    def get_all_users(self, requesting_user_role):
        """Get all users based on role permissions"""
        try:
            if requesting_user_role == 'admin':
                # Admins can see all users
                return self.users_data[['username', 'email', 'role', 'created_at', 'last_login']].copy()
            elif requesting_user_role == 'manager':
                # Managers can see viewers and themselves
                return self.users_data[
                    self.users_data['role'].isin(['viewer', 'manager'])
                ][['username', 'email', 'role', 'created_at', 'last_login']].copy()
            else:
                return pd.DataFrame()  # Viewers cannot see other users
        except Exception as e:
            logger.error(f"Error getting users: {str(e)}")
            return pd.DataFrame()
    
    def show_login_form(self):
        """Display login form"""
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            login_button = st.form_submit_button("Login", type="primary")
            
            if login_button:
                if username and password:
                    success, message = self.authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_role = message
                        st.session_state.username = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {message}")
                else:
                    st.error("Please enter both username and password")
        
        # Display default credentials
        st.info("""
        **Default User Accounts:**
        - Admin: username=`admin`, password=`admin123`
        - Manager: username=`manager1`, password=`manager123`
        - Viewer: username=`viewer1`, password=`viewer123`
        """)
    
    def show_register_form(self):
        """Display registration form - only accessible to logged in users"""
        if not st.session_state.get('authenticated', False):
            st.error("❌ You must be logged in to register new users")
            return
            
        user_role = st.session_state.get('user_role', 'viewer')
        
        # Role hierarchy check
        if user_role not in ['admin', 'manager']:
            st.error("❌ Access denied. Only Admin or Manager can register new users.")
            return
        
        with st.form("register_form"):
            st.subheader("Register New User")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Role selection based on hierarchy
            if user_role == 'admin':
                available_roles = ["viewer", "manager", "admin"]
            else:  # manager
                available_roles = ["viewer"]
            
            role = st.selectbox("Role", available_roles)
            
            register_button = st.form_submit_button("Register", type="primary")
            
            if register_button:
                if not all([username, email, password, confirm_password]):
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    success, message = self.register_user(username, password, email, role)
                    if success:
                        st.success(message)
                        st.info("User registered successfully and can now login")
                    else:
                        st.error(f"Registration failed: {message}")
    
    def logout(self):
        """Logout user and clear session"""
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.success("Logged out successfully!")
        st.rerun()
    
    def get_users_data(self):
        """Get users data for admin dashboard"""
        return self.users_data.copy()
