"""
User Authentication and Authorization System

This module provides user authentication and role-based access control
for the Financial Irregularities Detection System.
"""

import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

class UserRole(Enum):
    """User role definitions with access levels."""
    PUBLIC = "public"
    AUDITOR = "auditor"
    ADMINISTRATOR = "administrator"

class AuthenticationManager:
    """
    Manages user authentication and authorization.
    """
    
    def __init__(self, users_file: str = "data/users.json"):
        self.users_file = users_file
        self.session_timeout = 3600  # 1 hour in seconds
        
        # Try to use database backend, fallback to JSON if unavailable
        try:
            from database import get_db_manager
            self.db = get_db_manager()
            self.use_database = True
            print("✅ Authentication using database backend")
        except Exception as e:
            print(f"⚠️ Database unavailable, using JSON fallback: {e}")
            self.db = None
            self.use_database = False
            
        self.users = self.load_users()
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_info' not in st.session_state:
            st.session_state.user_info = None
        if 'login_time' not in st.session_state:
            st.session_state.login_time = None
    
    def load_users(self) -> Dict:
        """Load users from file or create default users."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
        
        # Create default users if file doesn't exist
        default_users = self.create_default_users()
        self.save_users(default_users)
        return default_users
    
    def create_default_users(self) -> Dict:
        """Create default user accounts."""
        default_users = {
            "admin": {
                "username": "admin",
                "password": self.hash_password("admin123"),
                "role": UserRole.ADMINISTRATOR.value,
                "full_name": "System Administrator",
                "email": "admin@auditor-general.gov.gh",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "active": True
            },
            "auditor": {
                "username": "auditor",
                "password": self.hash_password("auditor123"),
                "role": UserRole.AUDITOR.value,
                "full_name": "Senior Auditor",
                "email": "auditor@auditor-general.gov.gh",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "active": True
            },
            "public": {
                "username": "public",
                "password": self.hash_password("public123"),
                "role": UserRole.PUBLIC.value,
                "full_name": "Public User",
                "email": "public@ghana.gov.gh",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "active": True
            }
        }
        return default_users
    
    def save_users(self, users: Dict) -> None:
        """Save users to file."""
        try:
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.hash_password(password) == hashed
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """Authenticate user credentials."""
        if self.use_database and self.db:
            # Use database authentication
            try:
                user_data = self.db.authenticate_user(username, password)
                if user_data:
                    return True, user_data
                return False, None
            except Exception as e:
                print(f"Database authentication failed, falling back to JSON: {e}")
                # Fall through to JSON authentication
        
        # JSON fallback authentication
        if username in self.users:
            user = self.users[username]
            if user.get('active', False) and self.verify_password(password, user['password']):
                # Update last login
                user['last_login'] = datetime.now().isoformat()
                self.users[username] = user
                self.save_users(self.users)
                return True, user
        return False, None
    
    def login(self, username: str, password: str) -> bool:
        """Login user and set session state."""
        success, user_info = self.authenticate_user(username, password)
        
        if success:
            st.session_state.authenticated = True
            st.session_state.user_info = user_info
            st.session_state.login_time = datetime.now()
            return True
        return False
    
    def logout(self) -> None:
        """Logout user and clear session state."""
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.login_time = None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated and session is valid."""
        if not st.session_state.authenticated:
            return False
        
        # Check session timeout
        if st.session_state.login_time:
            elapsed = datetime.now() - st.session_state.login_time
            if elapsed.total_seconds() > self.session_timeout:
                self.logout()
                return False
        
        return True
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user information."""
        if self.is_authenticated():
            return st.session_state.user_info
        return None
    
    def get_user_role(self) -> Optional[UserRole]:
        """Get current user role."""
        user = self.get_current_user()
        if user:
            try:
                # Convert role to lowercase to match UserRole enum values
                return UserRole(user['role'].lower())
            except ValueError:
                return None
        return None
    
    def has_permission(self, required_role: UserRole) -> bool:
        """Check if current user has required permission level."""
        current_role = self.get_user_role()
        if not current_role:
            return False
        
        # Role hierarchy: PUBLIC < AUDITOR < ADMINISTRATOR
        role_hierarchy = {
            UserRole.PUBLIC: 1,
            UserRole.AUDITOR: 2,
            UserRole.ADMINISTRATOR: 3
        }
        
        current_level = role_hierarchy.get(current_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return current_level >= required_level
    
    def require_authentication(self) -> bool:
        """Decorator-like function to require authentication."""
        if not self.is_authenticated():
            self.show_login_form()
            return False
        return True
    
    def require_role(self, required_role: UserRole) -> bool:
        """Decorator-like function to require specific role."""
        if not self.require_authentication():
            return False
        
        if not self.has_permission(required_role):
            self.show_access_denied(required_role)
            return False
        
        return True
    
    def show_login_form(self) -> None:
        """Display login form."""
        st.markdown("# 🔐 Login Required")
        st.markdown("Please log in to access the Financial Irregularities Detection System")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.markdown("### Login to Ghana Financial Irregularities Detection System")
                
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                if st.form_submit_button("Login", use_container_width=True):
                    if self.login(username, password):
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                
                # Show demo credentials
                st.markdown("---")
                st.markdown("### 🔍 Demo Credentials")
                st.info("""
                **Administrator:** admin / admin123  
                **Auditor:** auditor / auditor123  
                **Public:** public / public123
                """)
    
    def show_access_denied(self, required_role: UserRole) -> None:
        """Display access denied message."""
        current_user = self.get_current_user()
        user_name = current_user.get('full_name', 'Unknown') if current_user else 'Unknown'
        current_role = self.get_user_role()
        
        st.error("🚫 Access Denied")
        st.warning(f"""
        **User:** {user_name}  
        **Current Role:** {current_role.value.title() if current_role else 'Unknown'}  
        **Required Role:** {required_role.value.title()}
        
        You don't have sufficient permissions to access this section.
        """)
        
        if st.button("🔙 Go Back"):
            st.rerun()
    
    def show_user_info(self) -> None:
        """Display current user information in sidebar."""
        if self.is_authenticated():
            user = self.get_current_user()
            if user:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 👤 Current User")
                st.sidebar.write(f"**Name:** {user.get('full_name', 'Unknown')}")
                st.sidebar.write(f"**Role:** {user.get('role', 'Unknown').title()}")
                st.sidebar.write(f"**Email:** {user.get('email', 'Unknown')}")
                
                if st.sidebar.button("🚪 Logout"):
                    self.logout()
                    st.rerun()
                
                # Session info
                if st.session_state.login_time:
                    elapsed = datetime.now() - st.session_state.login_time
                    remaining = self.session_timeout - elapsed.total_seconds()
                    if remaining > 0:
                        st.sidebar.write(f"**Session:** {int(remaining/60)}min remaining")
                    else:
                        st.sidebar.write("**Session:** Expired")

def get_page_permissions() -> Dict[str, UserRole]:
    """Define page access permissions."""
    return {
        "app.py": UserRole.PUBLIC,  # Main application/data processing
        "01_home.py": UserRole.PUBLIC,  # Home page
        "02_dashboard.py": UserRole.PUBLIC,  # Dashboard
        "03_entity_profile.py": UserRole.PUBLIC,  # Entity profiles
        "04_case_viewer.py": UserRole.AUDITOR,  # Case analysis
        "05_admin.py": UserRole.ADMINISTRATOR,  # Admin panel
    }

def check_page_access(page_name: str, auth_manager: AuthenticationManager) -> bool:
    """Check if current user can access the page."""
    permissions = get_page_permissions()
    required_role = permissions.get(page_name, UserRole.ADMINISTRATOR)
    return auth_manager.require_role(required_role)

def init_authentication() -> AuthenticationManager:
    """Initialize authentication manager."""
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthenticationManager()
    return st.session_state.auth_manager

def create_navigation_menu(auth_manager: AuthenticationManager) -> None:
    """Create role-based navigation menu."""
    if not auth_manager.is_authenticated():
        return
    
    user_role = auth_manager.get_user_role()
    
    st.sidebar.markdown("### 🧭 Navigation")
    
    # Page order: Home → Dashboard → Entity Profile → Case Viewer → Admin
    # Always available pages
    st.sidebar.page_link("pages/01_home.py", label="🏠 Home Page")
    st.sidebar.page_link("pages/02_dashboard.py", label="📊 Dashboard")
    st.sidebar.page_link("pages/03_entity_profile.py", label="🏢 Entity Profile")
    
    # Auditor level and above
    if auth_manager.has_permission(UserRole.AUDITOR):
        st.sidebar.page_link("pages/04_case_viewer.py", label="🔍 Case Viewer")
    
    # Administrator only
    if auth_manager.has_permission(UserRole.ADMINISTRATOR):
        st.sidebar.page_link("pages/05_admin.py", label="⚙️ Admin Panel")
    
    # Main application (data processing pipeline) - additional option
    st.sidebar.markdown("---")
    st.sidebar.page_link("app.py", label="🔧 Main Application")

# Global authentication instance
_auth_manager = None

def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager

if __name__ == "__main__":
    # Test authentication system
    auth = AuthenticationManager()
    print("Authentication system initialized")
    print(f"Default users created: {list(auth.users.keys())}")
    
    # Test authentication
    success, user = auth.authenticate_user("admin", "admin123")
    print(f"Admin authentication test: {success}")
    if success:
        print(f"Admin user info: {user['full_name']} ({user['role']})")