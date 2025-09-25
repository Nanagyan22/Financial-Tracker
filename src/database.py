"""
Database persistence layer for Financial Irregularities Detection System.
Handles CRUD operations for users, predictions, sessions, and system configuration.
"""

import os
import json
import hashlib
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import uuid

class DatabaseManager:
    """
    Comprehensive database manager for the Financial Irregularities Detection System.
    Provides CRUD operations for all system entities with proper connection management.
    """
    
    def __init__(self):
        """Initialize database manager with connection parameters."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False) -> Any:
        """Execute SQL query with parameters and return results."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                
                # Always commit first to persist changes
                conn.commit()
                
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                
                return cursor.rowcount
    
    # ==================== USER MANAGEMENT ====================
    
    def create_user(self, username: str, password: str, role: str, full_name: str, email: str) -> int:
        """Create a new user with hashed password."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        query = """
        INSERT INTO users (username, password_hash, role, full_name, email, created_at) 
        VALUES (%s, %s, %s, %s, %s, %s) 
        RETURNING id
        """
        params = (username, password_hash, role.lower(), full_name, email, datetime.now())
        
        result = self.execute_query(query, params, fetch_one=True)
        return result[0] if result else None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        query = """
        SELECT id, username, role, full_name, email, created_at, last_login 
        FROM users 
        WHERE username = %s AND password_hash = %s
        """
        
        result = self.execute_query(query, (username, password_hash), fetch_one=True)
        
        if result:
            # Update last login
            self.execute_query(
                "UPDATE users SET last_login = %s WHERE id = %s",
                (datetime.now(), result[0])
            )
            
            return {
                'id': result[0],
                'username': result[1],
                'role': result[2],
                'full_name': result[3],
                'email': result[4],
                'created_at': result[5],
                'last_login': result[6]
            }
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        query = """
        SELECT id, username, role, full_name, email, created_at, last_login 
        FROM users 
        WHERE id = %s
        """
        
        result = self.execute_query(query, (user_id,), fetch_one=True)
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'role': result[2],
                'full_name': result[3],
                'email': result[4],
                'created_at': result[5],
                'last_login': result[6]
            }
        return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all users."""
        query = """
        SELECT id, username, role, full_name, email, created_at, last_login 
        FROM users 
        ORDER BY created_at
        """
        
        results = self.execute_query(query, fetch_all=True)
        
        return [
            {
                'id': row[0],
                'username': row[1],
                'role': row[2],
                'full_name': row[3],
                'email': row[4],
                'created_at': row[5],
                'last_login': row[6]
            }
            for row in results
        ]
    
    # ==================== SESSION MANAGEMENT ====================
    
    def create_session(self, user_id: int, expires_in_hours: int = 1) -> str:
        """Create a new user session."""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        query = """
        INSERT INTO user_sessions (session_id, user_id, expires_at) 
        VALUES (%s, %s, %s)
        """
        
        self.execute_query(query, (session_id, user_id, expires_at))
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate session and return user data."""
        query = """
        SELECT u.id, u.username, u.role, u.full_name, s.expires_at
        FROM user_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_id = %s AND s.is_active = TRUE AND s.expires_at > %s
        """
        
        result = self.execute_query(query, (session_id, datetime.now()), fetch_one=True)
        
        if result:
            # Update last activity
            self.execute_query(
                "UPDATE user_sessions SET last_activity = %s WHERE session_id = %s",
                (datetime.now(), session_id)
            )
            
            return {
                'id': result[0],
                'username': result[1],
                'role': result[2],
                'full_name': result[3],
                'expires_at': result[4]
            }
        return None
    
    def invalidate_session(self, session_id: str):
        """Invalidate a session (logout)."""
        query = "UPDATE user_sessions SET is_active = FALSE WHERE session_id = %s"
        self.execute_query(query, (session_id,))
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        query = "DELETE FROM user_sessions WHERE expires_at < %s"
        count = self.execute_query(query, (datetime.now(),))
        return count
    
    # ==================== PREDICTIONS MANAGEMENT ====================
    
    def store_prediction(self, entity_name: str, entity_id: str, prediction: bool, 
                        confidence: float, risk_score: float, model_version: str = 'v1.0',
                        features: Dict = None, created_by: int = None) -> int:
        """Store a model prediction."""
        query = """
        INSERT INTO predictions (entity_name, entity_id, prediction, confidence, risk_score,
                               model_version, features, created_by) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
        RETURNING id
        """
        params = (entity_name, entity_id, prediction, confidence, risk_score, 
                 model_version, json.dumps(features) if features else None, created_by)
        
        result = self.execute_query(query, params, fetch_one=True)
        return result[0] if result else None
    
    def get_predictions(self, entity_name: str = None, limit: int = 100) -> List[Dict]:
        """Get predictions with optional filtering."""
        base_query = """
        SELECT p.id, p.entity_name, p.entity_id, p.prediction, p.confidence, p.risk_score,
               p.model_version, p.features, p.created_at, u.username
        FROM predictions p
        LEFT JOIN users u ON p.created_by = u.id
        """
        
        if entity_name:
            query = base_query + " WHERE p.entity_name = %s ORDER BY p.created_at DESC LIMIT %s"
            params = (entity_name, limit)
        else:
            query = base_query + " ORDER BY p.created_at DESC LIMIT %s"
            params = (limit,)
        
        results = self.execute_query(query, params, fetch_all=True)
        
        return [
            {
                'id': row[0],
                'entity_name': row[1],
                'entity_id': row[2],
                'prediction': row[3],
                'confidence': row[4],
                'risk_score': row[5],
                'model_version': row[6],
                'features': row[7] if row[7] else None,  # PostgreSQL JSONB already parsed
                'created_at': row[8],
                'created_by': row[9]
            }
            for row in results
        ]
    
    def get_prediction_statistics(self) -> Dict:
        """Get prediction statistics."""
        stats_query = """
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction = TRUE THEN 1 ELSE 0 END) as positive_predictions,
            AVG(confidence) as avg_confidence,
            AVG(risk_score) as avg_risk_score,
            COUNT(DISTINCT entity_name) as unique_entities
        FROM predictions
        """
        
        result = self.execute_query(stats_query, fetch_one=True)
        
        if result:
            return {
                'total_predictions': result[0],
                'positive_predictions': result[1],
                'negative_predictions': result[0] - result[1],
                'avg_confidence': float(result[2]) if result[2] else 0,
                'avg_risk_score': float(result[3]) if result[3] else 0,
                'unique_entities': result[4]
            }
        return {}
    
    # ==================== SYSTEM CONFIGURATION ====================
    
    def get_config(self, key: str) -> Optional[str]:
        """Get configuration value by key."""
        query = "SELECT config_value FROM system_config WHERE config_key = %s"
        result = self.execute_query(query, (key,), fetch_one=True)
        return result[0] if result else None
    
    def set_config(self, key: str, value: str, description: str = None, user_id: int = None):
        """Set or update configuration value."""
        query = """
        INSERT INTO system_config (config_key, config_value, description, updated_by) 
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (config_key) 
        DO UPDATE SET 
            config_value = EXCLUDED.config_value,
            description = COALESCE(EXCLUDED.description, system_config.description),
            updated_at = CURRENT_TIMESTAMP,
            updated_by = EXCLUDED.updated_by
        """
        self.execute_query(query, (key, value, description, user_id))
    
    def get_all_config(self) -> Dict:
        """Get all configuration values."""
        query = "SELECT config_key, config_value FROM system_config"
        results = self.execute_query(query, fetch_all=True)
        return {row[0]: row[1] for row in results}
    
    # ==================== AUDIT LOGGING ====================
    
    def log_activity(self, user_id: int, action: str, resource_type: str = None, 
                    resource_id: str = None, details: Dict = None):
        """Log user activity."""
        query = """
        INSERT INTO audit_log (user_id, action, resource_type, resource_id, details) 
        VALUES (%s, %s, %s, %s, %s)
        """
        params = (user_id, action, resource_type, resource_id, 
                 json.dumps(details) if details else None)
        
        self.execute_query(query, params)
    
    def get_audit_log(self, user_id: int = None, limit: int = 100) -> List[Dict]:
        """Get audit log entries."""
        base_query = """
        SELECT a.id, a.user_id, u.username, a.action, a.resource_type, a.resource_id,
               a.details, a.created_at
        FROM audit_log a
        LEFT JOIN users u ON a.user_id = u.id
        """
        
        if user_id:
            query = base_query + " WHERE a.user_id = %s ORDER BY a.created_at DESC LIMIT %s"
            params = (user_id, limit)
        else:
            query = base_query + " ORDER BY a.created_at DESC LIMIT %s"
            params = (limit,)
        
        results = self.execute_query(query, params, fetch_all=True)
        
        return [
            {
                'id': row[0],
                'user_id': row[1],
                'username': row[2],
                'action': row[3],
                'resource_type': row[4],
                'resource_id': row[5],
                'details': json.loads(row[6]) if row[6] else None,
                'created_at': row[7]
            }
            for row in results
        ]
    
    # ==================== UTILITY METHODS ====================
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        tables = ['users', 'predictions', 'user_sessions', 'system_config', 'audit_log']
        stats = {}
        
        for table in tables:
            query = f"SELECT COUNT(*) FROM {table}"
            result = self.execute_query(query, fetch_one=True)
            stats[table] = result[0] if result else 0
        
        return stats
    
    def health_check(self) -> bool:
        """Perform database health check."""
        try:
            self.execute_query("SELECT 1", fetch_one=True)
            return True
        except Exception:
            return False
    
    def initialize_schema(self) -> bool:
        """Initialize database schema with tables and indexes."""
        schema_sql = """
        -- Create Users Table
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) NOT NULL DEFAULT 'public',
            full_name VARCHAR(100),
            email VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );

        -- Create Predictions Table
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            entity_name VARCHAR(100) NOT NULL,
            entity_id VARCHAR(50),
            prediction BOOLEAN NOT NULL,
            confidence FLOAT NOT NULL,
            risk_score FLOAT,
            model_version VARCHAR(50) DEFAULT 'v1.0',
            features JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER REFERENCES users(id)
        );

        -- Create User Sessions Table
        CREATE TABLE IF NOT EXISTS user_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );

        -- Create System Configuration Table
        CREATE TABLE IF NOT EXISTS system_config (
            id SERIAL PRIMARY KEY,
            config_key VARCHAR(100) UNIQUE NOT NULL,
            config_value TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by INTEGER REFERENCES users(id)
        );

        -- Create Audit Log Table
        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            action VARCHAR(100) NOT NULL,
            resource_type VARCHAR(50),
            resource_id VARCHAR(100),
            details JSONB,
            ip_address INET,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_predictions_entity ON predictions(entity_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);
        CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
        CREATE INDEX IF NOT EXISTS idx_config_key ON system_config(config_key);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(schema_sql)
                conn.commit()
            print("✅ Database schema initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Schema initialization failed: {e}")
            return False

# Global database manager instance
db_manager = None

def get_db_manager():
    """Get or create database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def init_database():
    """Initialize database connection, schema, and seed data."""
    try:
        db = get_db_manager()
        if not db.health_check():
            print("❌ Database health check failed")
            return False
            
        print("✅ Database connection successful")
        
        # Initialize schema
        if not db.initialize_schema():
            print("❌ Schema initialization failed")
            return False
            
        # Migrate users from JSON if needed
        migrate_users_from_json(db)
        
        # Seed initial configuration
        seed_initial_config(db)
        
        # Show final stats
        stats = db.get_database_stats()
        print(f"📊 Database Stats: {stats}")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def migrate_users_from_json(db):
    """Migrate users from JSON file to database."""
    try:
        # Check if users already exist in database
        existing_users = db.get_all_users()
        if len(existing_users) >= 3:
            print("✅ Users already migrated to database")
            return
            
        # Load users from JSON
        json_file = "data/users.json"
        if not os.path.exists(json_file):
            print("⚠️ No JSON user file found to migrate")
            return
            
        with open(json_file, 'r') as f:
            users_data = json.load(f)
        
        # Migrate each user
        migrated_count = 0
        for username, user_info in users_data.items():
            try:
                # Insert user with existing password hash
                query = """
                INSERT INTO users (username, password_hash, role, full_name, email, created_at, last_login) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (username) DO NOTHING
                """
                params = (
                    user_info['username'],
                    user_info['password'],  # Already hashed
                    user_info['role'].lower(),
                    user_info['full_name'],
                    user_info['email'],
                    user_info['created_at'].split('T')[0] if 'T' in user_info['created_at'] else user_info['created_at'],
                    user_info.get('last_login', user_info['created_at']).split('T')[0] if user_info.get('last_login') and 'T' in user_info.get('last_login') else user_info.get('last_login')
                )
                
                if db.execute_query(query, params):
                    migrated_count += 1
                    
            except Exception as e:
                print(f"⚠️ Failed to migrate user {username}: {e}")
                
        print(f"✅ Migrated {migrated_count} users from JSON to database")
        
    except Exception as e:
        print(f"❌ User migration failed: {e}")

def seed_initial_config(db):
    """Seed initial system configuration."""
    try:
        initial_config = {
            'model_version': 'v1.0',
            'prediction_threshold': '0.5',
            'session_timeout': '3600',
            'monitoring_alerts': 'true',
            'precision_k_values': '10,20,50'
        }
        
        for key, value in initial_config.items():
            db.set_config(key, value, f"Initial configuration for {key}")
            
        print("✅ Initial system configuration seeded")
        
    except Exception as e:
        print(f"⚠️ Configuration seeding failed: {e}")

if __name__ == "__main__":
    # Test database connection
    if init_database():
        db = get_db_manager()
        
        # Test user operations
        users = db.get_all_users()
        print(f"Found {len(users)} users")
        
        # Test configuration
        configs = db.get_all_config()
        print(f"Configuration: {configs}")
        
        print("🎉 Database layer fully operational!")
    else:
        print("💥 Database layer failed to initialize")