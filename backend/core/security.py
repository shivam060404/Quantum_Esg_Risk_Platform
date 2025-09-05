from datetime import datetime, timedelta
from typing import Any, Union, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib
import hmac
from .config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Security
security = HTTPBearer()

class SecurityManager:
    """Centralized security management"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(
        subject: Union[str, Any], 
        expires_delta: timedelta = None,
        additional_claims: dict = None
    ) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.access_token_expire_minutes
            )
        
        to_encode = {
            "exp": expire,
            "sub": str(subject),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        if additional_claims:
            to_encode.update(additional_claims)
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(subject: Union[str, Any]) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
        
        to_encode = {
            "exp": expire,
            "sub": str(subject),
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.secret_key,
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm]
            )
            
            # Check token type
            token_type = payload.get("type")
            if token_type not in ["access", "refresh"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            return payload
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not validate credentials: {str(e)}"
            )
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_secure_hash(data: str, salt: str = None) -> str:
        """Generate secure hash with optional salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return f"{salt}:{hash_obj.hex()}"
    
    @staticmethod
    def verify_secure_hash(data: str, hashed_data: str) -> bool:
        """Verify data against secure hash"""
        try:
            salt, hash_hex = hashed_data.split(':')
            hash_obj = hashlib.pbkdf2_hmac(
                'sha256',
                data.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return hmac.compare_digest(hash_obj.hex(), hash_hex)
        except ValueError:
            return False

class RoleBasedAccessControl:
    """Role-based access control system"""
    
    # Define role hierarchy
    ROLES = {
        "admin": 100,
        "manager": 80,
        "analyst": 60,
        "user": 40,
        "viewer": 20
    }
    
    # Define permissions
    PERMISSIONS = {
        "portfolio.create": ["admin", "manager", "analyst"],
        "portfolio.read": ["admin", "manager", "analyst", "user", "viewer"],
        "portfolio.update": ["admin", "manager", "analyst"],
        "portfolio.delete": ["admin", "manager"],
        
        "quantum.optimize": ["admin", "manager", "analyst"],
        "quantum.view_results": ["admin", "manager", "analyst", "user"],
        
        "blockchain.verify": ["admin", "manager", "analyst"],
        "blockchain.view_history": ["admin", "manager", "analyst", "user"],
        
        "compliance.run_analysis": ["admin", "manager", "analyst"],
        "compliance.view_reports": ["admin", "manager", "analyst", "user"],
        
        "climate.stress_test": ["admin", "manager", "analyst"],
        "climate.view_results": ["admin", "manager", "analyst", "user"],
        
        "admin.user_management": ["admin"],
        "admin.system_config": ["admin"],
        "admin.audit_logs": ["admin", "manager"]
    }
    
    @classmethod
    def has_permission(cls, user_role: str, permission: str) -> bool:
        """Check if user role has specific permission"""
        allowed_roles = cls.PERMISSIONS.get(permission, [])
        return user_role in allowed_roles
    
    @classmethod
    def has_role_level(cls, user_role: str, required_level: int) -> bool:
        """Check if user role meets minimum level requirement"""
        user_level = cls.ROLES.get(user_role, 0)
        return user_level >= required_level
    
    @classmethod
    def get_user_permissions(cls, user_role: str) -> list:
        """Get all permissions for a user role"""
        permissions = []
        for permission, allowed_roles in cls.PERMISSIONS.items():
            if user_role in allowed_roles:
                permissions.append(permission)
        return permissions

class DataEncryption:
    """Data encryption utilities"""
    
    @staticmethod
    def encrypt_sensitive_data(data: str, key: str = None) -> str:
        """Encrypt sensitive data"""
        from cryptography.fernet import Fernet
        
        if key is None:
            key = settings.secret_key[:32].ljust(32, '0')  # Ensure 32 bytes
        
        # Generate Fernet key from secret
        import base64
        fernet_key = base64.urlsafe_b64encode(key.encode()[:32].ljust(32, b'0'))
        f = Fernet(fernet_key)
        
        encrypted_data = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, key: str = None) -> str:
        """Decrypt sensitive data"""
        from cryptography.fernet import Fernet
        import base64
        
        if key is None:
            key = settings.secret_key[:32].ljust(32, '0')
        
        fernet_key = base64.urlsafe_b64encode(key.encode()[:32].ljust(32, b'0'))
        f = Fernet(fernet_key)
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = f.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {str(e)}")

class AuditLogger:
    """Security audit logging"""
    
    @staticmethod
    async def log_security_event(
        event_type: str,
        user_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        details: dict = None
    ):
        """Log security-related events"""
        from database.models import AuditLog
        from database.database import AsyncSessionLocal
        import uuid
        
        try:
            async with AsyncSessionLocal() as session:
                audit_entry = AuditLog(
                    id=uuid.uuid4(),
                    user_id=uuid.UUID(user_id) if user_id else None,
                    action=event_type,
                    entity_type="security",
                    description=f"Security event: {event_type}",
                    metadata=details or {},
                    ip_address=ip_address,
                    user_agent=user_agent,
                    created_at=datetime.utcnow()
                )
                
                session.add(audit_entry)
                await session.commit()
                
        except Exception as e:
            # Log to file if database logging fails
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to log security event: {str(e)}")

class SecurityMiddleware:
    """Security middleware for request validation"""
    
    @staticmethod
    def validate_request_signature(request_body: str, signature: str, secret: str) -> bool:
        """Validate request signature for webhook security"""
        expected_signature = hmac.new(
            secret.encode(),
            request_body.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
    
    @staticmethod
    def check_rate_limit(identifier: str, limit: int = 100, window: int = 60) -> bool:
        """Check rate limiting (simplified implementation)"""
        # In production, use Redis or similar for distributed rate limiting
        import time
        from collections import defaultdict
        
        # Simple in-memory rate limiting (not suitable for production)
        if not hasattr(SecurityMiddleware, '_rate_limit_store'):
            SecurityMiddleware._rate_limit_store = defaultdict(list)
        
        now = time.time()
        requests = SecurityMiddleware._rate_limit_store[identifier]
        
        # Remove old requests outside the window
        requests[:] = [req_time for req_time in requests if now - req_time < window]
        
        # Check if limit exceeded
        if len(requests) >= limit:
            return False
        
        # Add current request
        requests.append(now)
        return True
    
    @staticmethod
    def sanitize_input(input_data: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        import html
        import re
        
        # HTML escape
        sanitized = html.escape(input_data)
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'onclick='
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized

# Utility functions for common security operations
def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    return SecurityManager.verify_token(token)

def create_access_token(subject: str, **kwargs) -> str:
    """Create access token"""
    return SecurityManager.create_access_token(subject, **kwargs)

def hash_password(password: str) -> str:
    """Hash password"""
    return SecurityManager.get_password_hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return SecurityManager.verify_password(plain_password, hashed_password)

def check_permission(user_role: str, permission: str) -> bool:
    """Check if user has permission"""
    return RoleBasedAccessControl.has_permission(user_role, permission)

# Security decorators
from functools import wraps
from fastapi import Request

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from request context
            # This would be implemented based on your authentication system
            user_role = kwargs.get('current_user', {}).get('role', 'viewer')
            
            if not check_permission(user_role, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(required_role: str):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_role = kwargs.get('current_user', {}).get('role', 'viewer')
            required_level = RoleBasedAccessControl.ROLES.get(required_role, 100)
            
            if not RoleBasedAccessControl.has_role_level(user_role, required_level):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient role level. Required: {required_role}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Export commonly used functions
__all__ = [
    "SecurityManager",
    "RoleBasedAccessControl",
    "DataEncryption",
    "AuditLogger",
    "SecurityMiddleware",
    "verify_token",
    "create_access_token",
    "hash_password",
    "verify_password",
    "check_permission",
    "require_permission",
    "require_role"
]