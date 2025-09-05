from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Application
    app_name: str = "Quantum-Enhanced ESG Risk Assessment Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    backend_cors_origins: List[str] = [
        "http://localhost:3000",
        "https://localhost:3000",
        "http://localhost:8000",
        "https://localhost:8000"
    ]
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/esg_platform"
    database_echo: bool = False
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # API Keys
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    
    # Blockchain
    blockchain_rpc_url: str = "http://localhost:8545"
    ethereum_private_key: Optional[str] = None
    polygon_rpc_url: str = "https://polygon-rpc.com"
    contract_address: Optional[str] = None
    
    # Quantum Computing
    ibm_quantum_token: Optional[str] = None
    qiskit_backend: str = "qasm_simulator"
    quantum_max_qubits: int = 16
    
    # ESG Data Sources
    bloomberg_api_key: Optional[str] = None
    refinitiv_api_key: Optional[str] = None
    msci_api_key: Optional[str] = None
    
    # File Storage
    upload_dir: str = "./uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [".pdf", ".csv", ".xlsx", ".json"]
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # AI/ML Configuration
    max_portfolio_size: int = 1000  # Maximum number of holdings
    default_risk_tolerance: float = 0.5
    quantum_simulation_shots: int = 8192
    monte_carlo_simulations: int = 10000
    
    # Climate Risk
    default_time_horizon: int = 10  # years
    climate_scenarios: List[str] = ["baseline", "moderate", "severe", "extreme"]
    
    # Compliance
    compliance_check_frequency: int = 30  # days
    regulatory_frameworks: List[str] = ["SEC", "EU_TAXONOMY", "TCFD", "SASB"]
    
    # Performance
    worker_processes: int = 4
    max_concurrent_requests: int = 100
    request_timeout: int = 300  # seconds
    
    # Feature Flags
    enable_quantum_optimization: bool = True
    enable_blockchain_verification: bool = True
    enable_ai_compliance: bool = True
    enable_climate_stress_testing: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_database_url(self) -> str:
        """Get database URL with proper formatting"""
        return self.database_url
    
    def get_async_database_url(self) -> str:
        """Get async database URL"""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() == "production"
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_production():
            # In production, only allow specific domains
            return [
                "https://your-production-domain.com",
                "https://app.your-domain.com"
            ]
        return self.backend_cors_origins

class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    environment: str = "development"
    database_echo: bool = True
    log_level: str = "DEBUG"

class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    environment: str = "production"
    database_echo: bool = False
    log_level: str = "WARNING"
    
class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    environment: str = "testing"
    database_url: str = "postgresql://postgres:password@localhost:5432/esg_platform_test"
    
@lru_cache()
def get_settings() -> Settings:
    """Get application settings based on environment"""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Global settings instance
settings = get_settings()

# Configuration validation
def validate_configuration():
    """Validate critical configuration settings"""
    errors = []
    
    # Check required API keys for production
    if settings.is_production():
        required_keys = [
            ("openai_api_key", "OpenAI API key"),
            ("secret_key", "Secret key"),
        ]
        
        for key, description in required_keys:
            if not getattr(settings, key) or getattr(settings, key) == "your-secret-key-change-in-production":
                errors.append(f"Missing or default {description} in production")
    
    # Check database URL format
    if not settings.database_url.startswith(("postgresql://", "sqlite://")):
        errors.append("Invalid database URL format")
    
    # Check quantum configuration
    if settings.enable_quantum_optimization and settings.quantum_max_qubits < 4:
        errors.append("Quantum max qubits should be at least 4 for meaningful optimization")
    
    # Check file upload limits
    if settings.max_file_size > 100 * 1024 * 1024:  # 100MB
        errors.append("Max file size should not exceed 100MB")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

# Environment-specific configurations
class QuantumConfig:
    """Quantum computing specific configuration"""
    
    @staticmethod
    def get_backend_config(backend_name: str = None) -> dict:
        """Get quantum backend configuration"""
        backend_name = backend_name or settings.qiskit_backend
        
        configs = {
            "qasm_simulator": {
                "shots": settings.quantum_simulation_shots,
                "memory": True,
                "max_parallel_threads": 4
            },
            "statevector_simulator": {
                "shots": 1,
                "memory": False
            },
            "ibmq_qasm_simulator": {
                "shots": min(settings.quantum_simulation_shots, 8192),
                "memory": True
            }
        }
        
        return configs.get(backend_name, configs["qasm_simulator"])

class BlockchainConfig:
    """Blockchain specific configuration"""
    
    @staticmethod
    def get_network_config(network: str = "development") -> dict:
        """Get blockchain network configuration"""
        configs = {
            "development": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 1337,
                "gas_limit": 6721975,
                "gas_price": 20000000000  # 20 gwei
            },
            "polygon": {
                "rpc_url": settings.polygon_rpc_url,
                "chain_id": 137,
                "gas_limit": 20000000,
                "gas_price": 30000000000  # 30 gwei
            },
            "ethereum": {
                "rpc_url": "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
                "chain_id": 1,
                "gas_limit": 21000,
                "gas_price": 50000000000  # 50 gwei
            }
        }
        
        return configs.get(network, configs["development"])

class AIConfig:
    """AI/ML specific configuration"""
    
    @staticmethod
    def get_model_config(model_type: str) -> dict:
        """Get AI model configuration"""
        configs = {
            "compliance_agent": {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
                "timeout": 60
            },
            "climate_agent": {
                "model": "gpt-4",
                "temperature": 0.2,
                "max_tokens": 1500,
                "timeout": 90
            },
            "portfolio_optimizer": {
                "algorithm": "QAOA",
                "max_iterations": 100,
                "convergence_threshold": 1e-6
            }
        }
        
        return configs.get(model_type, {})

# Export commonly used settings
__all__ = [
    "settings",
    "get_settings",
    "validate_configuration",
    "QuantumConfig",
    "BlockchainConfig",
    "AIConfig"
]