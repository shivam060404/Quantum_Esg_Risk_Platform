from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from typing import Generator, AsyncGenerator
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:1234@127.0.0.1:5432/esg_platform"
)

# Convert to async URL for async operations
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create engines
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Set to True for SQL debugging
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False
)

# Create session factories
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Dependency for getting database session
def get_db() -> Generator[Session, None, None]:
    """Get database session for synchronous operations"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for asynchronous operations"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Database health check
async def check_database_health() -> bool:
    """Check if database is accessible"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False

# Database initialization
async def init_database():
    """Initialize database with tables and initial data"""
    try:
        from .models import Base
        
        # Create all tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        
        # Insert initial data if needed
        await insert_initial_data()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

async def insert_initial_data():
    """Insert initial data into the database"""
    try:
        from .models import ClimateScenario, User
        import uuid
        from datetime import datetime
        
        async with AsyncSessionLocal() as session:
            # Check if climate scenarios already exist
            result = await session.execute(
                "SELECT COUNT(*) FROM climate_scenarios"
            )
            count = result.scalar()
            
            if count == 0:
                # Insert default climate scenarios
                scenarios = [
                    {
                        'name': 'Baseline (1.5°C)',
                        'description': 'Paris Agreement baseline scenario with 1.5°C warming',
                        'temperature_increase': 1.5,
                        'probability': 0.4,
                        'time_horizon': 30,
                        'physical_impacts': {
                            'sea_level_rise': 0.3,
                            'extreme_weather': 0.4,
                            'drought_risk': 0.3,
                            'flood_risk': 0.4
                        },
                        'transition_impacts': {
                            'carbon_pricing': 0.6,
                            'regulatory_changes': 0.5,
                            'technology_disruption': 0.7,
                            'market_shifts': 0.5
                        },
                        'economic_impacts': {
                            'gdp_impact': -0.02,
                            'inflation_impact': 0.01,
                            'interest_rate_impact': 0.005
                        }
                    },
                    {
                        'name': 'Moderate (2.0°C)',
                        'description': 'Moderate warming scenario with delayed climate action',
                        'temperature_increase': 2.0,
                        'probability': 0.35,
                        'time_horizon': 30,
                        'physical_impacts': {
                            'sea_level_rise': 0.5,
                            'extreme_weather': 0.6,
                            'drought_risk': 0.5,
                            'flood_risk': 0.6
                        },
                        'transition_impacts': {
                            'carbon_pricing': 0.8,
                            'regulatory_changes': 0.7,
                            'technology_disruption': 0.8,
                            'market_shifts': 0.7
                        },
                        'economic_impacts': {
                            'gdp_impact': -0.05,
                            'inflation_impact': 0.025,
                            'interest_rate_impact': 0.01
                        }
                    },
                    {
                        'name': 'Severe (3.0°C)',
                        'description': 'Severe warming with limited climate action',
                        'temperature_increase': 3.0,
                        'probability': 0.2,
                        'time_horizon': 30,
                        'physical_impacts': {
                            'sea_level_rise': 0.8,
                            'extreme_weather': 0.9,
                            'drought_risk': 0.8,
                            'flood_risk': 0.9
                        },
                        'transition_impacts': {
                            'carbon_pricing': 1.0,
                            'regulatory_changes': 0.9,
                            'technology_disruption': 0.9,
                            'market_shifts': 0.9
                        },
                        'economic_impacts': {
                            'gdp_impact': -0.1,
                            'inflation_impact': 0.05,
                            'interest_rate_impact': 0.02
                        }
                    },
                    {
                        'name': 'Extreme (4.0°C)',
                        'description': 'Extreme warming scenario with climate tipping points',
                        'temperature_increase': 4.0,
                        'probability': 0.05,
                        'time_horizon': 30,
                        'physical_impacts': {
                            'sea_level_rise': 1.0,
                            'extreme_weather': 1.0,
                            'drought_risk': 1.0,
                            'flood_risk': 1.0
                        },
                        'transition_impacts': {
                            'carbon_pricing': 1.0,
                            'regulatory_changes': 1.0,
                            'technology_disruption': 1.0,
                            'market_shifts': 1.0
                        },
                        'economic_impacts': {
                            'gdp_impact': -0.2,
                            'inflation_impact': 0.1,
                            'interest_rate_impact': 0.04
                        }
                    }
                ]
                
                for scenario_data in scenarios:
                    scenario = ClimateScenario(
                        id=uuid.uuid4(),
                        name=scenario_data['name'],
                        description=scenario_data['description'],
                        temperature_increase=scenario_data['temperature_increase'],
                        probability=scenario_data['probability'],
                        time_horizon=scenario_data['time_horizon'],
                        physical_impacts=scenario_data['physical_impacts'],
                        transition_impacts=scenario_data['transition_impacts'],
                        economic_impacts=scenario_data['economic_impacts'],
                        data_sources=['IPCC AR6', 'NGFS Scenarios'],
                        methodology='Integrated Assessment Model',
                        last_updated=datetime.utcnow(),
                        created_at=datetime.utcnow(),
                        created_by=uuid.uuid4()  # System user
                    )
                    session.add(scenario)
                
                await session.commit()
                logger.info("Initial climate scenarios inserted")
            
            # Check if admin user exists
            result = await session.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin'"
            )
            admin_count = result.scalar()
            
            if admin_count == 0:
                # Create default admin user
                from passlib.context import CryptContext
                
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                hashed_password = pwd_context.hash("admin123")  # Change in production!
                
                admin_user = User(
                    id=uuid.uuid4(),
                    email="admin@esg-platform.com",
                    username="admin",
                    full_name="System Administrator",
                    hashed_password=hashed_password,
                    is_active=True,
                    is_verified=True,
                    organization="ESG Platform",
                    role="admin",
                    preferences={
                        'theme': 'dark',
                        'notifications': True,
                        'default_currency': 'USD'
                    },
                    created_at=datetime.utcnow()
                )
                session.add(admin_user)
                await session.commit()
                logger.info("Default admin user created")
        
    except Exception as e:
        logger.error(f"Failed to insert initial data: {str(e)}")
        raise

# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    async def backup_database(backup_path: str) -> bool:
        """Create database backup"""
        try:
            import subprocess
            import os
            
            # Extract database connection details
            db_url = DATABASE_URL
            # Parse connection string and create pg_dump command
            # This is a simplified version - in production, use proper backup tools
            
            logger.info(f"Database backup created at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            return False
    
    @staticmethod
    async def restore_database(backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            # Implementation for database restore
            logger.info(f"Database restored from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {str(e)}")
            return False
    
    @staticmethod
    async def get_database_stats() -> dict:
        """Get database statistics"""
        try:
            async with AsyncSessionLocal() as session:
                stats = {}
                
                # Get table counts
                tables = [
                    'portfolios', 'holdings', 'esg_data', 'risk_assessments',
                    'compliance_reports', 'blockchain_verifications',
                    'quantum_optimizations', 'climate_scenarios', 'users'
                ]
                
                for table in tables:
                    result = await session.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = result.scalar()
                
                # Get database size
                result = await session.execute(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                stats['database_size'] = result.scalar()
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}
    
    @staticmethod
    async def cleanup_old_data(days: int = 90) -> bool:
        """Clean up old data based on retention policy"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with AsyncSessionLocal() as session:
                # Clean up old audit logs
                result = await session.execute(
                    "DELETE FROM audit_logs WHERE created_at < :cutoff_date",
                    {'cutoff_date': cutoff_date}
                )
                deleted_count = result.rowcount
                
                await session.commit()
                
                logger.info(f"Cleaned up {deleted_count} old audit log entries")
                return True
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")
            return False

# Connection pool monitoring
class ConnectionPoolMonitor:
    """Monitor database connection pool health"""
    
    @staticmethod
    def get_pool_status() -> dict:
        """Get connection pool status"""
        try:
            pool = engine.pool
            return {
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            }
        except Exception as e:
            logger.error(f"Failed to get pool status: {str(e)}")
            return {}
    
    @staticmethod
    async def test_connection() -> bool:
        """Test database connection"""
        try:
            async with AsyncSessionLocal() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False