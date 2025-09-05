from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

Base = declarative_base()

class Portfolio(Base):
    """Portfolio model for ESG risk assessment"""
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    total_value = Column(Float, nullable=False, default=0.0)
    currency = Column(String(3), default="USD")
    investment_strategy = Column(String(100))
    risk_tolerance = Column(String(20), default="medium")
    esg_mandate = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="portfolio")
    compliance_reports = relationship("ComplianceReport", back_populates="portfolio")
    
    # Indexes
    __table_args__ = (
        Index('idx_portfolio_user_id', 'user_id'),
        Index('idx_portfolio_created_at', 'created_at'),
    )

class Holding(Base):
    """Individual holdings within a portfolio"""
    __tablename__ = "holdings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    name = Column(String(255))
    asset_type = Column(String(50), default="equity")  # equity, bond, commodity, etc.
    sector = Column(String(100))
    industry = Column(String(100))
    country = Column(String(3))  # ISO country code
    weight = Column(Float, nullable=False)  # Portfolio weight (0-1)
    shares = Column(Float)
    market_value = Column(Float)
    cost_basis = Column(Float)
    currency = Column(String(3), default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")
    esg_data = relationship("ESGData", back_populates="holding")
    
    # Indexes
    __table_args__ = (
        Index('idx_holding_portfolio_id', 'portfolio_id'),
        Index('idx_holding_symbol', 'symbol'),
        Index('idx_holding_sector', 'sector'),
    )

class ESGData(Base):
    """ESG data for individual holdings"""
    __tablename__ = "esg_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    holding_id = Column(UUID(as_uuid=True), ForeignKey("holdings.id"), nullable=False)
    company_id = Column(String(50))  # External company identifier
    
    # ESG Scores
    environmental_score = Column(Float)
    social_score = Column(Float)
    governance_score = Column(Float)
    overall_esg_score = Column(Float)
    
    # Environmental Metrics
    carbon_emissions = Column(Float)  # tCO2e
    carbon_intensity = Column(Float)  # tCO2e per $M revenue
    water_usage = Column(Float)  # ML
    waste_generation = Column(Float)  # tonnes
    renewable_energy_usage = Column(Float)  # percentage
    
    # Social Metrics
    employee_satisfaction = Column(Float)
    diversity_ratio = Column(Float)
    safety_incidents = Column(Integer)
    community_investment = Column(Float)
    
    # Governance Metrics
    board_independence = Column(Float)
    executive_compensation_ratio = Column(Float)
    audit_quality_score = Column(Float)
    transparency_score = Column(Float)
    
    # Data Quality
    data_quality_score = Column(Float)
    last_updated = Column(DateTime)
    data_sources = Column(JSONB)  # List of data sources
    verification_status = Column(String(20), default="pending")
    blockchain_hash = Column(String(64))  # Blockchain verification hash
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    holding = relationship("Holding", back_populates="esg_data")
    
    # Indexes
    __table_args__ = (
        Index('idx_esg_holding_id', 'holding_id'),
        Index('idx_esg_company_id', 'company_id'),
        Index('idx_esg_overall_score', 'overall_esg_score'),
        Index('idx_esg_verification_status', 'verification_status'),
    )

class RiskAssessment(Base):
    """Risk assessment results for portfolios"""
    __tablename__ = "risk_assessments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    assessment_type = Column(String(50), nullable=False)  # climate, quantum, compliance
    
    # Risk Metrics
    overall_risk_score = Column(Float)
    risk_level = Column(String(20))  # low, medium, high, critical
    var_95 = Column(Float)  # Value at Risk 95%
    var_99 = Column(Float)  # Value at Risk 99%
    expected_shortfall = Column(Float)
    max_drawdown = Column(Float)
    
    # Climate Risk Specific
    physical_risk_score = Column(Float)
    transition_risk_score = Column(Float)
    climate_var = Column(Float)
    
    # Quantum Analysis
    quantum_enhanced = Column(Boolean, default=False)
    quantum_speedup = Column(Float)
    quantum_circuit_depth = Column(Integer)
    
    # Assessment Details
    methodology = Column(String(100))
    scenarios_tested = Column(JSONB)  # List of scenarios
    assessment_parameters = Column(JSONB)
    detailed_results = Column(JSONB)
    
    # Metadata
    execution_time = Column(Float)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="risk_assessments")
    
    # Indexes
    __table_args__ = (
        Index('idx_risk_portfolio_id', 'portfolio_id'),
        Index('idx_risk_assessment_type', 'assessment_type'),
        Index('idx_risk_created_at', 'created_at'),
        Index('idx_risk_level', 'risk_level'),
    )

class ComplianceReport(Base):
    """Compliance monitoring reports"""
    __tablename__ = "compliance_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    
    # Compliance Scores
    overall_compliance_score = Column(Float)
    compliance_status = Column(String(30))  # compliant, partially_compliant, non_compliant
    
    # Regulatory Compliance
    sec_compliance = Column(JSONB)
    eu_taxonomy_compliance = Column(JSONB)
    tcfd_alignment = Column(JSONB)
    sasb_coverage = Column(JSONB)
    
    # Issues and Recommendations
    identified_issues = Column(JSONB)
    recommendations = Column(JSONB)
    remediation_actions = Column(JSONB)
    
    # AI Analysis
    ai_analysis_summary = Column(Text)
    risk_drivers = Column(JSONB)
    compliance_gaps = Column(JSONB)
    
    # Review Information
    next_review_date = Column(DateTime)
    review_frequency = Column(String(20), default="quarterly")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="compliance_reports")
    
    # Indexes
    __table_args__ = (
        Index('idx_compliance_portfolio_id', 'portfolio_id'),
        Index('idx_compliance_status', 'compliance_status'),
        Index('idx_compliance_created_at', 'created_at'),
        Index('idx_compliance_next_review', 'next_review_date'),
    )

class BlockchainVerification(Base):
    """Blockchain verification records"""
    __tablename__ = "blockchain_verifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(50), nullable=False)  # esg_data, portfolio, assessment
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    
    # Blockchain Details
    transaction_hash = Column(String(66), unique=True)
    block_number = Column(Integer)
    block_hash = Column(String(66))
    contract_address = Column(String(42))
    gas_used = Column(Integer)
    
    # Verification Details
    data_hash = Column(String(64), nullable=False)
    verification_score = Column(Float)
    verification_status = Column(String(20), default="pending")
    
    # External Verification
    external_sources_verified = Column(Integer, default=0)
    consensus_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    verified_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_blockchain_entity', 'entity_type', 'entity_id'),
        Index('idx_blockchain_tx_hash', 'transaction_hash'),
        Index('idx_blockchain_status', 'verification_status'),
        Index('idx_blockchain_created_at', 'created_at'),
    )

class QuantumOptimization(Base):
    """Quantum optimization results"""
    __tablename__ = "quantum_optimizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    
    # Optimization Parameters
    optimization_type = Column(String(50), default="portfolio_allocation")
    objective_function = Column(String(100))
    constraints = Column(JSONB)
    esg_weights = Column(JSONB)
    
    # Quantum Details
    algorithm = Column(String(50), default="QAOA")
    num_qubits = Column(Integer)
    circuit_depth = Column(Integer)
    gate_count = Column(Integer)
    quantum_volume = Column(Integer)
    
    # Results
    optimal_allocations = Column(JSONB)
    optimization_score = Column(Float)
    convergence_iterations = Column(Integer)
    
    # Performance Metrics
    expected_return = Column(Float)
    portfolio_risk = Column(Float)
    sharpe_ratio = Column(Float)
    esg_score = Column(Float)
    
    # Execution Details
    execution_time = Column(Float)
    quantum_speedup = Column(Float)
    classical_comparison = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True))
    
    # Indexes
    __table_args__ = (
        Index('idx_quantum_portfolio_id', 'portfolio_id'),
        Index('idx_quantum_algorithm', 'algorithm'),
        Index('idx_quantum_created_at', 'created_at'),
    )

class ClimateScenario(Base):
    """Climate scenarios for stress testing"""
    __tablename__ = "climate_scenarios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Scenario Parameters
    temperature_increase = Column(Float)  # degrees Celsius
    probability = Column(Float)  # 0-1
    time_horizon = Column(Integer)  # years
    
    # Impact Factors
    physical_impacts = Column(JSONB)
    transition_impacts = Column(JSONB)
    economic_impacts = Column(JSONB)
    
    # Scenario Details
    data_sources = Column(JSONB)
    methodology = Column(String(100))
    last_updated = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True))
    
    # Indexes
    __table_args__ = (
        Index('idx_scenario_name', 'name'),
        Index('idx_scenario_temp_increase', 'temperature_increase'),
    )

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True)
    full_name = Column(String(255))
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Profile
    organization = Column(String(255))
    role = Column(String(50), default="user")
    preferences = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_username', 'username'),
        Index('idx_user_created_at', 'created_at'),
    )

class AuditLog(Base):
    """Audit log for tracking system activities"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True))
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(UUID(as_uuid=True))
    
    # Details
    description = Column(Text)
    changes = Column(JSONB)  # Before/after values
    extra_data = Column(JSONB)  # Additional context
    
    # Request Information
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_user_id', 'user_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_created_at', 'created_at'),
    )

# Database initialization function
async def init_db():
    """Initialize database tables"""
    from .database import engine
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("Database tables created successfully")