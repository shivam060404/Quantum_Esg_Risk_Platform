-- ESG Risk Assessment Platform Database Initialization
-- TimescaleDB extensions and initial data setup

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable additional extensions for advanced functionality
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create time-series tables for ESG data
CREATE TABLE IF NOT EXISTS esg_time_series (
    time TIMESTAMPTZ NOT NULL,
    company_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    environmental_score DOUBLE PRECISION,
    social_score DOUBLE PRECISION,
    governance_score DOUBLE PRECISION,
    overall_esg_score DOUBLE PRECISION,
    carbon_emissions DOUBLE PRECISION,
    carbon_intensity DOUBLE PRECISION,
    water_usage DOUBLE PRECISION,
    waste_generation DOUBLE PRECISION,
    renewable_energy_usage DOUBLE PRECISION,
    employee_satisfaction DOUBLE PRECISION,
    diversity_ratio DOUBLE PRECISION,
    board_independence DOUBLE PRECISION,
    data_source VARCHAR(100),
    data_quality_score DOUBLE PRECISION
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('esg_time_series', 'time', if_not_exists => TRUE);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_esg_time_series_company_time ON esg_time_series (company_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_esg_time_series_symbol_time ON esg_time_series (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_esg_time_series_overall_score ON esg_time_series (overall_esg_score, time DESC);

-- Create portfolio performance time series
CREATE TABLE IF NOT EXISTS portfolio_performance (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id UUID NOT NULL,
    total_value DOUBLE PRECISION,
    daily_return DOUBLE PRECISION,
    cumulative_return DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    esg_score DOUBLE PRECISION,
    carbon_footprint DOUBLE PRECISION,
    quantum_optimized BOOLEAN DEFAULT FALSE,
    optimization_score DOUBLE PRECISION
);

-- Convert to hypertable
SELECT create_hypertable('portfolio_performance', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_id_time ON portfolio_performance (portfolio_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_return ON portfolio_performance (daily_return, time DESC);

-- Create climate risk time series
CREATE TABLE IF NOT EXISTS climate_risk_data (
    time TIMESTAMPTZ NOT NULL,
    region VARCHAR(100) NOT NULL,
    risk_type VARCHAR(50) NOT NULL, -- physical, transition, acute, chronic
    temperature_anomaly DOUBLE PRECISION,
    precipitation_anomaly DOUBLE PRECISION,
    sea_level_change DOUBLE PRECISION,
    extreme_weather_events INTEGER,
    carbon_price DOUBLE PRECISION,
    policy_stringency_index DOUBLE PRECISION,
    renewable_energy_share DOUBLE PRECISION,
    economic_impact DOUBLE PRECISION,
    data_source VARCHAR(100)
);

-- Convert to hypertable
SELECT create_hypertable('climate_risk_data', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_climate_risk_region_time ON climate_risk_data (region, time DESC);
CREATE INDEX IF NOT EXISTS idx_climate_risk_type_time ON climate_risk_data (risk_type, time DESC);

-- Create quantum optimization results time series
CREATE TABLE IF NOT EXISTS quantum_optimization_history (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id UUID NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    num_qubits INTEGER,
    circuit_depth INTEGER,
    execution_time DOUBLE PRECISION,
    quantum_speedup DOUBLE PRECISION,
    optimization_score DOUBLE PRECISION,
    convergence_iterations INTEGER,
    expected_return DOUBLE PRECISION,
    portfolio_risk DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    esg_score DOUBLE PRECISION,
    success BOOLEAN DEFAULT TRUE
);

-- Convert to hypertable
SELECT create_hypertable('quantum_optimization_history', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_quantum_history_portfolio_time ON quantum_optimization_history (portfolio_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_quantum_history_algorithm ON quantum_optimization_history (algorithm, time DESC);

-- Create blockchain verification time series
CREATE TABLE IF NOT EXISTS blockchain_verification_history (
    time TIMESTAMPTZ NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    transaction_hash VARCHAR(66),
    block_number BIGINT,
    verification_score DOUBLE PRECISION,
    gas_used INTEGER,
    verification_time DOUBLE PRECISION,
    consensus_score DOUBLE PRECISION,
    external_sources_verified INTEGER,
    success BOOLEAN DEFAULT TRUE
);

-- Convert to hypertable
SELECT create_hypertable('blockchain_verification_history', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_blockchain_history_entity ON blockchain_verification_history (entity_type, entity_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_blockchain_history_tx_hash ON blockchain_verification_history (transaction_hash);

-- Create compliance monitoring time series
CREATE TABLE IF NOT EXISTS compliance_monitoring (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id UUID NOT NULL,
    regulation_type VARCHAR(50) NOT NULL, -- SEC, EU_TAXONOMY, TCFD, SASB
    compliance_score DOUBLE PRECISION,
    risk_level VARCHAR(20),
    issues_identified INTEGER,
    recommendations_count INTEGER,
    ai_confidence DOUBLE PRECISION,
    manual_review_required BOOLEAN DEFAULT FALSE
);

-- Convert to hypertable
SELECT create_hypertable('compliance_monitoring', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_compliance_portfolio_time ON compliance_monitoring (portfolio_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_compliance_regulation_time ON compliance_monitoring (regulation_type, time DESC);

-- Create continuous aggregates for performance optimization

-- Daily ESG score aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS esg_daily_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    company_id,
    symbol,
    AVG(environmental_score) AS avg_environmental_score,
    AVG(social_score) AS avg_social_score,
    AVG(governance_score) AS avg_governance_score,
    AVG(overall_esg_score) AS avg_overall_esg_score,
    AVG(carbon_intensity) AS avg_carbon_intensity,
    COUNT(*) AS data_points
FROM esg_time_series
GROUP BY day, company_id, symbol;

-- Weekly portfolio performance aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS portfolio_weekly_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', time) AS week,
    portfolio_id,
    AVG(total_value) AS avg_total_value,
    AVG(daily_return) AS avg_daily_return,
    STDDEV(daily_return) AS volatility,
    AVG(sharpe_ratio) AS avg_sharpe_ratio,
    MIN(max_drawdown) AS max_drawdown,
    AVG(esg_score) AS avg_esg_score,
    COUNT(*) AS trading_days
FROM portfolio_performance
GROUP BY week, portfolio_id;

-- Monthly climate risk aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS climate_monthly_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', time) AS month,
    region,
    risk_type,
    AVG(temperature_anomaly) AS avg_temperature_anomaly,
    AVG(precipitation_anomaly) AS avg_precipitation_anomaly,
    SUM(extreme_weather_events) AS total_extreme_events,
    AVG(carbon_price) AS avg_carbon_price,
    AVG(economic_impact) AS avg_economic_impact
FROM climate_risk_data
GROUP BY month, region, risk_type;

-- Create refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('esg_daily_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('portfolio_weekly_summary',
    start_offset => INTERVAL '2 weeks',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

SELECT add_continuous_aggregate_policy('climate_monthly_summary',
    start_offset => INTERVAL '2 months',
    end_offset => INTERVAL '1 week',
    schedule_interval => INTERVAL '1 week');

-- Create retention policies for data management
SELECT add_retention_policy('esg_time_series', INTERVAL '5 years');
SELECT add_retention_policy('portfolio_performance', INTERVAL '10 years');
SELECT add_retention_policy('climate_risk_data', INTERVAL '20 years');
SELECT add_retention_policy('quantum_optimization_history', INTERVAL '3 years');
SELECT add_retention_policy('blockchain_verification_history', INTERVAL '7 years');
SELECT add_retention_policy('compliance_monitoring', INTERVAL '7 years');

-- Insert sample ESG data
INSERT INTO esg_time_series (
    time, company_id, symbol, environmental_score, social_score, governance_score,
    overall_esg_score, carbon_emissions, carbon_intensity, water_usage,
    renewable_energy_usage, employee_satisfaction, diversity_ratio,
    board_independence, data_source, data_quality_score
) VALUES
    (NOW() - INTERVAL '1 day', 'AAPL_001', 'AAPL', 85.2, 88.1, 92.3, 88.5, 25300000, 45.2, 1200000, 0.75, 0.89, 0.42, 0.87, 'Bloomberg ESG', 0.95),
    (NOW() - INTERVAL '1 day', 'MSFT_001', 'MSFT', 89.1, 91.2, 94.5, 91.6, 11200000, 32.1, 890000, 0.82, 0.91, 0.48, 0.91, 'Refinitiv ESG', 0.97),
    (NOW() - INTERVAL '1 day', 'TSLA_001', 'TSLA', 92.3, 78.9, 81.2, 84.1, 8900000, 28.7, 650000, 0.95, 0.82, 0.35, 0.78, 'MSCI ESG', 0.93),
    (NOW() - INTERVAL '1 day', 'JNJ_001', 'JNJ', 78.5, 94.2, 89.7, 87.5, 15600000, 52.3, 1450000, 0.68, 0.95, 0.52, 0.88, 'Sustainalytics', 0.96),
    (NOW() - INTERVAL '1 day', 'PG_001', 'PG', 81.2, 87.6, 91.8, 86.9, 12800000, 48.9, 1100000, 0.72, 0.88, 0.45, 0.89, 'Bloomberg ESG', 0.94);

-- Insert sample portfolio performance data
INSERT INTO portfolio_performance (
    time, portfolio_id, total_value, daily_return, cumulative_return,
    volatility, sharpe_ratio, max_drawdown, esg_score, carbon_footprint,
    quantum_optimized, optimization_score
) VALUES
    (NOW() - INTERVAL '1 day', gen_random_uuid(), 10000000, 0.012, 0.127, 0.18, 1.42, -0.08, 87.2, 145.6, true, 0.89),
    (NOW() - INTERVAL '2 days', gen_random_uuid(), 9950000, -0.005, 0.115, 0.19, 1.38, -0.08, 86.8, 147.2, true, 0.87),
    (NOW() - INTERVAL '3 days', gen_random_uuid(), 9980000, 0.008, 0.120, 0.17, 1.45, -0.06, 87.5, 143.9, true, 0.91);

-- Insert sample climate risk data
INSERT INTO climate_risk_data (
    time, region, risk_type, temperature_anomaly, precipitation_anomaly,
    sea_level_change, extreme_weather_events, carbon_price, policy_stringency_index,
    renewable_energy_share, economic_impact, data_source
) VALUES
    (NOW() - INTERVAL '1 day', 'North America', 'physical', 1.2, -0.15, 3.2, 12, 85.50, 0.72, 0.28, -0.02, 'NOAA Climate Data'),
    (NOW() - INTERVAL '1 day', 'Europe', 'transition', 1.1, 0.08, 2.8, 8, 92.30, 0.85, 0.42, -0.01, 'Copernicus Climate'),
    (NOW() - INTERVAL '1 day', 'Asia Pacific', 'physical', 1.4, -0.22, 4.1, 18, 78.20, 0.68, 0.31, -0.03, 'JMA Climate Data');

-- Insert sample quantum optimization history
INSERT INTO quantum_optimization_history (
    time, portfolio_id, algorithm, num_qubits, circuit_depth, execution_time,
    quantum_speedup, optimization_score, convergence_iterations, expected_return,
    portfolio_risk, sharpe_ratio, esg_score, success
) VALUES
    (NOW() - INTERVAL '1 hour', gen_random_uuid(), 'QAOA', 8, 12, 45.2, 2.3, 0.89, 85, 0.127, 0.18, 1.42, 87.2, true),
    (NOW() - INTERVAL '2 hours', gen_random_uuid(), 'VQE', 6, 8, 32.1, 1.8, 0.87, 92, 0.115, 0.19, 1.38, 86.8, true),
    (NOW() - INTERVAL '3 hours', gen_random_uuid(), 'QAOA', 10, 15, 67.8, 2.7, 0.91, 78, 0.135, 0.17, 1.48, 88.1, true);

-- Insert sample blockchain verification history
INSERT INTO blockchain_verification_history (
    time, entity_type, entity_id, transaction_hash, block_number,
    verification_score, gas_used, verification_time, consensus_score,
    external_sources_verified, success
) VALUES
    (NOW() - INTERVAL '30 minutes', 'esg_data', gen_random_uuid(), '0x1234567890abcdef1234567890abcdef12345678', 12345678, 0.95, 150000, 12.3, 0.92, 3, true),
    (NOW() - INTERVAL '1 hour', 'portfolio', gen_random_uuid(), '0xabcdef1234567890abcdef1234567890abcdef12', 12345677, 0.88, 180000, 15.7, 0.85, 2, true),
    (NOW() - INTERVAL '2 hours', 'esg_data', gen_random_uuid(), '0x567890abcdef1234567890abcdef1234567890ab', 12345676, 0.92, 165000, 11.9, 0.89, 4, true);

-- Insert sample compliance monitoring data
INSERT INTO compliance_monitoring (
    time, portfolio_id, regulation_type, compliance_score, risk_level,
    issues_identified, recommendations_count, ai_confidence, manual_review_required
) VALUES
    (NOW() - INTERVAL '1 day', gen_random_uuid(), 'SEC', 0.89, 'low', 1, 3, 0.94, false),
    (NOW() - INTERVAL '1 day', gen_random_uuid(), 'EU_TAXONOMY', 0.76, 'medium', 3, 5, 0.87, true),
    (NOW() - INTERVAL '1 day', gen_random_uuid(), 'TCFD', 0.92, 'low', 0, 2, 0.96, false),
    (NOW() - INTERVAL '1 day', gen_random_uuid(), 'SASB', 0.84, 'low', 2, 4, 0.91, false);

-- Create functions for common queries

-- Function to get latest ESG scores for a company
CREATE OR REPLACE FUNCTION get_latest_esg_score(company_symbol VARCHAR(20))
RETURNS TABLE (
    environmental_score DOUBLE PRECISION,
    social_score DOUBLE PRECISION,
    governance_score DOUBLE PRECISION,
    overall_esg_score DOUBLE PRECISION,
    last_updated TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.environmental_score,
        e.social_score,
        e.governance_score,
        e.overall_esg_score,
        e.time
    FROM esg_time_series e
    WHERE e.symbol = company_symbol
    ORDER BY e.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate portfolio ESG score
CREATE OR REPLACE FUNCTION calculate_portfolio_esg_score(
    portfolio_uuid UUID,
    calculation_date TIMESTAMPTZ DEFAULT NOW()
)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    weighted_esg_score DOUBLE PRECISION;
BEGIN
    SELECT 
        SUM(h.weight * e.overall_esg_score) / SUM(h.weight)
    INTO weighted_esg_score
    FROM holdings h
    JOIN esg_time_series e ON h.symbol = e.symbol
    WHERE h.portfolio_id = portfolio_uuid
    AND e.time <= calculation_date
    AND e.time = (
        SELECT MAX(e2.time)
        FROM esg_time_series e2
        WHERE e2.symbol = e.symbol
        AND e2.time <= calculation_date
    );
    
    RETURN COALESCE(weighted_esg_score, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to get climate risk summary for a region
CREATE OR REPLACE FUNCTION get_climate_risk_summary(
    target_region VARCHAR(100),
    start_date TIMESTAMPTZ DEFAULT NOW() - INTERVAL '1 year',
    end_date TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    risk_type VARCHAR(50),
    avg_temperature_anomaly DOUBLE PRECISION,
    total_extreme_events BIGINT,
    avg_economic_impact DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.risk_type,
        AVG(c.temperature_anomaly),
        SUM(c.extreme_weather_events),
        AVG(c.economic_impact)
    FROM climate_risk_data c
    WHERE c.region = target_region
    AND c.time BETWEEN start_date AND end_date
    GROUP BY c.risk_type
    ORDER BY AVG(c.economic_impact) DESC;
END;
$$ LANGUAGE plpgsql;

-- Create views for common dashboard queries

-- Portfolio performance dashboard view
CREATE OR REPLACE VIEW portfolio_dashboard AS
SELECT 
    p.id,
    p.name,
    p.total_value,
    pp.daily_return,
    pp.cumulative_return,
    pp.sharpe_ratio,
    pp.esg_score,
    pp.quantum_optimized,
    pp.time as last_updated
FROM portfolios p
LEFT JOIN LATERAL (
    SELECT *
    FROM portfolio_performance pp2
    WHERE pp2.portfolio_id = p.id
    ORDER BY pp2.time DESC
    LIMIT 1
) pp ON true;

-- ESG leaders view
CREATE OR REPLACE VIEW esg_leaders AS
SELECT 
    symbol,
    company_id,
    overall_esg_score,
    environmental_score,
    social_score,
    governance_score,
    time as last_updated
FROM esg_time_series e1
WHERE e1.time = (
    SELECT MAX(e2.time)
    FROM esg_time_series e2
    WHERE e2.symbol = e1.symbol
)
AND overall_esg_score >= 80
ORDER BY overall_esg_score DESC;

-- Climate risk hotspots view
CREATE OR REPLACE VIEW climate_risk_hotspots AS
SELECT 
    region,
    AVG(temperature_anomaly) as avg_temp_anomaly,
    SUM(extreme_weather_events) as total_extreme_events,
    AVG(economic_impact) as avg_economic_impact,
    COUNT(*) as data_points
FROM climate_risk_data
WHERE time >= NOW() - INTERVAL '1 year'
GROUP BY region
HAVING AVG(economic_impact) < -0.01  -- Negative economic impact
ORDER BY avg_economic_impact ASC;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres;

-- Create database statistics
ANALYZE;

COMMIT;