from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel

# Import quantum optimizer
from quantum.portfolio_optimizer import QuantumPortfolioOptimizer
from core.security import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])

# Create a secondary router for frontend compatibility
compat_router = APIRouter(prefix="/api", tags=["portfolio-compat"])

# Pydantic models
class PortfolioRequest(BaseModel):
    assets: List[str]
    risk_tolerance: float
    esg_weight: float = 0.3
    expected_returns: Optional[List[float]] = None
    constraints: Optional[Dict[str, Any]] = None

class PortfolioResponse(BaseModel):
    portfolio_id: str
    weights: Dict[str, float]
    expected_return: float
    risk: float
    esg_score: float
    sharpe_ratio: float
    timestamp: str

class PortfolioSummary(BaseModel):
    portfolio_id: str
    name: str
    total_value: float
    esg_score: float
    risk_level: str
    last_updated: str

# Mock data for demonstration
MOCK_PORTFOLIOS = [
    {
        "portfolio_id": "PORTFOLIO_001",
        "name": "ESG Balanced Fund",
        "total_value": 1250000.00,
        "esg_score": 85.5,
        "risk_level": "Medium",
        "last_updated": datetime.utcnow().isoformat(),
        "assets": {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "TSLA": 0.08,
            "ESG_ETF_001": 0.25,
            "GREEN_BONDS": 0.20,
            "RENEWABLE_ETF": 0.20
        },
        "performance": {
            "ytd_return": 12.5,
            "volatility": 15.2,
            "sharpe_ratio": 0.82
        }
    },
    {
        "portfolio_id": "PORTFOLIO_002",
        "name": "Climate Tech Growth",
        "total_value": 850000.00,
        "esg_score": 92.1,
        "risk_level": "High",
        "last_updated": datetime.utcnow().isoformat(),
        "assets": {
            "CLEAN_ENERGY_ETF": 0.30,
            "SOLAR_STOCKS": 0.25,
            "WIND_ENERGY": 0.20,
            "BATTERY_TECH": 0.15,
            "CARBON_CREDITS": 0.10
        },
        "performance": {
            "ytd_return": 18.7,
            "volatility": 22.1,
            "sharpe_ratio": 0.85
        }
    }
]

@router.get("/list", response_model=List[PortfolioSummary])
async def get_portfolios(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get list of all portfolios"""
    try:
        # In production, verify token and get user-specific portfolios
        # verify_token(credentials.credentials)
        
        portfolios = [
            PortfolioSummary(
                portfolio_id=p["portfolio_id"],
                name=p["name"],
                total_value=p["total_value"],
                esg_score=p["esg_score"],
                risk_level=p["risk_level"],
                last_updated=p["last_updated"]
            )
            for p in MOCK_PORTFOLIOS
        ]
        
        return portfolios
        
    except Exception as e:
        logger.error(f"Error fetching portfolios: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolios")

@router.get("/{portfolio_id}")
async def get_portfolio(
    portfolio_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get detailed portfolio information"""
    try:
        # verify_token(credentials.credentials)
        
        portfolio = next((p for p in MOCK_PORTFOLIOS if p["portfolio_id"] == portfolio_id), None)
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return portfolio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio")

@router.post("/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(
    request: PortfolioRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Optimize portfolio using quantum computing"""
    try:
        # verify_token(credentials.credentials)
        
        # Initialize quantum optimizer
        optimizer = QuantumPortfolioOptimizer()
        
        # Mock optimization result for demonstration
        portfolio_id = f"OPT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate quantum optimization
        weights = {asset: 1.0/len(request.assets) for asset in request.assets}
        
        # Adjust weights based on ESG preference
        if request.esg_weight > 0.5:
            # Favor ESG assets
            esg_assets = [asset for asset in request.assets if 'ESG' in asset or 'GREEN' in asset]
            for asset in esg_assets:
                if asset in weights:
                    weights[asset] *= 1.2
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {asset: weight/total_weight for asset, weight in weights.items()}
        
        response = PortfolioResponse(
            portfolio_id=portfolio_id,
            weights=weights,
            expected_return=0.12 + (request.esg_weight * 0.02),
            risk=0.15 * (1 + request.risk_tolerance),
            esg_score=75.0 + (request.esg_weight * 20),
            sharpe_ratio=0.8 + (request.esg_weight * 0.1),
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Portfolio optimized: {portfolio_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize portfolio")

@router.get("/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: str,
    period: str = "1M",
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get portfolio performance metrics"""
    try:
        # verify_token(credentials.credentials)
        
        portfolio = next((p for p in MOCK_PORTFOLIOS if p["portfolio_id"] == portfolio_id), None)
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Mock performance data
        performance_data = {
            "portfolio_id": portfolio_id,
            "period": period,
            "returns": {
                "total_return": portfolio["performance"]["ytd_return"],
                "annualized_return": portfolio["performance"]["ytd_return"] * 1.1,
                "volatility": portfolio["performance"]["volatility"],
                "sharpe_ratio": portfolio["performance"]["sharpe_ratio"],
                "max_drawdown": -8.5
            },
            "esg_metrics": {
                "esg_score": portfolio["esg_score"],
                "environmental_score": portfolio["esg_score"] + 2,
                "social_score": portfolio["esg_score"] - 1,
                "governance_score": portfolio["esg_score"] + 1
            },
            "risk_metrics": {
                "var_95": -12.5,
                "cvar_95": -18.2,
                "beta": 0.95,
                "tracking_error": 3.2
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching performance for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio performance")

@router.post("/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: str,
    target_weights: Dict[str, float],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Rebalance portfolio to target weights"""
    try:
        # verify_token(credentials.credentials)
        
        # Validate weights sum to 1
        if abs(sum(target_weights.values()) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Target weights must sum to 1.0")
        
        # Mock rebalancing result
        rebalance_result = {
            "portfolio_id": portfolio_id,
            "status": "completed",
            "old_weights": MOCK_PORTFOLIOS[0]["assets"] if portfolio_id == "PORTFOLIO_001" else {},
            "new_weights": target_weights,
            "transaction_cost": 125.50,
            "execution_time": datetime.utcnow().isoformat(),
            "trades": [
                {
                    "asset": asset,
                    "action": "buy" if weight > 0.1 else "sell",
                    "quantity": weight * 1000,
                    "price": 100.0 + (hash(asset) % 50)
                }
                for asset, weight in target_weights.items()
            ]
        }
        
        logger.info(f"Portfolio {portfolio_id} rebalanced")
        return rebalance_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebalancing portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rebalance portfolio")

# Frontend compatibility endpoints (no auth required for development)
@compat_router.get("/portfolios", response_model=List[PortfolioSummary])
async def get_portfolios_compat():
    """Get list of all portfolios - frontend compatibility endpoint (no auth)"""
    try:
        portfolios = [
            PortfolioSummary(
                portfolio_id=p["portfolio_id"],
                name=p["name"],
                total_value=p["total_value"],
                esg_score=p["esg_score"],
                risk_level=p["risk_level"],
                last_updated=p["last_updated"]
            )
            for p in MOCK_PORTFOLIOS
        ]
        
        return portfolios
        
    except Exception as e:
        logger.error(f"Error fetching portfolios: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolios")

@compat_router.get("/portfolios/{portfolio_id}")
async def get_portfolio_compat(portfolio_id: str):
    """Get detailed portfolio information - frontend compatibility endpoint (no auth)"""
    try:
        portfolio = next((p for p in MOCK_PORTFOLIOS if p["portfolio_id"] == portfolio_id), None)
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return portfolio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio")