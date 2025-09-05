from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel

# Import blockchain oracle
from blockchain.esg_oracle import ESGOracle
from core.security import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/esg", tags=["esg"])

# Pydantic models
class ESGScoreRequest(BaseModel):
    company_symbol: str
    data_sources: Optional[List[str]] = ["msci", "sustainalytics", "refinitiv"]
    include_verification: bool = True

class ESGScore(BaseModel):
    company_symbol: str
    overall_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    score_date: str
    data_source: str
    verification_status: str

class ESGTrend(BaseModel):
    period: str
    score_change: float
    trend_direction: str
    key_factors: List[str]

class CompanyESGProfile(BaseModel):
    company_symbol: str
    company_name: str
    sector: str
    current_score: ESGScore
    historical_trends: List[ESGTrend]
    peer_comparison: Dict[str, float]
    risk_flags: List[str]
    opportunities: List[str]

# Mock ESG data
MOCK_ESG_DATA = {
    "AAPL": {
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "current_score": {
            "overall_score": 87.5,
            "environmental_score": 85.2,
            "social_score": 89.1,
            "governance_score": 88.3,
            "score_date": datetime.utcnow().isoformat(),
            "data_source": "msci",
            "verification_status": "verified"
        },
        "historical_trends": [
            {
                "period": "Q4_2023",
                "score_change": 2.3,
                "trend_direction": "improving",
                "key_factors": ["renewable_energy_adoption", "supply_chain_transparency"]
            },
            {
                "period": "Q3_2023",
                "score_change": 1.1,
                "trend_direction": "improving",
                "key_factors": ["carbon_neutrality_progress", "diversity_initiatives"]
            }
        ],
        "peer_comparison": {
            "sector_average": 78.2,
            "industry_leaders": 85.0,
            "percentile_rank": 92
        },
        "risk_flags": [],
        "opportunities": [
            "Further supply chain decarbonization",
            "Expanded recycling programs",
            "Enhanced worker rights in manufacturing"
        ]
    },
    "TSLA": {
        "company_name": "Tesla Inc.",
        "sector": "Automotive",
        "current_score": {
            "overall_score": 82.1,
            "environmental_score": 95.8,
            "social_score": 72.5,
            "governance_score": 78.0,
            "score_date": datetime.utcnow().isoformat(),
            "data_source": "sustainalytics",
            "verification_status": "verified"
        },
        "historical_trends": [
            {
                "period": "Q4_2023",
                "score_change": -1.2,
                "trend_direction": "declining",
                "key_factors": ["governance_concerns", "labor_relations"]
            }
        ],
        "peer_comparison": {
            "sector_average": 65.4,
            "industry_leaders": 80.0,
            "percentile_rank": 88
        },
        "risk_flags": [
            "Governance structure concerns",
            "Labor relations issues"
        ],
        "opportunities": [
            "Improved board independence",
            "Enhanced worker safety programs"
        ]
    }
}

@router.get("/companies", response_model=List[str])
async def get_esg_companies(
    sector: Optional[str] = None,
    min_score: Optional[float] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get list of companies with ESG data"""
    try:
        # verify_token(credentials.credentials)
        
        companies = list(MOCK_ESG_DATA.keys())
        
        # Filter by sector if provided
        if sector:
            companies = [
                symbol for symbol in companies 
                if MOCK_ESG_DATA[symbol]["sector"].lower() == sector.lower()
            ]
        
        # Filter by minimum score if provided
        if min_score:
            companies = [
                symbol for symbol in companies 
                if MOCK_ESG_DATA[symbol]["current_score"]["overall_score"] >= min_score
            ]
        
        return companies
        
    except Exception as e:
        logger.error(f"Error fetching ESG companies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ESG companies")

@router.get("/score/{company_symbol}", response_model=ESGScore)
async def get_esg_score(
    company_symbol: str,
    data_source: str = "msci",
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get ESG score for a specific company"""
    try:
        # verify_token(credentials.credentials)
        
        company_symbol = company_symbol.upper()
        
        if company_symbol not in MOCK_ESG_DATA:
            raise HTTPException(status_code=404, detail="Company ESG data not found")
        
        score_data = MOCK_ESG_DATA[company_symbol]["current_score"]
        
        esg_score = ESGScore(
            company_symbol=company_symbol,
            overall_score=score_data["overall_score"],
            environmental_score=score_data["environmental_score"],
            social_score=score_data["social_score"],
            governance_score=score_data["governance_score"],
            score_date=score_data["score_date"],
            data_source=data_source,
            verification_status=score_data["verification_status"]
        )
        
        return esg_score
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ESG score for {company_symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ESG score")

@router.get("/profile/{company_symbol}", response_model=CompanyESGProfile)
async def get_company_esg_profile(
    company_symbol: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get comprehensive ESG profile for a company"""
    try:
        # verify_token(credentials.credentials)
        
        company_symbol = company_symbol.upper()
        
        if company_symbol not in MOCK_ESG_DATA:
            raise HTTPException(status_code=404, detail="Company ESG data not found")
        
        company_data = MOCK_ESG_DATA[company_symbol]
        
        profile = CompanyESGProfile(
            company_symbol=company_symbol,
            company_name=company_data["company_name"],
            sector=company_data["sector"],
            current_score=ESGScore(
                company_symbol=company_symbol,
                **company_data["current_score"]
            ),
            historical_trends=[
                ESGTrend(**trend) for trend in company_data["historical_trends"]
            ],
            peer_comparison=company_data["peer_comparison"],
            risk_flags=company_data["risk_flags"],
            opportunities=company_data["opportunities"]
        )
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ESG profile for {company_symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ESG profile")

@router.post("/verify")
async def verify_esg_data(
    company_symbol: str,
    esg_data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Verify ESG data using blockchain oracle"""
    try:
        # verify_token(credentials.credentials)
        
        # Initialize blockchain oracle
        oracle = ESGOracle()
        
        # Mock verification process
        verification_result = {
            "company_symbol": company_symbol.upper(),
            "verification_id": f"VER_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "status": "verified",
            "confidence_score": 0.95,
            "data_sources_verified": ["msci", "sustainalytics"],
            "blockchain_hash": f"0x{hash(str(esg_data)) % (10**16):016x}",
            "verification_timestamp": datetime.utcnow().isoformat(),
            "discrepancies": [],
            "quality_score": 0.92
        }
        
        logger.info(f"ESG data verified for {company_symbol}")
        return verification_result
        
    except Exception as e:
        logger.error(f"Error verifying ESG data for {company_symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify ESG data")

@router.get("/sector-analysis/{sector}")
async def get_sector_esg_analysis(
    sector: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get ESG analysis for a specific sector"""
    try:
        # verify_token(credentials.credentials)
        
        # Filter companies by sector
        sector_companies = {
            symbol: data for symbol, data in MOCK_ESG_DATA.items()
            if data["sector"].lower() == sector.lower()
        }
        
        if not sector_companies:
            raise HTTPException(status_code=404, detail="No companies found for this sector")
        
        # Calculate sector statistics
        scores = [data["current_score"]["overall_score"] for data in sector_companies.values()]
        env_scores = [data["current_score"]["environmental_score"] for data in sector_companies.values()]
        social_scores = [data["current_score"]["social_score"] for data in sector_companies.values()]
        gov_scores = [data["current_score"]["governance_score"] for data in sector_companies.values()]
        
        analysis = {
            "sector": sector.title(),
            "company_count": len(sector_companies),
            "average_scores": {
                "overall": sum(scores) / len(scores),
                "environmental": sum(env_scores) / len(env_scores),
                "social": sum(social_scores) / len(social_scores),
                "governance": sum(gov_scores) / len(gov_scores)
            },
            "score_distribution": {
                "high_performers": len([s for s in scores if s >= 85]),
                "medium_performers": len([s for s in scores if 70 <= s < 85]),
                "low_performers": len([s for s in scores if s < 70])
            },
            "top_companies": sorted(
                [(symbol, data["current_score"]["overall_score"]) 
                 for symbol, data in sector_companies.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "common_risks": [
                "Climate transition risks",
                "Regulatory compliance",
                "Supply chain sustainability"
            ],
            "sector_trends": {
                "improving_areas": ["Environmental disclosure", "Carbon reduction"],
                "declining_areas": ["Board diversity", "Executive compensation"]
            },
            "analysis_date": datetime.utcnow().isoformat()
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing sector {sector}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze sector")

@router.get("/alerts")
async def get_esg_alerts(
    severity: Optional[str] = None,
    limit: int = 50,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get ESG-related alerts and notifications"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock ESG alerts
        alerts = [
            {
                "alert_id": "ESG_001",
                "company_symbol": "TSLA",
                "severity": "medium",
                "category": "governance",
                "title": "Board Independence Concerns",
                "description": "Recent governance changes may impact ESG score",
                "impact_score": -2.5,
                "created_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "status": "active"
            },
            {
                "alert_id": "ESG_002",
                "company_symbol": "AAPL",
                "severity": "low",
                "category": "environmental",
                "title": "Carbon Neutrality Progress",
                "description": "Company ahead of carbon neutrality targets",
                "impact_score": 1.8,
                "created_at": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                "status": "resolved"
            }
        ]
        
        # Filter by severity if provided
        if severity:
            alerts = [alert for alert in alerts if alert["severity"] == severity]
        
        # Limit results
        alerts = alerts[:limit]
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "active_count": len([a for a in alerts if a["status"] == "active"]),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching ESG alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ESG alerts")