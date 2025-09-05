from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel

# Import climate agent
from ai_agents.climate_agent import ClimateAgent
from core.security import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/climate", tags=["climate"])

# Pydantic models
class ClimateStressTestRequest(BaseModel):
    portfolio_id: str
    scenarios: List[str]
    time_horizon: int = 10  # years
    confidence_level: float = 0.95
    include_transition_risks: bool = True
    include_physical_risks: bool = True

class ClimateRiskMetrics(BaseModel):
    var_climate: float
    expected_shortfall: float
    transition_risk_score: float
    physical_risk_score: float
    stranded_assets_exposure: float
    carbon_intensity: float

class ClimateScenario(BaseModel):
    scenario_name: str
    temperature_increase: float
    carbon_price: float
    policy_stringency: str
    physical_risk_intensity: str
    description: str

class StressTestResult(BaseModel):
    portfolio_id: str
    scenario: ClimateScenario
    risk_metrics: ClimateRiskMetrics
    portfolio_impact: Dict[str, float]
    asset_level_impacts: Dict[str, float]
    recommendations: List[str]
    test_date: str

class ClimateRiskAssessment(BaseModel):
    asset_symbol: str
    overall_risk_score: float
    transition_risks: Dict[str, float]
    physical_risks: Dict[str, float]
    adaptation_measures: List[str]
    risk_factors: List[str]
    assessment_date: str

# Mock climate scenarios
CLIMATE_SCENARIOS = {
    "rcp26": ClimateScenario(
        scenario_name="RCP 2.6 (Paris Agreement)",
        temperature_increase=1.5,
        carbon_price=100.0,
        policy_stringency="high",
        physical_risk_intensity="low",
        description="Strong climate action scenario with rapid decarbonization"
    ),
    "rcp45": ClimateScenario(
        scenario_name="RCP 4.5 (Moderate Action)",
        temperature_increase=2.5,
        carbon_price=50.0,
        policy_stringency="medium",
        physical_risk_intensity="medium",
        description="Moderate climate action with gradual transition"
    ),
    "rcp85": ClimateScenario(
        scenario_name="RCP 8.5 (Business as Usual)",
        temperature_increase=4.0,
        carbon_price=10.0,
        policy_stringency="low",
        physical_risk_intensity="high",
        description="Limited climate action with high physical risks"
    )
}

# Mock climate risk data
MOCK_CLIMATE_RISKS = {
    "AAPL": {
        "overall_risk_score": 25.5,
        "transition_risks": {
            "policy_risk": 15.0,
            "technology_risk": 20.0,
            "market_risk": 30.0,
            "reputation_risk": 10.0
        },
        "physical_risks": {
            "acute_risks": 20.0,
            "chronic_risks": 35.0,
            "supply_chain_risks": 40.0
        },
        "adaptation_measures": [
            "Renewable energy transition",
            "Supply chain diversification",
            "Climate-resilient facilities"
        ],
        "risk_factors": [
            "Semiconductor supply chain vulnerability",
            "Regulatory changes in key markets"
        ]
    },
    "XOM": {
        "overall_risk_score": 85.2,
        "transition_risks": {
            "policy_risk": 90.0,
            "technology_risk": 85.0,
            "market_risk": 95.0,
            "reputation_risk": 80.0
        },
        "physical_risks": {
            "acute_risks": 70.0,
            "chronic_risks": 75.0,
            "supply_chain_risks": 60.0
        },
        "adaptation_measures": [
            "Carbon capture and storage",
            "Renewable energy investments",
            "Portfolio diversification"
        ],
        "risk_factors": [
            "High carbon intensity",
            "Stranded asset exposure",
            "Regulatory pressure"
        ]
    }
}

@router.get("/scenarios", response_model=List[ClimateScenario])
async def get_climate_scenarios(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get available climate scenarios for stress testing"""
    try:
        # verify_token(credentials.credentials)
        
        return list(CLIMATE_SCENARIOS.values())
        
    except Exception as e:
        logger.error(f"Error fetching climate scenarios: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch climate scenarios")

@router.post("/stress-test", response_model=List[StressTestResult])
async def run_climate_stress_test(
    request: ClimateStressTestRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Run climate stress test on portfolio"""
    try:
        # verify_token(credentials.credentials)
        
        # Initialize climate agent
        climate_agent = ClimateAgent()
        
        results = []
        
        for scenario_name in request.scenarios:
            if scenario_name not in CLIMATE_SCENARIOS:
                continue
                
            scenario = CLIMATE_SCENARIOS[scenario_name]
            
            # Mock stress test calculation
            base_impact = scenario.temperature_increase * 0.1
            
            risk_metrics = ClimateRiskMetrics(
                var_climate=base_impact * 0.15,
                expected_shortfall=base_impact * 0.25,
                transition_risk_score=scenario.carbon_price / 100.0 * 80,
                physical_risk_score=scenario.temperature_increase * 20,
                stranded_assets_exposure=base_impact * 0.3,
                carbon_intensity=500 - (scenario.carbon_price * 2)
            )
            
            portfolio_impact = {
                "total_portfolio_loss": base_impact * 0.12,
                "sector_energy_loss": base_impact * 0.35,
                "sector_tech_loss": base_impact * 0.05,
                "sector_finance_loss": base_impact * 0.08
            }
            
            asset_impacts = {
                "AAPL": base_impact * 0.03,
                "MSFT": base_impact * 0.02,
                "XOM": base_impact * 0.45,
                "JPM": base_impact * 0.08
            }
            
            recommendations = [
                "Reduce exposure to high-carbon assets",
                "Increase allocation to climate solutions",
                "Implement climate hedging strategies"
            ]
            
            if scenario.temperature_increase > 3.0:
                recommendations.extend([
                    "Consider physical risk insurance",
                    "Diversify geographically"
                ])
            
            result = StressTestResult(
                portfolio_id=request.portfolio_id,
                scenario=scenario,
                risk_metrics=risk_metrics,
                portfolio_impact=portfolio_impact,
                asset_level_impacts=asset_impacts,
                recommendations=recommendations,
                test_date=datetime.utcnow().isoformat()
            )
            
            results.append(result)
        
        logger.info(f"Climate stress test completed for portfolio {request.portfolio_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error running climate stress test: {e}")
        raise HTTPException(status_code=500, detail="Failed to run climate stress test")

@router.get("/risk-assessment/{asset_symbol}", response_model=ClimateRiskAssessment)
async def get_climate_risk_assessment(
    asset_symbol: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get climate risk assessment for a specific asset"""
    try:
        # verify_token(credentials.credentials)
        
        asset_symbol = asset_symbol.upper()
        
        if asset_symbol not in MOCK_CLIMATE_RISKS:
            # Generate default risk assessment
            risk_data = {
                "overall_risk_score": 45.0,
                "transition_risks": {
                    "policy_risk": 40.0,
                    "technology_risk": 35.0,
                    "market_risk": 50.0,
                    "reputation_risk": 30.0
                },
                "physical_risks": {
                    "acute_risks": 35.0,
                    "chronic_risks": 40.0,
                    "supply_chain_risks": 45.0
                },
                "adaptation_measures": [
                    "Climate risk disclosure",
                    "Operational efficiency improvements"
                ],
                "risk_factors": [
                    "Limited climate data availability"
                ]
            }
        else:
            risk_data = MOCK_CLIMATE_RISKS[asset_symbol]
        
        assessment = ClimateRiskAssessment(
            asset_symbol=asset_symbol,
            overall_risk_score=risk_data["overall_risk_score"],
            transition_risks=risk_data["transition_risks"],
            physical_risks=risk_data["physical_risks"],
            adaptation_measures=risk_data["adaptation_measures"],
            risk_factors=risk_data["risk_factors"],
            assessment_date=datetime.utcnow().isoformat()
        )
        
        return assessment
        
    except Exception as e:
        logger.error(f"Error fetching climate risk assessment for {asset_symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch climate risk assessment")

@router.get("/carbon-footprint/{portfolio_id}")
async def get_portfolio_carbon_footprint(
    portfolio_id: str,
    scope: str = "all",  # scope1, scope2, scope3, all
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get carbon footprint analysis for portfolio"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock carbon footprint data
        carbon_data = {
            "portfolio_id": portfolio_id,
            "total_emissions": 125000.5,  # tCO2e
            "emissions_intensity": 85.2,  # tCO2e per $M invested
            "scope_breakdown": {
                "scope1": 45000.2,
                "scope2": 35000.8,
                "scope3": 45000.5
            },
            "sector_breakdown": {
                "energy": 65000.0,
                "technology": 15000.0,
                "financials": 25000.0,
                "industrials": 20000.5
            },
            "top_emitters": [
                {"symbol": "XOM", "emissions": 45000.0, "percentage": 36.0},
                {"symbol": "CVX", "emissions": 20000.0, "percentage": 16.0},
                {"symbol": "COP", "emissions": 15000.0, "percentage": 12.0}
            ],
            "benchmarks": {
                "sector_average": 95.5,
                "market_average": 110.2,
                "best_in_class": 45.8
            },
            "trends": {
                "yoy_change": -8.5,  # percentage
                "trajectory": "improving",
                "target_alignment": "paris_aligned"
            },
            "calculation_date": datetime.utcnow().isoformat()
        }
        
        # Filter by scope if specified
        if scope != "all" and scope in carbon_data["scope_breakdown"]:
            carbon_data["filtered_emissions"] = carbon_data["scope_breakdown"][scope]
        
        return carbon_data
        
    except Exception as e:
        logger.error(f"Error fetching carbon footprint for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch carbon footprint")

@router.get("/transition-pathways")
async def get_transition_pathways(
    sector: Optional[str] = None,
    target_scenario: str = "rcp26",
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get climate transition pathways and targets"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock transition pathway data
        pathways = {
            "scenario": target_scenario,
            "global_targets": {
                "carbon_budget": 400,  # GtCO2
                "peak_emissions_year": 2025,
                "net_zero_year": 2050,
                "renewable_share_2030": 0.65
            },
            "sector_pathways": {
                "energy": {
                    "decarbonization_rate": 0.15,  # per year
                    "key_technologies": ["renewable_energy", "carbon_capture", "hydrogen"],
                    "investment_required": 2.5e12,  # USD
                    "stranding_risk": "high"
                },
                "technology": {
                    "decarbonization_rate": 0.08,
                    "key_technologies": ["energy_efficiency", "circular_economy"],
                    "investment_required": 5e11,
                    "stranding_risk": "low"
                },
                "transport": {
                    "decarbonization_rate": 0.12,
                    "key_technologies": ["electric_vehicles", "sustainable_fuels"],
                    "investment_required": 1.2e12,
                    "stranding_risk": "medium"
                }
            },
            "policy_milestones": [
                {
                    "year": 2025,
                    "milestone": "Carbon pricing expansion",
                    "impact": "medium"
                },
                {
                    "year": 2030,
                    "milestone": "Fossil fuel phase-out acceleration",
                    "impact": "high"
                },
                {
                    "year": 2035,
                    "milestone": "Mandatory climate disclosures",
                    "impact": "medium"
                }
            ],
            "investment_opportunities": [
                "Renewable energy infrastructure",
                "Energy storage technologies",
                "Climate adaptation solutions",
                "Nature-based solutions"
            ],
            "analysis_date": datetime.utcnow().isoformat()
        }
        
        # Filter by sector if provided
        if sector and sector in pathways["sector_pathways"]:
            pathways["focused_sector"] = pathways["sector_pathways"][sector]
        
        return pathways
        
    except Exception as e:
        logger.error(f"Error fetching transition pathways: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch transition pathways")

@router.get("/physical-risks/{region}")
async def get_physical_risk_analysis(
    region: str,
    risk_type: Optional[str] = None,  # flood, drought, hurricane, heatwave, wildfire
    time_horizon: int = 30,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get physical climate risk analysis for a region"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock physical risk data
        risk_analysis = {
            "region": region,
            "time_horizon": time_horizon,
            "overall_risk_score": 65.5,
            "risk_categories": {
                "flood": {
                    "current_risk": 45.0,
                    "future_risk": 72.0,
                    "trend": "increasing",
                    "key_drivers": ["sea_level_rise", "extreme_precipitation"]
                },
                "drought": {
                    "current_risk": 35.0,
                    "future_risk": 58.0,
                    "trend": "increasing",
                    "key_drivers": ["temperature_increase", "precipitation_changes"]
                },
                "hurricane": {
                    "current_risk": 55.0,
                    "future_risk": 75.0,
                    "trend": "increasing",
                    "key_drivers": ["ocean_warming", "atmospheric_changes"]
                },
                "heatwave": {
                    "current_risk": 40.0,
                    "future_risk": 80.0,
                    "trend": "rapidly_increasing",
                    "key_drivers": ["global_warming", "urban_heat_island"]
                },
                "wildfire": {
                    "current_risk": 50.0,
                    "future_risk": 70.0,
                    "trend": "increasing",
                    "key_drivers": ["temperature_increase", "drought_conditions"]
                }
            },
            "economic_impact": {
                "annual_expected_loss": 2.5e9,  # USD
                "infrastructure_at_risk": 0.15,  # percentage
                "population_affected": 1.2e6,
                "adaptation_cost": 5.8e8
            },
            "adaptation_measures": [
                "Flood defense systems",
                "Drought-resistant infrastructure",
                "Early warning systems",
                "Climate-resilient building codes"
            ],
            "data_sources": [
                "IPCC climate models",
                "Regional climate projections",
                "Historical disaster data"
            ],
            "analysis_date": datetime.utcnow().isoformat()
        }
        
        # Filter by risk type if provided
        if risk_type and risk_type in risk_analysis["risk_categories"]:
            risk_analysis["focused_risk"] = risk_analysis["risk_categories"][risk_type]
        
        return risk_analysis
        
    except Exception as e:
        logger.error(f"Error fetching physical risk analysis for {region}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch physical risk analysis")