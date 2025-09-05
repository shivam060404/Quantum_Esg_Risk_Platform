from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel
from enum import Enum

# Import compliance agent
from ai_agents.compliance_agent import ComplianceAgent
from core.security import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])

# Enums
class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"

class RegulationType(str, Enum):
    ESG_DISCLOSURE = "esg_disclosure"
    CLIMATE_RISK = "climate_risk"
    TAXONOMY = "taxonomy"
    STEWARDSHIP = "stewardship"
    FIDUCIARY_DUTY = "fiduciary_duty"
    ANTI_GREENWASHING = "anti_greenwashing"

class ReportType(str, Enum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    AD_HOC = "ad_hoc"

# Pydantic models
class RegulationInfo(BaseModel):
    regulation_id: str
    name: str
    jurisdiction: str
    type: RegulationType
    effective_date: str
    description: str
    requirements: List[str]
    penalties: Dict[str, Any]
    compliance_deadline: Optional[str] = None

class ComplianceCheck(BaseModel):
    check_id: str
    regulation_id: str
    requirement: str
    status: ComplianceStatus
    score: float  # 0-100
    findings: List[str]
    recommendations: List[str]
    evidence: List[str]
    last_checked: str
    next_review: str

class ComplianceReport(BaseModel):
    report_id: str
    portfolio_id: str
    report_type: ReportType
    period_start: str
    period_end: str
    overall_score: float
    compliance_checks: List[ComplianceCheck]
    summary: Dict[str, Any]
    generated_date: str
    status: str

class AuditTrail(BaseModel):
    audit_id: str
    timestamp: str
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None

class ComplianceAlert(BaseModel):
    alert_id: str
    severity: str  # low, medium, high, critical
    regulation_id: str
    title: str
    description: str
    affected_portfolios: List[str]
    deadline: Optional[str] = None
    status: str  # active, resolved, dismissed
    created_date: str
    updated_date: str

# Mock regulations data
MOCK_REGULATIONS = {
    "eu_sfdr": RegulationInfo(
        regulation_id="eu_sfdr",
        name="EU Sustainable Finance Disclosure Regulation (SFDR)",
        jurisdiction="European Union",
        type=RegulationType.ESG_DISCLOSURE,
        effective_date="2021-03-10",
        description="Regulation on sustainability‐related disclosures in the financial services sector",
        requirements=[
            "Principal adverse impact disclosures",
            "Product-level sustainability disclosures",
            "Entity-level sustainability disclosures",
            "Taxonomy alignment reporting"
        ],
        penalties={
            "administrative_fines": "Up to 5% of annual turnover",
            "public_warnings": "Reputational damage",
            "suspension": "Temporary business suspension"
        },
        compliance_deadline="2023-01-01"
    ),
    "eu_taxonomy": RegulationInfo(
        regulation_id="eu_taxonomy",
        name="EU Taxonomy Regulation",
        jurisdiction="European Union",
        type=RegulationType.TAXONOMY,
        effective_date="2022-01-01",
        description="Classification system for environmentally sustainable economic activities",
        requirements=[
            "Taxonomy alignment assessment",
            "Do no significant harm evaluation",
            "Minimum safeguards compliance",
            "Technical screening criteria verification"
        ],
        penalties={
            "administrative_fines": "Up to 2% of annual turnover",
            "corrective_measures": "Mandatory remediation"
        }
    ),
    "tcfd": RegulationInfo(
        regulation_id="tcfd",
        name="Task Force on Climate-related Financial Disclosures (TCFD)",
        jurisdiction="Global",
        type=RegulationType.CLIMATE_RISK,
        effective_date="2017-06-29",
        description="Framework for climate-related financial risk disclosures",
        requirements=[
            "Governance disclosures",
            "Strategy disclosures",
            "Risk management disclosures",
            "Metrics and targets disclosures"
        ],
        penalties={
            "reputational_risk": "Market confidence impact",
            "regulatory_scrutiny": "Increased oversight"
        }
    )
}

# Mock compliance data
MOCK_COMPLIANCE_CHECKS = {
    "portfolio_1": [
        ComplianceCheck(
            check_id="check_001",
            regulation_id="eu_sfdr",
            requirement="Principal adverse impact disclosures",
            status=ComplianceStatus.COMPLIANT,
            score=95.0,
            findings=["All required PAI indicators disclosed", "Data quality meets standards"],
            recommendations=["Consider additional voluntary indicators"],
            evidence=["PAI_report_2024.pdf", "data_quality_assessment.xlsx"],
            last_checked="2024-01-15T10:00:00Z",
            next_review="2024-04-15T10:00:00Z"
        ),
        ComplianceCheck(
            check_id="check_002",
            regulation_id="eu_taxonomy",
            requirement="Taxonomy alignment assessment",
            status=ComplianceStatus.PARTIALLY_COMPLIANT,
            score=72.0,
            findings=["65% of investments assessed", "Missing data for some holdings"],
            recommendations=["Improve data collection for remaining 35%", "Engage with investee companies"],
            evidence=["taxonomy_assessment_Q4.pdf"],
            last_checked="2024-01-10T14:30:00Z",
            next_review="2024-02-10T14:30:00Z"
        )
    ]
}

@router.get("/regulations", response_model=List[RegulationInfo])
async def get_regulations(
    jurisdiction: Optional[str] = None,
    regulation_type: Optional[RegulationType] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get list of applicable regulations"""
    try:
        # verify_token(credentials.credentials)
        
        regulations = list(MOCK_REGULATIONS.values())
        
        # Filter by jurisdiction
        if jurisdiction:
            regulations = [r for r in regulations if r.jurisdiction.lower() == jurisdiction.lower()]
        
        # Filter by type
        if regulation_type:
            regulations = [r for r in regulations if r.type == regulation_type]
        
        return regulations
        
    except Exception as e:
        logger.error(f"Error fetching regulations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch regulations")

@router.get("/regulations/{regulation_id}", response_model=RegulationInfo)
async def get_regulation_details(
    regulation_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get detailed information about a specific regulation"""
    try:
        # verify_token(credentials.credentials)
        
        if regulation_id not in MOCK_REGULATIONS:
            raise HTTPException(status_code=404, detail="Regulation not found")
        
        return MOCK_REGULATIONS[regulation_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching regulation {regulation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch regulation details")

@router.get("/check/{portfolio_id}", response_model=List[ComplianceCheck])
async def run_compliance_check(
    portfolio_id: str,
    regulation_ids: Optional[List[str]] = Query(None),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Run compliance checks for a portfolio"""
    try:
        # verify_token(credentials.credentials)
        
        # Initialize compliance agent
        compliance_agent = ComplianceAgent()
        
        # Get existing checks or generate new ones
        if portfolio_id in MOCK_COMPLIANCE_CHECKS:
            checks = MOCK_COMPLIANCE_CHECKS[portfolio_id]
        else:
            # Generate mock checks for new portfolio
            checks = [
                ComplianceCheck(
                    check_id=f"check_{portfolio_id}_001",
                    regulation_id="eu_sfdr",
                    requirement="ESG disclosure requirements",
                    status=ComplianceStatus.UNDER_REVIEW,
                    score=0.0,
                    findings=["Assessment in progress"],
                    recommendations=["Pending analysis completion"],
                    evidence=[],
                    last_checked=datetime.utcnow().isoformat(),
                    next_review=(datetime.utcnow() + timedelta(days=30)).isoformat()
                )
            ]
        
        # Filter by regulation IDs if provided
        if regulation_ids:
            checks = [c for c in checks if c.regulation_id in regulation_ids]
        
        logger.info(f"Compliance check completed for portfolio {portfolio_id}")
        return checks
        
    except Exception as e:
        logger.error(f"Error running compliance check for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to run compliance check")

@router.post("/report/generate", response_model=ComplianceReport)
async def generate_compliance_report(
    portfolio_id: str,
    report_type: ReportType,
    period_start: str,
    period_end: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate compliance report for a portfolio"""
    try:
        # verify_token(credentials.credentials)
        
        # Get compliance checks for the portfolio
        checks = MOCK_COMPLIANCE_CHECKS.get(portfolio_id, [])
        
        # Calculate overall score
        if checks:
            overall_score = sum(check.score for check in checks) / len(checks)
        else:
            overall_score = 0.0
        
        # Generate summary
        summary = {
            "total_checks": len(checks),
            "compliant_checks": len([c for c in checks if c.status == ComplianceStatus.COMPLIANT]),
            "non_compliant_checks": len([c for c in checks if c.status == ComplianceStatus.NON_COMPLIANT]),
            "partially_compliant_checks": len([c for c in checks if c.status == ComplianceStatus.PARTIALLY_COMPLIANT]),
            "under_review_checks": len([c for c in checks if c.status == ComplianceStatus.UNDER_REVIEW]),
            "compliance_rate": overall_score,
            "key_findings": [
                "Strong ESG disclosure practices",
                "Room for improvement in taxonomy alignment",
                "Climate risk disclosures need enhancement"
            ],
            "priority_actions": [
                "Complete taxonomy alignment assessment",
                "Enhance climate scenario analysis",
                "Improve data collection processes"
            ]
        }
        
        report = ComplianceReport(
            report_id=f"report_{portfolio_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            portfolio_id=portfolio_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            overall_score=overall_score,
            compliance_checks=checks,
            summary=summary,
            generated_date=datetime.utcnow().isoformat(),
            status="completed"
        )
        
        logger.info(f"Compliance report generated for portfolio {portfolio_id}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating compliance report for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")

@router.get("/alerts", response_model=List[ComplianceAlert])
async def get_compliance_alerts(
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, le=100),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get compliance alerts"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock alerts data
        alerts = [
            ComplianceAlert(
                alert_id="alert_001",
                severity="high",
                regulation_id="eu_sfdr",
                title="SFDR Disclosure Deadline Approaching",
                description="Annual SFDR disclosure deadline is in 30 days",
                affected_portfolios=["portfolio_1", "portfolio_2"],
                deadline="2024-03-31T23:59:59Z",
                status="active",
                created_date="2024-01-15T09:00:00Z",
                updated_date="2024-01-15T09:00:00Z"
            ),
            ComplianceAlert(
                alert_id="alert_002",
                severity="medium",
                regulation_id="eu_taxonomy",
                title="Taxonomy Data Gap Identified",
                description="Missing taxonomy alignment data for 15% of holdings",
                affected_portfolios=["portfolio_1"],
                status="active",
                created_date="2024-01-10T14:30:00Z",
                updated_date="2024-01-12T10:15:00Z"
            ),
            ComplianceAlert(
                alert_id="alert_003",
                severity="low",
                regulation_id="tcfd",
                title="TCFD Report Enhancement Opportunity",
                description="Consider adding quantitative climate metrics to improve disclosure quality",
                affected_portfolios=["portfolio_2", "portfolio_3"],
                status="active",
                created_date="2024-01-08T11:20:00Z",
                updated_date="2024-01-08T11:20:00Z"
            )
        ]
        
        # Filter by severity
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Filter by status
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        # Apply limit
        alerts = alerts[:limit]
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error fetching compliance alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch compliance alerts")

@router.get("/audit-trail", response_model=List[AuditTrail])
async def get_audit_trail(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = Query(100, le=1000),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get audit trail for compliance activities"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock audit trail data
        audit_entries = [
            AuditTrail(
                audit_id="audit_001",
                timestamp="2024-01-15T10:30:00Z",
                user_id="user_123",
                action="compliance_check_run",
                resource="portfolio_1",
                details={
                    "regulation_ids": ["eu_sfdr", "eu_taxonomy"],
                    "checks_performed": 5,
                    "overall_score": 85.2
                },
                ip_address="192.168.1.100"
            ),
            AuditTrail(
                audit_id="audit_002",
                timestamp="2024-01-15T09:15:00Z",
                user_id="user_456",
                action="report_generated",
                resource="compliance_report_001",
                details={
                    "report_type": "quarterly",
                    "portfolio_id": "portfolio_1",
                    "period": "Q4_2023"
                },
                ip_address="192.168.1.101"
            ),
            AuditTrail(
                audit_id="audit_003",
                timestamp="2024-01-14T16:45:00Z",
                user_id="user_789",
                action="alert_dismissed",
                resource="alert_002",
                details={
                    "reason": "Data gap resolved",
                    "resolution_notes": "Updated taxonomy data received from data provider"
                },
                ip_address="192.168.1.102"
            )
        ]
        
        # Apply filters
        if user_id:
            audit_entries = [a for a in audit_entries if a.user_id == user_id]
        
        if action:
            audit_entries = [a for a in audit_entries if a.action == action]
        
        if start_date:
            audit_entries = [a for a in audit_entries if a.timestamp >= start_date]
        
        if end_date:
            audit_entries = [a for a in audit_entries if a.timestamp <= end_date]
        
        # Apply limit
        audit_entries = audit_entries[:limit]
        
        return audit_entries
        
    except Exception as e:
        logger.error(f"Error fetching audit trail: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch audit trail")

@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: str,
    reason: str,
    notes: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Dismiss a compliance alert"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock dismissal logic
        dismissal_record = {
            "alert_id": alert_id,
            "dismissed_by": "current_user",  # Would get from token
            "dismissed_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "notes": notes,
            "status": "dismissed"
        }
        
        logger.info(f"Alert {alert_id} dismissed by user")
        return dismissal_record
        
    except Exception as e:
        logger.error(f"Error dismissing alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to dismiss alert")

@router.get("/dashboard")
async def get_compliance_dashboard(
    portfolio_id: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get compliance dashboard data"""
    try:
        # verify_token(credentials.credentials)
        
        # Mock dashboard data
        dashboard = {
            "overall_compliance_score": 82.5,
            "compliance_trend": "improving",  # improving, declining, stable
            "total_regulations": len(MOCK_REGULATIONS),
            "active_alerts": 3,
            "high_priority_alerts": 1,
            "upcoming_deadlines": [
                {
                    "regulation": "EU SFDR",
                    "requirement": "Annual disclosure",
                    "deadline": "2024-03-31",
                    "days_remaining": 45
                },
                {
                    "regulation": "EU Taxonomy",
                    "requirement": "Quarterly reporting",
                    "deadline": "2024-04-30",
                    "days_remaining": 75
                }
            ],
            "compliance_by_regulation": {
                "eu_sfdr": 95.0,
                "eu_taxonomy": 72.0,
                "tcfd": 88.0
            },
            "recent_activities": [
                {
                    "activity": "Compliance check completed",
                    "portfolio": "portfolio_1",
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                {
                    "activity": "Report generated",
                    "portfolio": "portfolio_2",
                    "timestamp": "2024-01-14T16:45:00Z"
                }
            ],
            "recommendations": [
                "Complete taxonomy alignment assessment for portfolio_1",
                "Enhance climate scenario analysis documentation",
                "Update ESG data collection procedures"
            ],
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error fetching compliance dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch compliance dashboard")