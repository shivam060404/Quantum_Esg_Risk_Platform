from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Dict, Any
import logging
from datetime import datetime
import json
import asyncio
from services.data_streaming import data_streaming_service, DataStreamType

# Import custom modules
from quantum.portfolio_optimizer import QuantumPortfolioOptimizer
from blockchain.esg_oracle import ESGOracle
from ai_agents.compliance_agent import ComplianceAgent
from ai_agents.climate_agent import ClimateAgent
from ai_agents.esg_agent import ESGAgent
from ai_agents.portfolio_agent import PortfolioAgent
from database.models import init_db
# from api.routes import portfolio, esg_data, compliance, climate  # TODO: Create API routes
from core.config import settings
from core.security import verify_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        message = json.dumps(data)
        await self.broadcast(message)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Quantum-Enhanced ESG Risk Assessment Platform")
    
    # Initialize database
    await init_db()
    
    # Initialize quantum components
    app.state.quantum_optimizer = QuantumPortfolioOptimizer()
    
    # Initialize blockchain oracle
    app.state.esg_oracle = ESGOracle()
    
    # Initialize AI agents
    app.state.compliance_agent = ComplianceAgent()
    app.state.climate_agent = ClimateAgent()
    app.state.esg_agent = ESGAgent()
    app.state.portfolio_agent = PortfolioAgent()
    
    logger.info("AI agents initialized successfully")
    
    logger.info("Platform initialization complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down platform")

# Create FastAPI application
app = FastAPI(
    title="Quantum-Enhanced ESG Risk Assessment Platform",
    description="Advanced ESG risk assessment with quantum computing, blockchain verification, and AI agents",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        payload = verify_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "quantum_optimizer": "operational",
            "blockchain_oracle": "operational",
            "ai_agents": "operational",
            "database": "operational"
        }
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await manager.connect(websocket)
    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "message": "Connected to ESG Risk Assessment Platform",
                "timestamp": datetime.utcnow().isoformat()
            }),
            websocket
        )
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                if message_data.get("type") == "subscribe":
                    # Handle subscription to specific data streams
                    await handle_subscription(websocket, message_data)
                elif message_data.get("type") == "unsubscribe":
                    await data_streaming_service.unsubscribe(websocket)
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "unsubscribed",
                            "message": "Unsubscribed from all data streams",
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        websocket
                    )
                elif message_data.get("type") == "ping":
                    # Handle ping/pong for connection health
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                        websocket
                    )
                elif message_data.get("type") == "get_status":
                    status = data_streaming_service.get_streaming_status()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "status",
                            "data": status,
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        websocket
                    )
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON format"}),
                    websocket
                )
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": str(e)}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await data_streaming_service.unsubscribe(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)
        await data_streaming_service.unsubscribe(websocket)

async def handle_subscription(websocket: WebSocket, message_data: Dict[str, Any]):
    """Handle WebSocket subscriptions to data streams"""
    try:
        subscription_types = message_data.get('types', ['market_data'])
        
        # Map subscription types to DataStreamType enums
        stream_type_mapping = {
            'market_data': DataStreamType.MARKET_DATA,
            'portfolio_updates': DataStreamType.PORTFOLIO_UPDATES,
            'risk_alerts': DataStreamType.RISK_ALERTS,
            'esg_alerts': DataStreamType.ESG_ALERTS,
            'climate_updates': DataStreamType.CLIMATE_UPDATES,
            'compliance_alerts': DataStreamType.COMPLIANCE_ALERTS,
            'news_sentiment': DataStreamType.NEWS_SENTIMENT,
            'volatility_updates': DataStreamType.VOLATILITY_UPDATES
        }
        
        # Convert subscription types to DataStreamType enums
        stream_types = []
        for sub_type in subscription_types:
            if sub_type in stream_type_mapping:
                stream_types.append(stream_type_mapping[sub_type])
        
        if not stream_types:
            stream_types = [DataStreamType.MARKET_DATA]  # Default subscription
        
        # Subscribe to data streaming service
        await data_streaming_service.subscribe(websocket, stream_types)
        
        # Send confirmation
        await websocket.send_text(json.dumps({
            "type": "subscription_confirmed",
            "subscriptions": subscription_types,
            "message": f"Subscribed to {len(stream_types)} data streams",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        logger.info(f"Client subscribed to streams: {subscription_types}")
        
    except Exception as e:
        logger.error(f"Error handling subscription: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Subscription failed: {str(e)}"
        }))

# Real-time data streaming is now handled by the DataStreamingService

# Start background task when app starts
@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup"""
    # Start the data streaming service
    asyncio.create_task(data_streaming_service.start_streaming())
    logger.info("Data streaming service started")
    logger.info("Real-time data simulation started")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with platform information"""
    return {
        "message": "Quantum-Enhanced ESG Risk Assessment Platform",
        "version": "1.0.0",
        "features": [
            "Quantum Portfolio Optimization",
            "Blockchain ESG Verification",
            "AI-Powered Compliance Monitoring",
            "Real-time Climate Risk Assessment"
        ],
        "endpoints": {
            "portfolio": "/api/v1/portfolio",
            "esg_data": "/api/v1/esg",
            "compliance": "/api/v1/compliance",
            "climate": "/api/v1/climate"
        }
    }

# Quantum portfolio optimization endpoint
@app.post("/api/v1/quantum/optimize-portfolio")
async def optimize_portfolio(
    portfolio_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Optimize portfolio using quantum algorithms"""
    try:
        optimizer = app.state.quantum_optimizer
        result = await optimizer.optimize(
            assets=portfolio_data.get("assets", []),
            constraints=portfolio_data.get("constraints", {}),
            esg_weights=portfolio_data.get("esg_weights", {})
        )
        return {
            "status": "success",
            "optimization_result": result,
            "quantum_advantage": result.get("quantum_speedup", 1.0),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Portfolio optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

# ESG data verification endpoint
@app.post("/api/v1/blockchain/verify-esg-data")
async def verify_esg_data(
    esg_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Verify ESG data using blockchain oracle"""
    try:
        oracle = app.state.esg_oracle
        verification_result = await oracle.verify_data(
            company_id=esg_data.get("company_id"),
            esg_metrics=esg_data.get("metrics", {}),
            data_sources=esg_data.get("sources", [])
        )
        return {
            "status": "success",
            "verification_result": verification_result,
            "blockchain_hash": verification_result.get("transaction_hash"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"ESG verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# AI compliance monitoring endpoint
@app.get("/api/v1/ai/compliance-check")
async def compliance_check(
    portfolio_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Check portfolio compliance using AI agents"""
    try:
        agent = app.state.compliance_agent
        compliance_result = await agent.analyze_compliance(
            portfolio_id=portfolio_id,
            user_id=current_user.get("user_id")
        )
        return {
            "status": "success",
            "compliance_analysis": compliance_result,
            "risk_level": compliance_result.get("risk_level"),
            "recommendations": compliance_result.get("recommendations", []),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Compliance check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")

# Climate stress testing endpoint
@app.post("/api/v1/climate/stress-test")
async def climate_stress_test(
    stress_test_params: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Perform climate stress testing with quantum-enhanced Monte Carlo"""
    try:
        agent = app.state.climate_agent
        stress_result = await agent.run_stress_test(
            portfolio_id=stress_test_params.get("portfolio_id"),
            scenarios=stress_test_params.get("scenarios", []),
            time_horizon=stress_test_params.get("time_horizon", 10),
            quantum_enhanced=stress_test_params.get("quantum_enhanced", True)
        )
        return {
            "status": "success",
            "stress_test_result": stress_result,
            "var_estimates": stress_result.get("var_estimates"),
            "scenario_impacts": stress_result.get("scenario_impacts"),
            "quantum_speedup": stress_result.get("quantum_speedup", 1.0),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Climate stress test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")

# Include API routers
from api.portfolio import router as portfolio_router, compat_router as portfolio_compat_router
from api.esg_data import router as esg_router
from api.climate import router as climate_router
from api.compliance import router as compliance_router
from api.ai_insights import router as ai_insights_router

app.include_router(portfolio_router)
app.include_router(portfolio_compat_router)  # Frontend compatibility routes
app.include_router(esg_router)
app.include_router(climate_router)
app.include_router(compliance_router)
app.include_router(ai_insights_router)

# Frontend compatibility endpoint for alerts
@app.get("/api/alerts")
async def get_alerts():
    """Get aggregated alerts from all sources - frontend compatibility endpoint"""
    try:
        from datetime import timedelta
        
        # Mock aggregated alerts data
        alerts = [
            {
                "id": "alert_001",
                "type": "warning",
                "category": "climate",
                "title": "Climate Risk Alert",
                "message": "Increased climate risk detected in energy sector holdings",
                "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                "read": False,
                "actionRequired": True,
                "source": "Climate Agent",
                "severity": "medium"
            },
            {
                "id": "alert_002",
                "type": "info",
                "category": "esg",
                "title": "ESG Score Update",
                "message": "Portfolio ESG score improved by 2.3 points",
                "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "read": False,
                "actionRequired": False,
                "source": "ESG Agent",
                "severity": "low"
            },
            {
                "id": "alert_003",
                "type": "critical",
                "category": "compliance",
                "title": "Compliance Violation",
                "message": "Potential regulatory compliance issue detected",
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "read": True,
                "actionRequired": True,
                "source": "Compliance Agent",
                "severity": "high"
            }
        ]
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )