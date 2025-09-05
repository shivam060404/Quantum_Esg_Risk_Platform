import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class DataStreamType(Enum):
    MARKET_DATA = "market_data"
    PORTFOLIO_UPDATES = "portfolio_updates"
    RISK_ALERTS = "risk_alerts"
    ESG_ALERTS = "esg_alerts"
    CLIMATE_UPDATES = "climate_updates"
    COMPLIANCE_ALERTS = "compliance_alerts"
    NEWS_SENTIMENT = "news_sentiment"
    VOLATILITY_UPDATES = "volatility_updates"

@dataclass
class MarketDataPoint:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    bid: float
    ask: float
    high_52w: float
    low_52w: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None

@dataclass
class RiskAlert:
    alert_id: str
    severity: str  # low, medium, high, critical
    risk_type: str
    description: str
    affected_assets: List[str]
    impact_score: float
    timestamp: datetime
    recommended_actions: List[str]
    confidence_level: float

@dataclass
class ESGAlert:
    alert_id: str
    company: str
    alert_type: str  # controversy, score_change, news, regulatory
    severity: str
    description: str
    esg_impact: Dict[str, float]  # E, S, G scores
    timestamp: datetime
    source: str
    credibility_score: float

@dataclass
class PortfolioUpdate:
    portfolio_id: str
    total_value: float
    daily_return: float
    daily_return_percent: float
    risk_metrics: Dict[str, float]
    top_performers: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]
    timestamp: datetime
    alerts_count: int

@dataclass
class ClimateUpdate:
    update_id: str
    event_type: str  # policy, physical_risk, transition_risk, opportunity
    description: str
    affected_sectors: List[str]
    impact_assessment: Dict[str, float]
    timestamp: datetime
    source: str
    geographic_scope: str

@dataclass
class ComplianceAlert:
    alert_id: str
    regulation: str
    alert_type: str  # violation, warning, update, deadline
    severity: str
    description: str
    affected_portfolios: List[str]
    deadline: Optional[datetime]
    required_actions: List[str]
    timestamp: datetime

@dataclass
class NewsSentiment:
    news_id: str
    headline: str
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    affected_symbols: List[str]
    categories: List[str]
    timestamp: datetime
    source: str
    summary: str

class DataStreamingService:
    """Service for generating and streaming real-time financial data"""
    
    def __init__(self):
        self.is_streaming = False
        self.subscribers = {}
        self.market_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM",
            "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", "NFLX",
            "CRM", "ORCL", "PFE", "KO", "PEP", "WMT", "BAC", "XOM", "CVX", "ABBV"
        ]
        self.base_prices = {symbol: random.uniform(50, 500) for symbol in self.market_symbols}
        self.price_history = {symbol: [] for symbol in self.market_symbols}
        
        # ESG companies for alerts
        self.esg_companies = [
            "AAPL", "MSFT", "GOOGL", "TSLA", "JNJ", "PG", "UNH", "DIS",
            "PFE", "KO", "PEP", "WMT", "XOM", "CVX"
        ]
        
        logger.info("DataStreamingService initialized")
    
    async def subscribe(self, websocket, stream_types: List[DataStreamType]):
        """Subscribe a WebSocket connection to specific data streams"""
        connection_id = id(websocket)
        self.subscribers[connection_id] = {
            'websocket': websocket,
            'stream_types': stream_types,
            'last_heartbeat': datetime.now()
        }
        logger.info(f"Client {connection_id} subscribed to streams: {[s.value for s in stream_types]}")
    
    async def unsubscribe(self, websocket):
        """Unsubscribe a WebSocket connection"""
        connection_id = id(websocket)
        if connection_id in self.subscribers:
            del self.subscribers[connection_id]
            logger.info(f"Client {connection_id} unsubscribed")
    
    async def start_streaming(self):
        """Start the data streaming service"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        logger.info("Starting data streaming service")
        
        # Start different streaming tasks
        tasks = [
            asyncio.create_task(self._stream_market_data()),
            asyncio.create_task(self._stream_portfolio_updates()),
            asyncio.create_task(self._stream_risk_alerts()),
            asyncio.create_task(self._stream_esg_alerts()),
            asyncio.create_task(self._stream_climate_updates()),
            asyncio.create_task(self._stream_compliance_alerts()),
            asyncio.create_task(self._stream_news_sentiment()),
            asyncio.create_task(self._cleanup_disconnected_clients())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in data streaming: {e}")
        finally:
            self.is_streaming = False
    
    async def stop_streaming(self):
        """Stop the data streaming service"""
        self.is_streaming = False
        logger.info("Data streaming service stopped")
    
    async def _stream_market_data(self):
        """Stream real-time market data"""
        while self.is_streaming:
            try:
                # Generate market data for random symbols
                symbols_to_update = random.sample(self.market_symbols, random.randint(3, 8))
                
                for symbol in symbols_to_update:
                    market_data = self._generate_market_data(symbol)
                    await self._broadcast_to_subscribers(
                        DataStreamType.MARKET_DATA,
                        {
                            'type': 'market_data',
                            'data': asdict(market_data)
                        }
                    )
                
                # Wait before next update (1-3 seconds)
                await asyncio.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error streaming market data: {e}")
                await asyncio.sleep(5)
    
    async def _stream_portfolio_updates(self):
        """Stream portfolio performance updates"""
        while self.is_streaming:
            try:
                # Generate portfolio updates every 30-60 seconds
                portfolio_update = self._generate_portfolio_update()
                await self._broadcast_to_subscribers(
                    DataStreamType.PORTFOLIO_UPDATES,
                    {
                        'type': 'portfolio_update',
                        'data': asdict(portfolio_update)
                    }
                )
                
                await asyncio.sleep(random.uniform(30, 60))
                
            except Exception as e:
                logger.error(f"Error streaming portfolio updates: {e}")
                await asyncio.sleep(30)
    
    async def _stream_risk_alerts(self):
        """Stream risk alerts and warnings"""
        while self.is_streaming:
            try:
                # Generate risk alerts every 2-5 minutes
                if random.random() < 0.3:  # 30% chance of generating an alert
                    risk_alert = self._generate_risk_alert()
                    await self._broadcast_to_subscribers(
                        DataStreamType.RISK_ALERTS,
                        {
                            'type': 'risk_alert',
                            'data': asdict(risk_alert)
                        }
                    )
                
                await asyncio.sleep(random.uniform(120, 300))
                
            except Exception as e:
                logger.error(f"Error streaming risk alerts: {e}")
                await asyncio.sleep(60)
    
    async def _stream_esg_alerts(self):
        """Stream ESG-related alerts and updates"""
        while self.is_streaming:
            try:
                # Generate ESG alerts every 5-10 minutes
                if random.random() < 0.2:  # 20% chance of generating an ESG alert
                    esg_alert = self._generate_esg_alert()
                    await self._broadcast_to_subscribers(
                        DataStreamType.ESG_ALERTS,
                        {
                            'type': 'esg_alert',
                            'data': asdict(esg_alert)
                        }
                    )
                
                await asyncio.sleep(random.uniform(300, 600))
                
            except Exception as e:
                logger.error(f"Error streaming ESG alerts: {e}")
                await asyncio.sleep(120)
    
    async def _stream_climate_updates(self):
        """Stream climate-related updates and events"""
        while self.is_streaming:
            try:
                # Generate climate updates every 10-20 minutes
                if random.random() < 0.15:  # 15% chance of generating a climate update
                    climate_update = self._generate_climate_update()
                    await self._broadcast_to_subscribers(
                        DataStreamType.CLIMATE_UPDATES,
                        {
                            'type': 'climate_update',
                            'data': asdict(climate_update)
                        }
                    )
                
                await asyncio.sleep(random.uniform(600, 1200))
                
            except Exception as e:
                logger.error(f"Error streaming climate updates: {e}")
                await asyncio.sleep(300)
    
    async def _stream_compliance_alerts(self):
        """Stream compliance alerts and regulatory updates"""
        while self.is_streaming:
            try:
                # Generate compliance alerts every 15-30 minutes
                if random.random() < 0.1:  # 10% chance of generating a compliance alert
                    compliance_alert = self._generate_compliance_alert()
                    await self._broadcast_to_subscribers(
                        DataStreamType.COMPLIANCE_ALERTS,
                        {
                            'type': 'compliance_alert',
                            'data': asdict(compliance_alert)
                        }
                    )
                
                await asyncio.sleep(random.uniform(900, 1800))
                
            except Exception as e:
                logger.error(f"Error streaming compliance alerts: {e}")
                await asyncio.sleep(300)
    
    async def _stream_news_sentiment(self):
        """Stream news sentiment analysis"""
        while self.is_streaming:
            try:
                # Generate news sentiment every 5-15 minutes
                if random.random() < 0.25:  # 25% chance of generating news sentiment
                    news_sentiment = self._generate_news_sentiment()
                    await self._broadcast_to_subscribers(
                        DataStreamType.NEWS_SENTIMENT,
                        {
                            'type': 'news_sentiment',
                            'data': asdict(news_sentiment)
                        }
                    )
                
                await asyncio.sleep(random.uniform(300, 900))
                
            except Exception as e:
                logger.error(f"Error streaming news sentiment: {e}")
                await asyncio.sleep(180)
    
    async def _cleanup_disconnected_clients(self):
        """Clean up disconnected WebSocket clients"""
        while self.is_streaming:
            try:
                current_time = datetime.now()
                disconnected_clients = []
                
                for connection_id, subscriber in self.subscribers.items():
                    try:
                        # Send heartbeat
                        await subscriber['websocket'].send_text(json.dumps({
                            'type': 'heartbeat',
                            'timestamp': current_time.isoformat()
                        }))
                        subscriber['last_heartbeat'] = current_time
                    except Exception:
                        disconnected_clients.append(connection_id)
                
                # Remove disconnected clients
                for connection_id in disconnected_clients:
                    if connection_id in self.subscribers:
                        del self.subscribers[connection_id]
                        logger.info(f"Removed disconnected client {connection_id}")
                
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in client cleanup: {e}")
                await asyncio.sleep(30)
    
    def _generate_market_data(self, symbol: str) -> MarketDataPoint:
        """Generate realistic market data for a symbol"""
        current_price = self.base_prices[symbol]
        
        # Generate price movement (random walk with slight upward bias)
        price_change = random.gauss(0.001, 0.02) * current_price  # 0.1% mean, 2% std
        new_price = max(current_price + price_change, 1.0)  # Minimum price of $1
        
        # Update base price and history
        self.base_prices[symbol] = new_price
        self.price_history[symbol].append(new_price)
        
        # Keep only last 100 prices for history
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Calculate change from previous price
        change = price_change
        change_percent = (change / current_price) * 100
        
        # Generate other market data
        volume = random.randint(100000, 10000000)
        spread = new_price * random.uniform(0.001, 0.005)  # 0.1% to 0.5% spread
        bid = new_price - spread / 2
        ask = new_price + spread / 2
        
        # 52-week high/low (mock)
        high_52w = new_price * random.uniform(1.1, 1.8)
        low_52w = new_price * random.uniform(0.6, 0.9)
        
        return MarketDataPoint(
            symbol=symbol,
            price=round(new_price, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            volume=volume,
            timestamp=datetime.now(),
            bid=round(bid, 2),
            ask=round(ask, 2),
            high_52w=round(high_52w, 2),
            low_52w=round(low_52w, 2),
            market_cap=random.randint(1000000000, 3000000000000),  # $1B to $3T
            pe_ratio=round(random.uniform(10, 50), 1)
        )
    
    def _generate_portfolio_update(self) -> PortfolioUpdate:
        """Generate portfolio performance update"""
        portfolio_id = f"portfolio_{random.randint(1, 10)}"
        
        # Generate portfolio metrics
        total_value = random.uniform(100000, 10000000)
        daily_return = random.gauss(0, 0.015) * total_value  # Normal distribution around 0
        daily_return_percent = (daily_return / total_value) * 100
        
        # Risk metrics
        risk_metrics = {
            'volatility': round(random.uniform(0.10, 0.30), 4),
            'var_95': round(random.uniform(-0.05, -0.01), 4),
            'beta': round(random.uniform(0.7, 1.3), 3),
            'sharpe_ratio': round(random.uniform(0.5, 2.0), 3)
        }
        
        # Top performers and losers
        symbols_sample = random.sample(self.market_symbols, 6)
        top_performers = [
            {
                'symbol': symbols_sample[i],
                'return': round(random.uniform(2, 8), 2),
                'weight': round(random.uniform(0.05, 0.15), 3)
            }
            for i in range(3)
        ]
        
        top_losers = [
            {
                'symbol': symbols_sample[i + 3],
                'return': round(random.uniform(-8, -2), 2),
                'weight': round(random.uniform(0.05, 0.15), 3)
            }
            for i in range(3)
        ]
        
        return PortfolioUpdate(
            portfolio_id=portfolio_id,
            total_value=round(total_value, 2),
            daily_return=round(daily_return, 2),
            daily_return_percent=round(daily_return_percent, 3),
            risk_metrics=risk_metrics,
            top_performers=top_performers,
            top_losers=top_losers,
            timestamp=datetime.now(),
            alerts_count=random.randint(0, 5)
        )
    
    def _generate_risk_alert(self) -> RiskAlert:
        """Generate risk alert"""
        risk_types = [
            "market_volatility", "concentration_risk", "liquidity_risk",
            "credit_risk", "operational_risk", "model_risk", "correlation_risk"
        ]
        
        severities = ["low", "medium", "high", "critical"]
        severity = random.choice(severities)
        risk_type = random.choice(risk_types)
        
        descriptions = {
            "market_volatility": "Elevated market volatility detected across multiple asset classes",
            "concentration_risk": "High concentration detected in single asset or sector",
            "liquidity_risk": "Liquidity constraints identified in portfolio holdings",
            "credit_risk": "Credit quality deterioration in fixed income holdings",
            "operational_risk": "Operational risk event affecting portfolio operations",
            "model_risk": "Model validation issues detected in risk calculations",
            "correlation_risk": "Unexpected correlation changes between asset classes"
        }
        
        affected_assets = random.sample(self.market_symbols, random.randint(1, 5))
        
        recommended_actions = {
            "low": ["Monitor situation", "Review risk limits"],
            "medium": ["Consider rebalancing", "Increase monitoring frequency", "Review hedging strategies"],
            "high": ["Immediate rebalancing required", "Implement hedging", "Reduce position sizes"],
            "critical": ["Emergency rebalancing", "Halt trading", "Escalate to risk committee"]
        }
        
        return RiskAlert(
            alert_id=f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            severity=severity,
            risk_type=risk_type,
            description=descriptions[risk_type],
            affected_assets=affected_assets,
            impact_score=random.uniform(0.1, 1.0),
            timestamp=datetime.now(),
            recommended_actions=recommended_actions[severity],
            confidence_level=random.uniform(0.7, 0.95)
        )
    
    def _generate_esg_alert(self) -> ESGAlert:
        """Generate ESG alert"""
        alert_types = ["controversy", "score_change", "news", "regulatory"]
        severities = ["low", "medium", "high"]
        
        company = random.choice(self.esg_companies)
        alert_type = random.choice(alert_types)
        severity = random.choice(severities)
        
        descriptions = {
            "controversy": f"ESG controversy identified for {company} related to environmental practices",
            "score_change": f"Significant ESG score change detected for {company}",
            "news": f"Negative ESG-related news coverage for {company}",
            "regulatory": f"New ESG regulatory requirements affecting {company}"
        }
        
        # ESG impact scores (E, S, G)
        esg_impact = {
            "environmental": round(random.uniform(-0.5, 0.5), 2),
            "social": round(random.uniform(-0.5, 0.5), 2),
            "governance": round(random.uniform(-0.5, 0.5), 2)
        }
        
        sources = ["Reuters", "Bloomberg ESG", "MSCI ESG", "Sustainalytics", "CDP"]
        
        return ESGAlert(
            alert_id=f"esg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            company=company,
            alert_type=alert_type,
            severity=severity,
            description=descriptions[alert_type],
            esg_impact=esg_impact,
            timestamp=datetime.now(),
            source=random.choice(sources),
            credibility_score=random.uniform(0.6, 0.95)
        )
    
    def _generate_climate_update(self) -> ClimateUpdate:
        """Generate climate update"""
        event_types = ["policy", "physical_risk", "transition_risk", "opportunity"]
        event_type = random.choice(event_types)
        
        descriptions = {
            "policy": "New climate policy announced affecting carbon pricing",
            "physical_risk": "Extreme weather event impacting supply chains",
            "transition_risk": "Technology disruption in renewable energy sector",
            "opportunity": "New green investment opportunity identified"
        }
        
        sectors = ["Energy", "Utilities", "Materials", "Industrials", "Transportation", "Real Estate"]
        affected_sectors = random.sample(sectors, random.randint(1, 3))
        
        impact_assessment = {
            "financial_impact": round(random.uniform(-0.1, 0.1), 3),
            "probability": round(random.uniform(0.1, 0.9), 2),
            "time_horizon": random.choice(["short", "medium", "long"])
        }
        
        geographic_scopes = ["Global", "North America", "Europe", "Asia-Pacific", "Emerging Markets"]
        
        return ClimateUpdate(
            update_id=f"climate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            event_type=event_type,
            description=descriptions[event_type],
            affected_sectors=affected_sectors,
            impact_assessment=impact_assessment,
            timestamp=datetime.now(),
            source="Climate Risk Analytics",
            geographic_scope=random.choice(geographic_scopes)
        )
    
    def _generate_compliance_alert(self) -> ComplianceAlert:
        """Generate compliance alert"""
        regulations = ["SFDR", "EU Taxonomy", "MiFID II", "AIFMD", "UCITS", "Basel III"]
        alert_types = ["violation", "warning", "update", "deadline"]
        severities = ["low", "medium", "high", "critical"]
        
        regulation = random.choice(regulations)
        alert_type = random.choice(alert_types)
        severity = random.choice(severities)
        
        descriptions = {
            "violation": f"Compliance violation detected under {regulation}",
            "warning": f"Potential compliance issue identified for {regulation}",
            "update": f"Regulatory update published for {regulation}",
            "deadline": f"Upcoming compliance deadline for {regulation}"
        }
        
        portfolios = [f"portfolio_{i}" for i in range(1, 11)]
        affected_portfolios = random.sample(portfolios, random.randint(1, 3))
        
        required_actions = {
            "violation": ["Immediate remediation", "File regulatory report", "Legal review"],
            "warning": ["Review compliance procedures", "Update documentation"],
            "update": ["Review new requirements", "Update policies", "Train staff"],
            "deadline": ["Complete required filings", "Submit documentation"]
        }
        
        # Set deadline for deadline alerts
        deadline = None
        if alert_type == "deadline":
            deadline = datetime.now() + timedelta(days=random.randint(7, 90))
        
        return ComplianceAlert(
            alert_id=f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            regulation=regulation,
            alert_type=alert_type,
            severity=severity,
            description=descriptions[alert_type],
            affected_portfolios=affected_portfolios,
            deadline=deadline,
            required_actions=required_actions[alert_type],
            timestamp=datetime.now()
        )
    
    def _generate_news_sentiment(self) -> NewsSentiment:
        """Generate news sentiment analysis"""
        headlines = [
            "Tech stocks rally on strong earnings reports",
            "Federal Reserve signals potential rate changes",
            "ESG investing gains momentum among institutional investors",
            "Climate change risks prompt portfolio reassessment",
            "Regulatory changes impact financial sector outlook",
            "Emerging markets show resilience amid global uncertainty",
            "Renewable energy sector attracts record investment",
            "Supply chain disruptions affect multiple industries"
        ]
        
        categories = [
            "earnings", "monetary_policy", "esg", "climate", "regulation",
            "emerging_markets", "energy", "supply_chain"
        ]
        
        headline = random.choice(headlines)
        sentiment_score = random.uniform(-1, 1)
        relevance_score = random.uniform(0.3, 1.0)
        
        affected_symbols = random.sample(self.market_symbols, random.randint(1, 5))
        news_categories = random.sample(categories, random.randint(1, 3))
        
        sources = ["Reuters", "Bloomberg", "Financial Times", "Wall Street Journal", "CNBC"]
        
        summaries = {
            "positive": "Positive market developments support investor confidence and asset valuations.",
            "negative": "Market concerns arise from regulatory and economic uncertainties affecting investor sentiment.",
            "neutral": "Mixed market signals provide balanced outlook for investment strategies."
        }
        
        sentiment_category = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
        
        return NewsSentiment(
            news_id=f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            headline=headline,
            sentiment_score=round(sentiment_score, 3),
            relevance_score=round(relevance_score, 3),
            affected_symbols=affected_symbols,
            categories=news_categories,
            timestamp=datetime.now(),
            source=random.choice(sources),
            summary=summaries[sentiment_category]
        )
    
    async def _broadcast_to_subscribers(self, stream_type: DataStreamType, message: Dict[str, Any]):
        """Broadcast message to all subscribers of a specific stream type"""
        if not self.subscribers:
            return
        
        disconnected_clients = []
        message_json = json.dumps(message, default=str)
        
        for connection_id, subscriber in self.subscribers.items():
            if stream_type in subscriber['stream_types']:
                try:
                    await subscriber['websocket'].send_text(message_json)
                except Exception as e:
                    logger.warning(f"Failed to send message to client {connection_id}: {e}")
                    disconnected_clients.append(connection_id)
        
        # Remove disconnected clients
        for connection_id in disconnected_clients:
            if connection_id in self.subscribers:
                del self.subscribers[connection_id]
    
    def get_subscriber_count(self) -> int:
        """Get current number of subscribers"""
        return len(self.subscribers)
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming service status"""
        return {
            'is_streaming': self.is_streaming,
            'subscriber_count': self.get_subscriber_count(),
            'active_streams': list(DataStreamType),
            'market_symbols': len(self.market_symbols),
            'uptime': datetime.now().isoformat()
        }

# Global instance
data_streaming_service = DataStreamingService()