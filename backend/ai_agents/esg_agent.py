import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json
import requests
from scipy import stats

logger = logging.getLogger(__name__)

class ESGAgent:
    """AI agent for ESG data analysis, scoring, and insights"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=openai_api_key
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # ESG scoring weights
        self.esg_weights = {
            'environmental': 0.4,
            'social': 0.3,
            'governance': 0.3
        }
        
        # ESG data sources and APIs
        self.data_sources = {
            'msci': {'weight': 0.3, 'available': True},
            'sustainalytics': {'weight': 0.25, 'available': True},
            'refinitiv': {'weight': 0.25, 'available': True},
            'bloomberg': {'weight': 0.2, 'available': True}
        }
        
        # Initialize tools
        self.tools = self._create_esg_tools()
        
        # Create agent
        self.agent = self._create_esg_agent()
        
        logger.info("Initialized ESGAgent with multi-source data integration")
    
    def _create_esg_tools(self) -> List[Tool]:
        """Create tools for ESG analysis"""
        
        tools = [
            Tool(
                name="esg_data_collector",
                description="Collect ESG data from multiple sources for a company",
                func=self._collect_esg_data
            ),
            Tool(
                name="esg_score_calculator",
                description="Calculate comprehensive ESG scores using multiple methodologies",
                func=self._calculate_esg_score
            ),
            Tool(
                name="esg_trend_analyzer",
                description="Analyze ESG trends and performance over time",
                func=self._analyze_esg_trends
            ),
            Tool(
                name="peer_comparison_analyzer",
                description="Compare ESG performance against industry peers",
                func=self._analyze_peer_comparison
            ),
            Tool(
                name="controversy_monitor",
                description="Monitor and assess ESG controversies and incidents",
                func=self._monitor_controversies
            ),
            Tool(
                name="impact_assessor",
                description="Assess real-world impact of ESG initiatives",
                func=self._assess_impact
            )
        ]
        
        return tools
    
    def _create_esg_agent(self) -> AgentExecutor:
        """Create the ESG analysis agent"""
        
        system_prompt = """
        You are an expert ESG (Environmental, Social, Governance) analyst with deep knowledge of:
        - ESG scoring methodologies and frameworks
        - Sustainability reporting standards (GRI, SASB, TCFD)
        - Impact measurement and management
        - ESG data quality and verification
        - Regulatory requirements and best practices
        
        Your role is to:
        1. Analyze ESG data from multiple sources
        2. Provide comprehensive ESG scores and insights
        3. Identify ESG risks and opportunities
        4. Monitor ESG trends and controversies
        5. Assess real-world impact of ESG initiatives
        6. Provide actionable recommendations for ESG improvement
        
        Always provide evidence-based analysis with clear reasoning and cite data sources.
        Consider both quantitative metrics and qualitative factors in your assessments.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5
        )
    
    async def _collect_esg_data(self, company_symbol: str) -> str:
        """Collect ESG data from multiple sources"""
        try:
            # Mock ESG data collection from multiple sources
            esg_data = {
                'company': company_symbol,
                'data_sources': {
                    'msci': {
                        'overall_score': np.random.uniform(3.0, 8.5),
                        'environmental': np.random.uniform(2.0, 9.0),
                        'social': np.random.uniform(3.0, 8.0),
                        'governance': np.random.uniform(4.0, 9.0),
                        'last_updated': datetime.now().isoformat()
                    },
                    'sustainalytics': {
                        'risk_score': np.random.uniform(10, 40),  # Lower is better
                        'risk_category': np.random.choice(['Low', 'Medium', 'High']),
                        'management_score': np.random.uniform(40, 90),
                        'last_updated': datetime.now().isoformat()
                    },
                    'refinitiv': {
                        'esg_score': np.random.uniform(30, 95),
                        'environmental_pillar': np.random.uniform(25, 90),
                        'social_pillar': np.random.uniform(35, 85),
                        'governance_pillar': np.random.uniform(40, 95),
                        'last_updated': datetime.now().isoformat()
                    }
                },
                'collection_timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(esg_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error collecting ESG data for {company_symbol}: {e}")
            return f"Error collecting ESG data: {str(e)}"
    
    async def _calculate_esg_score(self, esg_data: str) -> str:
        """Calculate comprehensive ESG score"""
        try:
            data = json.loads(esg_data)
            
            # Normalize scores from different providers
            normalized_scores = {}
            
            # MSCI (0-10 scale)
            if 'msci' in data['data_sources']:
                msci = data['data_sources']['msci']
                normalized_scores['msci'] = {
                    'overall': msci['overall_score'] * 10,  # Convert to 0-100
                    'environmental': msci['environmental'] * 10,
                    'social': msci['social'] * 10,
                    'governance': msci['governance'] * 10
                }
            
            # Sustainalytics (risk score - invert and normalize)
            if 'sustainalytics' in data['data_sources']:
                sust = data['data_sources']['sustainalytics']
                # Convert risk score to performance score (invert)
                risk_to_perf = max(0, 100 - sust['risk_score'] * 2)
                normalized_scores['sustainalytics'] = {
                    'overall': risk_to_perf,
                    'management': sust['management_score']
                }
            
            # Refinitiv (already 0-100 scale)
            if 'refinitiv' in data['data_sources']:
                ref = data['data_sources']['refinitiv']
                normalized_scores['refinitiv'] = {
                    'overall': ref['esg_score'],
                    'environmental': ref['environmental_pillar'],
                    'social': ref['social_pillar'],
                    'governance': ref['governance_pillar']
                }
            
            # Calculate weighted composite score
            composite_scores = {'environmental': 0, 'social': 0, 'governance': 0}
            total_weight = 0
            
            for source, scores in normalized_scores.items():
                if source in self.data_sources:
                    weight = self.data_sources[source]['weight']
                    total_weight += weight
                    
                    if 'environmental' in scores:
                        composite_scores['environmental'] += scores['environmental'] * weight
                    if 'social' in scores:
                        composite_scores['social'] += scores['social'] * weight
                    if 'governance' in scores:
                        composite_scores['governance'] += scores['governance'] * weight
            
            # Normalize by total weight
            if total_weight > 0:
                for pillar in composite_scores:
                    composite_scores[pillar] /= total_weight
            
            # Calculate overall ESG score
            overall_score = (
                composite_scores['environmental'] * self.esg_weights['environmental'] +
                composite_scores['social'] * self.esg_weights['social'] +
                composite_scores['governance'] * self.esg_weights['governance']
            )
            
            # Determine ESG rating
            if overall_score >= 80:
                rating = 'AAA'
            elif overall_score >= 70:
                rating = 'AA'
            elif overall_score >= 60:
                rating = 'A'
            elif overall_score >= 50:
                rating = 'BBB'
            elif overall_score >= 40:
                rating = 'BB'
            elif overall_score >= 30:
                rating = 'B'
            else:
                rating = 'CCC'
            
            result = {
                'company': data['company'],
                'composite_score': round(overall_score, 2),
                'rating': rating,
                'pillar_scores': {
                    'environmental': round(composite_scores['environmental'], 2),
                    'social': round(composite_scores['social'], 2),
                    'governance': round(composite_scores['governance'], 2)
                },
                'data_quality': {
                    'sources_used': len(normalized_scores),
                    'coverage': 'comprehensive' if len(normalized_scores) >= 3 else 'partial',
                    'confidence_level': min(95, 60 + len(normalized_scores) * 10)
                },
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error calculating ESG score: {e}")
            return f"Error calculating ESG score: {str(e)}"
    
    async def _analyze_esg_trends(self, company_symbol: str, period: str = "2Y") -> str:
        """Analyze ESG trends over time"""
        try:
            # Mock trend analysis
            periods = {
                "1Y": 12,
                "2Y": 24,
                "3Y": 36
            }
            
            months = periods.get(period, 24)
            
            # Generate mock historical ESG scores
            base_score = np.random.uniform(50, 80)
            trend = np.random.uniform(-0.5, 1.0)  # Monthly change
            noise = np.random.normal(0, 2, months)
            
            historical_scores = []
            current_score = base_score
            
            for i in range(months):
                current_score += trend + noise[i]
                current_score = max(0, min(100, current_score))  # Clamp to 0-100
                
                date = datetime.now() - timedelta(days=30 * (months - i))
                historical_scores.append({
                    'date': date.strftime('%Y-%m'),
                    'score': round(current_score, 2)
                })
            
            # Calculate trend statistics
            scores = [s['score'] for s in historical_scores]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(scores)), scores
            )
            
            trend_direction = 'improving' if slope > 0.1 else 'declining' if slope < -0.1 else 'stable'
            volatility = np.std(scores)
            
            result = {
                'company': company_symbol,
                'period': period,
                'historical_scores': historical_scores,
                'trend_analysis': {
                    'direction': trend_direction,
                    'slope': round(slope, 4),
                    'correlation': round(r_value, 3),
                    'volatility': round(volatility, 2),
                    'current_score': round(scores[-1], 2),
                    'period_change': round(scores[-1] - scores[0], 2)
                },
                'key_insights': [
                    f"ESG score has {trend_direction} over the {period} period",
                    f"Score volatility is {'high' if volatility > 5 else 'moderate' if volatility > 2 else 'low'}",
                    f"Trend correlation is {'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.4 else 'weak'}"
                ],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error analyzing ESG trends for {company_symbol}: {e}")
            return f"Error analyzing ESG trends: {str(e)}"
    
    async def _analyze_peer_comparison(self, company_symbol: str, sector: str = None) -> str:
        """Compare ESG performance against industry peers"""
        try:
            # Mock peer comparison
            if not sector:
                sector = np.random.choice(['Technology', 'Energy', 'Healthcare', 'Financials'])
            
            # Generate mock peer data
            num_peers = np.random.randint(8, 15)
            company_score = np.random.uniform(50, 85)
            
            peer_scores = np.random.normal(65, 12, num_peers)
            peer_scores = np.clip(peer_scores, 0, 100)
            
            peers = []
            for i, score in enumerate(peer_scores):
                peers.append({
                    'company': f"Peer_{i+1}",
                    'score': round(score, 2)
                })
            
            # Add target company
            peers.append({
                'company': company_symbol,
                'score': round(company_score, 2)
            })
            
            # Sort by score
            peers.sort(key=lambda x: x['score'], reverse=True)
            
            # Find company rank
            company_rank = next(i for i, p in enumerate(peers) if p['company'] == company_symbol) + 1
            
            # Calculate percentiles
            all_scores = [p['score'] for p in peers if p['company'] != company_symbol]
            percentile = stats.percentileofscore(all_scores, company_score)
            
            # Sector statistics
            sector_stats = {
                'mean': round(np.mean(all_scores), 2),
                'median': round(np.median(all_scores), 2),
                'std': round(np.std(all_scores), 2),
                'min': round(np.min(all_scores), 2),
                'max': round(np.max(all_scores), 2)
            }
            
            result = {
                'company': company_symbol,
                'sector': sector,
                'company_score': round(company_score, 2),
                'sector_rank': f"{company_rank}/{len(peers)}",
                'percentile': round(percentile, 1),
                'sector_statistics': sector_stats,
                'peer_comparison': peers,
                'performance_vs_sector': {
                    'vs_mean': round(company_score - sector_stats['mean'], 2),
                    'vs_median': round(company_score - sector_stats['median'], 2),
                    'relative_performance': 'above average' if company_score > sector_stats['mean'] else 'below average'
                },
                'key_insights': [
                    f"Ranks {company_rank} out of {len(peers)} companies in {sector} sector",
                    f"Scores in the {percentile:.0f}th percentile of sector peers",
                    f"Performance is {abs(company_score - sector_stats['mean']):.1f} points {'above' if company_score > sector_stats['mean'] else 'below'} sector average"
                ],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error analyzing peer comparison for {company_symbol}: {e}")
            return f"Error analyzing peer comparison: {str(e)}"
    
    async def _monitor_controversies(self, company_symbol: str) -> str:
        """Monitor ESG controversies and incidents"""
        try:
            # Mock controversy monitoring
            controversy_types = [
                'Environmental violations',
                'Labor disputes',
                'Data privacy breaches',
                'Governance scandals',
                'Product safety issues',
                'Human rights concerns'
            ]
            
            # Generate mock controversies
            num_controversies = np.random.randint(0, 4)
            controversies = []
            
            for i in range(num_controversies):
                severity = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2])
                controversy_type = np.random.choice(controversy_types)
                
                days_ago = np.random.randint(1, 365)
                date = datetime.now() - timedelta(days=days_ago)
                
                controversies.append({
                    'id': f"CONTR_{i+1:03d}",
                    'type': controversy_type,
                    'severity': severity,
                    'date': date.strftime('%Y-%m-%d'),
                    'description': f"{controversy_type} incident reported",
                    'status': np.random.choice(['Active', 'Resolved', 'Under Investigation']),
                    'impact_score': np.random.uniform(1, 10) if severity == 'High' else np.random.uniform(0.1, 5),
                    'media_coverage': np.random.choice(['Low', 'Medium', 'High'])
                })
            
            # Calculate controversy score
            if controversies:
                total_impact = sum(c['impact_score'] for c in controversies)
                active_controversies = len([c for c in controversies if c['status'] == 'Active'])
                high_severity = len([c for c in controversies if c['severity'] == 'High'])
            else:
                total_impact = 0
                active_controversies = 0
                high_severity = 0
            
            controversy_score = max(0, 100 - (total_impact * 5 + active_controversies * 10 + high_severity * 15))
            
            result = {
                'company': company_symbol,
                'controversy_score': round(controversy_score, 2),
                'total_controversies': len(controversies),
                'active_controversies': active_controversies,
                'high_severity_controversies': high_severity,
                'controversies': controversies,
                'risk_assessment': {
                    'overall_risk': 'Low' if controversy_score > 80 else 'Medium' if controversy_score > 60 else 'High',
                    'reputational_risk': 'High' if high_severity > 0 else 'Medium' if active_controversies > 0 else 'Low',
                    'regulatory_risk': 'High' if any(c['type'] in ['Environmental violations', 'Data privacy breaches'] for c in controversies) else 'Low'
                },
                'recommendations': [
                    "Monitor ongoing controversies closely",
                    "Implement proactive ESG risk management",
                    "Enhance stakeholder communication"
                ] if controversies else ["Maintain current ESG practices", "Continue monitoring for emerging risks"],
                'monitoring_timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error monitoring controversies for {company_symbol}: {e}")
            return f"Error monitoring controversies: {str(e)}"
    
    async def _assess_impact(self, company_symbol: str, initiative_type: str = None) -> str:
        """Assess real-world impact of ESG initiatives"""
        try:
            # Mock impact assessment
            initiative_types = [
                'Carbon reduction',
                'Renewable energy',
                'Diversity & inclusion',
                'Community investment',
                'Sustainable supply chain',
                'Waste reduction'
            ]
            
            if not initiative_type:
                initiative_type = np.random.choice(initiative_types)
            
            # Generate mock impact metrics
            impact_metrics = {
                'Carbon reduction': {
                    'metric': 'CO2 emissions reduced',
                    'value': np.random.uniform(10000, 100000),
                    'unit': 'tonnes CO2e',
                    'baseline_year': 2020,
                    'target_year': 2030,
                    'progress': np.random.uniform(0.2, 0.8)
                },
                'Renewable energy': {
                    'metric': 'Renewable energy capacity',
                    'value': np.random.uniform(50, 500),
                    'unit': 'MW',
                    'baseline_year': 2020,
                    'target_year': 2025,
                    'progress': np.random.uniform(0.3, 0.9)
                },
                'Diversity & inclusion': {
                    'metric': 'Women in leadership',
                    'value': np.random.uniform(25, 45),
                    'unit': '%',
                    'baseline_year': 2020,
                    'target_year': 2025,
                    'progress': np.random.uniform(0.4, 0.8)
                }
            }
            
            metrics = impact_metrics.get(initiative_type, {
                'metric': 'Impact score',
                'value': np.random.uniform(50, 90),
                'unit': 'points',
                'baseline_year': 2020,
                'target_year': 2025,
                'progress': np.random.uniform(0.3, 0.7)
            })
            
            # Calculate impact score
            impact_score = metrics['progress'] * 100
            
            result = {
                'company': company_symbol,
                'initiative_type': initiative_type,
                'impact_metrics': metrics,
                'impact_score': round(impact_score, 2),
                'assessment': {
                    'effectiveness': 'High' if impact_score > 70 else 'Medium' if impact_score > 40 else 'Low',
                    'progress_rate': 'On track' if metrics['progress'] > 0.5 else 'Behind schedule',
                    'materiality': 'High' if initiative_type in ['Carbon reduction', 'Renewable energy'] else 'Medium'
                },
                'stakeholder_benefits': [
                    'Environmental protection',
                    'Community development',
                    'Employee wellbeing',
                    'Investor confidence'
                ],
                'recommendations': [
                    'Continue current initiatives',
                    'Scale successful programs',
                    'Enhance measurement and reporting'
                ],
                'assessment_timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error assessing impact for {company_symbol}: {e}")
            return f"Error assessing impact: {str(e)}"
    
    async def analyze_company_esg(self, company_symbol: str) -> Dict[str, Any]:
        """Comprehensive ESG analysis for a company"""
        try:
            # Collect ESG data
            esg_data = await self._collect_esg_data(company_symbol)
            
            # Calculate ESG score
            esg_score = await self._calculate_esg_score(esg_data)
            
            # Analyze trends
            trends = await self._analyze_esg_trends(company_symbol)
            
            # Peer comparison
            peer_comparison = await self._analyze_peer_comparison(company_symbol)
            
            # Monitor controversies
            controversies = await self._monitor_controversies(company_symbol)
            
            # Assess impact
            impact = await self._assess_impact(company_symbol)
            
            # Combine all analyses
            comprehensive_analysis = {
                'company': company_symbol,
                'esg_data': json.loads(esg_data),
                'esg_score': json.loads(esg_score),
                'trends': json.loads(trends),
                'peer_comparison': json.loads(peer_comparison),
                'controversies': json.loads(controversies),
                'impact_assessment': json.loads(impact),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Completed comprehensive ESG analysis for {company_symbol}")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive ESG analysis for {company_symbol}: {e}")
            raise
    
    async def analyze_portfolio_esg(self, portfolio_holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze ESG metrics for an entire portfolio"""
        try:
            portfolio_analysis = {
                'portfolio_id': 'portfolio_analysis',
                'holdings_count': len(portfolio_holdings),
                'holdings_analysis': [],
                'portfolio_metrics': {},
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Analyze each holding
            total_weight = 0
            weighted_esg_score = 0
            
            for holding in portfolio_holdings:
                symbol = holding.get('symbol', '')
                weight = holding.get('weight', 0)
                
                if symbol and weight > 0:
                    # Get ESG analysis for holding
                    holding_analysis = await self.analyze_company_esg(symbol)
                    
                    # Extract key metrics
                    esg_score = holding_analysis['esg_score']['composite_score']
                    
                    portfolio_analysis['holdings_analysis'].append({
                        'symbol': symbol,
                        'weight': weight,
                        'esg_score': esg_score,
                        'esg_rating': holding_analysis['esg_score']['rating']
                    })
                    
                    # Calculate weighted metrics
                    weighted_esg_score += esg_score * weight
                    total_weight += weight
            
            # Calculate portfolio-level metrics
            if total_weight > 0:
                portfolio_esg_score = weighted_esg_score / total_weight
                
                # Determine portfolio ESG rating
                if portfolio_esg_score >= 80:
                    portfolio_rating = 'AAA'
                elif portfolio_esg_score >= 70:
                    portfolio_rating = 'AA'
                elif portfolio_esg_score >= 60:
                    portfolio_rating = 'A'
                elif portfolio_esg_score >= 50:
                    portfolio_rating = 'BBB'
                elif portfolio_esg_score >= 40:
                    portfolio_rating = 'BB'
                elif portfolio_esg_score >= 30:
                    portfolio_rating = 'B'
                else:
                    portfolio_rating = 'CCC'
                
                portfolio_analysis['portfolio_metrics'] = {
                    'weighted_esg_score': round(portfolio_esg_score, 2),
                    'portfolio_rating': portfolio_rating,
                    'total_weight_analyzed': round(total_weight, 4),
                    'coverage': round(total_weight * 100, 1) if total_weight <= 1 else 100
                }
            
            logger.info(f"Completed portfolio ESG analysis for {len(portfolio_holdings)} holdings")
            return portfolio_analysis
            
        except Exception as e:
            logger.error(f"Error in portfolio ESG analysis: {e}")
            raise
    
    def get_esg_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate ESG improvement recommendations based on analysis"""
        try:
            recommendations = []
            
            if 'esg_score' in analysis_results:
                score = analysis_results['esg_score']['composite_score']
                
                if score < 50:
                    recommendations.extend([
                        "Develop comprehensive ESG strategy and governance framework",
                        "Implement ESG data collection and reporting systems",
                        "Engage with ESG rating agencies to improve scores"
                    ])
                elif score < 70:
                    recommendations.extend([
                        "Enhance ESG disclosure and transparency",
                        "Set science-based targets for key ESG metrics",
                        "Improve stakeholder engagement processes"
                    ])
                else:
                    recommendations.extend([
                        "Maintain ESG leadership position",
                        "Share best practices with industry peers",
                        "Explore innovative ESG solutions"
                    ])
            
            if 'controversies' in analysis_results:
                controversy_score = analysis_results['controversies']['controversy_score']
                if controversy_score < 70:
                    recommendations.append("Implement proactive controversy monitoring and response system")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating ESG recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]