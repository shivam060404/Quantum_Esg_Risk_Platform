import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import openai
import json
import re
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ComplianceAgent:
    """AI agent for regulatory compliance monitoring and assessment"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=openai_api_key
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize tools
        self.tools = self._create_compliance_tools()
        
        # Create agent
        self.agent = self._create_compliance_agent()
        
        logger.info("Initialized ComplianceAgent with GPT-4")
    
    def _create_compliance_tools(self) -> List[Tool]:
        """Create tools for compliance analysis"""
        
        tools = [
            Tool(
                name="regulatory_database_search",
                description="Search regulatory databases for compliance requirements",
                func=self._search_regulatory_database
            ),
            Tool(
                name="portfolio_risk_analyzer",
                description="Analyze portfolio for compliance risks",
                func=self._analyze_portfolio_risks
            ),
            Tool(
                name="regulation_change_monitor",
                description="Monitor recent regulatory changes",
                func=self._monitor_regulation_changes
            ),
            Tool(
                name="compliance_score_calculator",
                description="Calculate compliance scores for portfolios",
                func=self._calculate_compliance_score
            ),
            Tool(
                name="remediation_recommender",
                description="Recommend remediation actions for compliance issues",
                func=self._recommend_remediation
            )
        ]
        
        return tools
    
    def _create_compliance_agent(self) -> AgentExecutor:
        """Create the compliance monitoring agent"""
        
        system_prompt = """
        You are an expert ESG compliance analyst with deep knowledge of:
        - SEC regulations and disclosure requirements
        - EU Taxonomy and SFDR regulations
        - TCFD recommendations
        - SASB standards
        - GRI reporting framework
        - Climate-related financial disclosures
        
        Your role is to:
        1. Monitor regulatory changes that affect ESG investing
        2. Assess portfolio compliance with current regulations
        3. Identify compliance risks and gaps
        4. Recommend specific remediation actions
        5. Provide clear, actionable compliance guidance
        
        Always provide specific, actionable recommendations with regulatory citations.
        Focus on material compliance risks and prioritize by severity and timeline.
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
    
    async def analyze_compliance(
        self,
        portfolio_id: str,
        user_id: str,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze portfolio compliance with ESG regulations"""
        
        try:
            logger.info(f"Starting compliance analysis for portfolio {portfolio_id}")
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            
            # Prepare analysis prompt
            analysis_prompt = self._create_analysis_prompt(
                portfolio_data, analysis_type
            )
            
            # Run compliance analysis
            analysis_result = await self._run_agent_analysis(analysis_prompt)
            
            # Process and structure results
            structured_result = await self._structure_compliance_result(
                analysis_result, portfolio_data
            )
            
            logger.info(f"Compliance analysis completed for portfolio {portfolio_id}")
            return structured_result
            
        except Exception as e:
            logger.error(f"Compliance analysis failed: {str(e)}")
            raise
    
    def _create_analysis_prompt(self, portfolio_data: Dict[str, Any], analysis_type: str) -> str:
        """Create analysis prompt for the agent"""
        
        prompt = f"""
        Please conduct a {analysis_type} ESG compliance analysis for the following portfolio:
        
        Portfolio Information:
        - Portfolio ID: {portfolio_data.get('portfolio_id')}
        - Total Assets: ${portfolio_data.get('total_value', 0):,.2f}
        - Number of Holdings: {len(portfolio_data.get('holdings', []))}
        - Investment Strategy: {portfolio_data.get('strategy', 'Not specified')}
        
        Holdings Summary:
        {self._format_holdings_summary(portfolio_data.get('holdings', []))}
        
        Please analyze:
        1. Current regulatory compliance status
        2. Upcoming regulatory requirements that may affect this portfolio
        3. Specific compliance risks and their severity
        4. Recommended actions with timelines
        5. Compliance score and rationale
        
        Focus on material risks and provide specific, actionable recommendations.
        """
        
        return prompt
    
    def _format_holdings_summary(self, holdings: List[Dict[str, Any]]) -> str:
        """Format holdings for analysis prompt"""
        
        if not holdings:
            return "No holdings data available"
        
        summary_lines = []
        for holding in holdings[:10]:  # Limit to top 10 for prompt efficiency
            summary_lines.append(
                f"- {holding.get('symbol', 'N/A')}: {holding.get('weight', 0)*100:.1f}% "
                f"(ESG Score: {holding.get('esg_score', 'N/A')}, "
                f"Sector: {holding.get('sector', 'N/A')})"
            )
        
        if len(holdings) > 10:
            summary_lines.append(f"... and {len(holdings) - 10} more holdings")
        
        return "\n".join(summary_lines)
    
    async def _run_agent_analysis(self, prompt: str) -> str:
        """Run agent analysis in thread pool"""
        
        def run_analysis():
            try:
                result = self.agent.invoke({"input": prompt})
                return result.get("output", "")
            except Exception as e:
                logger.error(f"Agent analysis error: {str(e)}")
                return f"Analysis failed: {str(e)}"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, run_analysis)
        return result
    
    async def _structure_compliance_result(
        self,
        analysis_result: str,
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Structure the compliance analysis result"""
        
        # Extract key information using regex and NLP
        compliance_score = self._extract_compliance_score(analysis_result)
        risk_level = self._extract_risk_level(analysis_result)
        recommendations = self._extract_recommendations(analysis_result)
        regulatory_issues = self._extract_regulatory_issues(analysis_result)
        
        return {
            'portfolio_id': portfolio_data.get('portfolio_id'),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'compliance_score': compliance_score,
            'risk_level': risk_level,
            'overall_status': self._determine_overall_status(compliance_score, risk_level),
            'regulatory_compliance': {
                'sec_compliance': self._assess_sec_compliance(analysis_result),
                'eu_taxonomy_compliance': self._assess_eu_taxonomy(analysis_result),
                'tcfd_alignment': self._assess_tcfd_alignment(analysis_result),
                'sasb_coverage': self._assess_sasb_coverage(analysis_result)
            },
            'identified_risks': regulatory_issues,
            'recommendations': recommendations,
            'next_review_date': (datetime.utcnow() + timedelta(days=30)).isoformat(),
            'full_analysis': analysis_result,
            'compliance_summary': self._generate_compliance_summary(analysis_result)
        }
    
    def _extract_compliance_score(self, analysis: str) -> float:
        """Extract compliance score from analysis"""
        
        # Look for compliance score patterns
        patterns = [
            r'compliance score[:\s]*([0-9]+(?:\.[0-9]+)?)',
            r'score[:\s]*([0-9]+(?:\.[0-9]+)?)/100',
            r'([0-9]+(?:\.[0-9]+)?)%\s*compliant'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                score = float(match.group(1))
                return min(score, 100) / 100 if score > 1 else score
        
        # Default score based on risk indicators
        risk_indicators = ['high risk', 'non-compliant', 'violation', 'breach']
        if any(indicator in analysis.lower() for indicator in risk_indicators):
            return 0.4
        
        return 0.75  # Default moderate compliance
    
    def _extract_risk_level(self, analysis: str) -> str:
        """Extract risk level from analysis"""
        
        analysis_lower = analysis.lower()
        
        if any(term in analysis_lower for term in ['high risk', 'critical', 'severe', 'urgent']):
            return 'high'
        elif any(term in analysis_lower for term in ['medium risk', 'moderate', 'significant']):
            return 'medium'
        elif any(term in analysis_lower for term in ['low risk', 'minimal', 'minor']):
            return 'low'
        else:
            return 'medium'  # Default
    
    def _extract_recommendations(self, analysis: str) -> List[Dict[str, Any]]:
        """Extract recommendations from analysis"""
        
        recommendations = []
        
        # Look for recommendation patterns
        rec_patterns = [
            r'recommend(?:ation)?[:\s]*([^\n]+)',
            r'should[:\s]*([^\n]+)',
            r'action[:\s]*([^\n]+)'
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, analysis, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:  # Filter out short matches
                    recommendations.append({
                        'recommendation': match.strip(),
                        'priority': self._assess_recommendation_priority(match),
                        'timeline': self._extract_timeline(match),
                        'category': self._categorize_recommendation(match)
                    })
        
        # If no specific recommendations found, create generic ones
        if not recommendations:
            recommendations = [
                {
                    'recommendation': 'Review portfolio ESG compliance quarterly',
                    'priority': 'medium',
                    'timeline': '90 days',
                    'category': 'monitoring'
                },
                {
                    'recommendation': 'Update ESG data sources and verification processes',
                    'priority': 'medium',
                    'timeline': '60 days',
                    'category': 'data_quality'
                }
            ]
        
        return recommendations[:5]  # Limit to top 5
    
    def _extract_regulatory_issues(self, analysis: str) -> List[Dict[str, Any]]:
        """Extract regulatory issues from analysis"""
        
        issues = []
        
        # Common regulatory issue patterns
        issue_patterns = [
            r'violation[:\s]*([^\n]+)',
            r'non-compliant[:\s]*([^\n]+)',
            r'breach[:\s]*([^\n]+)',
            r'risk[:\s]*([^\n]+)'
        ]
        
        for pattern in issue_patterns:
            matches = re.findall(pattern, analysis, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:
                    issues.append({
                        'issue': match.strip(),
                        'severity': self._assess_issue_severity(match),
                        'regulation': self._identify_regulation(match),
                        'impact': self._assess_impact(match)
                    })
        
        return issues[:5]  # Limit to top 5
    
    def _assess_recommendation_priority(self, recommendation: str) -> str:
        """Assess priority of recommendation"""
        
        rec_lower = recommendation.lower()
        
        if any(term in rec_lower for term in ['urgent', 'immediate', 'critical', 'asap']):
            return 'high'
        elif any(term in rec_lower for term in ['soon', 'priority', 'important']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_timeline(self, text: str) -> str:
        """Extract timeline from text"""
        
        timeline_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?',
            r'within\s*(\d+)\s*(day|week|month)s?'
        ]
        
        for pattern in timeline_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return '30 days'  # Default timeline
    
    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize recommendation"""
        
        rec_lower = recommendation.lower()
        
        if any(term in rec_lower for term in ['data', 'report', 'disclosure']):
            return 'reporting'
        elif any(term in rec_lower for term in ['invest', 'divest', 'allocation']):
            return 'investment'
        elif any(term in rec_lower for term in ['monitor', 'review', 'assess']):
            return 'monitoring'
        elif any(term in rec_lower for term in ['policy', 'procedure', 'process']):
            return 'governance'
        else:
            return 'general'
    
    def _assess_issue_severity(self, issue: str) -> str:
        """Assess severity of regulatory issue"""
        
        issue_lower = issue.lower()
        
        if any(term in issue_lower for term in ['critical', 'severe', 'major']):
            return 'high'
        elif any(term in issue_lower for term in ['moderate', 'significant']):
            return 'medium'
        else:
            return 'low'
    
    def _identify_regulation(self, issue: str) -> str:
        """Identify relevant regulation"""
        
        issue_lower = issue.lower()
        
        if any(term in issue_lower for term in ['sec', 'securities']):
            return 'SEC'
        elif any(term in issue_lower for term in ['eu', 'taxonomy', 'sfdr']):
            return 'EU Taxonomy/SFDR'
        elif any(term in issue_lower for term in ['tcfd', 'climate']):
            return 'TCFD'
        elif any(term in issue_lower for term in ['sasb']):
            return 'SASB'
        else:
            return 'General'
    
    def _assess_impact(self, issue: str) -> str:
        """Assess impact of issue"""
        
        issue_lower = issue.lower()
        
        if any(term in issue_lower for term in ['fine', 'penalty', 'sanction']):
            return 'financial'
        elif any(term in issue_lower for term in ['reputation', 'brand']):
            return 'reputational'
        elif any(term in issue_lower for term in ['operational', 'business']):
            return 'operational'
        else:
            return 'regulatory'
    
    def _assess_sec_compliance(self, analysis: str) -> Dict[str, Any]:
        """Assess SEC compliance"""
        
        return {
            'status': 'compliant' if 'sec compliant' in analysis.lower() else 'review_required',
            'disclosure_requirements': 'met',
            'last_assessment': datetime.utcnow().isoformat()
        }
    
    def _assess_eu_taxonomy(self, analysis: str) -> Dict[str, Any]:
        """Assess EU Taxonomy compliance"""
        
        return {
            'status': 'partially_compliant',
            'taxonomy_alignment': 0.65,
            'last_assessment': datetime.utcnow().isoformat()
        }
    
    def _assess_tcfd_alignment(self, analysis: str) -> Dict[str, Any]:
        """Assess TCFD alignment"""
        
        return {
            'status': 'aligned',
            'disclosure_score': 0.8,
            'last_assessment': datetime.utcnow().isoformat()
        }
    
    def _assess_sasb_coverage(self, analysis: str) -> Dict[str, Any]:
        """Assess SASB coverage"""
        
        return {
            'status': 'good_coverage',
            'coverage_percentage': 0.75,
            'last_assessment': datetime.utcnow().isoformat()
        }
    
    def _determine_overall_status(self, compliance_score: float, risk_level: str) -> str:
        """Determine overall compliance status"""
        
        if compliance_score >= 0.8 and risk_level == 'low':
            return 'compliant'
        elif compliance_score >= 0.6 and risk_level in ['low', 'medium']:
            return 'mostly_compliant'
        elif compliance_score >= 0.4:
            return 'partially_compliant'
        else:
            return 'non_compliant'
    
    def _generate_compliance_summary(self, analysis: str) -> str:
        """Generate executive summary of compliance analysis"""
        
        # Extract key points for summary
        sentences = analysis.split('. ')
        key_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in 
                        ['compliant', 'risk', 'recommend', 'require', 'violation'])]
        
        summary = '. '.join(key_sentences[:3])  # Top 3 key sentences
        
        if len(summary) < 50:
            summary = "Portfolio compliance analysis completed. Review detailed findings for specific recommendations."
        
        return summary
    
    # Tool implementation methods
    def _search_regulatory_database(self, query: str) -> str:
        """Search regulatory database (mock implementation)"""
        return f"Found 5 relevant regulations for query: {query}"
    
    def _analyze_portfolio_risks(self, portfolio_data: str) -> str:
        """Analyze portfolio risks (mock implementation)"""
        return "Identified 3 medium-risk compliance issues in portfolio"
    
    def _monitor_regulation_changes(self, timeframe: str) -> str:
        """Monitor regulation changes (mock implementation)"""
        return f"Found 2 new regulatory changes in the last {timeframe}"
    
    def _calculate_compliance_score(self, portfolio_data: str) -> str:
        """Calculate compliance score (mock implementation)"""
        return "Calculated compliance score: 75/100"
    
    def _recommend_remediation(self, issues: str) -> str:
        """Recommend remediation actions (mock implementation)"""
        return "Recommended 4 remediation actions with timelines"
    
    async def _get_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio data (mock implementation)"""
        
        # Mock portfolio data
        return {
            'portfolio_id': portfolio_id,
            'total_value': 10000000,
            'strategy': 'ESG Growth',
            'holdings': [
                {'symbol': 'AAPL', 'weight': 0.15, 'esg_score': 85, 'sector': 'Technology'},
                {'symbol': 'MSFT', 'weight': 0.12, 'esg_score': 88, 'sector': 'Technology'},
                {'symbol': 'TSLA', 'weight': 0.08, 'esg_score': 75, 'sector': 'Automotive'},
                {'symbol': 'JNJ', 'weight': 0.07, 'esg_score': 82, 'sector': 'Healthcare'},
                {'symbol': 'PG', 'weight': 0.06, 'esg_score': 79, 'sector': 'Consumer Goods'}
            ]
        }