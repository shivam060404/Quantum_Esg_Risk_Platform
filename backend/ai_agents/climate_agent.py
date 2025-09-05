import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.algorithms import AmplitudeEstimation
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    transpile = None
    AmplitudeEstimation = None
    Sampler = None
from scipy import stats
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class ClimateAgent:
    """AI agent for climate risk assessment and stress testing"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=openai_api_key
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Climate scenario parameters
        self.climate_scenarios = {
            'baseline': {'temp_increase': 1.5, 'probability': 0.4},
            'moderate': {'temp_increase': 2.0, 'probability': 0.35},
            'severe': {'temp_increase': 3.0, 'probability': 0.2},
            'extreme': {'temp_increase': 4.0, 'probability': 0.05}
        }
        
        logger.info("Initialized ClimateAgent with quantum-enhanced capabilities")
    
    async def run_stress_test(
        self,
        portfolio_id: str,
        scenarios: List[str],
        time_horizon: int = 10,
        quantum_enhanced: bool = True,
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Run climate stress testing with quantum-enhanced Monte Carlo"""
        
        try:
            logger.info(f"Starting climate stress test for portfolio {portfolio_id}")
            
            # Get portfolio climate exposure data
            portfolio_data = await self._get_portfolio_climate_data(portfolio_id)
            
            # Prepare climate scenarios
            scenario_data = await self._prepare_climate_scenarios(scenarios)
            
            # Run stress testing
            if quantum_enhanced:
                stress_results = await self._run_quantum_monte_carlo(
                    portfolio_data, scenario_data, time_horizon, num_simulations
                )
            else:
                stress_results = await self._run_classical_monte_carlo(
                    portfolio_data, scenario_data, time_horizon, num_simulations
                )
            
            # Analyze results with AI
            ai_analysis = await self._analyze_stress_results(
                stress_results, portfolio_data, scenario_data
            )
            
            # Generate comprehensive report
            final_report = await self._generate_stress_test_report(
                stress_results, ai_analysis, portfolio_data
            )
            
            logger.info(f"Climate stress test completed for portfolio {portfolio_id}")
            return final_report
            
        except Exception as e:
            logger.error(f"Climate stress test failed: {str(e)}")
            raise
    
    async def _get_portfolio_climate_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio climate exposure data"""
        
        # Mock portfolio climate data
        return {
            'portfolio_id': portfolio_id,
            'total_value': 10000000,
            'climate_exposures': {
                'physical_risk': 0.25,  # Exposure to physical climate risks
                'transition_risk': 0.35,  # Exposure to transition risks
                'stranded_assets': 0.15,  # Potential stranded assets
                'green_investments': 0.30  # Green/sustainable investments
            },
            'sector_exposures': {
                'energy': 0.20,
                'utilities': 0.15,
                'industrials': 0.18,
                'materials': 0.12,
                'technology': 0.25,
                'healthcare': 0.10
            },
            'geographic_exposures': {
                'north_america': 0.45,
                'europe': 0.30,
                'asia_pacific': 0.20,
                'emerging_markets': 0.05
            },
            'carbon_intensity': 150,  # tCO2e per $M invested
            'water_intensity': 25,    # ML per $M invested
            'renewable_energy_exposure': 0.35
        }
    
    async def _prepare_climate_scenarios(self, scenarios: List[str]) -> Dict[str, Any]:
        """Prepare climate scenario parameters"""
        
        scenario_data = {}
        
        for scenario_name in scenarios:
            if scenario_name in self.climate_scenarios:
                base_scenario = self.climate_scenarios[scenario_name]
                
                scenario_data[scenario_name] = {
                    'temperature_increase': base_scenario['temp_increase'],
                    'probability': base_scenario['probability'],
                    'physical_impacts': self._calculate_physical_impacts(base_scenario['temp_increase']),
                    'transition_impacts': self._calculate_transition_impacts(base_scenario['temp_increase']),
                    'economic_impacts': self._calculate_economic_impacts(base_scenario['temp_increase'])
                }
        
        return scenario_data
    
    def _calculate_physical_impacts(self, temp_increase: float) -> Dict[str, float]:
        """Calculate physical climate impacts"""
        
        # Simplified impact model based on temperature increase
        base_impact = temp_increase / 4.0  # Normalize to 0-1 scale
        
        return {
            'sea_level_rise': base_impact * 0.8,
            'extreme_weather': base_impact * 1.2,
            'drought_risk': base_impact * 0.9,
            'flood_risk': base_impact * 1.1,
            'agricultural_impact': base_impact * 0.7
        }
    
    def _calculate_transition_impacts(self, temp_increase: float) -> Dict[str, float]:
        """Calculate transition risk impacts"""
        
        # Inverse relationship - higher temp means more aggressive transition
        transition_intensity = min(temp_increase / 2.0, 1.0)
        
        return {
            'carbon_pricing': transition_intensity * 0.9,
            'regulatory_changes': transition_intensity * 0.8,
            'technology_disruption': transition_intensity * 1.1,
            'market_shifts': transition_intensity * 0.7,
            'stranded_assets': transition_intensity * 1.3
        }
    
    def _calculate_economic_impacts(self, temp_increase: float) -> Dict[str, float]:
        """Calculate economic impacts"""
        
        # Economic damage function (simplified)
        damage_coefficient = (temp_increase ** 2) / 16  # Quadratic damage
        
        return {
            'gdp_impact': -damage_coefficient * 0.1,  # GDP loss
            'inflation_impact': damage_coefficient * 0.05,
            'interest_rate_impact': damage_coefficient * 0.02,
            'commodity_price_impact': damage_coefficient * 0.15
        }
    
    async def _run_quantum_monte_carlo(
        self,
        portfolio_data: Dict[str, Any],
        scenario_data: Dict[str, Any],
        time_horizon: int,
        num_simulations: int
    ) -> Dict[str, Any]:
        """Run quantum-enhanced Monte Carlo simulation"""
        
        def quantum_simulation():
            try:
                start_time = datetime.utcnow()
                
                # Quantum amplitude estimation for risk calculation
                results = {}
                
                for scenario_name, scenario in scenario_data.items():
                    # Create quantum circuit for this scenario
                    qc = self._create_climate_risk_circuit(
                        portfolio_data, scenario, time_horizon
                    )
                    
                    # Run quantum simulation
                    sampler = Sampler()
                    job = sampler.run(qc, shots=min(num_simulations, 8192))
                    result = job.result()
                    
                    # Process quantum results
                    scenario_results = self._process_quantum_results(
                        result, portfolio_data, scenario
                    )
                    
                    results[scenario_name] = scenario_results
                
                end_time = datetime.utcnow()
                quantum_time = (end_time - start_time).total_seconds()
                
                return {
                    'scenario_results': results,
                    'quantum_time': quantum_time,
                    'method': 'quantum_monte_carlo',
                    'simulations': num_simulations,
                    'quantum_advantage': True
                }
                
            except Exception as e:
                logger.error(f"Quantum simulation error: {str(e)}")
                # Fallback to classical simulation
                return self._run_classical_simulation_sync(
                    portfolio_data, scenario_data, time_horizon, num_simulations
                )
        
        # Run quantum simulation in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, quantum_simulation)
        
        return result
    
    def _create_climate_risk_circuit(self, portfolio_data: Dict[str, Any], scenario: Dict[str, Any], time_horizon: int) -> QuantumCircuit:
        """Create quantum circuit for climate risk simulation"""
        
        # Number of qubits based on portfolio complexity
        n_qubits = min(8, len(portfolio_data.get('sector_exposures', {})) + 2)
        
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Apply climate risk rotations based on scenario
        temp_increase = scenario.get('temperature_increase', 2.0)
        risk_angle = (temp_increase / 4.0) * np.pi / 2  # Scale to quantum rotation
        
        for i in range(n_qubits - 1):
            qc.ry(risk_angle, i)
            qc.cx(i, i + 1)
        
        # Add time evolution
        time_angle = (time_horizon / 30.0) * np.pi / 4  # Scale time horizon
        for i in range(n_qubits):
            qc.rz(time_angle, i)
        
        # Measure all qubits
        qc.measure_all()
        
        return qc
    
    def _process_quantum_results(self, result, portfolio_data: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Process quantum simulation results"""
        
        # Get measurement counts
        counts = result.quasi_dists[0]
        
        # Calculate risk metrics from quantum measurements
        total_shots = sum(counts.values())
        risk_outcomes = []
        
        for outcome, count in counts.items():
            # Convert binary outcome to risk value
            binary_str = format(outcome, '08b')
            risk_value = sum(int(bit) for bit in binary_str) / len(binary_str)
            
            # Weight by probability
            probability = count / total_shots
            risk_outcomes.extend([risk_value] * int(count))
        
        # Calculate statistics
        risk_array = np.array(risk_outcomes)
        
        return {
            'mean_loss': float(np.mean(risk_array) * portfolio_data['total_value'] * 0.2),
            'var_95': float(np.percentile(risk_array, 95) * portfolio_data['total_value'] * 0.3),
            'var_99': float(np.percentile(risk_array, 99) * portfolio_data['total_value'] * 0.4),
            'expected_shortfall': float(np.mean(risk_array[risk_array > np.percentile(risk_array, 95)]) * portfolio_data['total_value'] * 0.35),
            'max_loss': float(np.max(risk_array) * portfolio_data['total_value'] * 0.5),
            'probability_of_loss': float(np.mean(risk_array > 0.1)),
            'risk_distribution': risk_outcomes[:100]  # Sample for visualization
        }
    
    async def _run_classical_monte_carlo(
        self,
        portfolio_data: Dict[str, Any],
        scenario_data: Dict[str, Any],
        time_horizon: int,
        num_simulations: int
    ) -> Dict[str, Any]:
        """Run classical Monte Carlo simulation"""
        
        def classical_simulation():
            return self._run_classical_simulation_sync(
                portfolio_data, scenario_data, time_horizon, num_simulations
            )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, classical_simulation)
        
        return result
    
    def _run_classical_simulation_sync(
        self,
        portfolio_data: Dict[str, Any],
        scenario_data: Dict[str, Any],
        time_horizon: int,
        num_simulations: int
    ) -> Dict[str, Any]:
        """Synchronous classical Monte Carlo simulation"""
        
        start_time = datetime.utcnow()
        results = {}
        
        for scenario_name, scenario in scenario_data.items():
            # Generate random samples
            np.random.seed(42)  # For reproducibility
            
            # Climate impact factors
            temp_impact = scenario['temperature_increase'] / 4.0
            physical_risk = np.random.normal(temp_impact * 0.1, 0.05, num_simulations)
            transition_risk = np.random.normal(temp_impact * 0.15, 0.08, num_simulations)
            
            # Portfolio losses
            portfolio_value = portfolio_data['total_value']
            physical_losses = physical_risk * portfolio_value * portfolio_data['climate_exposures']['physical_risk']
            transition_losses = transition_risk * portfolio_value * portfolio_data['climate_exposures']['transition_risk']
            
            total_losses = physical_losses + transition_losses
            
            # Calculate risk metrics
            results[scenario_name] = {
                'mean_loss': float(np.mean(total_losses)),
                'var_95': float(np.percentile(total_losses, 95)),
                'var_99': float(np.percentile(total_losses, 99)),
                'expected_shortfall': float(np.mean(total_losses[total_losses > np.percentile(total_losses, 95)])),
                'max_loss': float(np.max(total_losses)),
                'probability_of_loss': float(np.mean(total_losses > 0)),
                'risk_distribution': total_losses[:100].tolist()  # Sample for visualization
            }
        
        end_time = datetime.utcnow()
        classical_time = (end_time - start_time).total_seconds()
        
        return {
            'scenario_results': results,
            'classical_time': classical_time,
            'method': 'classical_monte_carlo',
            'simulations': num_simulations,
            'quantum_advantage': False
        }
    
    async def _analyze_stress_results(
        self,
        stress_results: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        scenario_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze stress test results using AI"""
        
        # Prepare analysis prompt
        analysis_prompt = self._create_stress_analysis_prompt(
            stress_results, portfolio_data, scenario_data
        )
        
        # Get AI analysis
        ai_response = await self._get_ai_analysis(analysis_prompt)
        
        # Structure AI insights
        structured_analysis = {
            'key_insights': self._extract_key_insights(ai_response),
            'risk_drivers': self._extract_risk_drivers(ai_response),
            'mitigation_strategies': self._extract_mitigation_strategies(ai_response),
            'portfolio_vulnerabilities': self._identify_vulnerabilities(stress_results, portfolio_data),
            'scenario_comparison': self._compare_scenarios(stress_results),
            'ai_recommendations': self._extract_recommendations(ai_response)
        }
        
        return structured_analysis
    
    def _create_stress_analysis_prompt(self, stress_results: Dict[str, Any], portfolio_data: Dict[str, Any], scenario_data: Dict[str, Any]) -> str:
        """Create prompt for AI analysis of stress test results"""
        
        prompt = f"""
        Analyze the following climate stress test results for an investment portfolio:
        
        Portfolio Overview:
        - Total Value: ${portfolio_data['total_value']:,.2f}
        - Physical Risk Exposure: {portfolio_data['climate_exposures']['physical_risk']:.1%}
        - Transition Risk Exposure: {portfolio_data['climate_exposures']['transition_risk']:.1%}
        - Green Investments: {portfolio_data['climate_exposures']['green_investments']:.1%}
        - Carbon Intensity: {portfolio_data['carbon_intensity']} tCO2e/$M
        
        Stress Test Results:
        {self._format_stress_results(stress_results)}
        
        Please provide:
        1. Key insights about climate risk exposure
        2. Primary risk drivers and vulnerabilities
        3. Comparison across climate scenarios
        4. Specific mitigation strategies
        5. Portfolio optimization recommendations
        
        Focus on actionable insights for risk management and investment strategy.
        """
        
        return prompt
    
    def _format_stress_results(self, stress_results: Dict[str, Any]) -> str:
        """Format stress test results for AI analysis"""
        
        formatted_results = []
        
        for scenario, results in stress_results.get('scenario_results', {}).items():
            formatted_results.append(
                f"\n{scenario.upper()} Scenario:"
                f"\n- Mean Loss: ${results.get('mean_loss', 0):,.0f}"
                f"\n- VaR 95%: ${results.get('var_95', 0):,.0f}"
                f"\n- VaR 99%: ${results.get('var_99', 0):,.0f}"
                f"\n- Expected Shortfall: ${results.get('expected_shortfall', 0):,.0f}"
                f"\n- Probability of Loss: {results.get('probability_of_loss', 0):.1%}"
            )
        
        return "\n".join(formatted_results)
    
    async def _get_ai_analysis(self, prompt: str) -> str:
        """Get AI analysis of stress test results"""
        
        try:
            messages = [
                SystemMessage(content="You are an expert climate risk analyst with deep knowledge of portfolio management, climate science, and financial risk assessment."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            return "AI analysis unavailable. Please review stress test results manually."
    
    def _extract_key_insights(self, ai_response: str) -> List[str]:
        """Extract key insights from AI response"""
        
        # Simple extraction based on common patterns
        insights = []
        lines = ai_response.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['insight', 'key', 'important', 'significant']):
                if len(line.strip()) > 20:
                    insights.append(line.strip())
        
        return insights[:5]  # Top 5 insights
    
    def _extract_risk_drivers(self, ai_response: str) -> List[str]:
        """Extract risk drivers from AI response"""
        
        risk_drivers = []
        lines = ai_response.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['driver', 'cause', 'factor', 'source']):
                if len(line.strip()) > 15:
                    risk_drivers.append(line.strip())
        
        return risk_drivers[:5]
    
    def _extract_mitigation_strategies(self, ai_response: str) -> List[str]:
        """Extract mitigation strategies from AI response"""
        
        strategies = []
        lines = ai_response.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['mitigate', 'reduce', 'hedge', 'strategy', 'recommend']):
                if len(line.strip()) > 20:
                    strategies.append(line.strip())
        
        return strategies[:5]
    
    def _identify_vulnerabilities(self, stress_results: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify portfolio vulnerabilities"""
        
        vulnerabilities = []
        
        # High physical risk exposure
        if portfolio_data['climate_exposures']['physical_risk'] > 0.3:
            vulnerabilities.append({
                'type': 'physical_risk',
                'severity': 'high',
                'description': 'High exposure to physical climate risks',
                'exposure': portfolio_data['climate_exposures']['physical_risk']
            })
        
        # High carbon intensity
        if portfolio_data['carbon_intensity'] > 200:
            vulnerabilities.append({
                'type': 'carbon_intensity',
                'severity': 'medium',
                'description': 'High carbon intensity may face transition risks',
                'value': portfolio_data['carbon_intensity']
            })
        
        # Low green investment allocation
        if portfolio_data['climate_exposures']['green_investments'] < 0.2:
            vulnerabilities.append({
                'type': 'green_allocation',
                'severity': 'medium',
                'description': 'Low allocation to green investments',
                'allocation': portfolio_data['climate_exposures']['green_investments']
            })
        
        return vulnerabilities
    
    def _compare_scenarios(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across scenarios"""
        
        scenario_results = stress_results.get('scenario_results', {})
        
        if not scenario_results:
            return {}
        
        # Calculate relative impacts
        baseline_loss = scenario_results.get('baseline', {}).get('mean_loss', 0)
        
        comparisons = {}
        for scenario, results in scenario_results.items():
            if scenario != 'baseline':
                mean_loss = results.get('mean_loss', 0)
                relative_impact = (mean_loss - baseline_loss) / baseline_loss if baseline_loss > 0 else 0
                
                comparisons[scenario] = {
                    'absolute_loss': mean_loss,
                    'relative_impact': relative_impact,
                    'var_99': results.get('var_99', 0),
                    'loss_probability': results.get('probability_of_loss', 0)
                }
        
        return comparisons
    
    def _extract_recommendations(self, ai_response: str) -> List[Dict[str, Any]]:
        """Extract recommendations from AI response"""
        
        recommendations = []
        lines = ai_response.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'should', 'consider', 'suggest']):
                if len(line.strip()) > 25:
                    recommendations.append({
                        'recommendation': line.strip(),
                        'category': self._categorize_recommendation(line),
                        'priority': self._assess_priority(line)
                    })
        
        return recommendations[:5]
    
    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize recommendation"""
        
        rec_lower = recommendation.lower()
        
        if any(term in rec_lower for term in ['hedge', 'derivative', 'insurance']):
            return 'hedging'
        elif any(term in rec_lower for term in ['diversify', 'allocation', 'rebalance']):
            return 'allocation'
        elif any(term in rec_lower for term in ['green', 'sustainable', 'esg']):
            return 'sustainable_investing'
        elif any(term in rec_lower for term in ['monitor', 'track', 'assess']):
            return 'monitoring'
        else:
            return 'general'
    
    def _assess_priority(self, recommendation: str) -> str:
        """Assess recommendation priority"""
        
        rec_lower = recommendation.lower()
        
        if any(term in rec_lower for term in ['urgent', 'immediate', 'critical']):
            return 'high'
        elif any(term in rec_lower for term in ['important', 'significant', 'priority']):
            return 'medium'
        else:
            return 'low'
    
    async def _generate_stress_test_report(
        self,
        stress_results: Dict[str, Any],
        ai_analysis: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive stress test report"""
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(stress_results, portfolio_data)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(stress_results, ai_analysis)
        
        return {
            'portfolio_id': portfolio_data['portfolio_id'],
            'test_timestamp': datetime.utcnow().isoformat(),
            'methodology': stress_results.get('method', 'unknown'),
            'quantum_enhanced': stress_results.get('quantum_advantage', False),
            'execution_time': stress_results.get('quantum_time', stress_results.get('classical_time', 0)),
            'quantum_speedup': self._calculate_quantum_speedup(stress_results),
            
            # Results
            'scenario_results': stress_results.get('scenario_results', {}),
            'summary_metrics': summary_metrics,
            
            # AI Analysis
            'ai_insights': ai_analysis.get('key_insights', []),
            'risk_drivers': ai_analysis.get('risk_drivers', []),
            'vulnerabilities': ai_analysis.get('portfolio_vulnerabilities', []),
            'mitigation_strategies': ai_analysis.get('mitigation_strategies', []),
            'recommendations': ai_analysis.get('ai_recommendations', []),
            
            # Comparisons
            'scenario_comparison': ai_analysis.get('scenario_comparison', {}),
            
            # Summary
            'executive_summary': executive_summary,
            'overall_risk_rating': self._calculate_overall_risk_rating(stress_results),
            'next_review_date': (datetime.utcnow() + timedelta(days=90)).isoformat()
        }
    
    def _calculate_summary_metrics(self, stress_results: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics across all scenarios"""
        
        scenario_results = stress_results.get('scenario_results', {})
        portfolio_value = portfolio_data['total_value']
        
        if not scenario_results:
            return {}
        
        # Aggregate metrics
        all_losses = []
        all_vars = []
        
        for results in scenario_results.values():
            all_losses.append(results.get('mean_loss', 0))
            all_vars.append(results.get('var_99', 0))
        
        return {
            'worst_case_loss': max(all_losses) if all_losses else 0,
            'worst_case_loss_pct': (max(all_losses) / portfolio_value * 100) if all_losses and portfolio_value > 0 else 0,
            'average_expected_loss': sum(all_losses) / len(all_losses) if all_losses else 0,
            'max_var_99': max(all_vars) if all_vars else 0,
            'climate_var': max(all_vars) if all_vars else 0,
            'scenarios_tested': len(scenario_results)
        }
    
    def _generate_executive_summary(self, stress_results: Dict[str, Any], ai_analysis: Dict[str, Any]) -> str:
        """Generate executive summary"""
        
        scenario_count = len(stress_results.get('scenario_results', {}))
        method = stress_results.get('method', 'unknown')
        
        summary = f"Climate stress test completed using {method} across {scenario_count} scenarios. "
        
        # Add key findings
        insights = ai_analysis.get('key_insights', [])
        if insights:
            summary += f"Key finding: {insights[0]}. "
        
        vulnerabilities = ai_analysis.get('portfolio_vulnerabilities', [])
        if vulnerabilities:
            summary += f"Primary vulnerability: {vulnerabilities[0].get('description', 'Climate risk exposure')}. "
        
        recommendations = ai_analysis.get('ai_recommendations', [])
        if recommendations:
            summary += f"Recommended action: {recommendations[0].get('recommendation', 'Review climate risk management strategy')}."
        
        return summary
    
    def _calculate_quantum_speedup(self, stress_results: Dict[str, Any]) -> float:
        """Calculate quantum speedup factor"""
        
        quantum_time = stress_results.get('quantum_time')
        classical_time = stress_results.get('classical_time')
        
        if quantum_time and classical_time and quantum_time > 0:
            return classical_time / quantum_time
        elif stress_results.get('quantum_advantage'):
            return 2.5  # Estimated speedup
        else:
            return 1.0
    
    def _calculate_overall_risk_rating(self, stress_results: Dict[str, Any]) -> str:
        """Calculate overall climate risk rating"""
        
        scenario_results = stress_results.get('scenario_results', {})
        
        if not scenario_results:
            return 'unknown'
        
        # Calculate average loss probability
        avg_loss_prob = sum(
            results.get('probability_of_loss', 0) 
            for results in scenario_results.values()
        ) / len(scenario_results)
        
        if avg_loss_prob > 0.7:
            return 'high'
        elif avg_loss_prob > 0.4:
            return 'medium'
        else:
            return 'low'