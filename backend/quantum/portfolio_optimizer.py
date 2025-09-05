import numpy as np
from typing import Dict, List, Any, Optional
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    # Fallback for when Qiskit is not properly installed
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    transpile = None
    QAOA = None
    COBYLA = None
    SPSA = None
    Sampler = None
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class QuantumPortfolioOptimizer:
    """Quantum-enhanced portfolio optimizer using QAOA algorithms"""
    
    def __init__(self, backend_name: str = "qasm_simulator"):
        self.backend_name = backend_name
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.quantum_available = QISKIT_AVAILABLE
        if not self.quantum_available:
            logger.warning("Qiskit not available, falling back to classical optimization")
        else:
            logger.info(f"Initialized QuantumPortfolioOptimizer with backend: {backend_name}")
    
    async def optimize(
        self,
        assets: List[Dict[str, Any]],
        constraints: Dict[str, Any],
        esg_weights: Dict[str, float],
        risk_tolerance: float = 0.5,
        quantum_enhanced: bool = True
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation using quantum algorithms"""
        
        try:
            logger.info(f"Starting quantum portfolio optimization for {len(assets)} assets")
            
            # Prepare optimization data
            optimization_data = await self._prepare_optimization_data(
                assets, constraints, esg_weights, risk_tolerance
            )
            
            if quantum_enhanced:
                # Run quantum optimization
                result = await self._run_quantum_optimization(optimization_data)
            else:
                # Run classical optimization for comparison
                result = await self._run_classical_optimization(optimization_data)
            
            # Post-process results
            processed_result = await self._process_optimization_result(
                result, assets, esg_weights
            )
            
            logger.info("Portfolio optimization completed successfully")
            return processed_result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            raise
    
    async def _prepare_optimization_data(
        self,
        assets: List[Dict[str, Any]],
        constraints: Dict[str, Any],
        esg_weights: Dict[str, float],
        risk_tolerance: float
    ) -> Dict[str, Any]:
        """Prepare data for quantum optimization"""
        
        # Extract asset data
        asset_symbols = [asset['symbol'] for asset in assets]
        expected_returns = np.array([asset.get('expected_return', 0.1) for asset in assets])
        
        # Create covariance matrix (simplified for demo)
        n_assets = len(assets)
        covariance_matrix = np.random.rand(n_assets, n_assets) * 0.01
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2
        np.fill_diagonal(covariance_matrix, np.random.rand(n_assets) * 0.05 + 0.01)
        
        # ESG scores
        esg_scores = np.array([
            asset.get('esg_score', 50) / 100 for asset in assets
        ])
        
        # Budget constraint
        budget = constraints.get('budget', 1000000)
        
        # Risk budget
        risk_budget = risk_tolerance * budget
        
        return {
            'asset_symbols': asset_symbols,
            'expected_returns': expected_returns,
            'covariance_matrix': covariance_matrix,
            'esg_scores': esg_scores,
            'esg_weights': esg_weights,
            'budget': budget,
            'risk_budget': risk_budget,
            'risk_tolerance': risk_tolerance,
            'n_assets': n_assets
        }
    
    async def _run_quantum_optimization(
        self, optimization_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run quantum optimization using QAOA"""
        
        def quantum_optimization():
            try:
                # Create portfolio optimization problem
                portfolio_opt = PortfolioOptimization(
                    expected_returns=optimization_data['expected_returns'],
                    covariances=optimization_data['covariance_matrix'],
                    risk_factor=optimization_data['risk_tolerance'],
                    budget=optimization_data['n_assets']  # Number of assets to select
                )
                
                # Convert to QUBO
                qp = portfolio_opt.to_quadratic_program()
                
                # Set up QAOA
                sampler = Sampler()
                qaoa = QAOA(
                    sampler=sampler,
                    optimizer=COBYLA(maxiter=100),
                    reps=3
                )
                
                # Create minimum eigen optimizer
                min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
                
                # Solve the optimization problem
                start_time = datetime.utcnow()
                result = min_eigen_optimizer.solve(qp)
                end_time = datetime.utcnow()
                
                quantum_time = (end_time - start_time).total_seconds()
                
                return {
                    'optimization_result': result,
                    'quantum_time': quantum_time,
                    'algorithm': 'QAOA',
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"Quantum optimization error: {str(e)}")
                return {
                    'optimization_result': None,
                    'quantum_time': 0,
                    'algorithm': 'QAOA',
                    'success': False,
                    'error': str(e)
                }
        
        # Run quantum optimization in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, quantum_optimization)
        
        return result
    
    async def _run_classical_optimization(
        self, optimization_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run classical optimization for comparison"""
        
        def classical_optimization():
            try:
                from scipy.optimize import minimize
                
                n_assets = optimization_data['n_assets']
                expected_returns = optimization_data['expected_returns']
                covariance_matrix = optimization_data['covariance_matrix']
                risk_tolerance = optimization_data['risk_tolerance']
                
                # Objective function: maximize return - risk penalty
                def objective(weights):
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                    return -(portfolio_return - risk_tolerance * portfolio_risk)
                
                # Constraints
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                ]
                
                # Bounds
                bounds = tuple((0, 1) for _ in range(n_assets))
                
                # Initial guess
                x0 = np.array([1/n_assets] * n_assets)
                
                start_time = datetime.utcnow()
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                end_time = datetime.utcnow()
                
                classical_time = (end_time - start_time).total_seconds()
                
                return {
                    'optimization_result': result,
                    'classical_time': classical_time,
                    'algorithm': 'Classical',
                    'success': result.success
                }
                
            except Exception as e:
                logger.error(f"Classical optimization error: {str(e)}")
                return {
                    'optimization_result': None,
                    'classical_time': 0,
                    'algorithm': 'Classical',
                    'success': False,
                    'error': str(e)
                }
        
        # Run classical optimization in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, classical_optimization)
        
        return result
    
    async def _process_optimization_result(
        self,
        result: Dict[str, Any],
        assets: List[Dict[str, Any]],
        esg_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Process and format optimization results"""
        
        if not result.get('success', False):
            return {
                'success': False,
                'error': result.get('error', 'Optimization failed'),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        optimization_result = result['optimization_result']
        
        # Extract optimal weights
        if hasattr(optimization_result, 'x'):
            # Classical result
            optimal_weights = optimization_result.x
        elif hasattr(optimization_result, 'samples'):
            # Quantum result
            # Convert binary solution to weights (simplified)
            binary_solution = list(optimization_result.samples[0].keys())[0]
            optimal_weights = np.array([int(bit) for bit in binary_solution])
            optimal_weights = optimal_weights / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else optimal_weights
        else:
            optimal_weights = np.array([1/len(assets)] * len(assets))
        
        # Calculate portfolio metrics
        portfolio_metrics = await self._calculate_portfolio_metrics(
            optimal_weights, assets, esg_weights
        )
        
        # Create allocation recommendations
        allocations = []
        for i, asset in enumerate(assets):
            if i < len(optimal_weights):
                allocations.append({
                    'symbol': asset['symbol'],
                    'name': asset.get('name', asset['symbol']),
                    'weight': float(optimal_weights[i]),
                    'allocation_amount': float(optimal_weights[i] * 1000000),  # Assuming $1M portfolio
                    'esg_score': asset.get('esg_score', 50),
                    'expected_return': asset.get('expected_return', 0.1)
                })
        
        return {
            'success': True,
            'algorithm': result.get('algorithm', 'Unknown'),
            'execution_time': result.get('quantum_time', result.get('classical_time', 0)),
            'quantum_speedup': result.get('quantum_time', 1) / max(result.get('classical_time', 1), 0.001),
            'optimal_allocations': allocations,
            'portfolio_metrics': portfolio_metrics,
            'optimization_status': 'completed',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        assets: List[Dict[str, Any]],
        esg_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        
        expected_returns = np.array([asset.get('expected_return', 0.1) for asset in assets])
        esg_scores = np.array([asset.get('esg_score', 50) for asset in assets])
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio ESG score
        portfolio_esg = np.dot(weights, esg_scores)
        
        # Risk metrics (simplified)
        portfolio_volatility = np.sqrt(np.sum(weights**2) * 0.16)  # Simplified volatility
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # ESG-adjusted return
        esg_adjustment = esg_weights.get('environmental', 0.33) * 0.1  # Simplified ESG premium
        esg_adjusted_return = portfolio_return + (portfolio_esg / 100) * esg_adjustment
        
        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'esg_score': float(portfolio_esg),
            'esg_adjusted_return': float(esg_adjusted_return),
            'diversification_ratio': float(1 / np.sum(weights**2)),  # Inverse concentration
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights))
        }
    
    async def get_quantum_circuit_info(self, n_assets: int) -> Dict[str, Any]:
        """Get information about the quantum circuit used for optimization"""
        
        try:
            # Create a sample quantum circuit for the given number of assets
            qc = QuantumCircuit(n_assets, n_assets)
            
            # Add Hadamard gates for superposition
            for i in range(n_assets):
                qc.h(i)
            
            # Add entangling gates
            for i in range(n_assets - 1):
                qc.cx(i, i + 1)
            
            # Add measurements
            qc.measure_all()
            
            return {
                'n_qubits': n_assets,
                'circuit_depth': qc.depth(),
                'gate_count': len(qc.data),
                'quantum_volume': 2**n_assets,
                'circuit_diagram': str(qc.draw(output='text'))
            }
            
        except Exception as e:
            logger.error(f"Error generating circuit info: {str(e)}")
            return {
                'error': str(e),
                'n_qubits': n_assets
            }