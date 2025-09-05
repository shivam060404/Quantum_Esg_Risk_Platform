import json
import hashlib
from typing import Dict, List, Any, Optional
from web3 import Web3
from eth_account import Account
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp

logger = logging.getLogger(__name__)

class ESGOracle:
    """Blockchain-based ESG data verification oracle"""
    
    def __init__(self, rpc_url: str = "http://localhost:8545", private_key: Optional[str] = None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.private_key = private_key
        self.account = Account.from_key(private_key) if private_key else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Smart contract ABI (simplified for demo)
        self.contract_abi = [
            {
                "inputs": [
                    {"name": "companyId", "type": "string"},
                    {"name": "dataHash", "type": "bytes32"},
                    {"name": "esgScore", "type": "uint256"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "sources", "type": "string[]"}
                ],
                "name": "submitESGData",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "companyId", "type": "string"}],
                "name": "getESGData",
                "outputs": [
                    {"name": "dataHash", "type": "bytes32"},
                    {"name": "esgScore", "type": "uint256"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "verified", "type": "bool"}
                ],
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "companyId", "type": "string"},
                    {"name": "dataHash", "type": "bytes32"}
                ],
                "name": "verifyESGData",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            }
        ]
        
        # Contract address (would be deployed)
        self.contract_address = "0x1234567890123456789012345678901234567890"  # Placeholder
        
        logger.info(f"Initialized ESGOracle with RPC: {rpc_url}")
    
    async def verify_data(
        self,
        company_id: str,
        esg_metrics: Dict[str, Any],
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """Verify ESG data and store on blockchain"""
        
        try:
            logger.info(f"Starting ESG data verification for company: {company_id}")
            
            # Calculate data hash
            data_hash = self._calculate_data_hash(esg_metrics)
            
            # Verify data integrity
            integrity_check = await self._verify_data_integrity(esg_metrics, data_sources)
            
            # Submit to blockchain
            blockchain_result = await self._submit_to_blockchain(
                company_id, data_hash, esg_metrics, data_sources
            )
            
            # Cross-reference with external sources
            external_verification = await self._cross_reference_external_sources(
                company_id, esg_metrics, data_sources
            )
            
            verification_result = {
                'company_id': company_id,
                'data_hash': data_hash,
                'integrity_verified': integrity_check['verified'],
                'blockchain_verified': blockchain_result['success'],
                'external_verification': external_verification,
                'transaction_hash': blockchain_result.get('transaction_hash'),
                'block_number': blockchain_result.get('block_number'),
                'verification_score': self._calculate_verification_score(
                    integrity_check, blockchain_result, external_verification
                ),
                'timestamp': datetime.utcnow().isoformat(),
                'sources_verified': len([s for s in external_verification.get('source_results', []) if s.get('verified', False)])
            }
            
            logger.info(f"ESG data verification completed for {company_id}")
            return verification_result
            
        except Exception as e:
            logger.error(f"ESG data verification failed: {str(e)}")
            raise
    
    def _calculate_data_hash(self, esg_metrics: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of ESG data"""
        
        # Normalize data for consistent hashing
        normalized_data = json.dumps(esg_metrics, sort_keys=True, separators=(',', ':'))
        data_hash = hashlib.sha256(normalized_data.encode()).hexdigest()
        
        return data_hash
    
    async def _verify_data_integrity(self, esg_metrics: Dict[str, Any], data_sources: List[str]) -> Dict[str, Any]:
        """Verify data integrity and consistency"""
        
        integrity_checks = {
            'data_completeness': self._check_data_completeness(esg_metrics),
            'value_ranges': self._check_value_ranges(esg_metrics),
            'temporal_consistency': self._check_temporal_consistency(esg_metrics),
            'source_credibility': self._assess_source_credibility(data_sources)
        }
        
        # Calculate overall integrity score
        integrity_score = sum(check['score'] for check in integrity_checks.values()) / len(integrity_checks)
        
        return {
            'verified': integrity_score >= 0.7,  # 70% threshold
            'integrity_score': integrity_score,
            'checks': integrity_checks,
            'issues': [check['issues'] for check in integrity_checks.values() if check.get('issues')]
        }
    
    def _check_data_completeness(self, esg_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all required ESG metrics are present"""
        
        required_fields = [
            'environmental_score', 'social_score', 'governance_score',
            'carbon_emissions', 'water_usage', 'waste_generation',
            'employee_satisfaction', 'diversity_ratio', 'board_independence'
        ]
        
        present_fields = [field for field in required_fields if field in esg_metrics]
        completeness_ratio = len(present_fields) / len(required_fields)
        
        return {
            'score': completeness_ratio,
            'present_fields': present_fields,
            'missing_fields': [field for field in required_fields if field not in esg_metrics],
            'issues': [] if completeness_ratio >= 0.8 else ['Incomplete data']
        }
    
    def _check_value_ranges(self, esg_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if values are within expected ranges"""
        
        range_checks = {
            'environmental_score': (0, 100),
            'social_score': (0, 100),
            'governance_score': (0, 100),
            'diversity_ratio': (0, 1),
            'board_independence': (0, 1)
        }
        
        issues = []
        valid_count = 0
        total_count = 0
        
        for field, (min_val, max_val) in range_checks.items():
            if field in esg_metrics:
                total_count += 1
                value = esg_metrics[field]
                if isinstance(value, (int, float)) and min_val <= value <= max_val:
                    valid_count += 1
                else:
                    issues.append(f"{field} value {value} outside range [{min_val}, {max_val}]")
        
        score = valid_count / total_count if total_count > 0 else 1.0
        
        return {
            'score': score,
            'valid_fields': valid_count,
            'total_fields': total_count,
            'issues': issues
        }
    
    def _check_temporal_consistency(self, esg_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check temporal consistency of data"""
        
        # Simplified temporal consistency check
        reporting_date = esg_metrics.get('reporting_date')
        current_date = datetime.utcnow().isoformat()
        
        issues = []
        if not reporting_date:
            issues.append('Missing reporting date')
            score = 0.5
        else:
            try:
                report_dt = datetime.fromisoformat(reporting_date.replace('Z', '+00:00'))
                current_dt = datetime.utcnow()
                days_diff = (current_dt - report_dt).days
                
                if days_diff > 365:  # Data older than 1 year
                    issues.append('Data is more than 1 year old')
                    score = 0.6
                elif days_diff < 0:  # Future date
                    issues.append('Reporting date is in the future')
                    score = 0.3
                else:
                    score = 1.0
            except Exception:
                issues.append('Invalid reporting date format')
                score = 0.4
        
        return {
            'score': score,
            'reporting_date': reporting_date,
            'issues': issues
        }
    
    def _assess_source_credibility(self, data_sources: List[str]) -> Dict[str, Any]:
        """Assess credibility of data sources"""
        
        credible_sources = [
            'bloomberg', 'refinitiv', 'msci', 'sustainalytics',
            'cdp', 'gri', 'sasb', 'tcfd', 'sec', 'epa'
        ]
        
        credible_count = 0
        for source in data_sources:
            if any(credible in source.lower() for credible in credible_sources):
                credible_count += 1
        
        credibility_ratio = credible_count / len(data_sources) if data_sources else 0
        
        return {
            'score': credibility_ratio,
            'credible_sources': credible_count,
            'total_sources': len(data_sources),
            'issues': [] if credibility_ratio >= 0.5 else ['Low source credibility']
        }
    
    async def _submit_to_blockchain(
        self,
        company_id: str,
        data_hash: str,
        esg_metrics: Dict[str, Any],
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """Submit ESG data to blockchain smart contract"""
        
        def blockchain_submission():
            try:
                if not self.w3.is_connected():
                    return {
                        'success': False,
                        'error': 'Blockchain connection failed'
                    }
                
                # Calculate overall ESG score
                esg_score = int((
                    esg_metrics.get('environmental_score', 0) +
                    esg_metrics.get('social_score', 0) +
                    esg_metrics.get('governance_score', 0)
                ) / 3)
                
                # Convert data hash to bytes32
                data_hash_bytes = bytes.fromhex(data_hash)
                
                # Simulate blockchain transaction (in real implementation, would use actual contract)
                transaction_hash = f"0x{hashlib.sha256(f'{company_id}{data_hash}{datetime.utcnow().timestamp()}'.encode()).hexdigest()}"
                block_number = 12345678  # Simulated block number
                
                return {
                    'success': True,
                    'transaction_hash': transaction_hash,
                    'block_number': block_number,
                    'gas_used': 150000,
                    'contract_address': self.contract_address
                }
                
            except Exception as e:
                logger.error(f"Blockchain submission error: {str(e)}")
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Run blockchain submission in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, blockchain_submission)
        
        return result
    
    async def _cross_reference_external_sources(
        self,
        company_id: str,
        esg_metrics: Dict[str, Any],
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """Cross-reference ESG data with external sources"""
        
        try:
            # Simulate external API calls
            source_results = []
            
            for source in data_sources[:3]:  # Limit to 3 sources for demo
                result = await self._verify_with_external_source(company_id, esg_metrics, source)
                source_results.append(result)
            
            # Calculate consensus score
            verified_sources = [r for r in source_results if r.get('verified', False)]
            consensus_score = len(verified_sources) / len(source_results) if source_results else 0
            
            return {
                'consensus_score': consensus_score,
                'source_results': source_results,
                'verified_sources': len(verified_sources),
                'total_sources_checked': len(source_results)
            }
            
        except Exception as e:
            logger.error(f"External verification error: {str(e)}")
            return {
                'consensus_score': 0,
                'source_results': [],
                'error': str(e)
            }
    
    async def _verify_with_external_source(
        self,
        company_id: str,
        esg_metrics: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """Verify data with a specific external source"""
        
        try:
            # Simulate external API call
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Mock verification result
            import random
            verified = random.random() > 0.3  # 70% success rate
            confidence = random.uniform(0.6, 0.95) if verified else random.uniform(0.1, 0.5)
            
            return {
                'source': source,
                'verified': verified,
                'confidence': confidence,
                'response_time': 0.1,
                'data_points_matched': random.randint(5, 15) if verified else random.randint(0, 5)
            }
            
        except Exception as e:
            return {
                'source': source,
                'verified': False,
                'error': str(e),
                'confidence': 0
            }
    
    def _calculate_verification_score(
        self,
        integrity_check: Dict[str, Any],
        blockchain_result: Dict[str, Any],
        external_verification: Dict[str, Any]
    ) -> float:
        """Calculate overall verification score"""
        
        weights = {
            'integrity': 0.4,
            'blockchain': 0.3,
            'external': 0.3
        }
        
        integrity_score = integrity_check.get('integrity_score', 0)
        blockchain_score = 1.0 if blockchain_result.get('success', False) else 0.0
        external_score = external_verification.get('consensus_score', 0)
        
        overall_score = (
            weights['integrity'] * integrity_score +
            weights['blockchain'] * blockchain_score +
            weights['external'] * external_score
        )
        
        return round(overall_score, 3)
    
    async def get_verification_history(
        self,
        company_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get verification history for a company"""
        
        try:
            # In real implementation, would query blockchain and database
            # For demo, return mock data
            history = []
            for i in range(min(limit, 5)):
                history.append({
                    'verification_id': f"ver_{company_id}_{i}",
                    'timestamp': (datetime.utcnow().timestamp() - i * 86400),
                    'verification_score': round(0.7 + (i * 0.05), 2),
                    'transaction_hash': f"0x{hashlib.sha256(f'{company_id}_{i}'.encode()).hexdigest()}",
                    'block_number': 12345678 - i
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving verification history: {str(e)}")
            return []
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get blockchain network statistics"""
        
        try:
            if self.w3.is_connected():
                latest_block = self.w3.eth.block_number
                gas_price = self.w3.eth.gas_price
                
                return {
                    'connected': True,
                    'latest_block': latest_block,
                    'gas_price': gas_price,
                    'network_id': self.w3.net.version,
                    'peer_count': self.w3.net.peer_count
                }
            else:
                return {
                    'connected': False,
                    'error': 'Not connected to blockchain network'
                }
                
        except Exception as e:
            logger.error(f"Error getting network stats: {str(e)}")
            return {
                'connected': False,
                'error': str(e)
            }