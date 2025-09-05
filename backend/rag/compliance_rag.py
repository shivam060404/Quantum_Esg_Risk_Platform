import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import AsyncCallbackHandler
import pinecone
import weaviate
import requests
from bs4 import BeautifulSoup
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncio
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class ComplianceRAGSystem:
    """RAG system for regulatory compliance synthesis and analysis"""
    
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str = None,
        weaviate_url: str = None,
        vector_store_type: str = "pinecone"
    ):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.weaviate_url = weaviate_url
        self.vector_store_type = vector_store_type
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        self.retrieval_qa = None
        
        # Regulatory sources configuration
        self.regulatory_sources = {
            "SEC": {
                "base_url": "https://www.sec.gov",
                "search_endpoints": [
                    "/rules/final",
                    "/rules/proposed",
                    "/divisions/corpfin/guidance"
                ],
                "keywords": ["ESG", "climate", "sustainability", "disclosure", "environmental"]
            },
            "EU_TAXONOMY": {
                "base_url": "https://ec.europa.eu",
                "search_endpoints": [
                    "/info/business-economy-euro/banking-and-finance/sustainable-finance",
                    "/info/business-economy-euro/banking-and-finance/sustainable-finance/eu-taxonomy-sustainable-activities"
                ],
                "keywords": ["taxonomy", "sustainable", "SFDR", "green", "environmental"]
            },
            "TCFD": {
                "base_url": "https://www.fsb-tcfd.org",
                "search_endpoints": [
                    "/recommendations",
                    "/publications",
                    "/guidance"
                ],
                "keywords": ["climate", "disclosure", "risk", "governance", "strategy"]
            },
            "SASB": {
                "base_url": "https://www.sasb.org",
                "search_endpoints": [
                    "/standards",
                    "/guidance",
                    "/implementation"
                ],
                "keywords": ["standards", "materiality", "sustainability", "disclosure"]
            }
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized ComplianceRAGSystem with {vector_store_type} vector store")
    
    async def initialize_vector_store(self):
        """Initialize the vector store based on configuration"""
        try:
            if self.vector_store_type == "pinecone":
                await self._initialize_pinecone()
            elif self.vector_store_type == "weaviate":
                await self._initialize_weaviate()
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
            
            # Initialize retrieval QA chain
            self._initialize_retrieval_qa()
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def _initialize_pinecone(self):
        """Initialize Pinecone vector store"""
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.pinecone_api_key,
            environment="us-west1-gcp-free"  # Change based on your Pinecone environment
        )
        
        # Create or connect to index
        index_name = "esg-compliance-rag"
        
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
            logger.info(f"Created new Pinecone index: {index_name}")
        
        # Initialize vector store
        self.vector_store = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )
    
    async def _initialize_weaviate(self):
        """Initialize Weaviate vector store"""
        if not self.weaviate_url:
            self.weaviate_url = "http://localhost:8080"
        
        # Initialize Weaviate client
        client = weaviate.Client(url=self.weaviate_url)
        
        # Create schema if it doesn't exist
        schema = {
            "classes": [{
                "class": "ComplianceDocument",
                "description": "Regulatory compliance documents",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "Document source"
                    },
                    {
                        "name": "regulation_type",
                        "dataType": ["string"],
                        "description": "Type of regulation"
                    },
                    {
                        "name": "last_updated",
                        "dataType": ["date"],
                        "description": "Last update date"
                    }
                ]
            }]
        }
        
        try:
            client.schema.create(schema)
            logger.info("Created Weaviate schema")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Schema creation warning: {str(e)}")
        
        # Initialize vector store
        self.vector_store = Weaviate(
            client=client,
            index_name="ComplianceDocument",
            text_key="content",
            embedding=self.embeddings
        )
    
    def _initialize_retrieval_qa(self):
        """Initialize the retrieval QA chain"""
        # Create custom prompt template
        prompt_template = """
        You are an expert regulatory compliance analyst specializing in ESG (Environmental, Social, and Governance) regulations.
        
        Use the following regulatory documents and context to answer questions about compliance requirements, 
        regulatory changes, and best practices. Always cite specific regulations and provide actionable guidance.
        
        Context from regulatory documents:
        {context}
        
        Question: {question}
        
        Please provide a comprehensive answer that includes:
        1. Relevant regulatory requirements
        2. Specific compliance obligations
        3. Implementation recommendations
        4. Potential risks of non-compliance
        5. Recent regulatory updates if applicable
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    async def ingest_regulatory_documents(self, force_update: bool = False):
        """Ingest regulatory documents from various sources"""
        try:
            logger.info("Starting regulatory document ingestion")
            
            all_documents = []
            
            # Ingest from each regulatory source
            for regulation_type, config in self.regulatory_sources.items():
                logger.info(f"Ingesting documents from {regulation_type}")
                
                documents = await self._ingest_from_source(
                    regulation_type, config, force_update
                )
                all_documents.extend(documents)
            
            # Add documents to vector store
            if all_documents:
                logger.info(f"Adding {len(all_documents)} documents to vector store")
                await self._add_documents_to_vector_store(all_documents)
            
            logger.info(f"Successfully ingested {len(all_documents)} regulatory documents")
            return len(all_documents)
            
        except Exception as e:
            logger.error(f"Failed to ingest regulatory documents: {str(e)}")
            raise
    
    async def _ingest_from_source(
        self, 
        regulation_type: str, 
        config: Dict[str, Any], 
        force_update: bool
    ) -> List[Document]:
        """Ingest documents from a specific regulatory source"""
        documents = []
        
        try:
            # Check if we need to update based on last ingestion time
            if not force_update and not await self._should_update_source(regulation_type):
                logger.info(f"Skipping {regulation_type} - recently updated")
                return documents
            
            # Scrape documents from web sources
            web_documents = await self._scrape_regulatory_website(
                regulation_type, config
            )
            documents.extend(web_documents)
            
            # Load local regulatory documents if available
            local_documents = await self._load_local_regulatory_docs(regulation_type)
            documents.extend(local_documents)
            
            # Update last ingestion timestamp
            await self._update_ingestion_timestamp(regulation_type)
            
        except Exception as e:
            logger.error(f"Error ingesting from {regulation_type}: {str(e)}")
        
        return documents
    
    async def _scrape_regulatory_website(
        self, 
        regulation_type: str, 
        config: Dict[str, Any]
    ) -> List[Document]:
        """Scrape regulatory documents from official websites"""
        documents = []
        
        try:
            base_url = config["base_url"]
            endpoints = config["search_endpoints"]
            keywords = config["keywords"]
            
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints:
                    url = urljoin(base_url, endpoint)
                    
                    try:
                        # Get page content
                        async with session.get(url, timeout=30) as response:
                            if response.status == 200:
                                content = await response.text()
                                
                                # Parse and extract relevant content
                                soup = BeautifulSoup(content, 'html.parser')
                                
                                # Extract text content
                                text_content = soup.get_text()
                                
                                # Filter content based on keywords
                                if any(keyword.lower() in text_content.lower() for keyword in keywords):
                                    # Create document
                                    doc = Document(
                                        page_content=text_content,
                                        metadata={
                                            "source": url,
                                            "regulation_type": regulation_type,
                                            "last_updated": datetime.utcnow().isoformat(),
                                            "content_type": "web_page"
                                        }
                                    )
                                    documents.append(doc)
                                    
                                    logger.info(f"Scraped document from {url}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to scrape {url}: {str(e)}")
                        continue
                    
                    # Rate limiting
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error scraping {regulation_type} website: {str(e)}")
        
        return documents
    
    async def _load_local_regulatory_docs(self, regulation_type: str) -> List[Document]:
        """Load local regulatory documents"""
        documents = []
        
        try:
            # Define paths for local regulatory documents
            doc_paths = {
                "SEC": [
                    "./regulatory_docs/sec/climate_disclosure_rules.pdf",
                    "./regulatory_docs/sec/esg_guidance.pdf"
                ],
                "EU_TAXONOMY": [
                    "./regulatory_docs/eu/taxonomy_regulation.pdf",
                    "./regulatory_docs/eu/sfdr_regulation.pdf"
                ],
                "TCFD": [
                    "./regulatory_docs/tcfd/recommendations.pdf",
                    "./regulatory_docs/tcfd/implementation_guidance.pdf"
                ],
                "SASB": [
                    "./regulatory_docs/sasb/standards_overview.pdf",
                    "./regulatory_docs/sasb/implementation_guide.pdf"
                ]
            }
            
            paths = doc_paths.get(regulation_type, [])
            
            for path in paths:
                try:
                    if path.endswith('.pdf'):
                        loader = PyPDFLoader(path)
                    elif path.endswith('.txt'):
                        loader = TextLoader(path)
                    elif path.endswith('.html'):
                        loader = UnstructuredHTMLLoader(path)
                    else:
                        continue
                    
                    # Load document
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            "regulation_type": regulation_type,
                            "last_updated": datetime.utcnow().isoformat(),
                            "content_type": "local_document"
                        })
                    
                    documents.extend(docs)
                    logger.info(f"Loaded local document: {path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error loading local docs for {regulation_type}: {str(e)}")
        
        return documents
    
    async def _add_documents_to_vector_store(self, documents: List[Document]):
        """Add documents to the vector store"""
        try:
            # Split documents into chunks
            split_docs = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                split_docs.extend(chunks)
            
            # Add to vector store
            if self.vector_store_type == "pinecone":
                self.vector_store.add_documents(split_docs)
            elif self.vector_store_type == "weaviate":
                self.vector_store.add_documents(split_docs)
            
            logger.info(f"Added {len(split_docs)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise
    
    async def query_compliance(
        self, 
        question: str, 
        regulation_types: List[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system for compliance information"""
        try:
            if not self.retrieval_qa:
                raise ValueError("RAG system not initialized. Call initialize_vector_store() first.")
            
            logger.info(f"Processing compliance query: {question[:100]}...")
            
            # Enhance query with regulation type filters if specified
            enhanced_query = question
            if regulation_types:
                reg_filter = " ".join(regulation_types)
                enhanced_query = f"{question} (Focus on: {reg_filter})"
            
            # Run the query
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.retrieval_qa,
                {"query": enhanced_query}
            )
            
            # Process results
            response = {
                "answer": result["result"],
                "query": question,
                "timestamp": datetime.utcnow().isoformat(),
                "regulation_types": regulation_types or ["all"]
            }
            
            if include_sources and "source_documents" in result:
                sources = []
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
                
                response["sources"] = sources
                response["source_count"] = len(sources)
            
            logger.info(f"Successfully processed compliance query")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process compliance query: {str(e)}")
            raise
    
    async def get_regulatory_updates(
        self, 
        regulation_types: List[str] = None,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get recent regulatory updates"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Query for recent updates
            query = f"recent regulatory updates and changes since {cutoff_date.strftime('%Y-%m-%d')}"
            
            if regulation_types:
                query += f" related to {', '.join(regulation_types)}"
            
            result = await self.query_compliance(
                query,
                regulation_types=regulation_types,
                include_sources=True
            )
            
            # Extract and format updates
            updates = []
            if "sources" in result:
                for source in result["sources"]:
                    metadata = source["metadata"]
                    last_updated = metadata.get("last_updated")
                    
                    if last_updated:
                        try:
                            update_date = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                            if update_date >= cutoff_date:
                                updates.append({
                                    "regulation_type": metadata.get("regulation_type"),
                                    "source": metadata.get("source"),
                                    "last_updated": last_updated,
                                    "content_preview": source["content"],
                                    "relevance_score": 0.8  # Placeholder
                                })
                        except Exception:
                            continue
            
            # Sort by date
            updates.sort(key=lambda x: x["last_updated"], reverse=True)
            
            logger.info(f"Found {len(updates)} recent regulatory updates")
            return updates
            
        except Exception as e:
            logger.error(f"Failed to get regulatory updates: {str(e)}")
            return []
    
    async def analyze_compliance_gap(
        self, 
        portfolio_data: Dict[str, Any],
        regulation_types: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze compliance gaps for a given portfolio"""
        try:
            # Prepare portfolio context
            portfolio_context = self._format_portfolio_context(portfolio_data)
            
            # Create compliance gap analysis query
            query = f"""
            Analyze the compliance gaps and requirements for the following portfolio:
            
            {portfolio_context}
            
            Please identify:
            1. Current compliance status
            2. Missing disclosures or requirements
            3. Recommended actions to achieve full compliance
            4. Timeline for implementation
            5. Potential risks of non-compliance
            """
            
            if regulation_types:
                query += f"\n\nFocus on these regulations: {', '.join(regulation_types)}"
            
            # Get compliance analysis
            result = await self.query_compliance(
                query,
                regulation_types=regulation_types,
                include_sources=True
            )
            
            # Parse and structure the analysis
            analysis = {
                "portfolio_id": portfolio_data.get("portfolio_id"),
                "analysis_date": datetime.utcnow().isoformat(),
                "regulation_types": regulation_types or ["all"],
                "compliance_analysis": result["answer"],
                "gaps_identified": self._extract_compliance_gaps(result["answer"]),
                "recommendations": self._extract_recommendations(result["answer"]),
                "risk_level": self._assess_risk_level(result["answer"]),
                "sources": result.get("sources", [])
            }
            
            logger.info(f"Completed compliance gap analysis for portfolio {portfolio_data.get('portfolio_id')}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze compliance gap: {str(e)}")
            raise
    
    def _format_portfolio_context(self, portfolio_data: Dict[str, Any]) -> str:
        """Format portfolio data for compliance analysis"""
        context = f"""
        Portfolio ID: {portfolio_data.get('portfolio_id', 'N/A')}
        Total Value: ${portfolio_data.get('total_value', 0):,.2f}
        Number of Holdings: {len(portfolio_data.get('holdings', []))}
        ESG Score: {portfolio_data.get('esg_score', 'N/A')}
        Investment Strategy: {portfolio_data.get('strategy', 'N/A')}
        
        Top Holdings:
        """
        
        holdings = portfolio_data.get('holdings', [])
        for i, holding in enumerate(holdings[:5]):
            context += f"\n- {holding.get('symbol', 'N/A')}: {holding.get('weight', 0)*100:.1f}% (ESG: {holding.get('esg_score', 'N/A')})"
        
        return context
    
    def _extract_compliance_gaps(self, analysis_text: str) -> List[str]:
        """Extract compliance gaps from analysis text"""
        gaps = []
        
        # Simple pattern matching for gaps
        gap_patterns = [
            r"missing\s+([^.]+)",
            r"lack\s+of\s+([^.]+)",
            r"insufficient\s+([^.]+)",
            r"non-compliant\s+([^.]+)"
        ]
        
        for pattern in gap_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE)
            gaps.extend([match.strip() for match in matches])
        
        return list(set(gaps))[:10]  # Limit to top 10 unique gaps
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract recommendations from analysis text"""
        recommendations = []
        
        # Split by common recommendation indicators
        lines = analysis_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in 
                   ['recommend', 'should', 'must', 'need to', 'implement', 'establish']):
                if len(line) > 20:  # Filter out short lines
                    recommendations.append(line)
        
        return recommendations[:10]  # Limit to top 10
    
    def _assess_risk_level(self, analysis_text: str) -> str:
        """Assess compliance risk level from analysis"""
        text_lower = analysis_text.lower()
        
        high_risk_indicators = ['critical', 'severe', 'high risk', 'non-compliant', 'violation']
        medium_risk_indicators = ['moderate', 'medium risk', 'partially compliant', 'gaps']
        low_risk_indicators = ['low risk', 'compliant', 'minor', 'minimal']
        
        if any(indicator in text_lower for indicator in high_risk_indicators):
            return 'high'
        elif any(indicator in text_lower for indicator in medium_risk_indicators):
            return 'medium'
        elif any(indicator in text_lower for indicator in low_risk_indicators):
            return 'low'
        else:
            return 'medium'  # Default
    
    async def _should_update_source(self, regulation_type: str) -> bool:
        """Check if a regulatory source should be updated"""
        # Simple implementation - update daily
        # In production, this could check actual timestamps
        return True
    
    async def _update_ingestion_timestamp(self, regulation_type: str):
        """Update the last ingestion timestamp for a source"""
        # Implementation would store timestamp in database
        pass
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            stats = {
                "vector_store_type": self.vector_store_type,
                "initialized": self.vector_store is not None,
                "regulatory_sources": len(self.regulatory_sources),
                "last_update": datetime.utcnow().isoformat()
            }
            
            # Add vector store specific stats
            if self.vector_store:
                if self.vector_store_type == "pinecone":
                    # Get Pinecone index stats
                    try:
                        index_stats = pinecone.Index("esg-compliance-rag").describe_index_stats()
                        stats["document_count"] = index_stats.get("total_vector_count", 0)
                    except Exception:
                        stats["document_count"] = "unknown"
                else:
                    stats["document_count"] = "unknown"
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {str(e)}")
            return {"error": str(e)}