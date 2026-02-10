import os
import json
from typing import List, Dict, Any
import google.generativeai as genai
from data_config import (GOOGLE_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, 
                         LLM_MAX_TOKENS, RETRIEVAL_CONTEXT_LIMIT, CONTEXT_WINDOW_SIZE, RAG_TEST_QUERIES)

class LLMManager:
    """Manages LLM interactions for the RAG pipeline"""
    
    def __init__(self):
        self.api_key = GOOGLE_API_KEY or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or update data_config.py")
        
        # Configure Google AI
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL_NAME,
            generation_config=genai.types.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            )
        )
        
        print(f"✅ LLM Manager initialized with model: {LLM_MODEL_NAME}")
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(retrieved_docs[:RETRIEVAL_CONTEXT_LIMIT]):
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            section = doc.metadata.get('section_heading', 'N/A')
            
            # Create context entry
            context_entry = f"[Source {i+1}: {source} - {section}]\n{content}\n"
            
            # Check if adding this would exceed context window
            if total_chars + len(context_entry) > CONTEXT_WINDOW_SIZE:
                break
                
            context_parts.append(context_entry)
            total_chars += len(context_entry)
        
        return "\n".join(context_parts)
    
    def create_rag_prompt(self, query: str, context: str) -> str:
        """Create a comprehensive RAG prompt"""
        prompt = f"""You are an expert assistant for mango inspection and export regulations. Use the provided context to answer questions accurately and comprehensively.

CONTEXT:
{context}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. Be specific and cite relevant sections when possible
3. If the context doesn't contain enough information, say so clearly
4. Focus on practical, actionable information
5. Use bullet points or numbered lists for clarity when appropriate
6. Include relevant details about procedures, requirements, or standards

QUESTION: {query}

ANSWER:"""
        return prompt
    
    def query_llm(self, prompt: str) -> str:
        """Send query to LLM and return response"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Error querying LLM: {e}")
            return f"Error: Could not generate response. {str(e)}"
    
    def rag_query(self, query: str, retriever) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve, format context, generate answer"""
        print(f"\n🔍 Processing query: {query}")
        
        # Step 1: Retrieve relevant documents
        try:
            retrieved_docs = retriever.invoke(query)
            print(f"📄 Retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            return {
                "query": query,
                "status": "error",
                "error": f"Retrieval failed: {str(e)}",
                "answer": "Could not retrieve relevant documents.",
                "sources": []
            }
        
        # Step 2: Format context
        context = self.format_context(retrieved_docs)
        
        # Step 3: Create prompt
        prompt = self.create_rag_prompt(query, context)
        
        # Step 4: Generate answer
        answer = self.query_llm(prompt)
        
        # Step 5: Prepare response with sources
        sources = []
        for i, doc in enumerate(retrieved_docs[:RETRIEVAL_CONTEXT_LIMIT]):
            sources.append({
                "index": i + 1,
                "source": doc.metadata.get('source', 'Unknown'),
                "section": doc.metadata.get('section_heading', 'N/A'),
                "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                "content_preview": doc.page_content[:200] + "..."
            })
        
        return {
            "query": query,
            "status": "success",
            "answer": answer,
            "sources": sources,
            "context_length": len(context),
            "num_retrieved": len(retrieved_docs)
        }

def test_llm_integration(retriever, test_queries: List[str] = None) -> List[Dict[str, Any]]:
    """Test the LLM integration with sample queries"""
    if test_queries is None:
        test_queries = RAG_TEST_QUERIES
    
    try:
        llm_manager = LLMManager()
    except ValueError as e:
        print(f"❌ Could not initialize LLM Manager: {e}")
        return []
    
    results = []
    print("\n" + "="*80)
    print("🤖 TESTING LLM INTEGRATION WITH RAG PIPELINE")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} Query {i}/{len(test_queries)} {'='*20}")
        
        # Get RAG response
        result = llm_manager.rag_query(query, retriever)
        results.append(result)
        
        # Display results
        if result["status"] == "success":
            print(f"💬 Answer:")
            print(f"{result['answer']}\n")
            
            print(f"📊 Metadata:")
            print(f"   - Retrieved: {result['num_retrieved']} documents")
            print(f"   - Context length: {result['context_length']} characters")
            print(f"   - Sources: {len(result['sources'])} documents")
            
            print(f"\n📚 Sources:")
            for source in result['sources']:
                print(f"   {source['index']}. {source['source']} - {source['section']}")
                
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print("-" * 80)
    
    return results

def save_rag_results(results: List[Dict[str, Any]], output_file: str = "rag_test_results.json"):
    """Save RAG test results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 RAG test results saved to {output_file}")