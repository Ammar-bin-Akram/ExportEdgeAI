import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from data_config import (LM_STUDIO_BASE_URL, LLM_MODEL_NAME, LLM_TEMPERATURE, 
                         LLM_MAX_TOKENS, RETRIEVAL_CONTEXT_LIMIT, CONTEXT_WINDOW_SIZE, RAG_TEST_QUERIES)

class LLMManager:
    """Manages LLM interactions for the RAG pipeline"""
    
    def __init__(self):
        self.base_url = LM_STUDIO_BASE_URL or os.getenv('LM_STUDIO_BASE_URL', 'http://192.168.0.101:1234/v1')
        
        # Initialize OpenAI-compatible client pointing at LM Studio
        # Longer timeout needed — local models can be slow on large prompts
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="lm-studio",  # LM Studio doesn't need a real key
            timeout=300.0,        # 5 min timeout for local inference
        )
        self.model_name = LLM_MODEL_NAME
        
        print(f" LLM Manager initialized — LM Studio @ {self.base_url}")
        print(f"   Model: {self.model_name}")
    
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

    def create_export_prompt(self, inspection_metadata: Dict[str, Any],
                             context: str) -> str:
        """
        Build a prompt that injects structured vision-pipeline inspection results
        as a clearly labelled metadata block, separate from the retrieval query.

        Args:
            inspection_metadata: Dict produced by ExportAdvisor (scores,
                                  defect counts, grade, disease %, etc.)
            context: Retrieved document context string from format_context()

        Returns:
            Complete prompt string ready to send to the LLM.
        """
        # ── Build the structured inspection block ──────────────────────────
        quality     = inspection_metadata.get("surface_quality_score", 0)
        uniformity  = inspection_metadata.get("color_uniformity_score", 0)
        defect_pct  = inspection_metadata.get("total_defect_percentage", 0.0)
        dark        = inspection_metadata.get("dark_spot_count", 0)
        brown       = inspection_metadata.get("brown_spot_count", 0)
        impact      = inspection_metadata.get("export_grade_impact", "unknown")
        grade       = inspection_metadata.get("export_grade", "C")
        disease_pct = inspection_metadata.get("disease_percentage", 0.0)

        inspection_block = f"""\
MANGO INSPECTION RESULTS (from automated vision analysis):
  Surface quality score:   {quality:.0f} / 100
  Colour uniformity score: {uniformity:.0f} / 100
  Defect area:             {defect_pct:.2f}%  (dark spots: {dark}, brown spots: {brown})
  Export grade impact:     {impact}
  Estimated export grade:  Grade {grade}
  Disease coverage (seg):  {disease_pct:.2f}%"""

        prompt = f"""You are an expert in mango export regulations, quality standards, and international trade requirements. \
Use the provided inspection results and regulatory context to give precise country-specific export advice.

{inspection_block}

REGULATORY CONTEXT (retrieved from export standards documents):
{context}

INSTRUCTIONS:
Answer in EXACTLY the following structured format, using these section headers:

RECOMMENDED COUNTRIES:
- List each suitable country/market on its own bullet, with the applicable grade or tolerance in parentheses.

NOT RECOMMENDED:
- List countries that would likely reject this mango, with a brief reason.

ACTIONABLE STEPS:
- Numbered list of concrete actions the exporter should take (e.g. sorting, treatment, labelling, documentation).

CONDITIONS:
- Any pre-export conditions, certifications, or thresholds that must be met.

Rules:
- Be concise — one line per bullet, no long paragraphs.
- Cite the relevant standard or document in parentheses where possible.
- If a section has no items, write "None" under it.

QUESTION: Which export markets are most suitable for this mango, and what conditions or actions (if any) are required?

ANSWER:"""
        return prompt

    def export_rag_query(self, inspection_metadata: Dict[str, Any],
                         retriever,
                         retrieval_query: Optional[str] = None) -> Dict[str, Any]:
        """
        RAG query tailored for export-country prediction.

        Uses a two-part strategy:
          - Retrieval query: focused on grade / defect keyword (for FAISS relevance)
          - Prompt:          full structured metadata block (for LLM clarity)

        Args:
            inspection_metadata: Dict from ExportAdvisor.build_metadata()
            retriever:            LangChain retriever (FAISS-backed)
            retrieval_query:      Override the FAISS query string (optional)

        Returns:
            Same structure as rag_query() — query, status, answer, sources, …
        """
        # Build a keyword-focused retrieval query (numbers confuse embedding models)
        grade  = inspection_metadata.get("export_grade", "B")
        impact = inspection_metadata.get("export_grade_impact", "moderate")
        if retrieval_query is None:
            retrieval_query = (
                f"mango export grade {grade} {impact} defect tolerance "
                f"country requirements inspection standards"
            )

        # Retrieve
        try:
            retrieved_docs = retriever.invoke(retrieval_query)
        except Exception as e:
            return {
                "query": retrieval_query,
                "status": "error",
                "error": f"Retrieval failed: {str(e)}",
                "answer": "Could not retrieve relevant documents.",
                "sources": [],
                "inspection_metadata": inspection_metadata,
            }

        context = self.format_context(retrieved_docs)
        prompt  = self.create_export_prompt(inspection_metadata, context)
        answer  = self.query_llm(prompt)

        sources = []
        for i, doc in enumerate(retrieved_docs[:RETRIEVAL_CONTEXT_LIMIT]):
            sources.append({
                "index": i + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section_heading", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "content_preview": doc.page_content[:200] + "...",
            })

        return {
            "query": retrieval_query,
            "status": "success",
            "answer": answer,
            "sources": sources,
            "context_length": len(context),
            "num_retrieved": len(retrieved_docs),
            "inspection_metadata": inspection_metadata,
        }

    
    def query_llm(self, prompt: str) -> str:
        """Send query to LLM via OpenAI-compatible API (LM Studio)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert assistant for mango inspection and export regulations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            text = response.choices[0].message.content or ""
            # Strip DeepSeek R1 <think>...</think> reasoning blocks
            import re
            text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
            return text
        except Exception as e:
            print(f" Error querying LLM: {e}")
            return f"Error: Could not generate response. {str(e)}"
    
    def rag_query(self, query: str, retriever) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve, format context, generate answer"""
        print(f"\n Processing query: {query}")
        
        # Step 1: Retrieve relevant documents
        try:
            retrieved_docs = retriever.invoke(query)
            print(f" Retrieved {len(retrieved_docs)} documents")
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
        print(f" Could not initialize LLM Manager: {e}")
        return []
    
    results = []
    print("\n" + "="*80)
    print(" TESTING LLM INTEGRATION WITH RAG PIPELINE")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} Query {i}/{len(test_queries)} {'='*20}")
        
        # Get RAG response
        result = llm_manager.rag_query(query, retriever)
        results.append(result)
        
        # Display results
        if result["status"] == "success":
            print(f" Answer:")
            print(f"{result['answer']}\n")
            
            print(f" Metadata:")
            print(f"   - Retrieved: {result['num_retrieved']} documents")
            print(f"   - Context length: {result['context_length']} characters")
            print(f"   - Sources: {len(result['sources'])} documents")
            
            print(f"\n Sources:")
            for source in result['sources']:
                print(f"   {source['index']}. {source['source']} - {source['section']}")
                
        else:
            print(f" Error: {result.get('error', 'Unknown error')}")
        
        print("-" * 80)
    
    return results

def save_rag_results(results: List[Dict[str, Any]], output_file: str = "rag_test_results.json"):
    """Save RAG test results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f" RAG test results saved to {output_file}")