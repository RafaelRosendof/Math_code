#from IPython.display import Markdown, display
import gradio as gr
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

system_prompt = """You are an elite software engineering assistant with deep expertise across:
- Event driven architectures (Kafka, RabbitMQ, AWS SQS/SNS, etc.)
- Software Engineering (Python, Go, Java/Spring, API Design, Distributed Systems)
- Good software design principles (SOLID, DRY, KISS, etc.)
- Design patterns (Factory, Singleton, Observer, etc.)

Your answers are grounded EXCLUSIVELY in the retrieved context provided to you.

═══════════════════════════════════════
CORE RULES
═══════════════════════════════════════

1. TRUTH-GROUNDING
   - Use ONLY information from the retrieved context.
   - If the answer is not in the context, say exactly:
     "I did not find specific information regarding this in the loaded documents."
   - Never infer, hallucinate, or supplement with outside knowledge.

2. MATH & THEORY
   - Render ALL mathematical expressions in LaTeX:
     · Inline: $expression$
     · Block:  $$expression$$
   - Preserve proof structure and logical flow when present.
   - Define every symbol on first use.
   - Use standard terminology: Martingales, Itô's Lemma, σ-algebras,
     Filtrations, Risk-Neutral Measure, etc.

3. CODE
   - Always specify the language in fenced code blocks:
``````python
`````go
````java
   - Explain what the code does BEFORE showing it.
   - If the context shows a specific implementation pattern, follow it exactly.
   - Highlight key lines with inline comments.

4. STRUCTURE
   - Use clear markdown headings (##, ###).
   - Prefer structured answers: definitions → theory → examples → code.
   - Use bullet points for lists, numbered steps for sequences.
   - End complex answers with a ## Summary section.

5. CROSS-DOMAIN ANSWERS
   - If the question touches both math and code (e.g., implementing a 
     stochastic model), address the theory first, then the implementation.
   - Draw explicit connections between the mathematical concept and its
     code representation.

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

## [Topic]

### Definition / Concept
...

### Mathematical Formulation
$$...$$

### Implementation
```language
...
```

### Summary
...

### TOOLS 

- Now you can have access to a set of tools that allow you to interact with a Firefox instance via the Marionette protocol.
These tools can be used to automate browser actions, retrieve information from web pages, and more.
- If you need to complement your answer with information from the web, you can use these tools to interact with the Firefox browser and retrieve the necessary data.


Respond in the language of the user's question.
"""

collection_name = "figas_software"
client = QdrantClient("http://localhost:6333")

def search_figas(pergunta, system_prompt, top_k=5, domain=None):
    resp_emb = ollama.embed(model="qwen3-embedding", input=pergunta)
    query_vector = resp_emb.embeddings[0]

    query_filter = None
    if domain:
        query_filter = Filter(must=[
            FieldCondition(key="domain", match=MatchValue(value=domain))
        ])

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=0.42,
        query_filter=query_filter
    ).points

    if not search_result:
        return "No relevant context found in the loaded documents.", []

    contexto_formatado = ""
    for hit in search_result:
        source = hit.payload.get("source", "unknown")
        domain_tag = hit.payload.get("domain", "?")
        score = hit.score
        contexto_formatado += f"\n--- Source: {source} | Domain: {domain_tag} | Score: {score:.2f} ---\n"
        contexto_formatado += hit.payload["text"] + "\n"

    user_content = f"""CONTEXT RECOVERED:
{contexto_formatado}

QUESTION: {pergunta}

ANSWER THE QUESTION BASED ONLY ON THE CONTEXT ABOVE. If the context contains mathematical formulas, explain them in detail.
"""
    return user_content, search_result

    #final_resp = ollama.chat(
    #    model="qwen3:14b",
    #    messages=[
    #        {"role": "system", "content": system_prompt},
    #        {"role": "user", "content": user_content}
    #    ]
    #)
#
    #return final_resp.message.content, search_result

resposta = " How can i'm do a udp connection in rust with not the http servers os libs but with the sockets of the system? I want to do a simple connection with a server and send a message to it, and then receive the response from the server. I want to do this in a simple way, without using any external libraries, just the standard library of rust."

print(type(resposta))

async def chat_wrapper(message, history):

    server_params = StdioServerParameters(
        command="npx",
        args=["firefox-devtools-mcp", "--connect-existing", "--marionette-port", "2828"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            mcp_tool_resp = await session.list_tools()
            
            ollama_tools = []
            
            for tool in mcp_tool_resp.tools:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                })
                
            context_rag, fonts = search_figas(message, system_prompt)
            
            user_context = f"""CONTEXT RECOVERED (if any):
            {context_rag}
            Fonts{fonts}
            QUESTION: {message}
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ]
            
            resp = ollama.chat(
                model="qwen3:14b",
                messages=messages,
                tools=ollama_tools
            )
            
            if resp.message.tool_calls:
                messages.append(resp.message)
                
                for tool_call in resp.message.tool_calls:
                    print(f"🔧 O LLM acionou o Firefox: {tool_call.function.name}")
                    
                    mcp_result = await session.call_tool(
                        tool_call.function.name,
                        tool_call.function.arguments
                    )

                    tool_result_text = "\n".join([c.text for c in mcp_result.content if c.type == "text"])
                    
                    messages.append({
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": tool_result_text
                    })
                    
                final_resp = ollama.chat(
                    model="qwen3:14b",
                    messages=messages
                )
                
                return final_resp.message.content + "\n\n*(Ação executada no Firefox)*"
            
            return resp.message.content
        
                    
def chat_wrapper(message = resposta, system_prompt=system_prompt): 
    resposta, fontes = search_figas(message, system_prompt)
    
    source_text = "\n\n**Sources:**\n" + "\n".join(
        [f"· {h.payload.get('source')} (Score: {h.score:.2f})" for h in fontes]
    )
    
    return resposta + source_text

demo = gr.ChatInterface(
    fn=chat_wrapper, 
    title="Figas Software Assistant",
    description="Elite Engineering RAG System"
)

if __name__ == "__main__":
    demo.launch()