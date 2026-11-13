import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class PrivacyMultiAgentRAG:
    def __init__(self):
        # 100% Local Embedding Model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.model = "qwen2:0.5b"

    def ingest_document(self, file_path):
        """Reads a private PDF, chunks it, and embeds it into a local ChromaDB."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        self.vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings, 
            persist_directory="./chroma_local_db"
        )
        return len(splits)

    def _researcher(self, query):
        """AGENT 1: Retrieves semantic context from the private database."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

    def _synthesizer(self, query, context, feedback=""):
        """AGENT 2: Drafts the answer. Forced to rely ONLY on provided context."""
        prompt = f"Private Context: {context}\n\nUser Query: {query}\n"
        if feedback:
            prompt += f"\nCritic Feedback to fix: {feedback}\n"
        prompt += "\nDraft a concise, accurate answer based EXCLUSIVELY on the Private Context. Do not use outside knowledge."
        
        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': 'You are a Synthesizer Agent. You summarize private enterprise data.'},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']

    def _critic(self, query, draft, context):
        """AGENT 3: Validates the draft to prevent AI hallucinations."""
        prompt = f"Context: {context}\n\nQuery: {query}\n\nDraft Answer: {draft}\n\nReview the draft. Does it accurately and directly answer the query using ONLY the provided context? If perfectly accurate, reply exactly with 'APPROVED'. If it contains hallucinations or outside knowledge, reply with 1 sentence of strict feedback."
        
        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': 'You are a strict QA Critic Agent. Your job is to prevent data leakage and hallucinations.'},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']

    def run_orchestration(self, query):
        """Coordinates the multi-agent workflow and yields live status updates."""
        if not self.vector_store:
            yield {"agent": "System", "status": "Error: Please upload a document first."}
            return

        yield {"agent": "Researcher", "status": "🔍 Querying local vector database..."}
        context = self._researcher(query)
        
        draft = ""
        feedback = ""
        
        # The Synthesizer gets 2 attempts to pass the Critic's strict review
        for attempt in range(2):
            yield {"agent": "Synthesizer", "status": f"✍️ Drafting response (Attempt {attempt+1})..."}
            draft = self._synthesizer(query, context, feedback)
            
            yield {"agent": "Critic", "status": "🕵️ Reviewing draft against source documents..."}
            critique = self._critic(query, draft, context)
            
            if "APPROVED" in critique.upper():
                yield {"agent": "Critic", "status": "✅ Draft APPROVED. Zero hallucinations detected."}
                yield {"agent": "Final", "content": draft}
                return
            else:
                feedback = critique
                yield {"agent": "Critic", "status": f"❌ Draft REJECTED. Feedback: {feedback}"}
        
        # Fallback if it fails twice
        yield {"agent": "Critic", "status": "⚠️ Output forced after max retries."}
        yield {"agent": "Final", "content": draft}