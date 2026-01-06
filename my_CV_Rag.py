import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer
from openai import OpenAI
import PyPDF2
from datetime import datetime

class CVRAGSystem:
    def __init__(self, openai_api_key: str):
        """
        Initialize the CV RAG system
        
        Args:
            openai_api_key: Your OpenAI API key
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Store CV data
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Chunk configuration
        self.chunk_size = 512
        self.chunk_overlap = 50
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT file: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
                
        return chunks
    
    def parse_cv_sections(self, text: str) -> Dict[str, str]:
        """Parse CV into structured sections"""
        sections = {
            'contact': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'projects': '',
            'certifications': ''
        }
        
        # Simple parser - you can enhance this based on your CV format
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Detect section headers
            if 'contact' in line_lower or 'email' in line_lower or 'phone' in line_lower:
                current_section = 'contact'
            elif 'summary' in line_lower or 'objective' in line_lower:
                current_section = 'summary'
            elif 'experience' in line_lower or 'work history' in line_lower:
                current_section = 'experience'
            elif 'education' in line_lower:
                current_section = 'education'
            elif 'skill' in line_lower:
                current_section = 'skills'
            elif 'project' in line_lower:
                current_section = 'projects'
            elif 'certification' in line_lower or 'certificate' in line_lower:
                current_section = 'certifications'
            
            # Add line to current section
            if current_section and line.strip():
                sections[current_section] += line + '\n'
        
        return sections
    
    def load_cv(self, file_path: str, source_type: str = 'auto'):
        """
        Load CV from file
        
        Args:
            file_path: Path to CV file
            source_type: 'pdf', 'txt', or 'auto' for automatic detection
        """
        # Detect file type
        if source_type == 'auto':
            if file_path.lower().endswith('.pdf'):
                source_type = 'pdf'
            elif file_path.lower().endswith('.txt'):
                source_type = 'txt'
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        
        # Extract text
        if source_type == 'pdf':
            text = self.extract_text_from_pdf(file_path)
        elif source_type == 'txt':
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        if not text.strip():
            print(f"No text extracted from {file_path}")
            return
        
        # Parse sections
        sections = self.parse_cv_sections(text)
        
        # Create chunks for each section
        for section_name, section_text in sections.items():
            if section_text.strip():
                chunks = self.chunk_text(section_text)
                for i, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.metadata.append({
                        'section': section_name,
                        'source': file_path,
                        'chunk_id': i,
                        'timestamp': datetime.now().isoformat()
                    })
        
        print(f"Loaded {len(self.documents)} chunks from {file_path}")
    
    def create_embeddings(self):
        """Create embeddings for all CV chunks"""
        if not self.documents:
            print("No documents loaded!")
            return
        
        self.embeddings = self.embedding_model.encode(self.documents)
        print(f"Created embeddings for {len(self.documents)} chunks")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        if self.embeddings is None:
            self.create_embeddings()
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        if len(self.embeddings.shape) == 1:
            similarities = np.dot(self.embeddings, query_embedding)
        else:
            similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def answer_question(self, query: str, top_k: int = 3) -> str:
        """
        Answer question about the CV
        
        Args:
            query: User question
            top_k: Number of chunks to use
            
        Returns:
            Answer based on CV content
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k)
        
        if not relevant_chunks:
            return "No relevant information found in the CV."
        
        # Format context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"[Source: {chunk['metadata']['section']}]\n{chunk['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an AI assistant analyzing a CV/Resume. Answer the question based ONLY on the provided CV context.

CV Context:
{context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided CV context
2. If the information is not in the CV, say "This information is not available in the CV"
3. Be concise and professional
4. When mentioning experience or skills, cite which section they came from
5. Format dates, skills, and achievements clearly

Answer:"""
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful CV/Resume analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def search_cv(self, query: str, search_type: str = "semantic", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search CV with different strategies
        
        Args:
            query: Search query
            search_type: "semantic" (embedding-based) or "keyword" (text search)
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if search_type == "semantic":
            return self.retrieve_relevant_chunks(query, top_k)
        
        elif search_type == "keyword":
            # Simple keyword search (case-insensitive)
            query_lower = query.lower()
            results = []
            
            for idx, doc in enumerate(self.documents):
                if query_lower in doc.lower():
                    results.append({
                        'text': doc,
                        'metadata': self.metadata[idx],
                        'match_type': 'keyword'
                    })
            
            return results[:top_k]
        
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    
    def get_cv_summary(self) -> str:
        """Generate a summary of the CV"""
        if not self.documents:
            return "No CV loaded."
        
        # Combine all documents
        full_cv = "\n".join(self.documents)
        
        prompt = f"""Based on the following CV content, provide a comprehensive summary including:
        1. Professional title/role
        2. Key skills and expertise
        3. Years of experience
        4. Education background
        5. Notable achievements or projects

CV Content:
{full_cv[:4000]}  # Limit context length

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional CV analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"


# Example usage
def main():
    # Initialize with your OpenAI API key
    OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual key
    
    # Create RAG system
    cv_rag = CVRAGSystem(openai_api_key=OPENAI_API_KEY)
    
    # Load your CV (replace with your actual file path)
    cv_rag.load_cv("halimah_CV.pdf")  # or .txt file
    
    # Create embeddings
    cv_rag.create_embeddings()
    
    # Example queries
    print("=" * 60)
    print("CV RAG SYSTEM - Ask questions about your CV")
    print("=" * 60)
    
    while True:
        print("\nWhat would you like to know about the CV?")
        print("1. Ask a question")
        print("2. Get CV summary")
        print("3. Search for specific information")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            query = input("\nEnter your question: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            answer = cv_rag.answer_question(query)
            print("\n" + "=" * 40)
            print("ANSWER:")
            print("=" * 40)
            print(answer)
            
        elif choice == "2":
            summary = cv_rag.get_cv_summary()
            print("\n" + "=" * 40)
            print("CV SUMMARY:")
            print("=" * 40)
            print(summary)
            
        elif choice == "3":
            query = input("\nEnter search query: ").strip()
            results = cv_rag.search_cv(query, search_type="semantic", top_k=3)
            
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [Section: {result['metadata']['section']}]")
                print(f"Similarity: {result.get('similarity', 'N/A'):.4f}")
                print(f"Content: {result['text'][:200]}...")
                
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
    