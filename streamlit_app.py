import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import anthropic
import asyncio
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
cx = sqlite3.connect("test.db")

"""
MVP Implementation Outline:

1. UPLOAD INTERFACE
- Upload JSON transcripts
- Validate JSON format matches expected structure
- Show upload status/confirmation
- Store raw transcripts

2. BASIC PROCESSING
- Parse uploaded JSON
- Extract segments based on existing structure
- Store processed segments
- Display processing status

3. CONTEXTUAL ENRICHMENT
- Send segments to Claude
- Generate context for each segment
- Handle batch processing
- Track enrichment progress

4. EMBEDDING CREATION
- Create embeddings for original segments
- Create embeddings for contextualized versions
- Handle batch embedding creation
- Track embedding progress

5. STORAGE
- Store original segments
- Store contextualized versions
- Store embeddings
- Enable retrieval of all versions

6. QUERY INTERFACE
- Text input for queries
- Convert query to embedding
- Search through stored embeddings
- Display matching segments

7. RESPONSE GENERATION
- Take query and relevant segments
- Format prompt for Claude
- Get response
- Display response with evidence

8. RESPONSE STORAGE
- Store query
- Store matching segments
- Store Claude's response
- Enable browsing of past queries/responses

9. DATA MANAGEMENT
- View stored transcripts
- View stored segments
- View enrichment status
- Basic data cleanup options
"""

class TranscriptUploader:
    def __init__(self):
        self.storage_path = Path("data/transcripts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_transcript(self, content: str, filename: str) -> bool:
        """Saves and validates uploaded transcript"""
        try:
            # Parse JSON to validate format
            transcript_data = json.loads(content)
            
            # Validate expected structure
            if not self._validate_structure(transcript_data):
                return False
                
            # Save to file
            file_path = self.storage_path / f"{Path(filename).stem}.json"
            with open(file_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)
                
            return True
            
        except json.JSONDecodeError:
            return False
            
    def _validate_structure(self, data: Dict) -> bool:
        """Validates transcript has required structure"""
        required_keys = {'dialogue', 'metadata'}
        if not all(key in data for key in required_keys):
            return False
            
        if not isinstance(data['dialogue'], list):
            return False
            
        if not data['dialogue']:  # Empty dialogue
            return False
            
        # Validate dialogue structure
        required_dialogue_keys = {'speaker', 'text', 'start', 'end'}
        return all(
            all(key in entry for key in required_dialogue_keys)
            for entry in data['dialogue']
        )
    
    def get_stored_transcripts(self) -> List[str]:
        """Returns list of stored transcript filenames"""
        return [f.stem for f in self.storage_path.glob('*.json')]
    
# Step 2: Basic Processing Implementation
class SegmentProcessor:
    def __init__(self):
        self.segments_path = Path("data/segments")
        self.segments_path.mkdir(parents=True, exist_ok=True)
    
    def process_transcript(self, filename: str, transcript_data: Dict) -> List[Dict]:
        """Process transcript into segments"""
        segments = []
        current_segment = []
        
        for entry in transcript_data['dialogue']:
            current_segment.append(entry)
            
            # Create segment every 5 dialogue entries
            # (adjust this number based on your needs)
            if len(current_segment) >= 5:
                segment = self._create_segment(
                    current_segment, 
                    transcript_data['metadata'],
                    len(segments)
                )
                segments.append(segment)
                current_segment = []
        
        # Handle any remaining dialogue
        if current_segment:
            segment = self._create_segment(
                current_segment, 
                transcript_data['metadata'],
                len(segments)
            )
            segments.append(segment)
        
        # Store segments
        self._store_segments(filename, segments)
        
        return segments
    
    def _create_segment(self, 
                       dialogue_entries: List[Dict], 
                       transcript_metadata: Dict,
                       segment_index: int) -> Dict:
        """Create a segment from dialogue entries"""
        return {
            'id': f"seg_{segment_index:04d}",
            'content': dialogue_entries,
            'speakers': list(set(entry['speaker'] for entry in dialogue_entries)),
            'start_time': dialogue_entries[0]['start'],
            'end_time': dialogue_entries[-1]['end'],
            'metadata': {
                'transcript_id': transcript_metadata.get('transcript_name'),
                'segment_index': segment_index,
                'processing_date': datetime.now().isoformat()
            }
        }
    
    def _store_segments(self, transcript_name: str, segments: List[Dict]):
        """Store processed segments"""
        file_path = self.segments_path / f"{transcript_name}_segments.json"
        with open(file_path, 'w') as f:
            json.dump(segments, f, indent=2)
    
    def get_processed_segments(self, transcript_name: str) -> Optional[List[Dict]]:
        """Retrieve processed segments for a transcript"""
        file_path = self.segments_path / f"{transcript_name}_segments.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

# Step 3: Contextual Enrichment Implementation
class ClaudeEnricher:
    def __init__(self):
        self.enriched_path = Path("data/enriched")
        self.enriched_path.mkdir(parents=True, exist_ok=True)
        self.claude = anthropic.Client(api_key=st.secrets["your_api_key_here"])
        
    async def enrich_segments(self, segments: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Enrich segments with context using Claude"""
        enriched_segments = []
        
        # Process in batches
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            tasks = [self._enrich_segment(segment) for segment in batch]
            batch_results = await asyncio.gather(*tasks)
            enriched_segments.extend(batch_results)
            
            # Update progress for Streamlit
            progress = (i + len(batch)) / len(segments)
            st.progress(progress)
        
        return enriched_segments
    
    async def _enrich_segment(self, segment: Dict) -> Dict:
        """Generate context for a single segment"""
        context = await self._generate_context(segment)
        
        enriched_segment = {
            **segment,
            'enrichment': {
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Create optimized text for embedding
        enriched_segment['embedding_text'] = self._create_embedding_text(
            segment, context
        )
        
        return enriched_segment
    
    async def _generate_context(self, segment: Dict) -> str:
        """Use Claude to generate context"""
        # Prepare segment content for prompt
        dialogue_text = "\n".join([
            f"{entry['speaker']}: {entry['text']}" 
            for entry in segment['content']
        ])
        
        prompt = f"""
        For this conversation segment:
        
        {dialogue_text}
        
        Provide brief, focused context about:
        1. The specific topic/issue being discussed
        2. Key points or concerns raised
        3. Important technical or domain-specific details
        
        Return only the contextual information, nothing else.
        """
        
        response = await self._get_claude_response(prompt)
        return response.content
    
    async def _get_claude_response(self, prompt: str) -> anthropic.types.Message:
        """Get response from Claude with error handling"""
        try:
            return await self.claude.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            st.error(f"Error getting Claude response: {str(e)}")
            raise
    
    def _create_embedding_text(self, segment: Dict, context: str) -> str:
        """Create optimized text for embedding"""
        dialogue_text = " ".join([
            f"{entry['speaker']}: {entry['text']}" 
            for entry in segment['content']
        ])
        
        return f"""
        Context: {context}
        Content: {dialogue_text}
        Speakers: {', '.join(segment['speakers'])}
        """.strip()
    
    def store_enriched_segments(self, transcript_name: str, segments: List[Dict]):
        """Store enriched segments"""
        file_path = self.enriched_path / f"{transcript_name}_enriched.json"
        with open(file_path, 'w') as f:
            json.dump(segments, f, indent=2)
    
    def get_enriched_segments(self, transcript_name: str) -> Optional[List[Dict]]:
        """Retrieve enriched segments"""
        file_path = self.enriched_path / f"{transcript_name}_enriched.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

# Step 4: Embedding Creation Implementation
class ChromaEmbedder:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Create collections for both original and enriched versions
        self.original_collection = self.client.create_collection(
            name="original_segments",
            embedding_function=self.embedding_fn
        )
        
        self.enriched_collection = self.client.create_collection(
            name="enriched_segments",
            embedding_function=self.embedding_fn
        )
    
    def create_embeddings(self, segments: List[Dict], is_enriched: bool = False) -> None:
        """Create embeddings for segments"""
        collection = self.enriched_collection if is_enriched else self.original_collection
        
        # Prepare batch data
        ids = [str(segment['id']) for segment in segments]
        
        # Create texts for embedding
        if is_enriched:
            texts = [segment['embedding_text'] for segment in segments]
        else:
            texts = [
                " ".join([entry['text'] for entry in segment['content']])
                for segment in segments
            ]
        
        # Create metadata
        metadatas = [
            {
                'transcript_id': segment['metadata']['transcript_id'],
                'segment_index': segment['metadata'].get('segment_index', 0),
                'is_enriched': is_enriched
            }
            for segment in segments
        ]
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def get_embedding_status(self, transcript_id: str) -> Dict:
        """Get embedding status for a transcript"""
        original_count = len(self.original_collection.get(
            where={"transcript_id": transcript_id}
        )['ids'])
        
        enriched_count = len(self.enriched_collection.get(
            where={"transcript_id": transcript_id}
        )['ids'])
        
        return {
            'original_segments': original_count,
            'enriched_segments': enriched_count
        }

# Step 5: Storage System Implementation
class StorageSystem:
    def __init__(self):
        self.base_path = Path("data")
        self.transcripts_path = self.base_path / "transcripts"
        self.segments_path = self.base_path / "segments" 
        self.enriched_path = self.base_path / "enriched"
        self.relationships_path = self.base_path / "relationships"
        
        # Create all required directories
        for path in [self.transcripts_path, self.segments_path, 
                    self.enriched_path, self.relationships_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize SQLite for relationships
        self.db_path = self.base_path / "storage.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self.setup_database()
        
    def setup_database(self):
        """Create necessary database tables"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY,
            transcript_id TEXT,
            content TEXT,
            metadata JSON,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS enriched_segments (
            id TEXT PRIMARY KEY,
            segment_id TEXT,
            context TEXT,
            embedding_text TEXT,
            metadata JSON,
            created_at TIMESTAMP,
            FOREIGN KEY (segment_id) REFERENCES segments(id)
        )
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS segment_relationships (
            from_segment_id TEXT,
            to_segment_id TEXT,
            relationship_type TEXT,
            confidence FLOAT,
            metadata JSON,
            created_at TIMESTAMP,
            PRIMARY KEY (from_segment_id, to_segment_id, relationship_type),
            FOREIGN KEY (from_segment_id) REFERENCES segments(id),
            FOREIGN KEY (to_segment_id) REFERENCES segments(id)
        )
        """)
        
        self.conn.commit()
    
    def store_transcript(self, content: Dict, filename: str) -> str:
        """Store original transcript"""
        transcript_id = f"trans_{int(datetime.now().timestamp())}"
        file_path = self.transcripts_path / f"{transcript_id}.json"
        
        # Add storage metadata
        content['storage_metadata'] = {
            'transcript_id': transcript_id,
            'original_filename': filename,
            'stored_at': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
            
        return transcript_id
    
    def store_segments(self, transcript_id: str, segments: List[Dict]) -> List[str]:
        """Store processed segments with SQL and JSON"""
        segment_ids = []
        
        for segment in segments:
            segment_id = segment['id']
            segment_ids.append(segment_id)
            
            # Store in SQLite
            self.conn.execute("""
            INSERT INTO segments 
            (id, transcript_id, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                segment_id,
                transcript_id,
                json.dumps(segment['content']),
                json.dumps(segment['metadata']),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            # Store full data as JSON
            file_path = self.segments_path / f"{segment_id}.json"
            with open(file_path, 'w') as f:
                json.dump(segment, f, indent=2)
        
        self.conn.commit()
        return segment_ids
    
    def store_enriched_segment(self, segment_id: str, enriched_data: Dict) -> str:
        """Store enriched segment data"""
        enriched_id = f"enr_{segment_id}"
        
        # Store in SQLite
        self.conn.execute("""
        INSERT INTO enriched_segments
        (id, segment_id, context, embedding_text, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            enriched_id,
            segment_id,
            enriched_data['enrichment']['context'],
            enriched_data['embedding_text'],
            json.dumps(enriched_data.get('metadata', {})),
            datetime.now().isoformat()
        ))
        
        # Store full data as JSON
        file_path = self.enriched_path / f"{enriched_id}.json"
        with open(file_path, 'w') as f:
            json.dump(enriched_data, f, indent=2)
            
        self.conn.commit()
        return enriched_id
    
    def add_relationship(self, 
                        from_segment_id: str,
                        to_segment_id: str,
                        relationship_type: str,
                        confidence: float = 1.0,
                        metadata: Dict = None):
        """Add relationship between segments"""
        self.conn.execute("""
        INSERT INTO segment_relationships
        (from_segment_id, to_segment_id, relationship_type, 
         confidence, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            from_segment_id,
            to_segment_id,
            relationship_type,
            confidence,
            json.dumps(metadata or {}),
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def get_transcript(self, transcript_id: str) -> Optional[Dict]:
        """Retrieve transcript data"""
        file_path = self.transcripts_path / f"{transcript_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_segment(self, segment_id: str) -> Optional[Dict]:
        """Retrieve segment data"""
        file_path = self.segments_path / f"{segment_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_enriched_segment(self, enriched_id: str) -> Optional[Dict]:
        """Retrieve enriched segment data"""
        file_path = self.enriched_path / f"{enriched_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_segment_relationships(self, segment_id: str) -> List[Dict]:
        """Get all relationships for a segment"""
        cursor = self.conn.execute("""
        SELECT * FROM segment_relationships
        WHERE from_segment_id = ? OR to_segment_id = ?
        """, (segment_id, segment_id))
        
        relationships = []
        for row in cursor.fetchall():
            relationships.append({
                'from_segment': row[0],
                'to_segment': row[1],
                'type': row[2],
                'confidence': row[3],
                'metadata': json.loads(row[4]),
                'created_at': row[5]
            })
            
        return relationships
    
    def get_transcript_segments(self, transcript_id: str) -> List[Dict]:
        """Get all segments for a transcript"""
        cursor = self.conn.execute("""
        SELECT id FROM segments
        WHERE transcript_id = ?
        ORDER BY created_at
        """, (transcript_id,))
        
        segments = []
        for (segment_id,) in cursor.fetchall():
            segment = self.get_segment(segment_id)
            if segment:
                segments.append(segment)
                
        return segments

#Step 6: Query Interface implementation
class QueryInterface:
    def __init__(self, chroma_embedder: ChromaEmbedder):
        self.embedder = chroma_embedder
        self.results_path = Path("data/query_results")
        self.results_path.mkdir(parents=True, exist_ok=True)

    async def search_segments(self, 
                            query: str, 
                            use_enriched: bool = True,
                            top_k: int = 10) -> List[Dict]:
        """Search for relevant segments using both collections"""
        collection = (self.embedder.enriched_collection if use_enriched 
                     else self.embedder.original_collection)

        # Search using query
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'score': float(results['distances'][0][i]),
                'metadata': results['metadatas'][0][i],
                'content': results['documents'][0][i]
            }
            formatted_results.append(result)

        return formatted_results

    def store_query_result(self, 
                          query: str,
                          results: List[Dict]) -> str:
        """Store query results for future reference"""
        query_id = f"query_{int(datetime.now().timestamp())}"
        
        result_data = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        file_path = self.results_path / f"{query_id}.json"
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        return query_id

    def get_query_history(self) -> List[Dict]:
        """Get history of past queries"""
        history = []
        for file_path in self.results_path.glob('*.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                history.append({
                    'id': file_path.stem,
                    'query': data['query'],
                    'timestamp': data['timestamp'],
                    'result_count': len(data['results'])
                })
        
        return sorted(history, 
                     key=lambda x: x['timestamp'], 
                     reverse=True)

    def get_query_result(self, query_id: str) -> Optional[Dict]:
        """Retrieve specific query results"""
        file_path = self.results_path / f"{query_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

# Step 7: RESPONSE GENERATION implementation
class ResponseGenerator:
    """Handles response generation using Claude and retrieved segments"""
    
    def __init__(self, claude_client, storage_system: StorageSystem):
        self.claude = claude_client
        self.storage = storage_system
        self.responses_path = Path("data/responses")
        self.responses_path.mkdir(parents=True, exist_ok=True)
    
    async def generate_response(self,
                              query: str,
                              retrieved_segments: List[Dict],
                              max_segments: int = 5) -> Dict:
        """Generate response using query and retrieved segments"""
        # Select top segments based on score
        top_segments = sorted(
            retrieved_segments,
            key=lambda x: x['score'],
            reverse=True
        )[:max_segments]
        
        # Create context from segments
        context = self._create_context(top_segments)
        
        # Generate response
        response = await self._get_claude_response(query, context)
        
        # Create response object
        response_data = {
            'query': query,
            'response': response.content,
            'evidence': [segment['id'] for segment in top_segments],
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'model': "claude-3-opus-20240229",
                'segments_used': len(top_segments)
            }
        }
        
        # Store response
        response_id = self._store_response(response_data)
        response_data['id'] = response_id
        
        return response_data
    
    def _create_context(self, segments: List[Dict]) -> str:
        """Create prompt context from segments"""
        context_parts = []
        
        for segment in segments:
            # Get full segment data if available
            full_segment = self.storage.get_segment(segment['id'])
            if full_segment:
                content = "\n".join([
                    f"{entry['speaker']}: {entry['text']}"
                    for entry in full_segment['content']
                ])
            else:
                content = segment['content']
            
            context_parts.append(f"""
Segment {segment['id']}:
{content}
---""")
            
        return "\n".join(context_parts)
    
    async def _get_claude_response(self, 
                                 query: str,
                                 context: str) -> anthropic.types.Message:
        """Get response from Claude"""
        prompt = f"""Based on these transcript segments, answer the following query:

Context segments:
{context}

Query: {query}

Provide a clear, direct answer that:
1. Specifically addresses the query
2. Uses information from the provided segments
3. References specific segments when possible
4. Acknowledges if information is incomplete

Response:"""
        
        try:
            return await self.claude.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            st.error(f"Error getting Claude response: {str(e)}")
            raise
    
    def _store_response(self, response_data: Dict) -> str:
        """Store generated response"""
        response_id = f"resp_{int(datetime.now().timestamp())}"
        file_path = self.responses_path / f"{response_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(response_data, f, indent=2)
            
        return response_id
    
    def get_response_history(self) -> List[Dict]:
        """Get history of generated responses"""
        history = []
        for file_path in self.responses_path.glob('*.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                history.append({
                    'id': file_path.stem,
                    'query': data['query'],
                    'timestamp': data['timestamp'],
                    'evidence_count': len(data['evidence'])
                })
        
        return sorted(history, 
                     key=lambda x: x['timestamp'], 
                     reverse=True)
    
    def get_response(self, response_id: str) -> Optional[Dict]:
        """Retrieve specific response"""
        file_path = self.responses_path / f"{response_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

class ResponseStorage:
    """Manages storage and retrieval of generated responses and their evidence"""
    
    def __init__(self, storage_system: StorageSystem):
        self.storage = storage_system
        self.responses_path = Path("data/responses")
        self.responses_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite for response tracking
        self.db_path = Path("data/responses.db")
        self.conn = sqlite3.connect(str(self.db_path))
        self.setup_database()
        
    def setup_database(self):
        """Create database schema for responses"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            query TEXT,
            response_text TEXT,
            model TEXT,
            confidence FLOAT,
            created_at TIMESTAMP,
            metadata JSON
        )
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS response_evidence (
            response_id TEXT,
            segment_id TEXT,
            relevance_score FLOAT,
            created_at TIMESTAMP,
            FOREIGN KEY (response_id) REFERENCES responses(id),
            FOREIGN KEY (segment_id) REFERENCES segments(id)
        )
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS response_feedback (
            response_id TEXT,
            feedback_type TEXT,
            feedback_value TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (response_id) REFERENCES responses(id)
        )
        """)
        
        self.conn.commit()
    
    def store_response(self, response_data: Dict) -> str:
        """Store complete response with evidence links"""
        response_id = f"resp_{int(datetime.now().timestamp())}"
        
        # Store in SQLite
        self.conn.execute("""
        INSERT INTO responses 
        (id, query, response_text, model, confidence, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            response_id,
            response_data['query'],
            response_data['response'],
            response_data['metadata']['model'],
            response_data['metadata'].get('confidence', 0.0),
            datetime.now().isoformat(),
            json.dumps(response_data['metadata'])
        ))
        
        # Store evidence links
        for segment_id in response_data['evidence']:
            self.conn.execute("""
            INSERT INTO response_evidence
            (response_id, segment_id, relevance_score, created_at)
            VALUES (?, ?, ?, ?)
            """, (
                response_id,
                segment_id,
                1.0,  # Default score
                datetime.now().isoformat()
            ))
        
        # Store full response data as JSON
        file_path = self.responses_path / f"{response_id}.json"
        with open(file_path, 'w') as f:
            json.dump({**response_data, 'id': response_id}, f, indent=2)
            
        self.conn.commit()
        return response_id
    
    def get_response(self, response_id: str) -> Optional[Dict]:
        """Retrieve complete response with evidence"""
        # Get basic response data
        cursor = self.conn.execute("""
        SELECT * FROM responses WHERE id = ?
        """, (response_id,))
        
        response_row = cursor.fetchone()
        if not response_row:
            return None
            
        # Get evidence
        cursor = self.conn.execute("""
        SELECT segment_id, relevance_score
        FROM response_evidence
        WHERE response_id = ?
        ORDER BY relevance_score DESC
        """, (response_id,))
        
        evidence = []
        for segment_id, score in cursor.fetchall():
            segment = self.storage.get_segment(segment_id)
            if segment:
                evidence.append({
                    'segment': segment,
                    'relevance_score': score
                })
        
        # Get feedback
        cursor = self.conn.execute("""
        SELECT feedback_type, feedback_value, created_at
        FROM response_feedback
        WHERE response_id = ?
        ORDER BY created_at DESC
        """, (response_id,))
        
        feedback = [
            {
                'type': row[0],
                'value': row[1],
                'timestamp': row[2]
            }
            for row in cursor.fetchall()
        ]
        
        # Combine all data
        return {
            'id': response_id,
            'query': response_row[1],
            'response': response_row[2],
            'model': response_row[3],
            'confidence': response_row[4],
            'created_at': response_row[5],
            'metadata': json.loads(response_row[6]),
            'evidence': evidence,
            'feedback': feedback
        }
    
    def add_feedback(self,
                    response_id: str,
                    feedback_type: str,
                    feedback_value: str):
        """Add feedback for a response"""
        self.conn.execute("""
        INSERT INTO response_feedback
        (response_id, feedback_type, feedback_value, created_at)
        VALUES (?, ?, ?, ?)
        """, (
            response_id,
            feedback_type,
            feedback_value,
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def get_responses_by_query(self, 
                             query_text: str,
                             limit: int = 10) -> List[Dict]:
        """Find similar previous responses"""
        cursor = self.conn.execute("""
        SELECT id, query, response_text, created_at
        FROM responses
        WHERE query LIKE ?
        ORDER BY created_at DESC
        LIMIT ?
        """, (f"%{query_text}%", limit))
        
        return [
            {
                'id': row[0],
                'query': row[1],
                'response': row[2],
                'timestamp': row[3]
            }
            for row in cursor.fetchall()
        ]
    
    def get_response_stats(self) -> Dict:
        """Get usage statistics for responses"""
        cursor = self.conn.execute("""
        SELECT 
            COUNT(*) as total_responses,
            COUNT(DISTINCT model) as models_used,
            AVG(confidence) as avg_confidence,
            MIN(created_at) as first_response,
            MAX(created_at) as last_response
        FROM responses
        """)
        
        row = cursor.fetchone()
        
        return {
            'total_responses': row[0],
            'models_used': row[1],
            'avg_confidence': row[2],
            'first_response': row[3],
            'last_response': row[4]
        }
    
    def cleanup_old_responses(self, days_old: int = 30):
        """Remove responses older than specified days"""
        cutoff = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        # Get responses to delete
        cursor = self.conn.execute("""
        SELECT id FROM responses
        WHERE created_at < ?
        """, (cutoff,))
        
        response_ids = [row[0] for row in cursor.fetchall()]
        
        # Delete from all tables
        for response_id in response_ids:
            # Delete evidence links
            self.conn.execute("""
            DELETE FROM response_evidence
            WHERE response_id = ?
            """, (response_id,))
            
            # Delete feedback
            self.conn.execute("""
            DELETE FROM response_feedback
            WHERE response_id = ?
            """, (response_id,))
            
            # Delete response
            self.conn.execute("""
            DELETE FROM responses
            WHERE id = ?
            """, (response_id,))
            
            # Delete JSON file
            file_path = self.responses_path / f"{response_id}.json"
            if file_path.exists():
                file_path.unlink()
        
        self.conn.commit()
        return len(response_ids)

# Step 9: Data Management Implementation
class DataManager:
    """Manages cleanup, monitoring, and maintenance of all stored data"""
    
    def __init__(self, 
                 storage_system: StorageSystem,
                 chroma_embedder: ChromaEmbedder):
        self.storage = storage_system
        self.embedder = chroma_embedder
        self.stats_path = Path("data/statistics")
        self.stats_path.mkdir(parents=True, exist_ok=True)
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            'transcripts': self._get_transcript_stats(),
            'segments': self._get_segment_stats(),
            'embeddings': self._get_embedding_stats(),
            'disk_usage': self._get_storage_stats(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Store stats history
        self._store_stats_snapshot(stats)
        
        return stats
    
    def _get_transcript_stats(self) -> Dict:
        """Get transcript statistics"""
        cursor = self.storage.conn.execute("""
        SELECT COUNT(*) as total_transcripts,
               MIN(created_at) as oldest,
               MAX(created_at) as newest
        FROM segments
        GROUP BY transcript_id
        """)
        
        results = cursor.fetchall()
        total = len(results) if results else 0
        
        return {
            'total_count': total,
            'oldest': results[0][1] if total > 0 else None,
            'newest': results[0][2] if total > 0 else None
        }
    
    def _get_segment_stats(self) -> Dict:
        """Get segment statistics"""
        cursor = self.storage.conn.execute("""
        SELECT COUNT(*) as total_segments,
               COUNT(DISTINCT transcript_id) as transcripts,
               AVG(LENGTH(content)) as avg_length
        FROM segments
        """)
        
        row = cursor.fetchone()
        
        # Get enrichment stats
        cursor = self.storage.conn.execute("""
        SELECT COUNT(*) as total_enriched
        FROM enriched_segments
        """)
        
        enriched_count = cursor.fetchone()[0]
        
        return {
            'total_segments': row[0],
            'transcript_count': row[1],
            'average_length': row[2],
            'enriched_count': enriched_count,
            'enrichment_ratio': enriched_count / row[0] if row[0] > 0 else 0
        }
    
    def _get_embedding_stats(self) -> Dict:
        """Get embedding statistics"""
        original_count = len(self.embedder.original_collection.get()['ids'])
        enriched_count = len(self.embedder.enriched_collection.get()['ids'])
        
        return {
            'original_embeddings': original_count,
            'enriched_embeddings': enriched_count,
            'total_embeddings': original_count + enriched_count
        }
    
    def _get_storage_stats(self) -> Dict:
        """Get storage usage statistics"""
        stats = {}
        
        for path in [self.storage.transcripts_path, 
                    self.storage.segments_path,
                    self.storage.enriched_path,
                    self.storage.relationships_path]:
            stats[path.name] = sum(
                f.stat().st_size for f in path.glob('**/*') if f.is_file()
            )
            
        return stats
    
    def _store_stats_snapshot(self, stats: Dict):
        """Store statistics snapshot for tracking"""
        snapshot_id = f"stats_{int(datetime.now().timestamp())}"
        file_path = self.stats_path / f"{snapshot_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def cleanup_data(self, 
                    older_than_days: int = 30,
                    transcript_ids: List[str] = None) -> Dict:
        """Clean up old or specified data"""
        results = {
            'transcripts_removed': 0,
            'segments_removed': 0,
            'embeddings_removed': 0,
            'storage_freed': 0
        }
        
        try:
            # Start transaction
            self.storage.conn.execute("BEGIN TRANSACTION")
            
            # Get transcripts to remove
            if transcript_ids:
                transcripts = transcript_ids
            else:
                cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                cursor = self.storage.conn.execute("""
                SELECT DISTINCT transcript_id 
                FROM segments 
                WHERE created_at < ?
                """, (cutoff,))
                transcripts = [row[0] for row in cursor.fetchall()]
            
            for transcript_id in transcripts:
                # Get segments
                cursor = self.storage.conn.execute("""
                SELECT id FROM segments
                WHERE transcript_id = ?
                """, (transcript_id,))
                
                segment_ids = [row[0] for row in cursor.fetchall()]
                
                # Remove embeddings
                for segment_id in segment_ids:
                    try:
                        self.embedder.original_collection.delete(ids=[segment_id])
                        self.embedder.enriched_collection.delete(ids=[f"enr_{segment_id}"])
                        results['embeddings_removed'] += 2
                    except:
                        pass
                
                # Remove from database
                self.storage.conn.execute("""
                DELETE FROM enriched_segments
                WHERE segment_id IN (
                    SELECT id FROM segments WHERE transcript_id = ?
                )
                """, (transcript_id,))
                
                self.storage.conn.execute("""
                DELETE FROM segment_relationships
                WHERE from_segment_id IN (
                    SELECT id FROM segments WHERE transcript_id = ?
                )
                OR to_segment_id IN (
                    SELECT id FROM segments WHERE transcript_id = ?
                )
                """, (transcript_id, transcript_id))
                
                self.storage.conn.execute("""
                DELETE FROM segments
                WHERE transcript_id = ?
                """, (transcript_id,))
                
                # Remove files
                transcript_path = self.storage.transcripts_path / f"{transcript_id}.json"
                if transcript_path.exists():
                    results['storage_freed'] += transcript_path.stat().st_size
                    transcript_path.unlink()
                
                for segment_id in segment_ids:
                    # Remove segment files
                    segment_path = self.storage.segments_path / f"{segment_id}.json"
                    if segment_path.exists():
                        results['storage_freed'] += segment_path.stat().st_size
                        segment_path.unlink()
                    
                    # Remove enriched files
                    enriched_path = self.storage.enriched_path / f"enr_{segment_id}.json"
                    if enriched_path.exists():
                        results['storage_freed'] += enriched_path.stat().st_size
                        enriched_path.unlink()
                
                results['transcripts_removed'] += 1
                results['segments_removed'] += len(segment_ids)
            
            # Commit transaction
            self.storage.conn.commit()
            
        except Exception as e:
            self.storage.conn.rollback()
            raise RuntimeError(f"Cleanup failed: {str(e)}")
        
        return results
    
    def get_integrity_report(self) -> Dict:
        """Check data integrity across all storage"""
        report = {
            'missing_files': [],
            'orphaned_segments': [],
            'mismatched_embeddings': [],
            'broken_relationships': []
        }
        
        # Check transcript files exist
        cursor = self.storage.conn.execute("""
        SELECT DISTINCT transcript_id FROM segments
        """)
        
        for (transcript_id,) in cursor.fetchall():
            transcript_path = self.storage.transcripts_path / f"{transcript_id}.json"
            if not transcript_path.exists():
                report['missing_files'].append(f"Transcript: {transcript_id}")
        
        # Check segments have files
        cursor = self.storage.conn.execute("SELECT id FROM segments")
        for (segment_id,) in cursor.fetchall():
            segment_path = self.storage.segments_path / f"{segment_id}.json"
            if not segment_path.exists():
                report['missing_files'].append(f"Segment: {segment_id}")
        
        # Check enriched segments have base segments
        cursor = self.storage.conn.execute("""
        SELECT id, segment_id FROM enriched_segments
        WHERE segment_id NOT IN (SELECT id FROM segments)
        """)
        
        for enriched_id, segment_id in cursor.fetchall():
            report['orphaned_segments'].append(
                f"Enriched {enriched_id} -> Missing {segment_id}"
            )
        
        # Check embeddings match segments
        original_ids = set(self.embedder.original_collection.get()['ids'])
        enriched_ids = set(self.embedder.enriched_collection.get()['ids'])
        
        cursor = self.storage.conn.execute("SELECT id FROM segments")
        for (segment_id,) in cursor.fetchall():
            if segment_id not in original_ids:
                report['mismatched_embeddings'].append(
                    f"Missing original embedding: {segment_id}"
                )
            
            enriched_id = f"enr_{segment_id}"
            if enriched_id not in enriched_ids:
                report['mismatched_embeddings'].append(
                    f"Missing enriched embedding: {enriched_id}"
                )
        
        # Check relationship integrity
        cursor = self.storage.conn.execute("""
        SELECT from_segment_id, to_segment_id 
        FROM segment_relationships
        WHERE from_segment_id NOT IN (SELECT id FROM segments)
        OR to_segment_id NOT IN (SELECT id FROM segments)
        """)
        
        for from_id, to_id in cursor.fetchall():
            report['broken_relationships'].append(
                f"Relationship: {from_id} -> {to_id}"
            )
        
        return report
    
    def vacuum_storage(self) -> Dict:
        """Optimize storage and free space"""
        initial_sizes = self._get_storage_stats()
        
        # Vacuum SQLite database
        self.storage.conn.execute("VACUUM")
        
        # Optimize ChromaDB collections
        self.embedder.original_collection.persist()
        self.embedder.enriched_collection.persist()
        
        # Get final sizes
        final_sizes = self._get_storage_stats()
        
        return {
            'initial_size': initial_sizes,
            'final_size': final_sizes,
            'space_saved': {
                k: initial_sizes[k] - final_sizes[k]
                for k in initial_sizes.keys()
            }
        }

def initialize_session_state():
    """Initialize session state variables"""
    if 'transcript_uploader' not in st.session_state:
        st.session_state.transcript_uploader = TranscriptUploader()
    if 'segment_processor' not in st.session_state:
        st.session_state.segment_processor = SegmentProcessor()
    if 'claude_enricher' not in st.session_state:
        st.session_state.claude_enricher = ClaudeEnricher()
    if 'chroma_embedder' not in st.session_state:
        st.session_state.chroma_embedder = ChromaEmbedder()
    if 'storage_system' not in st.session_state:
        st.session_state.storage_system = StorageSystem()
    if 'query_interface' not in st.session_state:
        st.session_state.query_interface = QueryInterface(st.session_state.chroma_embedder)
    if 'response_generator' not in st.session_state:
        st.session_state.response_generator = ResponseGenerator(
            anthropic.Client(api_key=st.secrets["ANTHROPIC_API_KEY"]),
            st.session_state.storage_system
        )
    if 'response_storage' not in st.session_state:
        st.session_state.response_storage = ResponseStorage(st.session_state.storage_system)
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager(
            st.session_state.storage_system,
            st.session_state.chroma_embedder
        )
