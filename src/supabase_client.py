"""Supabase client for storing OCR results and user data."""
from supabase import create_client, Client
import os
from typing import Optional, Dict, Any
from datetime import datetime

class SupabaseManager:
    """Manager for Supabase database operations."""
    
    def __init__(self):
        url = os.getenv("SUPABASE_URL", "https://gxhuencnzpaabiijvwzz.supabase.co")
        key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd4aHVlbmNuenBhYWJpaWp2d3p6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMxMjA0MjcsImV4cCI6MjA3ODY5NjQyN30.9dbx-r-HdX0WPtj8BqY1_No7kxd0pFWL9f9uF4WcJXE")
        self.client: Client = create_client(url, key)
    
    async def save_ocr_result(
        self,
        filename: str,
        text: str,
        stats: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save OCR processing result to Supabase."""
        try:
            data = {
                "filename": filename,
                "text": text,
                "pages": stats.get("pages", 0),
                "columns": stats.get("columns", 0),
                "characters": stats.get("characters", 0),
                "processing_time": stats.get("time", 0),
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            result = self.client.table("ocr_results").insert(data).execute()
            return {"success": True, "id": result.data[0]["id"] if result.data else None}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_user_history(self, user_id: str, limit: int = 10) -> list:
        """Retrieve user's OCR processing history."""
        try:
            result = self.client.table("ocr_results")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            return result.data
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []
    
    async def get_ocr_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific OCR result by ID."""
        try:
            result = self.client.table("ocr_results")\
                .select("*")\
                .eq("id", result_id)\
                .single()\
                .execute()
            return result.data
        except Exception as e:
            print(f"Error retrieving result: {e}")
            return None
    
    async def delete_ocr_result(self, result_id: str, user_id: str) -> bool:
        """Delete an OCR result (with user ownership check)."""
        try:
            self.client.table("ocr_results")\
                .delete()\
                .eq("id", result_id)\
                .eq("user_id", user_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error deleting result: {e}")
            return False
    
    async def update_usage_stats(self, user_id: str, pages_processed: int):
        """Update user's processing statistics."""
        try:
            # Check if user stats exist
            result = self.client.table("user_stats")\
                .select("*")\
                .eq("user_id", user_id)\
                .execute()
            
            if result.data:
                # Update existing stats
                current_pages = result.data[0].get("total_pages", 0)
                current_docs = result.data[0].get("total_documents", 0)
                
                self.client.table("user_stats")\
                    .update({
                        "total_pages": current_pages + pages_processed,
                        "total_documents": current_docs + 1,
                        "last_processed": datetime.utcnow().isoformat()
                    })\
                    .eq("user_id", user_id)\
                    .execute()
            else:
                # Create new stats
                self.client.table("user_stats")\
                    .insert({
                        "user_id": user_id,
                        "total_pages": pages_processed,
                        "total_documents": 1,
                        "last_processed": datetime.utcnow().isoformat()
                    })\
                    .execute()
            return True
        except Exception as e:
            print(f"Error updating stats: {e}")
            return False
    
    async def get_usage_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's processing statistics."""
        try:
            result = self.client.table("user_stats")\
                .select("*")\
                .eq("user_id", user_id)\
                .single()\
                .execute()
            return result.data
        except Exception as e:
            print(f"Error retrieving stats: {e}")
            return None
