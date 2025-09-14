"""
Resource management for the Video Annotation Tool.
Handles file cleanup, memory management, and upload optimizations.
"""

import os
import asyncio
import tempfile
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import psutil
import sqlite3
from contextlib import contextmanager

from config import settings, logger


class ResourceManager:
    """Manages system resources, file cleanup, and memory optimization."""
    
    def __init__(self):
        self.temp_dir = Path(settings.temp_dir) if settings.temp_dir else Path(tempfile.gettempdir()) / "video_annotator"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Track temporary files and their expiration
        self._temp_files: Dict[str, float] = {}
        
        # Upload progress tracking
        self._upload_progress: Dict[str, Dict] = {}
        
        # Memory usage tracking
        self._memory_threshold = 0.85  # 85% memory usage threshold
        
    async def cleanup_expired_files(self, max_age_hours: Optional[int] = None) -> int:
        """Clean up expired temporary files and old uploads."""
        if max_age_hours is None:
            max_age_hours = settings.cleanup_interval_hours
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        # Clean temporary files
        try:
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                    try:
                        temp_file.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned expired temp file: {temp_file}")
                    except OSError as e:
                        logger.warning(f"Could not delete temp file {temp_file}: {e}")
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
        
        # Clean upload progress tracking for old uploads
        current_time = time.time()
        expired_uploads = [
            upload_id for upload_id, info in self._upload_progress.items()
            if current_time - info.get('start_time', 0) > 3600  # 1 hour
        ]
        
        for upload_id in expired_uploads:
            del self._upload_progress[upload_id]
            
        return cleaned_count
    
    async def get_disk_usage(self) -> Dict[str, Dict]:
        """Get disk usage information for all configured directories."""
        disk_info = {}
        
        directories = [
            ("uploads", settings.upload_dir),
            ("static", settings.static_dir),
            ("temp", str(self.temp_dir)),
        ]
        
        if settings.log_file:
            log_dir = Path(settings.log_file).parent
            directories.append(("logs", str(log_dir)))
        
        for name, directory in directories:
            try:
                path = Path(directory)
                if path.exists():
                    # Get directory size
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    
                    # Get disk usage for the partition
                    disk_usage = shutil.disk_usage(directory)
                    
                    disk_info[name] = {
                        "path": str(path.absolute()),
                        "size_bytes": size,
                        "size_mb": size / (1024 * 1024),
                        "disk_total": disk_usage.total,
                        "disk_used": disk_usage.used,
                        "disk_free": disk_usage.free,
                        "disk_usage_percent": (disk_usage.used / disk_usage.total) * 100
                    }
                else:
                    disk_info[name] = {"path": directory, "exists": False}
            except Exception as e:
                logger.error(f"Error getting disk info for {directory}: {e}")
                disk_info[name] = {"path": directory, "error": str(e)}
        
        return disk_info
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percentage": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percentage": swap.percent,
                "is_high": memory.percent > (self._memory_threshold * 100)
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    def calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate SHA-256 hash of a file for duplicate detection."""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    async def detect_duplicate_files(self) -> List[Dict]:
        """Detect duplicate files in the upload directory."""
        upload_dir = Path(settings.upload_dir)
        if not upload_dir.exists():
            return []
        
        file_hashes = {}
        duplicates = []
        
        try:
            for file_path in upload_dir.glob("*"):
                if file_path.is_file():
                    file_hash = self.calculate_file_hash(str(file_path))
                    if file_hash:
                        if file_hash in file_hashes:
                            duplicates.append({
                                "hash": file_hash,
                                "original": file_hashes[file_hash],
                                "duplicate": str(file_path),
                                "size": file_path.stat().st_size
                            })
                        else:
                            file_hashes[file_hash] = str(file_path)
        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}")
        
        return duplicates
    
    async def optimize_upload_memory(self, content: bytes) -> Tuple[str, str]:
        """Optimize memory usage during upload by using temporary files for large uploads."""
        temp_file_path = None
        
        try:
            # If content is large, write to temporary file to free memory
            if len(content) > 50 * 1024 * 1024:  # 50MB threshold
                temp_file = tempfile.NamedTemporaryFile(
                    dir=self.temp_dir,
                    delete=False,
                    suffix=".tmp"
                )
                
                # Write in chunks to control memory usage
                chunk_size = 1024 * 1024  # 1MB chunks
                for i in range(0, len(content), chunk_size):
                    temp_file.write(content[i:i + chunk_size])
                
                temp_file_path = temp_file.name
                temp_file.close()
                
                # Calculate hash from file
                file_hash = self.calculate_file_hash(temp_file_path)
                
                logger.info(f"Large upload optimized using temp file: {temp_file_path}")
                
                return temp_file_path, file_hash
            else:
                # For smaller files, calculate hash directly from memory
                file_hash = hashlib.sha256(content).hexdigest()
                return None, file_hash
                
        except Exception as e:
            # Clean up temp file on error
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
            logger.error(f"Error during upload optimization: {e}")
            raise
    
    def track_upload_progress(self, upload_id: str, filename: str, total_size: int):
        """Track upload progress for monitoring."""
        self._upload_progress[upload_id] = {
            "filename": filename,
            "total_size": total_size,
            "start_time": time.time(),
            "status": "uploading"
        }
    
    def update_upload_progress(self, upload_id: str, bytes_processed: int):
        """Update upload progress."""
        if upload_id in self._upload_progress:
            info = self._upload_progress[upload_id]
            info["bytes_processed"] = bytes_processed
            info["progress_percent"] = (bytes_processed / info["total_size"]) * 100
            info["last_update"] = time.time()
    
    def complete_upload(self, upload_id: str, success: bool = True):
        """Mark upload as completed."""
        if upload_id in self._upload_progress:
            info = self._upload_progress[upload_id]
            info["status"] = "completed" if success else "failed"
            info["end_time"] = time.time()
            info["duration"] = info["end_time"] - info["start_time"]
    
    def get_upload_progress(self, upload_id: str) -> Optional[Dict]:
        """Get progress information for an upload."""
        return self._upload_progress.get(upload_id)
    
    def get_all_upload_progress(self) -> Dict:
        """Get progress information for all active uploads."""
        return dict(self._upload_progress)
    
    async def emergency_cleanup(self) -> Dict:
        """Perform emergency cleanup when system resources are low."""
        cleanup_results = {
            "temp_files_removed": 0,
            "memory_freed": 0,
            "actions_taken": []
        }
        
        try:
            # Check memory usage
            memory_info = self.get_memory_usage()
            
            if memory_info.get("is_high", False):
                logger.warning("High memory usage detected, performing emergency cleanup")
                cleanup_results["actions_taken"].append("high_memory_cleanup")
                
                # Clean all temporary files regardless of age
                temp_count = 0
                for temp_file in self.temp_dir.glob("*"):
                    if temp_file.is_file():
                        try:
                            size = temp_file.stat().st_size
                            temp_file.unlink()
                            temp_count += 1
                            cleanup_results["memory_freed"] += size
                        except OSError:
                            pass
                
                cleanup_results["temp_files_removed"] = temp_count
                
                # Clear upload progress tracking for completed uploads
                completed_uploads = [
                    upload_id for upload_id, info in self._upload_progress.items()
                    if info.get("status") in ["completed", "failed"]
                ]
                
                for upload_id in completed_uploads:
                    del self._upload_progress[upload_id]
                
                cleanup_results["actions_taken"].append(f"cleared_{len(completed_uploads)}_upload_records")
            
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {e}")
            cleanup_results["error"] = str(e)
        
        return cleanup_results
    
    async def get_resource_stats(self) -> Dict:
        """Get comprehensive resource statistics."""
        return {
            "memory": self.get_memory_usage(),
            "disk": await self.get_disk_usage(),
            "temp_files_count": len(list(self.temp_dir.glob("*"))),
            "active_uploads": len(self._upload_progress),
            "upload_progress": self.get_all_upload_progress()
        }


# Global resource manager instance
resource_manager = ResourceManager()


# Background task for periodic cleanup
async def periodic_cleanup_task():
    """Background task for periodic resource cleanup."""
    while True:
        try:
            # Wait for cleanup interval
            await asyncio.sleep(settings.cleanup_interval_hours * 3600)
            
            logger.info("Starting periodic cleanup")
            
            # Clean expired files
            cleaned_count = await resource_manager.cleanup_expired_files()
            logger.info(f"Periodic cleanup completed: {cleaned_count} files removed")
            
            # Check memory usage and perform emergency cleanup if needed
            memory_info = resource_manager.get_memory_usage()
            if memory_info.get("is_high", False):
                emergency_results = await resource_manager.emergency_cleanup()
                logger.warning(f"Emergency cleanup performed: {emergency_results}")
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")


def start_background_tasks(app):
    """Start background cleanup tasks."""
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting background resource management tasks")
        asyncio.create_task(periodic_cleanup_task())
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Performing final cleanup before shutdown")
        await resource_manager.cleanup_expired_files(max_age_hours=0)  # Clean all temp files