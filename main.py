from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import os
import json
import uuid
import shutil
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import sqlite3
from contextlib import contextmanager
import logging
from pathlib import Path
import base64
import io

# Try to import PIL with fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    print("Warning: PIL/Pillow not available - thumbnails will be disabled")

# Import configuration and resource management
from config import settings, logger
from resource_manager import resource_manager, start_background_tasks

# Thumbnail generation utilities
def generate_video_thumbnail(video_path: str, timestamp: float = 1.0, size: tuple = (200, 150)) -> str:
    """Generate a thumbnail from video at specified timestamp and return as base64"""
    if not PIL_AVAILABLE:
        logger.warning("PIL not available - cannot generate thumbnails")
        return None
        
    try:
        logger.info(f"Generating thumbnail for {video_path} at {timestamp}s")
        
        # Check if file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None
            
        # Get video properties for better frame selection
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Ensure timestamp is within video duration
        if duration > 0:
            timestamp = min(timestamp, duration - 0.1)  # Stay within bounds
            
        # Set the position to the desired timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        
        ret, frame = cap.read()
        if not ret:
            # Try a different timestamp if the first fails
            logger.warning(f"Could not read frame at {timestamp}s, trying 10% into video")
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, int(frame_count * 0.1)))
            ret, frame = cap.read()
            
        cap.release()
        
        if not ret or frame is None:
            logger.error(f"Could not read any frame from video: {video_path}")
            return None
            
        # Ensure frame has valid dimensions
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.error(f"Invalid frame dimensions: {frame.shape}")
            return None
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create PIL Image and resize
        pil_image = Image.fromarray(frame_rgb)
        
        # Calculate proper aspect ratio
        original_width, original_height = pil_image.size
        target_width, target_height = size
        
        # Calculate scaling to maintain aspect ratio
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_ratio = min(width_ratio, height_ratio)
        
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info(f"Successfully generated thumbnail for {video_path} ({len(img_base64)} bytes)")
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error generating thumbnail for {video_path}: {e}")
        import traceback
        logger.error(f"Thumbnail generation traceback: {traceback.format_exc()}")
        return None

def generate_multiple_thumbnails(video_path: str, count: int = 3, size: tuple = (150, 100)) -> List[str]:
    """Generate multiple thumbnails from different parts of the video"""
    if not PIL_AVAILABLE:
        return []
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        # Get video duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        if duration <= 0:
            return []
            
        # Calculate timestamps for thumbnails
        thumbnails = []
        for i in range(count):
            # Skip very beginning and end, distribute evenly
            timestamp = (duration * (i + 1)) / (count + 1)
            thumbnail = generate_video_thumbnail(video_path, timestamp, size)
            if thumbnail:
                thumbnails.append(thumbnail)
                
        return thumbnails
        
    except Exception as e:
        logger.error(f"Error generating multiple thumbnails for {video_path}: {e}")
        return []

# Annotation interpolation and tracking utilities
def interpolate_bbox(bbox1: dict, bbox2: dict, frame1: int, frame2: int, target_frame: int) -> dict:
    """Linear interpolation between two bounding boxes"""
    if frame1 == frame2:
        return bbox1.copy()
        
    # Calculate interpolation ratio
    ratio = (target_frame - frame1) / (frame2 - frame1)
    ratio = max(0, min(1, ratio))  # Clamp between 0 and 1
    
    # Interpolate each coordinate
    interpolated = {
        'x': bbox1['x'] + (bbox2['x'] - bbox1['x']) * ratio,
        'y': bbox1['y'] + (bbox2['y'] - bbox1['y']) * ratio,
        'width': bbox1['width'] + (bbox2['width'] - bbox1['width']) * ratio,
        'height': bbox1['height'] + (bbox2['height'] - bbox1['height']) * ratio
    }
    
    return interpolated

def interpolate_polygon(poly1: dict, poly2: dict, frame1: int, frame2: int, target_frame: int) -> dict:
    """Linear interpolation between two polygons"""
    if frame1 == frame2:
        return poly1.copy()
    
    points1 = poly1.get('points', [])
    points2 = poly2.get('points', [])
    
    # Only interpolate if both polygons have the same number of points
    if len(points1) != len(points2):
        return poly1.copy()  # Return first polygon if they don't match
    
    ratio = (target_frame - frame1) / (frame2 - frame1)
    ratio = max(0, min(1, ratio))
    
    interpolated_points = []
    for p1, p2 in zip(points1, points2):
        interpolated_points.append({
            'x': p1['x'] + (p2['x'] - p1['x']) * ratio,
            'y': p1['y'] + (p2['y'] - p1['y']) * ratio
        })
    
    return {'points': interpolated_points}

# Configuration from settings
MAX_FILE_SIZE = settings.max_file_size
ALLOWED_VIDEO_EXTENSIONS = set(settings.supported_formats)
MAX_FILENAME_LENGTH = settings.max_filename_length
MAX_CLASS_NAME_LENGTH = settings.max_class_name_length
MAX_FRAME_NUMBER = settings.max_frame_count

# Pydantic Models
class AnnotationCreate(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=36)
    frame_number: int = Field(..., ge=0, le=MAX_FRAME_NUMBER)
    annotation_type: str = Field(..., pattern='^(bbox|polygon)$')
    class_name: str = Field(..., min_length=1, max_length=MAX_CLASS_NAME_LENGTH)
    geometry: dict
    track_id: Optional[int] = Field(None, ge=1)
    
    @validator('class_name')
    def validate_class_name(cls, v):
        if not v.strip():
            raise ValueError('Class name cannot be empty or whitespace')
        return v.strip().lower()

class AnnotationUpdate(BaseModel):
    geometry: dict
    class_name: Optional[str] = Field(None, min_length=1, max_length=MAX_CLASS_NAME_LENGTH)
    
    @validator('class_name')
    def validate_class_name(cls, v):
        if v and not v.strip():
            raise ValueError('Class name cannot be empty or whitespace')
        return v.strip().lower() if v else None

class InterpolationRequest(BaseModel):
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)
    track_id: Optional[int] = Field(None, ge=1)
    interpolation_type: str = Field(default="linear", pattern='^(linear|bezier|auto)$')
    
class TrackingRequest(BaseModel):
    annotation_id: str
    target_frames: List[int] = Field(..., min_items=1)
    tracking_method: str = Field(default="optical_flow", pattern='^(optical_flow|template_matching|deep_learning)$')

class VideoInfo(BaseModel):
    id: str
    filename: str
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    upload_date: str

# Database setup
def init_database():
    """Initialize the database with tables and indexes"""
    try:
        conn = sqlite3.connect('annotations.db')
        cursor = conn.cursor()
        
        # Enable foreign key support
        cursor.execute('PRAGMA foreign_keys = ON')
        
        # Videos table with improved schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                duration REAL NOT NULL CHECK(duration >= 0),
                fps REAL NOT NULL CHECK(fps > 0),
                width INTEGER NOT NULL CHECK(width > 0),
                height INTEGER NOT NULL CHECK(height > 0),
                frame_count INTEGER NOT NULL CHECK(frame_count > 0),
                upload_date TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                thumbnail TEXT,
                preview_thumbnails TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Annotations table with improved schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL CHECK(frame_number >= 0),
                annotation_type TEXT NOT NULL CHECK(annotation_type IN ('bbox', 'polygon')),
                class_name TEXT NOT NULL,
                geometry TEXT NOT NULL,
                track_id INTEGER CHECK(track_id > 0),
                created_at TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for better performance
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_videos_filename ON videos(filename)',
            'CREATE INDEX IF NOT EXISTS idx_videos_upload_date ON videos(upload_date DESC)',
            'CREATE INDEX IF NOT EXISTS idx_annotations_video_id ON annotations(video_id)',
            'CREATE INDEX IF NOT EXISTS idx_annotations_frame_number ON annotations(frame_number)',
            'CREATE INDEX IF NOT EXISTS idx_annotations_video_frame ON annotations(video_id, frame_number)',
            'CREATE INDEX IF NOT EXISTS idx_annotations_class_name ON annotations(class_name)',
            'CREATE INDEX IF NOT EXISTS idx_annotations_track_id ON annotations(track_id)',
            'CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(annotation_type)',
            'CREATE INDEX IF NOT EXISTS idx_annotations_created_at ON annotations(created_at DESC)'
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        # Create a table for application metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert or update schema version
        cursor.execute('''
            INSERT OR REPLACE INTO app_metadata (key, value, updated_at)
            VALUES ('schema_version', '1.1', CURRENT_TIMESTAMP)
        ''')
        
        # Add triggers to update timestamps
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_videos_updated_at
            AFTER UPDATE ON videos
            BEGIN
                UPDATE videos SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_annotations_updated_at
            AFTER UPDATE ON annotations
            BEGIN
                UPDATE annotations SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        ''')
        
        # Add thumbnail columns if they don't exist (migration)
        cursor.execute("PRAGMA table_info(videos)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'thumbnail' not in columns:
            cursor.execute('ALTER TABLE videos ADD COLUMN thumbnail TEXT')
            logger.info("Added thumbnail column to videos table")
            
        if 'preview_thumbnails' not in columns:
            cursor.execute('ALTER TABLE videos ADD COLUMN preview_thumbnails TEXT')
            logger.info("Added preview_thumbnails column to videos table")
        
        conn.commit()
        logger.info("Database initialized successfully with indexes and triggers")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        conn.close()

@contextmanager
def get_db():
    """Database connection context manager with proper configuration"""
    # Extract database path from URL (simple SQLite URL parsing)
    db_path = settings.database_url.replace('sqlite:///', '').replace('sqlite:', '')
    
    conn = sqlite3.connect(
        db_path,
        timeout=settings.database_timeout,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA journal_mode = WAL')  # Better concurrency
    conn.execute('PRAGMA synchronous = NORMAL')  # Better performance
    conn.execute(f'PRAGMA cache_size = -{settings.cache_size_bytes // 1024}')  # Cache from settings
    
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        conn.close()

# Database utility functions
def get_database_stats():
    """Get database statistics"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get video count and total size
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM videos")
            video_count, total_size = cursor.fetchone()
            
            # Get annotation count
            cursor.execute("SELECT COUNT(*) FROM annotations")
            annotation_count = cursor.fetchone()[0]
            
            # Get unique class names
            cursor.execute("SELECT COUNT(DISTINCT class_name) FROM annotations")
            unique_classes = cursor.fetchone()[0]
            
            # Get schema version
            cursor.execute("SELECT value FROM app_metadata WHERE key = 'schema_version'")
            schema_version = cursor.fetchone()
            schema_version = schema_version[0] if schema_version else "1.0"
            
            return {
                "video_count": video_count,
                "total_file_size": total_size,
                "annotation_count": annotation_count,
                "unique_classes": unique_classes,
                "schema_version": schema_version
            }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return None

def cleanup_orphaned_files():
    """Clean up files that exist on disk but not in database"""
    try:
        uploads_dir = Path(settings.upload_dir)
        if not uploads_dir.exists():
            return 0
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM videos")
            db_files = {row[0] for row in cursor.fetchall()}
        
        disk_files = list(uploads_dir.glob("*"))
        orphaned_count = 0
        
        for file_path in disk_files:
            if str(file_path) not in db_files:
                try:
                    file_path.unlink()
                    orphaned_count += 1
                    logger.info(f"Removed orphaned file: {file_path}")
                except OSError as e:
                    logger.error(f"Failed to remove orphaned file {file_path}: {e}")
        
        return orphaned_count
    except Exception as e:
        logger.error(f"Failed to cleanup orphaned files: {e}")
        return 0

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name, 
    version=settings.app_version,
    debug=settings.debug
)

# CORS middleware - configurable origins
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

# Create directories from settings
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.static_dir, exist_ok=True)
if settings.temp_dir:
    os.makedirs(settings.temp_dir, exist_ok=True)

# Initialize database
init_database()

# Start background resource management tasks
start_background_tasks(app)

# Serve static files from configured directories
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main application page"""
    return get_html_content()

@app.post("/api/upload-video", response_model=VideoInfo)
async def upload_video(file: UploadFile = File(...)):
    """Upload and process a video file"""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check filename length
    if len(file.filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Filename too long")
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported video format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Check file size first before reading content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    # Generate upload ID for tracking
    upload_id = str(uuid.uuid4())
    resource_manager.track_upload_progress(upload_id, file.filename, len(content))
    
    # Generate unique ID and sanitize filename
    video_id = str(uuid.uuid4())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in '.-_').rstrip()
    file_path = f"{settings.upload_dir}/{video_id}_{safe_filename}"
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    try:
        # Optimize memory usage for large files
        temp_file_path, file_hash = await resource_manager.optimize_upload_memory(content)
        
        if temp_file_path:
            # Move temp file to final location
            shutil.move(temp_file_path, file_path)
            logger.info(f"Large file uploaded via temp file: {safe_filename} ({len(content)} bytes)")
        else:
            # Write directly for smaller files
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            logger.info(f"File uploaded: {safe_filename} ({len(content)} bytes)")
        
        # Update upload progress
        resource_manager.update_upload_progress(upload_id, len(content))
        
    except Exception as e:
        resource_manager.complete_upload(upload_id, success=False)
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Extract video metadata using OpenCV
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        # Clean up file on failure
        try:
            os.remove(file_path)
        except OSError:
            pass
        logger.error(f"Could not open video file: {file_path}")
        raise HTTPException(status_code=400, detail="Invalid video file or corrupted")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    # Validate video properties
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        try:
            os.remove(file_path)
        except OSError:
            pass
        logger.error(f"Invalid video properties: fps={fps}, frames={frame_count}, size={width}x{height}")
        raise HTTPException(status_code=400, detail="Invalid video properties")
    
    # Check reasonable video limits
    if frame_count > MAX_FRAME_NUMBER:
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=f"Video too long (max {MAX_FRAME_NUMBER} frames)")
    
    if width > settings.max_video_width or height > settings.max_video_height:
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=400, 
            detail=f"Video resolution too high (max {settings.max_video_width}x{settings.max_video_height})"
        )
    
    # Generate thumbnails (with error handling to not block upload)
    logger.info(f"Generating thumbnails for video: {safe_filename}")
    thumbnail = None
    preview_thumbnails_json = None
    
    try:
        thumbnail = generate_video_thumbnail(file_path, timestamp=1.0, size=(200, 150))
        preview_thumbnails = generate_multiple_thumbnails(file_path, count=3, size=(150, 100))
        preview_thumbnails_json = json.dumps(preview_thumbnails) if preview_thumbnails else None
        logger.info(f"Thumbnails generated successfully for {safe_filename}")
    except Exception as e:
        logger.warning(f"Failed to generate thumbnails for {safe_filename}: {e}")
        # Don't fail the upload if thumbnail generation fails
    
    # Save to database
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO videos (id, filename, file_path, duration, fps, width, height, frame_count, upload_date, file_size, thumbnail, preview_thumbnails)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (video_id, safe_filename, file_path, duration, fps, width, height, frame_count, datetime.now().isoformat(), len(content), thumbnail, preview_thumbnails_json))
            conn.commit()
            logger.info(f"Video metadata and thumbnails saved: {video_id} ({len(content)} bytes)")
            
            # Mark upload as successful
            resource_manager.complete_upload(upload_id, success=True)
            
    except sqlite3.IntegrityError as e:
        resource_manager.complete_upload(upload_id, success=False)
        # Clean up file on database error (e.g., duplicate file path)
        try:
            os.remove(file_path)
        except OSError:
            pass
        logger.error(f"Database integrity error: {e}")
        raise HTTPException(status_code=409, detail="Video with this path already exists")
    except Exception as e:
        # Mark upload as failed and clean up file
        resource_manager.complete_upload(upload_id, success=False)
        try:
            os.remove(file_path)
        except OSError:
            pass
        logger.error(f"Database error while saving video: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video metadata")
    
    return VideoInfo(
        id=video_id,
        filename=safe_filename,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        upload_date=datetime.now().isoformat()
    )

@app.get("/api/videos")
async def get_videos():
    """Get all uploaded videos with improved query performance"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Simple query first - get core video info
            cursor.execute("""
                SELECT id, filename, duration, fps, width, height, frame_count, upload_date,
                       COALESCE(file_size, 0) as file_size,
                       thumbnail,
                       COALESCE(preview_thumbnails, '[]') as preview_thumbnails
                FROM videos 
                ORDER BY upload_date DESC
            """)
            videos = cursor.fetchall()
            
            result = []
            for video in videos:
                # Handle preview_thumbnails parsing safely
                preview_thumbnails = []
                preview_data = video['preview_thumbnails']
                if preview_data and preview_data != '[]':
                    try:
                        preview_thumbnails = json.loads(preview_data)
                        if not isinstance(preview_thumbnails, list):
                            preview_thumbnails = []
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse preview_thumbnails for video {video['id']}: {e}")
                        preview_thumbnails = []
                
                video_info = {
                    "id": video['id'],
                    "filename": video['filename'],
                    "duration": float(video['duration']),
                    "fps": float(video['fps']),
                    "width": int(video['width']),
                    "height": int(video['height']),
                    "frame_count": int(video['frame_count']),
                    "upload_date": video['upload_date'],
                    "file_size": int(video['file_size']),
                    "thumbnail": video['thumbnail'],
                    "preview_thumbnails": preview_thumbnails
                }
                result.append(video_info)
                
            logger.info(f"Returning {len(result)} videos")
            return result
            
    except Exception as e:
        logger.error(f"Failed to retrieve videos: {e}")
        import traceback
        logger.error(f"Videos retrieval traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve videos: {str(e)}")

@app.get("/api/videos/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: str, timestamp: float = 1.0):
    """Generate a thumbnail for a specific video at a given timestamp"""
    # Validate video_id format
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
            video = cursor.fetchone()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            file_path = video['file_path']
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Video file not found")
            
            thumbnail = generate_video_thumbnail(file_path, timestamp)
            if not thumbnail:
                raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
            
            return {"thumbnail": thumbnail}
            
    except Exception as e:
        logger.error(f"Error generating thumbnail for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

@app.post("/api/videos/{video_id}/regenerate-thumbnails")
async def regenerate_video_thumbnails(video_id: str):
    """Regenerate thumbnails for an existing video"""
    # Validate video_id format
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
            video = cursor.fetchone()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            file_path = video['file_path']
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Video file not found")
            
            # Generate new thumbnails
            thumbnail = generate_video_thumbnail(file_path, timestamp=1.0, size=(200, 150))
            preview_thumbnails = generate_multiple_thumbnails(file_path, count=3, size=(150, 100))
            preview_thumbnails_json = json.dumps(preview_thumbnails) if preview_thumbnails else None
            
            # Update database
            cursor.execute("""
                UPDATE videos 
                SET thumbnail = ?, preview_thumbnails = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (thumbnail, preview_thumbnails_json, video_id))
            
            conn.commit()
            
            logger.info(f"Regenerated thumbnails for video {video_id}")
            
            return {
                "message": "Thumbnails regenerated successfully",
                "thumbnail": thumbnail,
                "preview_count": len(preview_thumbnails) if preview_thumbnails else 0
            }
            
    except Exception as e:
        logger.error(f"Error regenerating thumbnails for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to regenerate thumbnails")

@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all associated annotations and files"""
    # Validate video_id format
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get video info before deletion to access file path
            cursor.execute("SELECT file_path, filename FROM videos WHERE id = ?", (video_id,))
            video = cursor.fetchone()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            file_path = video['file_path']
            filename = video['filename']
            
            # Delete associated annotations first (CASCADE should handle this, but let's be explicit)
            cursor.execute("DELETE FROM annotations WHERE video_id = ?", (video_id,))
            deleted_annotations = cursor.rowcount
            
            # Delete the video record
            cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Video not found")
            
            conn.commit()
            
            # Delete the physical file
            file_deleted = False
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    file_deleted = True
                    logger.info(f"Deleted video file: {file_path}")
                else:
                    logger.warning(f"Video file not found for deletion: {file_path}")
            except OSError as e:
                logger.error(f"Failed to delete video file {file_path}: {e}")
                # Don't raise an exception here as the database record is already deleted
            
            logger.info(f"Video deleted: {video_id} ({filename}) with {deleted_annotations} annotations")
            
            return {
                "status": "deleted",
                "video_id": video_id,
                "filename": filename,
                "annotations_deleted": deleted_annotations,
                "file_deleted": file_deleted,
                "message": f"Video '{filename}' and {deleted_annotations} annotations deleted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete video")

@app.post("/api/annotations")
async def create_annotation(annotation: AnnotationCreate):
    """Create a new annotation"""
    annotation_id = str(uuid.uuid4())
    
    # Validate that the video exists
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM videos WHERE id = ?", (annotation.video_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Video not found")
            
            # Validate geometry based on annotation type
            if annotation.annotation_type == 'bbox':
                geometry = annotation.geometry
                required_keys = {'x', 'y', 'width', 'height'}
                if not all(key in geometry for key in required_keys):
                    raise HTTPException(status_code=400, detail="Bounding box must have x, y, width, height")
                
                # Validate numeric values
                for key in required_keys:
                    if not isinstance(geometry[key], (int, float)) or geometry[key] < 0:
                        raise HTTPException(status_code=400, detail=f"Invalid {key} value")
                        
            elif annotation.annotation_type == 'polygon':
                geometry = annotation.geometry
                if 'points' not in geometry or not isinstance(geometry['points'], list):
                    raise HTTPException(status_code=400, detail="Polygon must have points array")
                
                if len(geometry['points']) < 3:
                    raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")
                
                for i, point in enumerate(geometry['points']):
                    if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
                        raise HTTPException(status_code=400, detail=f"Invalid point {i} format")
                    if not isinstance(point['x'], (int, float)) or not isinstance(point['y'], (int, float)):
                        raise HTTPException(status_code=400, detail=f"Invalid coordinates in point {i}")
            
            cursor.execute('''
                INSERT INTO annotations (id, video_id, frame_number, annotation_type, class_name, geometry, track_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                annotation_id,
                annotation.video_id,
                annotation.frame_number,
                annotation.annotation_type,
                annotation.class_name,
                json.dumps(annotation.geometry),
                annotation.track_id,
                datetime.now().isoformat()
            ))
            conn.commit()
            logger.info(f"Annotation created: {annotation_id} for video {annotation.video_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create annotation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create annotation")
    
    return {"id": annotation_id, "status": "created"}

@app.get("/api/annotations/{video_id}")
async def get_annotations(video_id: str, frame_number: Optional[int] = None):
    """Get annotations for a video, optionally filtered by frame"""
    # Validate video_id format
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    # Validate frame_number if provided
    if frame_number is not None and (frame_number < 0 or frame_number > MAX_FRAME_NUMBER):
        raise HTTPException(status_code=400, detail="Invalid frame number")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Verify video exists
            cursor.execute("SELECT id FROM videos WHERE id = ?", (video_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Video not found")
            
            if frame_number is not None:
                cursor.execute('''
                    SELECT * FROM annotations 
                    WHERE video_id = ? AND frame_number = ?
                ''', (video_id, frame_number))
            else:
                cursor.execute('''
                    SELECT * FROM annotations 
                    WHERE video_id = ?
                    ORDER BY frame_number
                ''', (video_id,))
            
            annotations = cursor.fetchall()
            
            result = []
            for ann in annotations:
                try:
                    geometry = json.loads(ann['geometry'])
                except json.JSONDecodeError:
                    logger.error(f"Invalid geometry JSON in annotation {ann['id']}")
                    continue
                    
                result.append({
                    "id": ann['id'],
                    "video_id": ann['video_id'],
                    "frame_number": ann['frame_number'],
                    "annotation_type": ann['annotation_type'],
                    "class_name": ann['class_name'],
                    "geometry": geometry,
                    "track_id": ann['track_id'],
                    "created_at": ann['created_at']
                })
            
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get annotations for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve annotations")

@app.delete("/api/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation"""
    # Validate annotation_id format
    try:
        uuid.UUID(annotation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid annotation ID format")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
            conn.commit()
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Annotation not found")
            
            logger.info(f"Annotation deleted: {annotation_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete annotation {annotation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete annotation")
    
    return {"status": "deleted"}

@app.post("/api/annotations/interpolate")
async def interpolate_annotations(request: InterpolationRequest):
    """Generate interpolated annotations between keyframes"""
    try:
        # Validate frame range
        if request.start_frame >= request.end_frame:
            raise HTTPException(status_code=400, detail="Start frame must be before end frame")
        
        if request.end_frame - request.start_frame < 2:
            raise HTTPException(status_code=400, detail="Need at least one frame between keyframes")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Find annotations with the same track_id on start and end frames
            cursor.execute("""
                SELECT a1.video_id, a1.annotation_type, a1.class_name, a1.geometry as start_geom,
                       a2.geometry as end_geom
                FROM annotations a1
                JOIN annotations a2 ON a1.video_id = a2.video_id 
                                    AND a1.track_id = a2.track_id
                                    AND a1.class_name = a2.class_name
                                    AND a1.annotation_type = a2.annotation_type
                WHERE a1.frame_number = ? AND a2.frame_number = ?
                  AND a1.track_id = ?
            """, (request.start_frame, request.end_frame, request.track_id))
            
            keyframe_data = cursor.fetchone()
            if not keyframe_data:
                raise HTTPException(
                    status_code=404, 
                    detail="Could not find matching annotations on both keyframes with the same track_id"
                )
            
            video_id = keyframe_data['video_id']
            annotation_type = keyframe_data['annotation_type']
            class_name = keyframe_data['class_name']
            start_geometry = json.loads(keyframe_data['start_geom'])
            end_geometry = json.loads(keyframe_data['end_geom'])
            
            # Generate interpolated annotations
            created_annotations = []
            
            for frame_num in range(request.start_frame + 1, request.end_frame):
                # Check if annotation already exists for this frame
                cursor.execute("""
                    SELECT id FROM annotations 
                    WHERE video_id = ? AND frame_number = ? AND track_id = ?
                """, (video_id, frame_num, request.track_id))
                
                if cursor.fetchone():
                    continue  # Skip if annotation already exists
                
                # Interpolate geometry
                if annotation_type == 'bbox':
                    interpolated_geometry = interpolate_bbox(
                        start_geometry, end_geometry, 
                        request.start_frame, request.end_frame, frame_num
                    )
                elif annotation_type == 'polygon':
                    interpolated_geometry = interpolate_polygon(
                        start_geometry, end_geometry,
                        request.start_frame, request.end_frame, frame_num
                    )
                else:
                    continue
                
                # Create the interpolated annotation
                annotation_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO annotations 
                    (id, video_id, frame_number, annotation_type, class_name, geometry, track_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    annotation_id, video_id, frame_num, annotation_type, class_name,
                    json.dumps(interpolated_geometry), request.track_id, datetime.now().isoformat()
                ))
                
                created_annotations.append({
                    'id': annotation_id,
                    'frame_number': frame_num,
                    'geometry': interpolated_geometry
                })
            
            conn.commit()
            logger.info(f"Created {len(created_annotations)} interpolated annotations for track {request.track_id}")
            
            return {
                'status': 'success',
                'interpolated_count': len(created_annotations),
                'start_frame': request.start_frame,
                'end_frame': request.end_frame,
                'track_id': request.track_id,
                'created_annotations': created_annotations
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interpolation failed: {str(e)}")

@app.get("/api/export/{video_id}")
async def export_annotations(video_id: str, format: str = "coco"):
    """Export annotations in various formats"""
    # Validate video_id format
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    # Validate format
    allowed_formats = {"coco", "yolo"}
    if format.lower() not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported export format. Allowed: {', '.join(allowed_formats)}"
        )
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get video info
            cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            video = cursor.fetchone()
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            # Get annotations
            cursor.execute("SELECT * FROM annotations WHERE video_id = ?", (video_id,))
            annotations = cursor.fetchall()
            
            logger.info(f"Exporting {len(annotations)} annotations for video {video_id} in {format} format")
            
            if format.lower() == "coco":
                return export_coco_format(video, annotations)
            elif format.lower() == "yolo":
                return export_yolo_format(video, annotations)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export annotations for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export annotations")

@app.get("/api/stats")
async def get_stats():
    """Get database and application statistics"""
    stats = get_database_stats()
    if stats is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
    
    return stats

@app.post("/api/cleanup")
async def cleanup_files():
    """Clean up orphaned files (admin endpoint)"""
    try:
        cleaned_count = cleanup_orphaned_files()
        logger.info(f"Cleanup completed: removed {cleaned_count} orphaned files")
        return {
            "status": "completed",
            "files_removed": cleaned_count,
            "message": f"Removed {cleaned_count} orphaned files"
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup operation failed")

@app.get("/api/videos/{video_id}/stats")
async def get_video_stats(video_id: str):
    """Get statistics for a specific video"""
    # Validate video_id format
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get video info with file stats
            cursor.execute("""
                SELECT id, filename, duration, fps, width, height, frame_count, 
                       upload_date, file_size, created_at, updated_at
                FROM videos WHERE id = ?
            """, (video_id,))
            video = cursor.fetchone()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            # Get annotation statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_annotations,
                    COUNT(DISTINCT frame_number) as annotated_frames,
                    COUNT(DISTINCT class_name) as unique_classes,
                    COUNT(DISTINCT track_id) as unique_tracks
                FROM annotations WHERE video_id = ?
            """, (video_id,))
            ann_stats = cursor.fetchone()
            
            # Get class distribution
            cursor.execute("""
                SELECT class_name, COUNT(*) as count
                FROM annotations 
                WHERE video_id = ?
                GROUP BY class_name
                ORDER BY count DESC
            """, (video_id,))
            class_distribution = [{
                "class_name": row[0],
                "count": row[1]
            } for row in cursor.fetchall()]
            
            return {
                "video": dict(video),
                "annotation_stats": dict(ann_stats),
                "class_distribution": class_distribution,
                "completion_percentage": (ann_stats[1] / video['frame_count'] * 100) if video['frame_count'] > 0 else 0
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video stats for {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve video statistics")

@app.get("/api/config")
async def get_config():
    """Get public configuration information"""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "max_file_size_mb": settings.max_file_size_mb,
        "max_video_width": settings.max_video_width,
        "max_video_height": settings.max_video_height,
        "max_frame_count": settings.max_frame_count,
        "supported_formats": settings.supported_formats,
        "max_class_name_length": settings.max_class_name_length,
        "max_filename_length": settings.max_filename_length,
        "enable_metrics": settings.enable_metrics
    }

@app.get("/api/resources")
async def get_resource_stats():
    """Get comprehensive resource usage statistics"""
    try:
        stats = await resource_manager.get_resource_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get resource stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resource statistics")

@app.post("/api/resources/cleanup")
async def manual_cleanup():
    """Manually trigger resource cleanup"""
    try:
        cleaned_count = await resource_manager.cleanup_expired_files()
        logger.info(f"Manual cleanup completed: {cleaned_count} files removed")
        return {
            "status": "completed",
            "files_cleaned": cleaned_count,
            "message": f"Cleaned {cleaned_count} expired files"
        }
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup operation failed")

@app.post("/api/resources/emergency-cleanup")
async def emergency_cleanup():
    """Perform emergency cleanup when system resources are low"""
    try:
        results = await resource_manager.emergency_cleanup()
        return {
            "status": "completed",
            "results": results
        }
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Emergency cleanup failed")

@app.get("/api/resources/duplicates")
async def find_duplicate_files():
    """Find duplicate files in the upload directory"""
    try:
        duplicates = await resource_manager.detect_duplicate_files()
        return {
            "duplicates_found": len(duplicates),
            "duplicates": duplicates,
            "total_wasted_space": sum(dup["size"] for dup in duplicates)
        }
    except Exception as e:
        logger.error(f"Failed to detect duplicates: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect duplicate files")

@app.get("/api/uploads/{upload_id}/progress")
async def get_upload_progress(upload_id: str):
    """Get progress information for a specific upload"""
    try:
        uuid.UUID(upload_id)  # Validate UUID format
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid upload ID format")
    
    progress = resource_manager.get_upload_progress(upload_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return progress

@app.get("/api/uploads/progress")
async def get_all_upload_progress():
    """Get progress information for all active uploads"""
    return {
        "active_uploads": resource_manager.get_all_upload_progress()
    }

def export_coco_format(video, annotations):
    """Export in COCO format"""
    try:
        coco_data = {
            "info": {
                "description": f"Annotations for {video['filename']}",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Create images (frames)
        frame_ids = set()
        for ann in annotations:
            frame_ids.add(ann['frame_number'])
        
        for frame_id in sorted(frame_ids):
            coco_data["images"].append({
                "id": frame_id,
                "file_name": f"frame_{frame_id:06d}.jpg",
                "width": video['width'],
                "height": video['height']
            })
        
        # Create categories
        class_names = set()
        for ann in annotations:
            class_names.add(ann['class_name'])
        
        for i, class_name in enumerate(sorted(class_names)):
            coco_data["categories"].append({
                "id": i + 1,
                "name": class_name,
                "supercategory": "object"
            })
    
        # Create annotations
        class_name_to_id = {cat['name']: cat['id'] for cat in coco_data["categories"]}
        
        for i, ann in enumerate(annotations):
            try:
                geometry = json.loads(ann['geometry'])
                
                if ann['annotation_type'] == 'bbox':
                    if not all(key in geometry for key in ['x', 'y', 'width', 'height']):
                        logger.warning(f"Skipping invalid bbox annotation {ann['id']}")
                        continue
                    bbox = [geometry['x'], geometry['y'], geometry['width'], geometry['height']]
                    area = geometry['width'] * geometry['height']
                else:
                    # For polygon, calculate bbox and area
                    if 'points' not in geometry or not geometry['points']:
                        logger.warning(f"Skipping invalid polygon annotation {ann['id']}")
                        continue
                    points = geometry['points']
                    if len(points) < 3:
                        logger.warning(f"Skipping polygon with insufficient points: {ann['id']}")
                        continue
                    xs = [p['x'] for p in points if 'x' in p and 'y' in p]
                    ys = [p['y'] for p in points if 'x' in p and 'y' in p]
                    if len(xs) != len(points) or len(xs) < 3:
                        logger.warning(f"Skipping polygon with invalid points: {ann['id']}")
                        continue
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    area = bbox[2] * bbox[3]  # Approximation
                
                coco_data["annotations"].append({
                    "id": i + 1,
                    "image_id": ann['frame_number'],
                    "category_id": class_name_to_id[ann['class_name']],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error processing annotation {ann['id']}: {e}")
                continue
        
        return coco_data
    
    except Exception as e:
        logger.error(f"Error in COCO export: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate COCO format")

def export_yolo_format(video, annotations):
    """Export in YOLO format"""
    try:
        # Group annotations by frame
        frames = {}
        for ann in annotations:
            frame_num = ann['frame_number']
            if frame_num not in frames:
                frames[frame_num] = []
            frames[frame_num].append(ann)
        
        # Get unique class names
        class_names = sorted(set(ann['class_name'] for ann in annotations))
        
        yolo_data = {
            "classes": class_names,
            "frames": {}
        }
        
        class_name_to_id = {name: i for i, name in enumerate(class_names)}
    
        for frame_num, frame_annotations in frames.items():
            yolo_annotations = []
            
            for ann in frame_annotations:
                try:
                    geometry = json.loads(ann['geometry'])
                    class_id = class_name_to_id[ann['class_name']]
                    
                    if ann['annotation_type'] == 'bbox':
                        # Validate required keys
                        if not all(key in geometry for key in ['x', 'y', 'width', 'height']):
                            logger.warning(f"Skipping invalid bbox annotation {ann['id']}")
                            continue
                        
                        # Convert to YOLO format (normalized center coordinates and dimensions)
                        x_center = (geometry['x'] + geometry['width'] / 2) / video['width']
                        y_center = (geometry['y'] + geometry['height'] / 2) / video['height']
                        width = geometry['width'] / video['width']
                        height = geometry['height'] / video['height']
                        
                        # Validate normalized coordinates
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                            logger.warning(f"Skipping bbox with invalid normalized coordinates: {ann['id']}")
                            continue
                        
                        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    # Note: YOLO format typically doesn't support polygons, so we skip them
                except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
                    logger.error(f"Error processing annotation {ann['id']} for YOLO: {e}")
                    continue
            
            yolo_data["frames"][f"frame_{frame_num:06d}"] = yolo_annotations
        
        return yolo_data
    
    except Exception as e:
        logger.error(f"Error in YOLO export: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate YOLO format")

def get_html_content():
    """Return the HTML content for the frontend"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Annotation Tool</title>
        <style>
            :root {
                /* Light theme colors */
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary-gradient: linear-gradient(135deg, #764ba2, #667eea);
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --text-primary: #333333;
                --text-secondary: #666666;
                --border-color: #e0e0e0;
                --accent-color: #667eea;
                --success-color: #4CAF50;
                --error-color: #f44336;
                --warning-color: #ff9800;
                --shadow-light: rgba(0, 0, 0, 0.1);
                --shadow-medium: rgba(0, 0, 0, 0.15);
                --glass-bg: rgba(255, 255, 255, 0.95);
            }
            
            [data-theme="dark"] {
                /* Dark theme colors */
                --primary-gradient: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                --secondary-gradient: linear-gradient(135deg, #16213e, #1a1a2e);
                --bg-primary: #1e1e1e;
                --bg-secondary: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #cccccc;
                --border-color: #404040;
                --accent-color: #7b68ee;
                --success-color: #66bb6a;
                --error-color: #ef5350;
                --warning-color: #ffb74d;
                --shadow-light: rgba(0, 0, 0, 0.3);
                --shadow-medium: rgba(0, 0, 0, 0.4);
                --glass-bg: rgba(30, 30, 30, 0.95);
            }
            
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
                transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: var(--primary-gradient);
                min-height: 100vh;
                color: var(--text-primary);
            }
            
            .page-header {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-bottom: 2px solid var(--accent-color);
                padding: 1rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 2px 10px var(--shadow-light);
                position: sticky;
                top: 0;
                z-index: 1000;
            }
            
            .page-header h1 {
                color: var(--accent-color);
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .header-controls {
                display: flex;
                gap: 1rem;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px var(--shadow-light);
                text-align: center;
                position: relative;
            }
            
            .theme-toggle {
                position: absolute;
                top: 20px;
                right: 20px;
                background: var(--accent-color);
                color: white;
                border: none;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 15px var(--shadow-medium);
                transition: all 0.3s ease;
            }
            
            .theme-toggle:hover {
                transform: translateY(-2px) scale(1.1);
                box-shadow: 0 6px 20px var(--shadow-medium);
            }
            
            .header h1 {
                font-size: 2.5em;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            
            .header p {
                color: var(--text-secondary);
                font-size: 1.1em;
            }
            
            .toolbar {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px var(--shadow-light);
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            
            .toolbar-group {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px 12px;
                background: var(--bg-secondary);
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }
            
            .toolbar-group label {
                font-size: 12px;
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 350px;
                gap: 30px;
                margin-bottom: 30px;
            }
            
            .video-section {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px var(--shadow-light);
            }
            
            .controls-panel {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px var(--shadow-light);
            }
            
            .upload-area {
                border: 3px dashed var(--accent-color);
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                transition: all 0.3s ease;
                cursor: pointer;
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
            }
            
            .upload-area:hover {
                border-color: var(--accent-color);
                background: var(--bg-secondary);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px var(--shadow-light);
            }
            
            .upload-area.dragover {
                border-color: var(--accent-color);
                background: var(--bg-secondary);
                box-shadow: 0 8px 25px var(--shadow-medium);
            }
            
            #video-container {
                position: relative;
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 20px;
                min-height: 400px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            #video-player {
                width: 100%;
                height: auto;
                max-height: 500px;
            }
            
            #annotation-canvas {
                position: absolute;
                top: 0;
                left: 0;
                cursor: crosshair;
                pointer-events: all;
            }
            
            .video-controls {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            
            button {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            input, select {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s ease;
            }
            
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
                color: #333;
            }
            
            .annotation-tools {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            
            .annotation-tools button {
                flex: 1;
                padding: 10px;
                font-size: 12px;
            }
            
            .annotation-tools button.active {
                background: linear-gradient(135deg, #764ba2, #667eea);
            }
            
            .frame-info {
                background: rgba(102, 126, 234, 0.1);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            
            .annotations-list {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
            
            .annotation-item {
                background: #f8f9fa;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .annotation-item:last-child {
                margin-bottom: 0;
            }
            
            .annotation-info {
                flex: 1;
            }
            
            .annotation-info strong {
                color: #667eea;
            }
            
            .delete-btn {
                background: #ff6b6b;
                padding: 6px 12px;
                font-size: 12px;
                border-radius: 6px;
            }
            
            .delete-btn:hover {
                background: #ff5252;
            }
            
            .export-section {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px var(--shadow-light);
                margin-top: 30px;
            }
            
            .export-buttons {
                display: flex;
                gap: 15px;
                margin-top: 15px;
            }
            
            .status-message {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 25px;
                border-radius: 10px;
                color: white;
                font-weight: 500;
                z-index: 1000;
                transform: translateX(400px);
                transition: transform 0.3s ease;
            }
            
            .status-message.show {
                transform: translateX(0);
            }
            
            .status-message.success {
                background: linear-gradient(135deg, #4CAF50, #45a049);
            }
            
            .status-message.error {
                background: linear-gradient(135deg, #f44336, #d32f2f);
            }
            
            .video-list {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px var(--shadow-light);
                margin-bottom: 30px;
            }
            
            .video-list h2 {
                color: var(--text-primary);
                margin-bottom: 20px;
            }
            
            .video-item {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 15px;
                border: 1px solid var(--border-color);
                border-radius: 10px;
                margin-bottom: 15px;
                transition: all 0.3s ease;
                background: var(--bg-secondary);
            }
            
            .video-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px var(--shadow-light);
                border-color: var(--accent-color);
            }
            
            .video-item:last-child {
                margin-bottom: 0;
            }
            
            .video-thumbnail {
                width: 120px;
                height: 80px;
                border-radius: 8px;
                object-fit: cover;
                background: var(--bg-primary);
                border: 2px solid var(--border-color);
                flex-shrink: 0;
            }
            
            .video-thumbnail.loading {
                background: linear-gradient(90deg, var(--bg-primary) 25%, var(--bg-secondary) 50%, var(--bg-primary) 75%);
                background-size: 200% 100%;
                animation: loading 1.5s infinite;
            }
            
            .video-thumbnail-fallback {
                width: 120px;
                height: 80px;
                border-radius: 8px;
                background: var(--bg-secondary);
                border: 2px solid var(--border-color);
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: var(--text-secondary);
                cursor: pointer;
            }
            
            @keyframes loading {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            
            .video-info {
                flex: 1;
                min-width: 0;
            }
            
            .video-info h3 {
                color: var(--accent-color);
                margin-bottom: 5px;
                font-size: 16px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .video-meta {
                color: var(--text-secondary);
                font-size: 14px;
                margin-bottom: 8px;
            }
            
            .video-stats {
                display: flex;
                gap: 15px;
                font-size: 12px;
                color: var(--text-secondary);
            }
            
            .video-stat {
                display: flex;
                align-items: center;
                gap: 4px;
            }
            
            .preview-thumbnails {
                display: none;
                position: absolute;
                top: -10px;
                left: 50%;
                transform: translateX(-50%);
                background: var(--glass-bg);
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 4px 20px var(--shadow-medium);
                z-index: 100;
                gap: 5px;
            }
            
            .video-item:hover .preview-thumbnails {
                display: flex;
            }
            
            .preview-thumbnail {
                width: 60px;
                height: 40px;
                border-radius: 4px;
                object-fit: cover;
                border: 1px solid var(--border-color);
            }
            
            .video-actions {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .video-actions button {
                white-space: nowrap;
            }
            
            @media (max-width: 1024px) {
                .main-content {
                    grid-template-columns: 1fr;
                }
                
                .container {
                    padding: 15px;
                }
                
                .header h1 {
                    font-size: 2em;
                }
            }
        </style>
    </head>
    <body>
        <header class="page-header">
            <h1>JR Video Annotation Tool</h1>
            <div class="header-controls">
                <button id="theme-toggle" class="theme-toggle" title="Toggle Dark/Light Mode">
                    <span id="theme-icon">🌙</span>
                </button>
            </div>
        </header>
        
        <div class="container">

            <div class="video-list" id="video-list" style="display: none;">
                <h2>Uploaded Videos</h2>
                <div id="videos-container"></div>
            </div>

            <div class="upload-area" id="upload-area">
                <h3>Upload Video</h3>
                <p>Drag and drop your video file here or click to select</p>
                <input type="file" id="video-input" accept="video/*" style="display: none;">
            </div>

            <div class="main-content" id="main-content" style="display: none;">
                <div class="video-section">
                    <div class="toolbar">
                        <div class="toolbar-group">
                            <label>Playback</label>
                            <button id="play-pause" title="Space: Play/Pause">
                                <span>▶️</span>
                            </button>
                            <button id="step-backward" title="Left Arrow: Previous Frame">⏮️</button>
                            <button id="step-forward" title="Right Arrow: Next Frame">⏭️</button>
                        </div>
                        <div class="toolbar-group">
                            <label>Speed</label>
                            <select id="speed-selector" title="Playback Speed">
                                <option value="0.25">0.25x</option>
                                <option value="0.5">0.5x</option>
                                <option value="0.75">0.75x</option>
                                <option value="1" selected>1x</option>
                                <option value="1.25">1.25x</option>
                                <option value="1.5">1.5x</option>
                                <option value="2">2x</option>
                            </select>
                        </div>
                        <div class="toolbar-group">
                            <label>Jump</label>
                            <button onclick="jumpBackward()" title="Jump Back 10s">⏪</button>
                            <button onclick="jumpForward()" title="Jump Forward 10s">⏩</button>
                        </div>
                        <div class="toolbar-group">
                            <label>Tools</label>
                            <button id="fullscreen-toggle" title="F: Fullscreen">🔍</button>
                            <button id="reset-zoom" title="R: Reset View">🎯</button>
                        </div>
                    </div>
                    <div id="video-container">
                        <video id="video-player" controls style="display: none;">
                            Your browser does not support the video tag.
                        </video>
                        <canvas id="annotation-canvas"></canvas>
                        <div id="no-video" style="color: #999; font-size: 18px;">
                            No video loaded
                        </div>
                    </div>
                    
                    <div class="video-controls">
                        <button onclick="playPause()">Play/Pause</button>
                        <button onclick="previousFrame()">◀ Frame</button>
                        <button onclick="nextFrame()">Frame ▶</button>
                        <input type="range" id="timeline" min="0" max="100" value="0" 
                               style="flex: 1; margin: 0 15px;">
                        <span id="time-display">00:00 / 00:00</span>
                    </div>
                </div>

                <div class="controls-panel">
                    <h3>Annotation Controls</h3>
                    
                    <div class="frame-info" id="frame-info">
                        <strong>Frame:</strong> <span id="current-frame">0</span><br>
                        <strong>Time:</strong> <span id="current-time">00:00</span><br>
                        <strong>Total Frames:</strong> <span id="total-frames">0</span>
                    </div>

                    <div class="annotation-tools">
                        <button id="bbox-tool" class="active" onclick="setAnnotationTool('bbox')">
                            Bounding Box
                        </button>
                        <button id="polygon-tool" onclick="setAnnotationTool('polygon')">
                            Polygon
                        </button>
                    </div>

                    <div class="form-group">
                        <label for="class-name">Class Name:</label>
                        <input type="text" id="class-name" placeholder="e.g., person, car" value="person">
                    </div>

                    <div class="form-group">
                        <label for="track-id">Track ID (optional):</label>
                        <input type="number" id="track-id" placeholder="1, 2, 3...">
                    </div>

                    <button onclick="clearAnnotations()" style="background: #ff6b6b; width: 100%; margin-bottom: 20px;">
                        Clear Frame Annotations
                    </button>

                    <!-- Interpolation Controls -->
                    <div class="interpolation-panel" style="margin-top: 20px; padding: 15px; background: var(--bg-secondary); border-radius: 8px; border: 1px solid var(--border-color);">
                        <h4 style="margin-bottom: 15px; color: var(--accent-color);">🎯 Object Tracking & Interpolation</h4>
                        
                        <div class="form-group">
                            <label for="interp-track-id">Track ID for Interpolation:</label>
                            <input type="number" id="interp-track-id" placeholder="Enter track ID" min="1">
                        </div>
                        
                        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                            <div class="form-group" style="flex: 1; margin-bottom: 0;">
                                <label for="start-frame">Start Frame:</label>
                                <input type="number" id="start-frame" placeholder="0" min="0">
                            </div>
                            <div class="form-group" style="flex: 1; margin-bottom: 0;">
                                <label for="end-frame">End Frame:</label>
                                <input type="number" id="end-frame" placeholder="100" min="0">
                            </div>
                        </div>
                        
                        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                            <button onclick="setKeyframe('start')" style="flex: 1; background: var(--success-color); font-size: 12px;">
                                📍 Set Start Keyframe
                            </button>
                            <button onclick="setKeyframe('end')" style="flex: 1; background: var(--warning-color); font-size: 12px;">
                                📍 Set End Keyframe
                            </button>
                        </div>
                        
                        <button onclick="performInterpolation()" style="width: 100%; background: linear-gradient(135deg, var(--accent-color), #5b4fcf); font-size: 14px;">
                            ✨ Interpolate Between Keyframes
                        </button>
                        
                        <div id="interpolation-status" style="margin-top: 10px; font-size: 12px; color: var(--text-secondary);"></div>
                    </div>
                    
                    <h4>Current Frame Annotations</h4>
                    <div class="annotations-list" id="annotations-list">
                        <p style="color: #999; text-align: center;">No annotations</p>
                    </div>
                </div>
            </div>

            <div class="export-section" id="export-section" style="display: none;">
                <h3>Export Annotations</h3>
                <p>Download your annotations in industry-standard formats</p>
                <div class="export-buttons">
                    <button onclick="exportAnnotations('coco')">Export COCO Format</button>
                    <button onclick="exportAnnotations('yolo')">Export YOLO Format</button>
                </div>
            </div>
        </div>

        <div id="status-message" class="status-message"></div>

        <script>
            // Global variables
            let currentVideo = null;
            let annotations = [];
            let currentAnnotationTool = 'bbox';
            let isDrawing = false;
            let startPoint = null;
            let currentAnnotation = null;
            let polygonPoints = [];

            // Enhanced UI Variables
            let isDarkMode = localStorage.getItem('darkMode') === 'true';
            
            // Initialize the application
            document.addEventListener('DOMContentLoaded', function() {
                setupEventListeners();
                loadVideos();
                initializeTheme();
                setupEnhancedKeyboardShortcuts();
            });
            
            function initializeTheme() {
                if (isDarkMode) {
                    document.body.setAttribute('data-theme', 'dark');
                    document.getElementById('theme-icon').textContent = '☀️';
                } else {
                    document.body.removeAttribute('data-theme');
                    document.getElementById('theme-icon').textContent = '🌙';
                }
            }
            
            function toggleTheme() {
                isDarkMode = !isDarkMode;
                localStorage.setItem('darkMode', isDarkMode);
                initializeTheme();
                showStatus(isDarkMode ? 'Dark mode enabled' : 'Light mode enabled', 'success');
            }
            
            function setupEnhancedKeyboardShortcuts() {
                document.addEventListener('keydown', function(e) {
                    // Don't trigger shortcuts when typing in input fields
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                    
                    switch(e.key.toLowerCase()) {
                        case 'f':
                            e.preventDefault();
                            toggleFullscreen();
                            break;
                        case 'd':
                            e.preventDefault();
                            toggleTheme();
                            break;
                        case '1':
                            e.preventDefault();
                            setAnnotationTool('bbox');
                            break;
                        case '2':
                            e.preventDefault();
                            setAnnotationTool('polygon');
                            break;
                        case 'r':
                            e.preventDefault();
                            resetZoom();
                            break;
                        case '+':
                        case '=':
                            e.preventDefault();
                            adjustPlaybackSpeed(1);
                            break;
                        case '-':
                            e.preventDefault();
                            adjustPlaybackSpeed(-1);
                            break;
                    }
                });
            }

            function setupEventListeners() {
                const uploadArea = document.getElementById('upload-area');
                const videoInput = document.getElementById('video-input');
                const videoPlayer = document.getElementById('video-player');
                const timeline = document.getElementById('timeline');
                const canvas = document.getElementById('annotation-canvas');

                // Upload area events
                uploadArea.addEventListener('click', () => videoInput.click());
                uploadArea.addEventListener('dragover', handleDragOver);
                uploadArea.addEventListener('drop', handleDrop);
                uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));

                // File input change
                videoInput.addEventListener('change', handleFileSelect);

                // Video events
                videoPlayer.addEventListener('loadedmetadata', onVideoLoaded);
                videoPlayer.addEventListener('timeupdate', onTimeUpdate);
                videoPlayer.addEventListener('ended', () => videoPlayer.pause());

                // Timeline events
                timeline.addEventListener('input', seekToFrame);
                timeline.addEventListener('change', seekToFrame);

                // Canvas events for annotation
                canvas.addEventListener('mousedown', startAnnotation);
                canvas.addEventListener('mousemove', updateAnnotation);
                canvas.addEventListener('mouseup', finishAnnotation);
                canvas.addEventListener('dblclick', finishPolygon);

                // Keyboard shortcuts
                document.addEventListener('keydown', handleKeyPress);
                
                // Enhanced UI controls
                const themeToggle = document.getElementById('theme-toggle');
                const fullscreenToggle = document.getElementById('fullscreen-toggle');
                const resetZoomBtn = document.getElementById('reset-zoom');
                const playPause = document.getElementById('play-pause');
                const stepBackward = document.getElementById('step-backward');
                const stepForward = document.getElementById('step-forward');
                const speedSelector = document.getElementById('speed-selector');
                
                if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
                if (fullscreenToggle) fullscreenToggle.addEventListener('click', toggleFullscreen);
                if (resetZoomBtn) resetZoomBtn.addEventListener('click', resetZoom);
                if (playPause) playPause.addEventListener('click', togglePlayPause);
                if (stepBackward) stepBackward.addEventListener('click', stepBackwardFrame);
                if (stepForward) stepForward.addEventListener('click', stepForwardFrame);
                if (speedSelector) speedSelector.addEventListener('change', handleSpeedChange);
            }

            function handleDragOver(e) {
                e.preventDefault();
                document.getElementById('upload-area').classList.add('dragover');
            }

            function handleDrop(e) {
                e.preventDefault();
                document.getElementById('upload-area').classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uploadVideo(files[0]);
                }
            }

            function handleFileSelect(e) {
                if (e.target.files.length > 0) {
                    uploadVideo(e.target.files[0]);
                }
            }

            async function uploadVideo(file) {
                console.log('Starting upload for file:', file.name);
                const formData = new FormData();
                formData.append('file', file);

                showStatus('Uploading video...', 'success');

                try {
                    const response = await fetch('/api/upload-video', {
                        method: 'POST',
                        body: formData
                    });

                    console.log('Upload response status:', response.status);
                    
                    if (response.ok) {
                        const videoInfo = await response.json();
                        console.log('Upload successful, video info:', videoInfo);
                        showStatus('Video uploaded successfully!', 'success');
                        
                        // Load video if successful and refresh list
                        await loadVideos();
                        loadVideo(videoInfo.id);  // Use the ID instead of the whole object
                    } else {
                        const errorText = await response.text();
                        console.error('Upload failed with status:', response.status, errorText);
                        throw new Error(`Upload failed: ${response.status} ${errorText}`);
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                    showStatus('Upload failed: ' + error.message, 'error');
                }
            }

            async function loadVideos() {
                try {
                    console.log('Loading videos...');
                    const response = await fetch('/api/videos');
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const videos = await response.json();
                    console.log('Loaded videos:', videos.length, videos);
                    
                    const container = document.getElementById('videos-container');
                    const videoList = document.getElementById('video-list');

                    if (!videos || videos.length === 0) {
                        console.log('No videos found');
                        videoList.style.display = 'none';
                        return;
                    }

                    console.log('Displaying video list');
                    videoList.style.display = 'block';
                    container.innerHTML = '';

                    videos.forEach(video => {
                        const videoItem = document.createElement('div');
                        videoItem.className = 'video-item';
                        
                        // Create thumbnail element
                        const thumbnailHtml = video.thumbnail ? 
                            `<img src="${video.thumbnail}" alt="Video thumbnail" class="video-thumbnail" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">` +
                            `<div class="video-thumbnail-fallback" style="display: none;">📹</div>` :
                            `<div class="video-thumbnail-fallback">📹</div>`;
                        
                        // Create preview thumbnails for hover
                        const previewHtml = video.preview_thumbnails && video.preview_thumbnails.length > 0 ?
                            `<div class="preview-thumbnails">
                                ${video.preview_thumbnails.map(thumb => 
                                    `<img src="${thumb}" alt="Preview" class="preview-thumbnail">`
                                ).join('')}
                            </div>` : '';
                        
                        // Calculate file size display safely
                        const fileSizeMB = (video.file_size && video.file_size > 0) ? 
                            (video.file_size / (1024 * 1024)).toFixed(1) : 'Unknown';
                        
                        videoItem.innerHTML = `
                            ${thumbnailHtml}
                            <div class="video-info">
                                <h3 title="${video.filename}">${video.filename}</h3>
                                <div class="video-meta">
                                    Duration: ${formatTime(video.duration)} | Resolution: ${video.width}×${video.height}
                                </div>
                                <div class="video-stats">
                                    <div class="video-stat">
                                        <span>📊</span>
                                        <span>${video.fps.toFixed(1)} FPS</span>
                                    </div>
                                    <div class="video-stat">
                                        <span>🎞️</span>
                                        <span>${video.frame_count.toLocaleString()} frames</span>
                                    </div>
                                    <div class="video-stat">
                                        <span>💾</span>
                                        <span>${fileSizeMB} MB</span>
                                    </div>
                                </div>
                            </div>
                            <div class="video-actions">
                                <button onclick="loadVideo('${video.id}')">Load Video</button>
                                <button class="delete-btn" onclick="deleteVideo('${video.id}', '${video.filename}')">
                                    Delete
                                </button>
                            </div>
                            ${previewHtml}
                        `;
                        
                        // Add click handler to thumbnail for quick preview
                        const thumbnail = videoItem.querySelector('.video-thumbnail, .video-thumbnail-fallback');
                        if (thumbnail) {
                            thumbnail.style.cursor = 'pointer';
                            thumbnail.addEventListener('click', () => {
                                loadVideo(video.id);
                            });
                        }
                        
                        // If no thumbnail exists, try to generate one
                        if (!video.thumbnail) {
                            generateThumbnailForVideo(video.id, videoItem);
                        }
                        
                        container.appendChild(videoItem);
                    });
                } catch (error) {
                    console.error('Failed to load videos:', error);
                    showStatus('Failed to load videos: ' + error.message, 'error');
                }
            }
            
            async function generateThumbnailForVideo(videoId, videoItem) {
                try {
                    const response = await fetch(`/api/videos/${videoId}/regenerate-thumbnails`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        // Update the thumbnail in the video item
                        const fallback = videoItem.querySelector('.video-thumbnail-fallback');
                        if (fallback && result.thumbnail) {
                            fallback.outerHTML = `<img src="${result.thumbnail}" alt="Video thumbnail" class="video-thumbnail">`;
                        }
                    }
                } catch (error) {
                    console.log(`Could not generate thumbnail for video ${videoId}:`, error);
                }
            }
            
            // Enhanced Toolbar Control Functions
            function toggleFullscreen() {
                const container = document.getElementById('video-container');
                if (!document.fullscreenElement) {
                    if (container) {
                        container.requestFullscreen().then(() => {
                            showStatus('Fullscreen enabled. Press ESC to exit.', 'success');
                        }).catch(err => {
                            console.error('Error attempting to enable fullscreen:', err);
                            showStatus('Fullscreen not supported', 'error');
                        });
                    }
                } else {
                    document.exitFullscreen();
                }
            }
            
            function resetZoom() {
                const canvas = document.getElementById('annotation-canvas');
                const videoPlayer = document.getElementById('video-player');
                
                // Reset video and canvas transformations
                if (videoPlayer) {
                    videoPlayer.style.transform = 'scale(1)';
                }
                if (canvas) {
                    canvas.style.transform = 'scale(1)';
                }
                
                showStatus('Zoom reset to 100%', 'success');
            }
            
            function togglePlayPause() {
                const videoPlayer = document.getElementById('video-player');
                const playPauseIcon = document.getElementById('play-pause').querySelector('span');
                
                if (videoPlayer.paused) {
                    videoPlayer.play();
                    playPauseIcon.textContent = '⏸️';
                } else {
                    videoPlayer.pause();
                    playPauseIcon.textContent = '▶️';
                }
            }
            
            function stepBackwardFrame() {
                const videoPlayer = document.getElementById('video-player');
                if (videoPlayer && videoPlayer.readyState >= 2) {
                    videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - (1 / 30)); // Step back 1 frame at 30fps
                    updateTimeline();
                }
            }
            
            function stepForwardFrame() {
                const videoPlayer = document.getElementById('video-player');
                if (videoPlayer && videoPlayer.readyState >= 2) {
                    videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + (1 / 30)); // Step forward 1 frame at 30fps
                    updateTimeline();
                }
            }
            
            function handleSpeedChange(event) {
                const speed = parseFloat(event.target.value);
                const videoPlayer = document.getElementById('video-player');
                if (videoPlayer) {
                    videoPlayer.playbackRate = speed;
                    showStatus(`Playback speed: ${speed}x`, 'success');
                }
            }
            
            function adjustPlaybackSpeed(direction) {
                const speedSelector = document.getElementById('speed-selector');
                const currentSpeed = parseFloat(speedSelector.value);
                const speeds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
                const currentIndex = speeds.indexOf(currentSpeed);
                
                let newIndex = currentIndex + direction;
                if (newIndex < 0) newIndex = 0;
                if (newIndex >= speeds.length) newIndex = speeds.length - 1;
                
                const newSpeed = speeds[newIndex];
                speedSelector.value = newSpeed;
                
                const videoPlayer = document.getElementById('video-player');
                if (videoPlayer) {
                    videoPlayer.playbackRate = newSpeed;
                    showStatus(`Playback speed: ${newSpeed}x`, 'success');
                }
            }
            
            function jumpToTime(seconds) {
                const videoPlayer = document.getElementById('video-player');
                if (videoPlayer && videoPlayer.readyState >= 2) {
                    const newTime = Math.max(0, Math.min(videoPlayer.duration, videoPlayer.currentTime + seconds));
                    videoPlayer.currentTime = newTime;
                    updateTimeline();
                }
            }
            
            function jumpBackward() {
                jumpToTime(-10); // Jump back 10 seconds
            }
            
            function jumpForward() {
                jumpToTime(10); // Jump forward 10 seconds
            }
            
            function updateTimeline() {
                const videoPlayer = document.getElementById('video-player');
                const timeline = document.getElementById('timeline');
                
                if (videoPlayer && timeline && videoPlayer.duration) {
                    timeline.value = (videoPlayer.currentTime / videoPlayer.duration) * 100;
                    document.getElementById('time-display').textContent = 
                        `${formatTime(videoPlayer.currentTime)} / ${formatTime(videoPlayer.duration)}`;
                }
            }

            async function loadVideo(videoId) {
                try {
                    let videoInfo;
                    if (typeof videoId === 'string') {
                        const response = await fetch('/api/videos');
                        const videos = await response.json();
                        videoInfo = videos.find(v => v.id === videoId);
                    } else {
                        videoInfo = videoId;
                    }

                    if (!videoInfo) {
                        throw new Error('Video not found');
                    }

                    currentVideo = videoInfo;
                    
                    const videoPlayer = document.getElementById('video-player');
                    const canvas = document.getElementById('annotation-canvas');
                    const noVideo = document.getElementById('no-video');

                    videoPlayer.src = `/uploads/${videoInfo.id}_${videoInfo.filename}`;
                    videoPlayer.style.display = 'block';
                    noVideo.style.display = 'none';

                    document.getElementById('main-content').style.display = 'grid';
                    document.getElementById('export-section').style.display = 'block';

                    // Update frame info
                    document.getElementById('total-frames').textContent = videoInfo.frame_count;

                    // Load annotations for this video
                    await loadAnnotations(videoInfo.id);

                } catch (error) {
                    showStatus('Failed to load video: ' + error.message, 'error');
                }
            }
            
            async function deleteVideo(videoId, filename) {
                // Confirm deletion
                if (!confirm(`Are you sure you want to delete the video "${filename}"?\n\nThis will permanently delete:\n- The video file\n- All annotations for this video\n\nThis action cannot be undone.`)) {
                    return;
                }
                
                try {
                    showStatus('Deleting video...', 'success');
                    
                    const response = await fetch(`/api/videos/${videoId}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        showStatus(`Video "${filename}" deleted successfully!`, 'success');
                        
                        // If the deleted video is currently loaded, clear the interface
                        if (currentVideo && currentVideo.id === videoId) {
                            currentVideo = null;
                            const videoPlayer = document.getElementById('video-player');
                            const noVideo = document.getElementById('no-video');
                            
                            videoPlayer.style.display = 'none';
                            noVideo.style.display = 'block';
                            
                            document.getElementById('main-content').style.display = 'none';
                            document.getElementById('export-section').style.display = 'none';
                            
                            // Clear annotations list
                            const annotationsList = document.getElementById('annotations-list');
                            annotationsList.innerHTML = '<p style="color: #999; text-align: center;">No annotations</p>';
                        }
                        
                        // Refresh the video list
                        await loadVideos();
                        
                    } else {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Delete failed');
                    }
                } catch (error) {
                    showStatus('Failed to delete video: ' + error.message, 'error');
                }
            }

            function onVideoLoaded() {
                const video = document.getElementById('video-player');
                const canvas = document.getElementById('annotation-canvas');
                const timeline = document.getElementById('timeline');

                // Resize canvas to match video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.style.width = video.offsetWidth + 'px';
                canvas.style.height = video.offsetHeight + 'px';

                // Setup timeline
                timeline.max = currentVideo.frame_count - 1;
                timeline.value = 0;

                updateFrameInfo();
            }

            function onTimeUpdate() {
                const video = document.getElementById('video-player');
                const timeline = document.getElementById('timeline');
                
                if (currentVideo) {
                    const currentFrame = Math.floor(video.currentTime * currentVideo.fps);
                    timeline.value = currentFrame;
                    updateFrameInfo();
                    updateTimeline(); // Also update the enhanced timeline display
                    loadFrameAnnotations();
                    
                    // Update play/pause button icon
                    const playPauseIcon = document.getElementById('play-pause')?.querySelector('span');
                    if (playPauseIcon) {
                        playPauseIcon.textContent = video.paused ? '▶️' : '⏸️';
                    }
                }
            }

            function updateFrameInfo() {
                const video = document.getElementById('video-player');
                if (!currentVideo) return;

                const currentFrame = Math.floor(video.currentTime * currentVideo.fps);
                const currentTime = video.currentTime;
                const duration = video.duration || currentVideo.duration;

                document.getElementById('current-frame').textContent = currentFrame;
                document.getElementById('current-time').textContent = formatTime(currentTime);
                document.getElementById('time-display').textContent = 
                    `${formatTime(currentTime)} / ${formatTime(duration)}`;
            }

            function formatTime(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }

            function seekToFrame() {
                const video = document.getElementById('video-player');
                const timeline = document.getElementById('timeline');
                
                if (currentVideo) {
                    const frameNumber = parseInt(timeline.value);
                    const time = frameNumber / currentVideo.fps;
                    video.currentTime = time;
                    updateFrameInfo();
                    loadFrameAnnotations();
                }
            }

            function playPause() {
                const video = document.getElementById('video-player');
                if (video.paused) {
                    video.play();
                } else {
                    video.pause();
                }
            }

            function previousFrame() {
                const video = document.getElementById('video-player');
                const timeline = document.getElementById('timeline');
                
                if (currentVideo) {
                    let currentFrame = Math.floor(video.currentTime * currentVideo.fps);
                    currentFrame = Math.max(0, currentFrame - 1);
                    video.currentTime = currentFrame / currentVideo.fps;
                    timeline.value = currentFrame;
                    updateFrameInfo();
                    loadFrameAnnotations();
                }
            }

            function nextFrame() {
                const video = document.getElementById('video-player');
                const timeline = document.getElementById('timeline');
                
                if (currentVideo) {
                    let currentFrame = Math.floor(video.currentTime * currentVideo.fps);
                    currentFrame = Math.min(currentVideo.frame_count - 1, currentFrame + 1);
                    video.currentTime = currentFrame / currentVideo.fps;
                    timeline.value = currentFrame;
                    updateFrameInfo();
                    loadFrameAnnotations();
                }
            }

            function handleKeyPress(e) {
                if (e.target.tagName === 'INPUT') return;

                switch(e.key) {
                    case ' ':
                        e.preventDefault();
                        playPause();
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        previousFrame();
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        nextFrame();
                        break;
                }
            }

            function setAnnotationTool(tool) {
                currentAnnotationTool = tool;
                
                // Update button states
                document.getElementById('bbox-tool').classList.remove('active');
                document.getElementById('polygon-tool').classList.remove('active');
                document.getElementById(tool + '-tool').classList.add('active');

                // Reset any current annotation
                polygonPoints = [];
                drawAnnotations();
            }

            function getCanvasCoordinates(e) {
                const canvas = document.getElementById('annotation-canvas');
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                
                return {
                    x: (e.clientX - rect.left) * scaleX,
                    y: (e.clientY - rect.top) * scaleY
                };
            }

            function startAnnotation(e) {
                const coords = getCanvasCoordinates(e);
                
                if (currentAnnotationTool === 'bbox') {
                    isDrawing = true;
                    startPoint = coords;
                } else if (currentAnnotationTool === 'polygon') {
                    polygonPoints.push(coords);
                    drawAnnotations();
                }
            }

            function updateAnnotation(e) {
                if (!isDrawing || currentAnnotationTool !== 'bbox') return;
                
                const coords = getCanvasCoordinates(e);
                
                // Draw temporary rectangle
                drawAnnotations();
                const canvas = document.getElementById('annotation-canvas');
                const ctx = canvas.getContext('2d');
                
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    startPoint.x,
                    startPoint.y,
                    coords.x - startPoint.x,
                    coords.y - startPoint.y
                );
            }

            function finishAnnotation(e) {
                if (!isDrawing || currentAnnotationTool !== 'bbox') return;
                
                isDrawing = false;
                const coords = getCanvasCoordinates(e);
                
                const width = Math.abs(coords.x - startPoint.x);
                const height = Math.abs(coords.y - startPoint.y);
                
                if (width > 10 && height > 10) { // Minimum size threshold
                    const annotation = {
                        video_id: currentVideo.id,
                        frame_number: getCurrentFrame(),
                        annotation_type: 'bbox',
                        class_name: document.getElementById('class-name').value || 'object',
                        geometry: {
                            x: Math.min(startPoint.x, coords.x),
                            y: Math.min(startPoint.y, coords.y),
                            width: width,
                            height: height
                        },
                        track_id: parseInt(document.getElementById('track-id').value) || null
                    };
                    
                    saveAnnotation(annotation);
                }
                
                startPoint = null;
            }

            function finishPolygon(e) {
                if (currentAnnotationTool !== 'polygon' || polygonPoints.length < 3) return;
                
                const annotation = {
                    video_id: currentVideo.id,
                    frame_number: getCurrentFrame(),
                    annotation_type: 'polygon',
                    class_name: document.getElementById('class-name').value || 'object',
                    geometry: {
                        points: polygonPoints.slice()
                    },
                    track_id: parseInt(document.getElementById('track-id').value) || null
                };
                
                saveAnnotation(annotation);
                polygonPoints = [];
            }

            function getCurrentFrame() {
                const video = document.getElementById('video-player');
                return Math.floor(video.currentTime * currentVideo.fps);
            }

            async function saveAnnotation(annotation) {
                try {
                    const response = await fetch('/api/annotations', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(annotation)
                    });

                    if (response.ok) {
                        await loadFrameAnnotations();
                        showStatus('Annotation saved!', 'success');
                    } else {
                        throw new Error('Failed to save annotation');
                    }
                } catch (error) {
                    showStatus('Failed to save annotation: ' + error.message, 'error');
                }
            }

            async function loadAnnotations(videoId) {
                try {
                    const response = await fetch(`/api/annotations/${videoId}`);
                    annotations = await response.json();
                    loadFrameAnnotations();
                } catch (error) {
                    showStatus('Failed to load annotations', 'error');
                }
            }

            async function loadFrameAnnotations() {
                if (!currentVideo) return;
                
                const currentFrame = getCurrentFrame();
                const frameAnnotations = annotations.filter(ann => ann.frame_number === currentFrame);
                
                // Update UI
                const annotationsList = document.getElementById('annotations-list');
                
                if (frameAnnotations.length === 0) {
                    annotationsList.innerHTML = '<p style="color: #999; text-align: center;">No annotations</p>';
                } else {
                    annotationsList.innerHTML = frameAnnotations.map(ann => `
                        <div class="annotation-item">
                            <div class="annotation-info">
                                <strong>${ann.class_name}</strong><br>
                                <small>${ann.annotation_type}${ann.track_id ? ` (Track: ${ann.track_id})` : ''}</small>
                            </div>
                            <button class="delete-btn" onclick="deleteAnnotation('${ann.id}')">
                                Delete
                            </button>
                        </div>
                    `).join('');
                }
                
                // Redraw canvas
                drawAnnotations();
            }

            function drawAnnotations() {
                const canvas = document.getElementById('annotation-canvas');
                const ctx = canvas.getContext('2d');
                
                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw existing annotations for current frame
                const currentFrame = getCurrentFrame();
                const frameAnnotations = annotations.filter(ann => ann.frame_number === currentFrame);
                
                frameAnnotations.forEach((ann, index) => {
                    const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
                    ctx.strokeStyle = colors[index % colors.length];
                    ctx.fillStyle = colors[index % colors.length] + '30';
                    ctx.lineWidth = 2;
                    
                    if (ann.annotation_type === 'bbox') {
                        const g = ann.geometry;
                        ctx.strokeRect(g.x, g.y, g.width, g.height);
                        ctx.fillRect(g.x, g.y, g.width, g.height);
                        
                        // Draw label
                        ctx.fillStyle = colors[index % colors.length];
                        ctx.font = '14px Arial';
                        ctx.fillText(ann.class_name, g.x, g.y - 5);
                    } else if (ann.annotation_type === 'polygon') {
                        const points = ann.geometry.points;
                        if (points.length > 2) {
                            ctx.beginPath();
                            ctx.moveTo(points[0].x, points[0].y);
                            for (let i = 1; i < points.length; i++) {
                                ctx.lineTo(points[i].x, points[i].y);
                            }
                            ctx.closePath();
                            ctx.stroke();
                            ctx.fill();
                            
                            // Draw label
                            ctx.fillStyle = colors[index % colors.length];
                            ctx.font = '14px Arial';
                            ctx.fillText(ann.class_name, points[0].x, points[0].y - 5);
                        }
                    }
                });
                
                // Draw current polygon being created
                if (currentAnnotationTool === 'polygon' && polygonPoints.length > 0) {
                    ctx.strokeStyle = '#ff0000';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
                    for (let i = 1; i < polygonPoints.length; i++) {
                        ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
                    }
                    ctx.stroke();
                    
                    // Draw points
                    polygonPoints.forEach(point => {
                        ctx.fillStyle = '#ff0000';
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
                        ctx.fill();
                    });
                }
            }

            async function deleteAnnotation(annotationId) {
                try {
                    const response = await fetch(`/api/annotations/${annotationId}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        annotations = annotations.filter(ann => ann.id !== annotationId);
                        loadFrameAnnotations();
                        showStatus('Annotation deleted!', 'success');
                    } else {
                        throw new Error('Failed to delete annotation');
                    }
                } catch (error) {
                    showStatus('Failed to delete annotation: ' + error.message, 'error');
                }
            }

            async function clearAnnotations() {
                const currentFrame = getCurrentFrame();
                const frameAnnotations = annotations.filter(ann => ann.frame_number === currentFrame);
                
                if (frameAnnotations.length === 0) {
                    showStatus('No annotations to clear', 'error');
                    return;
                }
                
                if (!confirm(`Delete all ${frameAnnotations.length} annotations from this frame?`)) {
                    return;
                }
                
                try {
                    for (const ann of frameAnnotations) {
                        await fetch(`/api/annotations/${ann.id}`, { method: 'DELETE' });
                    }
                    
                    annotations = annotations.filter(ann => ann.frame_number !== currentFrame);
                    loadFrameAnnotations();
                    showStatus('Frame annotations cleared!', 'success');
                } catch (error) {
                    showStatus('Failed to clear annotations: ' + error.message, 'error');
                }
            }

            async function exportAnnotations(format) {
                if (!currentVideo) {
                    showStatus('No video loaded', 'error');
                    return;
                }
                
                try {
                    const response = await fetch(`/api/export/${currentVideo.id}?format=${format}`);
                    const data = await response.json();
                    
                    // Create and download file
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${currentVideo.filename.split('.')[0]}_annotations_${format}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    showStatus(`${format.toUpperCase()} export completed!`, 'success');
                } catch (error) {
                    showStatus('Export failed: ' + error.message, 'error');
                }
            }

            function showStatus(message, type) {
                const statusDiv = document.getElementById('status-message');
                statusDiv.textContent = message;
                statusDiv.className = `status-message ${type} show`;
                
                setTimeout(() => {
                    statusDiv.classList.remove('show');
                }, 3000);
            }
            
            // Interpolation and tracking functions
            function setKeyframe(type) {
                const videoPlayer = document.getElementById('video-player');
                if (!currentVideo || !videoPlayer) {
                    showStatus('No video loaded', 'error');
                    return;
                }
                
                const currentFrame = Math.floor(videoPlayer.currentTime * currentVideo.fps);
                const trackId = document.getElementById('track-id').value;
                
                if (!trackId) {
                    showStatus('Please set a Track ID first', 'error');
                    return;
                }
                
                if (type === 'start') {
                    document.getElementById('start-frame').value = currentFrame;
                    document.getElementById('interp-track-id').value = trackId;
                    showStatus(`Start keyframe set to frame ${currentFrame}`, 'success');
                } else if (type === 'end') {
                    document.getElementById('end-frame').value = currentFrame;
                    document.getElementById('interp-track-id').value = trackId;
                    showStatus(`End keyframe set to frame ${currentFrame}`, 'success');
                }
                
                updateInterpolationStatus();
            }
            
            function updateInterpolationStatus() {
                const startFrame = parseInt(document.getElementById('start-frame').value) || 0;
                const endFrame = parseInt(document.getElementById('end-frame').value) || 0;
                const trackId = document.getElementById('interp-track-id').value;
                const statusDiv = document.getElementById('interpolation-status');
                
                if (!trackId) {
                    statusDiv.textContent = 'Set a track ID to enable interpolation';
                    return;
                }
                
                if (startFrame >= endFrame) {
                    statusDiv.textContent = 'Start frame must be before end frame';
                    statusDiv.style.color = 'var(--error-color)';
                    return;
                }
                
                const frameCount = endFrame - startFrame - 1;
                if (frameCount > 0) {
                    statusDiv.textContent = `Will create ${frameCount} interpolated annotations`;
                    statusDiv.style.color = 'var(--success-color)';
                } else {
                    statusDiv.textContent = 'Need at least 2 frames gap for interpolation';
                    statusDiv.style.color = 'var(--warning-color)';
                }
            }
            
            async function performInterpolation() {
                const startFrame = parseInt(document.getElementById('start-frame').value);
                const endFrame = parseInt(document.getElementById('end-frame').value);
                const trackId = parseInt(document.getElementById('interp-track-id').value);
                
                if (!trackId || isNaN(trackId)) {
                    showStatus('Please enter a valid Track ID', 'error');
                    return;
                }
                
                if (isNaN(startFrame) || isNaN(endFrame)) {
                    showStatus('Please enter valid start and end frames', 'error');
                    return;
                }
                
                if (startFrame >= endFrame) {
                    showStatus('Start frame must be before end frame', 'error');
                    return;
                }
                
                if (endFrame - startFrame < 2) {
                    showStatus('Need at least 2 frames between keyframes', 'error');
                    return;
                }
                
                try {
                    showStatus('Generating interpolated annotations...', 'success');
                    
                    const response = await fetch('/api/annotations/interpolate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            start_frame: startFrame,
                            end_frame: endFrame,
                            track_id: trackId,
                            interpolation_type: 'linear'
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        showStatus(`Successfully created ${result.interpolated_count} interpolated annotations!`, 'success');
                        
                        // Update the status display
                        document.getElementById('interpolation-status').textContent = 
                            `✅ Created ${result.interpolated_count} annotations for track ${trackId}`;
                        document.getElementById('interpolation-status').style.color = 'var(--success-color)';
                        
                        // Refresh annotations for current frame
                        loadFrameAnnotations();
                    } else {
                        const error = await response.json();
                        throw new Error(error.detail || 'Interpolation failed');
                    }
                } catch (error) {
                    console.error('Interpolation error:', error);
                    showStatus(`Interpolation failed: ${error.message}`, 'error');
                }
            }
            
            // Add event listeners for interpolation inputs
            document.addEventListener('DOMContentLoaded', function() {
                const startFrameInput = document.getElementById('start-frame');
                const endFrameInput = document.getElementById('end-frame');
                const trackIdInput = document.getElementById('interp-track-id');
                
                if (startFrameInput) startFrameInput.addEventListener('input', updateInterpolationStatus);
                if (endFrameInput) endFrameInput.addEventListener('input', updateInterpolationStatus);
                if (trackIdInput) trackIdInput.addEventListener('input', updateInterpolationStatus);
            });
        </script>
    </body>
    </html>
    '''

# Run the application
if __name__ == "__main__":
    print(f"🎥 {settings.app_name} v{settings.app_version} Starting...")
    print("📊 Features: Bounding Box & Polygon Annotation, Video Tracking, COCO/YOLO Export")
    print(f"🌐 Access the application at: http://{settings.host}:{settings.port}")
    print(f"📁 Videos will be stored in: ./{settings.upload_dir}/")
    print(f"🗄️  Database: {settings.database_url}")
    if settings.log_file:
        print(f"📝 Logs will be written to: {settings.log_file}")
    print(f"🔧 Debug mode: {settings.debug}")
    print(f"📏 Max file size: {settings.max_file_size_mb}MB")
    print("\n" + "="*60)
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Configuration loaded from environment and .env file")
    
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port, 
        log_level=settings.log_level.lower(),
        reload=settings.auto_reload,
        workers=settings.workers if not settings.debug else 1
    )

