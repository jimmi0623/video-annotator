from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import sqlite3
from contextlib import contextmanager

# Pydantic Models
class AnnotationCreate(BaseModel):
    video_id: str
    frame_number: int
    annotation_type: str  # 'bbox', 'polygon'
    class_name: str
    geometry: dict
    track_id: Optional[int] = None

class AnnotationUpdate(BaseModel):
    geometry: dict
    class_name: Optional[str] = None

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
    conn = sqlite3.connect('annotations.db')
    cursor = conn.cursor()
    
    # Videos table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            duration REAL NOT NULL,
            fps REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            frame_count INTEGER NOT NULL,
            upload_date TEXT NOT NULL
        )
    ''')
    
    # Annotations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            annotation_type TEXT NOT NULL,
            class_name TEXT NOT NULL,
            geometry TEXT NOT NULL,
            track_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    conn = sqlite3.connect('annotations.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Initialize FastAPI app
app = FastAPI(title="Video Annotation Tool", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize database
init_database()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main application page"""
    return get_html_content()

@app.post("/api/upload-video", response_model=VideoInfo)
async def upload_video(file: UploadFile = File(...)):
    """Upload and process a video file"""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    
    # Generate unique ID and save file
    video_id = str(uuid.uuid4())
    file_path = f"uploads/{video_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Extract video metadata using OpenCV
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    # Save to database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO videos (id, filename, file_path, duration, fps, width, height, frame_count, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, file.filename, file_path, duration, fps, width, height, frame_count, datetime.now().isoformat()))
        conn.commit()
    
    return VideoInfo(
        id=video_id,
        filename=file.filename,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        upload_date=datetime.now().isoformat()
    )

@app.get("/api/videos", response_model=List[VideoInfo])
async def get_videos():
    """Get all uploaded videos"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY upload_date DESC")
        videos = cursor.fetchall()
        
        return [VideoInfo(
            id=video['id'],
            filename=video['filename'],
            duration=video['duration'],
            fps=video['fps'],
            width=video['width'],
            height=video['height'],
            frame_count=video['frame_count'],
            upload_date=video['upload_date']
        ) for video in videos]

@app.post("/api/annotations")
async def create_annotation(annotation: AnnotationCreate):
    """Create a new annotation"""
    annotation_id = str(uuid.uuid4())
    
    with get_db() as conn:
        cursor = conn.cursor()
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
    
    return {"id": annotation_id, "status": "created"}

@app.get("/api/annotations/{video_id}")
async def get_annotations(video_id: str, frame_number: Optional[int] = None):
    """Get annotations for a video, optionally filtered by frame"""
    with get_db() as conn:
        cursor = conn.cursor()
        
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
            result.append({
                "id": ann['id'],
                "video_id": ann['video_id'],
                "frame_number": ann['frame_number'],
                "annotation_type": ann['annotation_type'],
                "class_name": ann['class_name'],
                "geometry": json.loads(ann['geometry']),
                "track_id": ann['track_id'],
                "created_at": ann['created_at']
            })
        
        return result

@app.delete("/api/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Annotation not found")
    
    return {"status": "deleted"}

@app.get("/api/export/{video_id}")
async def export_annotations(video_id: str, format: str = "coco"):
    """Export annotations in various formats"""
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
        
        if format.lower() == "coco":
            return export_coco_format(video, annotations)
        elif format.lower() == "yolo":
            return export_yolo_format(video, annotations)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

def export_coco_format(video, annotations):
    """Export in COCO format"""
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
        geometry = json.loads(ann['geometry'])
        
        if ann['annotation_type'] == 'bbox':
            bbox = [geometry['x'], geometry['y'], geometry['width'], geometry['height']]
            area = geometry['width'] * geometry['height']
        else:
            # For polygon, calculate bbox and area
            points = geometry['points']
            xs = [p['x'] for p in points]
            ys = [p['y'] for p in points]
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
    
    return coco_data

def export_yolo_format(video, annotations):
    """Export in YOLO format"""
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
            geometry = json.loads(ann['geometry'])
            class_id = class_name_to_id[ann['class_name']]
            
            if ann['annotation_type'] == 'bbox':
                # Convert to YOLO format (normalized center coordinates and dimensions)
                x_center = (geometry['x'] + geometry['width'] / 2) / video['width']
                y_center = (geometry['y'] + geometry['height'] / 2) / video['height']
                width = geometry['width'] / video['width']
                height = geometry['height'] / video['height']
                
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        yolo_data["frames"][f"frame_{frame_num:06d}"] = yolo_annotations
    
    return yolo_data

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
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            
            .header p {
                color: #666;
                font-size: 1.1em;
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 350px;
                gap: 30px;
                margin-bottom: 30px;
            }
            
            .video-section {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .controls-panel {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background: rgba(102, 126, 234, 0.05);
                transform: translateY(-2px);
            }
            
            .upload-area.dragover {
                border-color: #764ba2;
                background: rgba(102, 126, 234, 0.1);
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
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
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
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
            }
            
            .video-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                margin-bottom: 15px;
                transition: all 0.3s ease;
            }
            
            .video-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }
            
            .video-item:last-child {
                margin-bottom: 0;
            }
            
            .video-info h3 {
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .video-meta {
                color: #666;
                font-size: 14px;
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
        <div class="container">
            <div class="header">
                <h1>Video Annotation Tool</h1>
                <p>Professional video annotation made simple and accessible</p>
            </div>

            <div class="video-list" id="video-list" style="display: none;">
                <h2>Uploaded Videos</h2>
                <div id="videos-container"></div>
            </div>

            <div class="upload-area" id="upload-area">
                <h3>Upload Video</h3>
                <p>Drag and drop your video file here or click to select</p>
                <input type="file" id="video-input" accept="video/*" style="display: none;">
                <button type="button" onclick="document.getElementById('video-input').click()">
                    Choose File
                </button>
            </div>

            <div class="main-content" id="main-content" style="display: none;">
                <div class="video-section">
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

            // Initialize the application
            document.addEventListener('DOMContentLoaded', function() {
                setupEventListeners();
                loadVideos();
            });

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
                const formData = new FormData();
                formData.append('file', file);

                showStatus('Uploading video...', 'success');

                try {
                    const response = await fetch('/api/upload-video', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const videoInfo = await response.json();
                        showStatus('Video uploaded successfully!', 'success');
                        loadVideo(videoInfo);
                        loadVideos();
                    } else {
                        throw new Error('Upload failed');
                    }
                } catch (error) {
                    showStatus('Upload failed: ' + error.message, 'error');
                }
            }

            async function loadVideos() {
                try {
                    const response = await fetch('/api/videos');
                    const videos = await response.json();
                    
                    const container = document.getElementById('videos-container');
                    const videoList = document.getElementById('video-list');

                    if (videos.length === 0) {
                        videoList.style.display = 'none';
                        return;
                    }

                    videoList.style.display = 'block';
                    container.innerHTML = '';

                    videos.forEach(video => {
                        const videoItem = document.createElement('div');
                        videoItem.className = 'video-item';
                        videoItem.innerHTML = `
                            <div class="video-info">
                                <h3>${video.filename}</h3>
                                <div class="video-meta">
                                    Duration: ${formatTime(video.duration)} | 
                                    FPS: ${video.fps.toFixed(1)} | 
                                    Resolution: ${video.width}×${video.height}
                                </div>
                            </div>
                            <button onclick="loadVideo('${video.id}')">Load Video</button>
                        `;
                        container.appendChild(videoItem);
                    });
                } catch (error) {
                    showStatus('Failed to load videos', 'error');
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
                    loadFrameAnnotations();
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
        </script>
    </body>
    </html>
    '''

# Run the application
if __name__ == "__main__":
    print("🎥 Video Annotation Tool by James Rono Starting...")
    print("📊 Features: Bounding Box & Polygon Annotation, Video Tracking, COCO/YOLO Export")
    print("🌐 Access the application at: http://localhost:8000")
    print("📁 Videos will be stored in: ./uploads/")
    print("🗄️  Database: ./annotations.db")
    print("\n" + "="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

