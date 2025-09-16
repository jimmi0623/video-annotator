# JR Video Annotation Tool 🎬

**A Professional-Grade Video Annotation Platform** built with FastAPI and modern web technologies. Create, track, and manage video annotations with advanced AI-ready export capabilities.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](#)


## 🚀 Key Features

### 🎯 **Advanced Annotation System**
- **Bounding Box Annotation** with precision controls
- **Polygon Annotation** for complex shapes  
- **✨ Object Tracking & Interpolation** - Track objects across frames with automatic interpolation
- **Track ID Management** for consistent object identification
- **Frame-by-Frame Navigation** with precise controls

### 🎬 **Professional Video Management**
- **Multi-Format Support** (MP4, AVI, MOV, MKV, WebM)
- **🖼️ Smart Video Thumbnails** - Auto-generated preview thumbnails with hover effects
- **📊 Video Statistics** - Duration, FPS, resolution, file size display
- **Drag & Drop Upload** with real-time progress tracking
- **Video List Management** with rich metadata

### ⚡ **Intelligent Workflows**
- **🎯 Linear Interpolation** between keyframes for object tracking
- **Automatic Annotation Generation** for video sequences  
- **Collision Detection** prevents annotation conflicts
- **Real-time Validation** and status feedback
- **Professional Keyboard Shortcuts** for efficient workflow

### 🎨 **Modern User Experience** 
- **🌓 Dark/Light Theme** with smooth transitions and localStorage persistence
- **Responsive Design** optimized for all screen sizes
- **Professional Toolbar** with advanced playback controls
- **Fullscreen Mode** for detailed annotation work
- **Visual Status Messages** with contextual feedback

### 📤 **Industry-Standard Exports**
- **COCO Format** for computer vision research
- **YOLO Format** for object detection training
- **Custom JSON** schemas with complete metadata
- **Pascal VOC XML** (coming soon)
- **TensorFlow Records** (coming soon)

### 🔧 **Production-Grade Infrastructure**
- **SQLite Database** with optimized indexes and foreign keys
- **Advanced Logging** with rotation and structured output
- **Resource Management** with automatic cleanup and monitoring
- **Comprehensive Error Handling** and recovery systems
- **Background Tasks** for performance optimization
- **Security Features** with input validation and sanitization

## 🆕 Latest Updates (v1.2.0)

### 🎯 **Object Tracking & Interpolation System**
- **Linear interpolation** between bounding boxes and polygons
- **One-click keyframe setting** for start/end frames
- **Automatic annotation generation** across frame ranges
- **Smart collision detection** to prevent overwrites
- **Track ID validation** and consistency checking

### 🖼️ **Smart Thumbnail System** 
- **Automatic thumbnail generation** using OpenCV and PIL
- **Multiple preview thumbnails** for video segments
- **Hover effects** with thumbnail previews
- **Fallback thumbnail display** with graceful error handling
- **Base64 embedded thumbnails** for fast loading

### 🎨 **Enhanced User Interface**
- **Professional page header** with branding
- **Modern video list layout** with rich statistics
- **Interpolation control panel** with visual feedback
- **Improved toolbar** with speed controls and navigation
- **Better error messages** and user guidance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jimmi0623/video-annotator.git
   cd video-annotator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

## 🎯 Usage Guide

### 📹 **Video Management**
1. **Upload**: Drag and drop videos or click to select
2. **Preview**: View auto-generated thumbnails and metadata
3. **Load**: Click thumbnails or "Load Video" to start annotating
4. **Delete**: Remove videos with confirmation dialog

### 🎯 **Creating Annotations**

#### Bounding Boxes
1. Select "Bounding Box" tool
2. Click and drag on the video frame
3. Enter class name and Track ID
4. Annotation saves automatically

#### Polygons  
1. Select "Polygon" tool
2. Click points to create vertices
3. Double-click to close polygon
4. Enter class name and Track ID

### ⚡ **Object Tracking Workflow**
1. **Create Start Keyframe**: Annotate object on starting frame with Track ID
2. **Navigate to End Frame**: Use timeline to jump to end position  
3. **Create End Keyframe**: Annotate same object with same Track ID
4. **Set Interpolation Points**: Click "📍 Set Start/End Keyframe" buttons
5. **Generate Interpolation**: Click "✨ Interpolate Between Keyframes"
6. **Review Results**: System creates smooth transitions automatically

### ⌨️ **Keyboard Shortcuts**
- **Space**: Play/Pause video
- **←/→**: Navigate frames  
- **F**: Toggle fullscreen
- **D**: Toggle dark/light theme
- **1/2**: Switch annotation tools
- **+/-**: Adjust playback speed
- **R**: Reset zoom

### 📤 **Exporting Annotations**
1. Complete annotations for your video
2. Choose format: COCO or YOLO
3. Download generated files
4. Use in your ML pipeline

## 🏗️ Architecture

### Backend (FastAPI)
- **API Endpoints**: RESTful API for video and annotation management
- **Database**: SQLite for storing video metadata and annotations
- **File Handling**: Secure video upload and storage
- **Export Logic**: COCO and YOLO format conversion

### Frontend (Vanilla JS)
- **Video Player**: HTML5 video with custom controls
- **Canvas Overlay**: HTML5 canvas for annotation drawing
- **Real-time Updates**: Dynamic annotation management
- **Responsive Design**: CSS Grid and Flexbox layout

### Database Schema
```sql
-- Videos table
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    duration REAL NOT NULL,
    fps REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    upload_date TEXT NOT NULL
);

-- Annotations table
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    annotation_type TEXT NOT NULL,
    class_name TEXT NOT NULL,
    geometry TEXT NOT NULL,
    track_id INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos (id)
);
```

## 🔧 API Reference

### Video Endpoints
- `POST /api/upload-video` - Upload with thumbnail generation
- `GET /api/videos` - List with thumbnails and metadata
- `DELETE /api/videos/{video_id}` - Delete with cleanup
- `GET /api/videos/{video_id}/thumbnail` - Generate custom thumbnails
- `POST /api/videos/{video_id}/regenerate-thumbnails` - Regenerate thumbnails

### Annotation Endpoints
- `POST /api/annotations` - Create new annotation
- `GET /api/annotations/{video_id}` - Get annotations for video
- `DELETE /api/annotations/{annotation_id}` - Delete annotation
- `POST /api/annotations/interpolate` - Generate interpolated annotations

### Export & Stats
- `GET /api/export/{video_id}?format=coco` - COCO export
- `GET /api/export/{video_id}?format=yolo` - YOLO export
- `GET /api/videos/{video_id}/stats` - Video statistics
- `GET /api/resources` - System resource usage

## 🛠️ Development

### Project Structure
```
video-annotator/
├── main.py              # Main FastAPI application
├── config.py            # Configuration management  
├── resource_manager.py  # Resource monitoring
├── requirements.txt     # Dependencies
├── README.md           # Documentation
├── .env.example        # Environment template
├── uploads/            # Video storage (ignored)
└── annotations.db      # SQLite database (ignored)
```

### Running in Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Run the application (default port 8000)
python main.py

# Run on different port
$env:PORT="8080"; python main.py  # Windows PowerShell
export PORT=8080 && python main.py  # Linux/Mac

# Test video upload via curl
curl -X POST "http://localhost:8080/api/upload-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_video.mp4"

# Test API endpoints
curl http://localhost:8080/api/stats      # Database statistics
curl http://localhost:8080/api/resources  # Resource usage
curl http://localhost:8080/api/config     # Public configuration
```

### Dependencies
```
fastapi>=0.100.0           # Web framework
uvicorn[standard]>=0.20.0  # ASGI server
python-multipart>=0.0.6    # File upload support
opencv-python>=4.7.0       # Video processing
Pillow>=11.0.0             # Image processing for thumbnails
pydantic>=2.0.0            # Data validation
pydantic-settings>=2.0.0   # Configuration management
python-json-logger>=2.0.0  # Structured logging
aiofiles>=23.1.0           # Async file operations
python-dotenv>=1.0.0       # Environment variables
psutil>=5.9.0              # System monitoring
```

### Tested Environments
- ✅ **Windows 11** with Python 3.13.7
- ✅ **Memory optimization** for large files (64MB+ tested)
- ✅ **Database operations** with SQLite
- ✅ **Video streaming** with HTTP 206 partial content
- ✅ **All API endpoints** functional
- ✅ **Background tasks** and resource management

## 🚀 Deployment

### Docker (Coming Soon)
```dockerfile
# Dockerfile will be added in future updates
FROM python:3.9-slim
# ... deployment configuration
```

### Production Considerations
- Set up proper CORS origins
- Use PostgreSQL for production database
- Implement file storage service (AWS S3, etc.)
- Add authentication and authorization
- Set up monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🛫 Roadmap

### 🔄 Current (v1.2.0) - COMPLETED ✅
- ✅ Advanced object tracking and interpolation
- ✅ Smart thumbnail generation system
- ✅ Enhanced UI with dark/light themes  
- ✅ Professional toolbar and controls
- ✅ Database schema improvements

### 📊 Next Release (v1.3.0)
- [ ] **Analytics Dashboard** - Annotation statistics and progress tracking
- [ ] **Advanced Annotation Tools** - Keypoint and line annotations
- [ ] **Pascal VOC XML Export** - Additional export format
- [ ] **Batch Operations** - Multi-frame annotation tools
- [ ] **Annotation Validation** - Quality scoring system

### 🎯 Future (v1.4.0+)
- [ ] **Video Preprocessing** - Trimming, format conversion
- [ ] **Collaborative Features** - Multi-user annotation
- [ ] **AI-Assisted Annotation** - Auto-tracking integration
- [ ] **Cloud Storage** - S3/GCS integration
- [ ] **Mobile Support** - Responsive mobile interface

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- OpenCV for video processing capabilities
- The open-source community for inspiration and tools

## 📧 Contact

**James Rono** - jimmironno@gmail.com - https://www.linkedin.com/in/mijj0623

Project Link: [https://github.com/jimmi0623/video-annotator](https://github.com/jimmi0623/video-annotator)

---

⭐ **Star this repository if you find it helpful!**
