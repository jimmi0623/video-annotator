# Video Annotation Tool

A professional web-based video annotation tool built with FastAPI and modern web technologies. Designed for creating precise bounding box and polygon annotations on video frames with export capabilities to industry-standard formats.

![Video Annotation Tool](https://img.shields.io/badge/FastAPI-0.104.1-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### Core Functionality
- **Video Upload & Processing**: Drag-and-drop video upload with automatic metadata extraction
- **Multiple Annotation Types**: 
  - Bounding boxes for object detection
  - Polygons for precise shape annotation
- **Frame-by-Frame Navigation**: Precise video control with timeline scrubbing
- **Real-time Visualization**: Live annotation overlay on video frames
- **Track ID Support**: Object tracking across frames

### Export Capabilities
- **COCO Format**: Industry-standard format for object detection datasets
- **YOLO Format**: Popular format for YOLO model training
- **JSON Export**: Complete annotation data with metadata

### User Experience
- **Modern UI**: Professional gradient design with responsive layout
- **Keyboard Shortcuts**: Space (play/pause), Arrow keys (frame navigation)
- **Visual Feedback**: Status notifications and progress indicators
- **Drag & Drop**: Intuitive file upload experience

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/video-annotator.git
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

## 🎯 Usage

### Uploading Videos
1. Drag and drop a video file onto the upload area, or click to select
2. Supported formats: MP4, AVI, MOV, MKV
3. Video metadata is automatically extracted and stored

### Creating Annotations

#### Bounding Boxes
1. Select "Bounding Box" tool
2. Click and drag on the video frame to create a rectangle
3. Enter class name (e.g., "person", "car")
4. Optionally add track ID for object tracking

#### Polygons
1. Select "Polygon" tool
2. Click points around the object to create vertices
3. Double-click to close the polygon
4. Enter class name and optional track ID

### Navigation
- **Play/Pause**: Space bar or play button
- **Frame Navigation**: Arrow keys or frame buttons
- **Timeline**: Click or drag to jump to specific frames
- **Current Frame Info**: Displayed in the control panel

### Exporting Annotations
1. Complete your annotations
2. Choose export format (COCO or YOLO)
3. Download the generated JSON file
4. Use in your machine learning pipeline

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
- `POST /api/upload-video` - Upload a new video
- `GET /api/videos` - List all uploaded videos

### Annotation Endpoints
- `POST /api/annotations` - Create new annotation
- `GET /api/annotations/{video_id}` - Get annotations for a video
- `DELETE /api/annotations/{annotation_id}` - Delete an annotation

### Export Endpoints
- `GET /api/export/{video_id}?format=coco` - Export COCO format
- `GET /api/export/{video_id}?format=yolo` - Export YOLO format

## 🛠️ Development

### Project Structure
```
video-annotator/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules
├── uploads/            # Video file storage (ignored)
├── static/             # Static assets (ignored)
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
# Run the application
python main.py

# Test video upload via curl
curl -X POST "http://localhost:8000/api/upload-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_video.mp4"
```

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

## 🎯 Roadmap

### Version 1.1 (Coming Soon)
- [ ] User authentication and authorization
- [ ] Batch annotation tools
- [ ] Video compression and optimization
- [ ] Advanced export options
- [ ] Annotation validation tools

### Version 1.2
- [ ] Real-time collaboration
- [ ] Machine learning integration
- [ ] Advanced polygon editing
- [ ] Custom class management
- [ ] Annotation statistics and analytics

### Version 2.0
- [ ] Multi-user support
- [ ] Cloud storage integration
- [ ] Advanced video processing
- [ ] Mobile responsive design
- [ ] API rate limiting and security

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- OpenCV for video processing capabilities
- The open-source community for inspiration and tools

## 📧 Contact

**James Rono** - [Your Email] - [Your LinkedIn]

Project Link: [https://github.com/yourusername/video-annotator](https://github.com/yourusername/video-annotator)

---

⭐ **Star this repository if you find it helpful!**