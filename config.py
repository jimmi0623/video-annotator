"""
Configuration management for the Video Annotation Tool.
Supports environment variables, configuration files, and defaults.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="Video Annotation Tool", env="APP_NAME")
    app_version: str = Field(default="1.2.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    allowed_origins: List[str] = Field(
        default=["http://localhost:8000", "http://127.0.0.1:8000"], 
        env="ALLOWED_ORIGINS"
    )
    max_file_size_mb: int = Field(default=500, env="MAX_FILE_SIZE_MB")
    
    # Database settings
    database_url: str = Field(default="sqlite:///annotations.db", env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_timeout: float = Field(default=30.0, env="DATABASE_TIMEOUT")
    
    # File storage settings
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    static_dir: str = Field(default="static", env="STATIC_DIR")
    temp_dir: str = Field(default="temp", env="TEMP_DIR")
    
    # Video processing settings
    max_video_width: int = Field(default=4096, env="MAX_VIDEO_WIDTH")
    max_video_height: int = Field(default=4096, env="MAX_VIDEO_HEIGHT")
    max_frame_count: int = Field(default=999999, env="MAX_FRAME_COUNT")
    supported_formats: List[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm"], 
        env="SUPPORTED_FORMATS"
    )
    
    # Annotation settings
    max_class_name_length: int = Field(default=50, env="MAX_CLASS_NAME_LENGTH")
    max_filename_length: int = Field(default=255, env="MAX_FILENAME_LENGTH")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_max_size: int = Field(default=10 * 1024 * 1024, env="LOG_MAX_SIZE")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Performance settings
    cleanup_interval_hours: int = Field(default=24, env="CLEANUP_INTERVAL_HOURS")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    cache_size_mb: int = Field(default=64, env="CACHE_SIZE_MB")
    
    # Development settings
    auto_reload: bool = Field(default=False, env="AUTO_RELOAD")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    
    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_origins(cls, v):
        """Parse comma-separated origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    @field_validator('supported_formats', mode='before')
    @classmethod
    def parse_formats(cls, v):
        """Parse comma-separated formats from environment variable."""
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(',') if fmt.strip()]
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level. Must be one of: {valid_levels}')
        return v.upper()
    
    @property
    def max_file_size(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def cache_size_bytes(self) -> int:
        """Get cache size in bytes."""
        return self.cache_size_mb * 1024 * 1024
    
    def ensure_directories(self):
        """Create necessary directories."""
        directories = [self.upload_dir, self.static_dir]
        if self.temp_dir:
            directories.append(self.temp_dir)
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


def setup_logging(settings: Settings) -> logging.Logger:
    """Setup logging configuration."""
    import logging.handlers
    
    # Create logs directory if logging to file
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if settings.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file,
            maxBytes=settings.log_max_size,
            backupCount=settings.log_backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Get application logger
    logger = logging.getLogger("video_annotator")
    
    return logger


def get_settings() -> Settings:
    """Get application settings singleton."""
    if not hasattr(get_settings, '_settings'):
        get_settings._settings = Settings()
        # Ensure directories exist
        get_settings._settings.ensure_directories()
    return get_settings._settings


# Global settings instance
settings = get_settings()

# Application logger
logger = setup_logging(settings)