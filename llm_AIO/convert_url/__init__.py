"""
文件上传与URL转换工具 (FastAPI版本)

简洁高效的文件上传工具，可轻松集成到FastAPI项目。
"""

from .core import FileConverter, File, Base, DatasetMetadata, TrainingJob

# 向后兼容的别名
def VideoConverter(session, **kwargs):
    return FileConverter(session, file_category='video', **kwargs)

def AudioConverter(session, **kwargs):
    return FileConverter(session, file_category='audio', **kwargs)

def ImageConverter(session, **kwargs):
    return FileConverter(session, file_category='image', **kwargs)

Video = Audio = Image = File

__version__ = '2.0.0'
__all__ = ['FileConverter', 'File', 'Base', 'VideoConverter', 'Video', 'AudioConverter', 'Audio', 'ImageConverter', 'Image']