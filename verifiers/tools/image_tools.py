import base64
import io
import json
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2


def crop_image(image_data: str, x: int, y: int, width: int, height: int) -> str:
    """
    Crop an image to the specified region.
    
    Args:
        image_data: Base64 encoded image string
        x: Left coordinate of crop region
        y: Top coordinate of crop region  
        width: Width of crop region
        height: Height of crop region
    
    Returns:
        Base64 encoded cropped image
        
    Examples:
        crop_image(image_data, 100, 100, 200, 200)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Crop image
        cropped = image.crop((x, y, x + width, y + height))
        
        # Encode back to base64
        buffer = io.BytesIO()
        cropped.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error cropping image: {str(e)}"


def zoom_image(image_data: str, zoom_factor: float, center_x: Optional[int] = None, center_y: Optional[int] = None) -> str:
    """
    Zoom into an image by the specified factor.
    
    Args:
        image_data: Base64 encoded image string
        zoom_factor: Factor to zoom by (>1 zooms in, <1 zooms out)
        center_x: X coordinate of zoom center (default: image center)
        center_y: Y coordinate of zoom center (default: image center)
    
    Returns:
        Base64 encoded zoomed image
        
    Examples:
        zoom_image(image_data, 2.0)
        zoom_image(image_data, 1.5, 300, 200)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        width, height = image.size
        center_x = center_x if center_x is not None else width // 2
        center_y = center_y if center_y is not None else height // 2
        
        # Calculate new dimensions
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        
        # Calculate crop box for zoom
        left = max(0, center_x - new_width // 2)
        top = max(0, center_y - new_height // 2)
        right = min(width, left + new_width)
        bottom = min(height, top + new_height)
        
        # Crop and resize
        cropped = image.crop((left, top, right, bottom))
        zoomed = cropped.resize((width, height), Image.Resampling.LANCZOS)
        
        # Encode back to base64
        buffer = io.BytesIO()
        zoomed.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error zooming image: {str(e)}"


def rotate_image(image_data: str, angle: float, expand: bool = True) -> str:
    """
    Rotate an image by the specified angle.
    
    Args:
        image_data: Base64 encoded image string
        angle: Rotation angle in degrees (positive = counterclockwise)
        expand: Whether to expand the image to fit the rotated content
    
    Returns:
        Base64 encoded rotated image
        
    Examples:
        rotate_image(image_data, 90)
        rotate_image(image_data, 45, expand=False)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Rotate image
        rotated = image.rotate(angle, expand=expand, fillcolor='white')
        
        # Encode back to base64
        buffer = io.BytesIO()
        rotated.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error rotating image: {str(e)}"


def flip_image(image_data: str, direction: str) -> str:
    """
    Flip an image horizontally or vertically.
    
    Args:
        image_data: Base64 encoded image string
        direction: 'horizontal' or 'vertical'
    
    Returns:
        Base64 encoded flipped image
        
    Examples:
        flip_image(image_data, 'horizontal')
        flip_image(image_data, 'vertical')
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Flip image
        if direction.lower() == 'horizontal':
            flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif direction.lower() == 'vertical':
            flipped = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            return f"Error: direction must be 'horizontal' or 'vertical', got '{direction}'"
        
        # Encode back to base64
        buffer = io.BytesIO()
        flipped.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error flipping image: {str(e)}"


def adjust_brightness(image_data: str, factor: float) -> str:
    """
    Adjust the brightness of an image.
    
    Args:
        image_data: Base64 encoded image string
        factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
    
    Returns:
        Base64 encoded brightness-adjusted image
        
    Examples:
        adjust_brightness(image_data, 1.5)
        adjust_brightness(image_data, 0.7)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        enhanced = enhancer.enhance(factor)
        
        # Encode back to base64
        buffer = io.BytesIO()
        enhanced.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error adjusting brightness: {str(e)}"


def adjust_contrast(image_data: str, factor: float) -> str:
    """
    Adjust the contrast of an image.
    
    Args:
        image_data: Base64 encoded image string
        factor: Contrast factor (1.0 = no change, >1.0 = higher contrast, <1.0 = lower contrast)
    
    Returns:
        Base64 encoded contrast-adjusted image
        
    Examples:
        adjust_contrast(image_data, 1.3)
        adjust_contrast(image_data, 0.8)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(factor)
        
        # Encode back to base64
        buffer = io.BytesIO()
        enhanced.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error adjusting contrast: {str(e)}"


def apply_blur(image_data: str, radius: float) -> str:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image_data: Base64 encoded image string
        radius: Blur radius (higher = more blur)
    
    Returns:
        Base64 encoded blurred image
        
    Examples:
        apply_blur(image_data, 2.0)
        apply_blur(image_data, 5.0)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply blur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Encode back to base64
        buffer = io.BytesIO()
        blurred.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error applying blur: {str(e)}"


def apply_sharpen(image_data: str, factor: float = 1.0) -> str:
    """
    Apply sharpening to an image.
    
    Args:
        image_data: Base64 encoded image string
        factor: Sharpening factor (1.0 = default sharpening)
    
    Returns:
        Base64 encoded sharpened image
        
    Examples:
        apply_sharpen(image_data)
        apply_sharpen(image_data, 2.0)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply sharpening
        enhancer = ImageEnhance.Sharpness(image)
        sharpened = enhancer.enhance(1.0 + factor)
        
        # Encode back to base64
        buffer = io.BytesIO()
        sharpened.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error applying sharpen: {str(e)}"


def convert_grayscale(image_data: str) -> str:
    """
    Convert an image to grayscale.
    
    Args:
        image_data: Base64 encoded image string
    
    Returns:
        Base64 encoded grayscale image
        
    Examples:
        convert_grayscale(image_data)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        grayscale = ImageOps.grayscale(image)
        
        # Encode back to base64
        buffer = io.BytesIO()
        grayscale.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error converting to grayscale: {str(e)}"


def resize_image(image_data: str, width: int, height: int, maintain_aspect: bool = True) -> str:
    """
    Resize an image to specified dimensions.
    
    Args:
        image_data: Base64 encoded image string
        width: Target width
        height: Target height
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Base64 encoded resized image
        
    Examples:
        resize_image(image_data, 800, 600)
        resize_image(image_data, 400, 400, maintain_aspect=False)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if maintain_aspect:
            # Calculate aspect ratio preserving dimensions
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            resized = image
        else:
            # Resize to exact dimensions
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Encode back to base64
        buffer = io.BytesIO()
        resized.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return f"Error resizing image: {str(e)}"


def get_image_info(image_data: str) -> str:
    """
    Get information about an image (dimensions, format, etc.).
    
    Args:
        image_data: Base64 encoded image string
    
    Returns:
        JSON string with image information
        
    Examples:
        get_image_info(image_data)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        info = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "size_bytes": len(image_bytes)
        }
        
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error getting image info: {str(e)}"


# List of all available image tools
IMAGE_TOOLS = [
    crop_image,
    zoom_image,
    rotate_image,
    flip_image,
    adjust_brightness,
    adjust_contrast,
    apply_blur,
    apply_sharpen,
    convert_grayscale,
    resize_image,
    get_image_info
]