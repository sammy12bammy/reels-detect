# getting data
import os
import time
import uuid
import cv2


class ImageCapture:
    """Capture images from webcam for training data collection."""
    
    def __init__(self, images_path: str = os.path.join('data', 'images'), camera_id: int = 0):
        """
        Initialize the image capture.
        
        Args:
            images_path: Directory to save captured images
            camera_id: Camera device ID (0 for built-in webcam)
        """
        self.images_path = images_path
        self.camera_id = camera_id
        
        # Create directory if it doesn't exist
        os.makedirs(self.images_path, exist_ok=True)
    
    def capture(self, num_images: int = 30, delay: float = 0.5, show_preview: bool = True):
        """
        Capture images from the webcam.
        
        Args:
            num_images: Number of images to capture
            delay: Delay between captures in seconds
            show_preview: Whether to show preview window
            
        Returns:
            List of saved image paths
        """
        saved_images = []
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        try:
            for img_num in range(num_images):
                print(f'Taking image {img_num + 1}/{num_images}')
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Warning: Failed to capture image {img_num + 1}")
                    continue
                
                # Generate unique filename
                img_name = os.path.join(self.images_path, f'{str(uuid.uuid1())}.jpg')
                cv2.imwrite(img_name, frame)
                saved_images.append(img_name)
                
                # Show preview
                if show_preview:
                    cv2.imshow('frame', frame)
                
                time.sleep(delay)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Capture cancelled by user")
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f'Finished taking {len(saved_images)} images')
        return saved_images


# Allow running directly for standalone use
if __name__ == "__main__":
    capture = ImageCapture()
    capture.capture(num_images=30)
