import tensorflow as tf
from tensorflow.keras.models import Model

class FaceTracker(Model):
    """
    Custom training model for 3-head face detection and mouth tracking.
    
    Handles:
        - Face classification (binary)
        - Face bounding box regression  
        - Mouth open classification (binary)
    """
    
    def __init__(self, model, **kwargs):
        """
        Args:
            model: The compiled 3-head Keras model (VGG16-based)
        """
        super().__init__(**kwargs)
        self.model = model

    def compile(self, opt, classloss, localizationloss, mouth_classloss, **kwargs):
        """
        Compile the model with optimizer and loss functions.
        
        Args:
            opt: Optimizer (e.g., Adam)
            classloss: Binary cross-entropy for face classification
            localizationloss: Custom loss for bounding box regression
            mouth_classloss: Binary cross-entropy for mouth open classification
        """
        super().compile(**kwargs)
        self.classloss = classloss
        self.lloss = localizationloss
        self.mouth_classloss = mouth_classloss  # NEW: mouth open loss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        """
        Custom training step for 3-head model.
        
        Returns:
            Dictionary of losses for monitoring
        """
        images, labels = batch
        
        with tf.GradientTape() as tape: 
            # Get predictions from all 3 heads
            face_class, face_bbox, mouth_open = self.model(images, training=True)
            
            # Calculate losses for each head
            batch_face_classloss = self.classloss(labels[0], face_class)
            batch_bbox_loss = self.lloss(tf.cast(labels[1], tf.float32), face_bbox)
            batch_mouth_loss = self.mouth_classloss(labels[2], mouth_open)  # NEW
            
            # Combined weighted loss
            total_loss = batch_bbox_loss + 0.5*batch_face_classloss + 0.5*batch_mouth_loss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {
            "total_loss": total_loss,
            "face_class_loss": batch_face_classloss,
            "bbox_loss": batch_bbox_loss,
            "mouth_loss": batch_mouth_loss  # NEW: track mouth loss separately
        }

    def test_step(self, batch, **kwargs):
        """
        Custom validation/test step for 3-head model.
        
        Returns:
            Dictionary of losses for monitoring
        """
        images, labels = batch
        
        # Get predictions from all 3 heads
        face_class, face_bbox, mouth_open = self.model(images, training=False)

        # Calculate losses for each head
        batch_face_classloss = self.classloss(labels[0], face_class)
        batch_bbox_loss = self.lloss(tf.cast(labels[1], tf.float32), face_bbox)
        batch_mouth_loss = self.mouth_classloss(labels[2], mouth_open)  # NEW
        
        # Combined weighted loss
        total_loss = batch_bbox_loss + 0.5*batch_face_classloss + 0.5*batch_mouth_loss

        return {
            "total_loss": total_loss,
            "face_class_loss": batch_face_classloss,
            "bbox_loss": batch_bbox_loss,
            "mouth_loss": batch_mouth_loss  # NEW: track mouth loss separately
        }

    def call(self, images, **kwargs):
        """Forward pass through the model."""
        return self.model(images, **kwargs)