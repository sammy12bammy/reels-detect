import tensorflow as tf
from tensorflow.keras.models import Model # base model

class FaceTracker(Model):
    # pass through pre built neural network 
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    # pass through loss and optimizer
    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.classloss = classloss
        self.lloss = localizationloss
        self.opt = opt

    # hard core training step
    def train_step(self, batch, **kwargs):
        images, labels = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(images, training=True) # make predictions
            
            # pass through losses
            batch_classloss = self.classloss(labels[0], classes) # classifaction loss
            batch_lloss = self.lloss(tf.cast(labels[1], tf.float32), coords) # localization loss
            
            total_loss = batch_lloss+0.5*batch_classloss # total loss metric
            
            grad = tape.gradient(total_loss, self.model.trainable_variables) # apply gradients
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables)) # gradient descent. apply one step of gradient descent
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_lloss}

    def test_step(self, batch, **kwargs):
        images, labels = batch
        classes, coords = self.model(images, training=False)

        batch_classloss = self.classloss(labels[0], classes)
        batch_lloss = self.lloss(tf.cast(labels[1], tf.float32), coords)
        total_loss = batch_lloss+0.5*batch_classloss

        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_lloss}

    def call(self, images, **kwargs):
        return self.model(images, **kwargs)