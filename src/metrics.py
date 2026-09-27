
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class DiceMetric(tf.keras.metrics.Metric):

    def __init__(self, smooth=1e-6, name='dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)

        self.smooth = smooth

        # Running state across batches
        self.total_intersection = self.add_weight(
            name='total_intersection',
            initializer='zeros'
        )

        self.total_true = self.add_weight(
            name='total_true',
            initializer='zeros'
        )

        self.total_pred = self.add_weight(
            name='total_pred',
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate batch-level global statistics
        batch_intersection = tf.reduce_sum(y_true * y_pred)
        batch_true = tf.reduce_sum(y_true)
        batch_pred = tf.reduce_sum(y_pred)

        # Add this batch's statistics to the running state
        self.total_intersection.assign_add(batch_intersection)
        self.total_true.assign_add(batch_true)
        self.total_pred.assign_add(batch_pred)

    def result(self):

        numerator = (
            2.0 * self.total_intersection
            + self.smooth
        )

        denominator = (
            self.total_true
            + self.total_pred
            + self.smooth
        )

        return numerator / denominator

    def reset_state(self):

        self.total_intersection.assign(0.0)
        self.total_true.assign(0.0)
        self.total_pred.assign(0.0)

    def get_config(self):

        config = super().get_config()

        config.update({
            'smooth': self.smooth
        })

        return config

if __name__=="__main__":
    gt = tf.convert_to_tensor([[[[1],[1]],[[0],[0]]]], dtype=tf.float32)
    pred = tf.convert_to_tensor([[[[1],[0.5]],[[0.2],[0.1]]]], dtype=tf.float32)
   