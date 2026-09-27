import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class DiceLoss(tf.keras.Loss):
    def __init__(self, smooth=1e-6, name='dice_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        intersection = tf.reduce_sum(y_true*y_pred, axis=(1,2,3))
        gt_sum = tf.reduce_sum(y_true, axis=(1,2,3))
        pred_sum = tf.reduce_sum(y_pred, axis=(1,2,3))
        coef =(2*intersection+self.smooth)/(gt_sum+pred_sum+self.smooth)
        mean_coef = tf.reduce_mean(coef)
        return 1-mean_coef
    def get_config(self):
        config = super().get_config()
        config.update({'smooth':self.smooth})
        return config

@keras.saving.register_keras_serializable()
class BCEDiceLoss(tf.keras.Loss):
    def __init__(self, bce_weight=0.3, dice_weight=0.7, smooth=1e-6, name='bce_dice_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError('weights cannot be in negative')
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError('both weights cannot be 0')
    def call(self, y_true, y_pred):
        
        dice_loss = self.dice_loss(y_true, y_pred)
        bce_loss =  tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true, y_pred)
            )
        total_loss = (self.bce_weight*bce_loss)+(self.dice_weight*dice_loss)
        return total_loss
    def get_config(self):
        config = super().get_config()
        config.update({"smooth":self.smooth,
                       "dice_weight":self.dice_weight,
                       "bce_weight":self.bce_weight})
        return config


if __name__=='__main__':
    gt = tf.convert_to_tensor([[[[1],[1]],[[0],[0]]]], dtype=tf.float32)
    pred = tf.convert_to_tensor([[[[1],[0.5]],[[0.2],[0.1]]]], dtype=tf.float32)
    perfect_wrong = tf.convert_to_tensor([[[[0],[0]],[[1],[1]]]], dtype=tf.float32)
    Dice_loss = DiceLoss()
    BCE_dice_loss = BCEDiceLoss()
    dc = Dice_loss(gt, pred)
    bce_dice = BCE_dice_loss(gt, pred)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt, pred))
    print('General pattern', "="*10)
    print(f'Dice loss is: {dc}')
    print(f'BCE + Dice loss is: {bce_dice}')
    print(f'BCE loss: {bce}')

    print('Perfect right', "="*10)
    print(f'Dice loss is: {Dice_loss(gt, gt)}')
    print(f'BCE + Dice loss is: {BCE_dice_loss(gt, gt)}')
    print(f'BCE loss: {tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt, gt))}')

    print('Perfect wrong', "="*10)
    print(f'Dice loss is: {Dice_loss(gt, perfect_wrong)}')
    print(f'BCE + Dice loss is: {BCE_dice_loss(gt, perfect_wrong)}')
    print(f'BCE loss: {tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt, perfect_wrong))}')