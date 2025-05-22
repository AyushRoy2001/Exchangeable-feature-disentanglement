#IMPORT LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import AdamW
import tensorflow_probability as tfp
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import Constraint

#DATA LOADING
def load_image(path, size, mask=False):
    image = Image.open(path)
    image = image.resize((size, size))

    if mask:
        image = image.convert('L')  # Convert to grayscale
    else:
        image = image.convert('RGB')  # Convert to RGB
    
    image = np.array(image)
    return image

def load_data_testing(root_path, size):
    images = []
    masks = []

    image_folder = os.path.join(root_path, 'TissueImages')
    mask_folder = os.path.join(root_path, 'GroundTruth')

    for image_path in sorted(glob(os.path.join(image_folder, '*tif'))):
        img_id = os.path.basename(image_path).split('.')[0]
        mask_path = os.path.join(mask_folder, f'{img_id}_bin_mask.png')

        img = load_image(image_path, size) / 255.0
        mask = load_image(mask_path, size, mask=True) / 255.0

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

def load_data_training(root_path, size):
    images = []
    masks = []

    image_folder = os.path.join(root_path, 'TissueImages')
    mask_folder = os.path.join(root_path, 'GroundTruth')

    for image_path in sorted(glob(os.path.join(image_folder, '*png'))):
        img_id = os.path.basename(image_path).split('.')[0]
        mask_path = os.path.join(mask_folder, f'{img_id}_bin_mask.png')

        img = load_image(image_path, size) / 255.0
        mask = load_image(mask_path, size, mask=True) / 255.0

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

size = 512   # image size: 512x512
root_path = '/kaggle/input/tnbc-seg/MonuSeg/MonuSeg/Training'
X_train, y_train = load_data_training(root_path, size)

size = 512   # image size: 512x512
root_path = '/kaggle/input/tnbc-seg/MonuSeg/MonuSeg/Test'
X_test, y_test = load_data_testing(root_path, size)
print(f"X shape: {X_train.shape}     |  y shape: {y_train.shape}")
# X = np.expand_dims(X, -1)
y_train = np.expand_dims(y_train, -1)
print(f"\nX shape: {X_train.shape}  |  y shape: {y_train.shape}")
print(f"X shape: {X_test.shape}     |  y shape: {y_test.shape}")
# X = np.expand_dims(X, -1)
y_test = np.expand_dims(y_test, -1)
print(f"\nX shape: {X_test.shape}  |  y shape: {y_test.shape}")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_val shape:',X_val.shape)
print('X_test shape:',X_test.shape)
print('y_test shape:',y_test.shape)

# DATA AUGMENTATION
def horizontal_flip(image, mask):
    flipped_image = np.fliplr(image)
    flipped_mask = np.fliplr(mask)
    return flipped_image, flipped_mask

def vertical_flip(image, mask):
    flipped_image = np.flipud(image)
    flipped_mask = np.flipud(mask)
    return flipped_image, flipped_mask

def rotate_90(image, mask):
    rotated_image = np.rot90(image)
    rotated_mask = np.rot90(mask)
    return rotated_image, rotated_mask

def rotate_270(image, mask):
    rotated_image = np.rot90(image, -1)
    rotated_mask = np.rot90(mask, -1)
    return rotated_image, rotated_mask

augmented_X_train = []
augmented_y_train = []

for i in range(len(X_train)):
    image = X_train[i]
    mask = y_train[i]
    
    augmented_X_train.append(image)
    augmented_y_train.append(mask)

    # Horizontal flip
    h_flip_image, h_flip_mask = horizontal_flip(image, mask)
    augmented_X_train.append(h_flip_image)
    augmented_y_train.append(h_flip_mask)

    # Vertical flip
    v_flip_image, v_flip_mask = vertical_flip(image, mask)
    augmented_X_train.append(v_flip_image)
    augmented_y_train.append(v_flip_mask)

    # Rotate 90 degrees
    rot_90_image, rot_90_mask = rotate_90(image, mask)
    augmented_X_train.append(rot_90_image)
    augmented_y_train.append(rot_90_mask)

    # Rotate 270 degrees
    rot_270_image, rot_270_mask = rotate_270(image, mask)
    augmented_X_train.append(rot_270_image)
    augmented_y_train.append(rot_270_mask)

augmented_X_train = np.array(augmented_X_train)
augmented_y_train = np.array(augmented_y_train)
print("Augmented X_train shape:", augmented_X_train.shape)
print("Augmented y_train shape:", augmented_y_train.shape)

def horizontal_flip(image, mask):
    flipped_image = np.fliplr(image)
    flipped_mask = np.fliplr(mask)
    return flipped_image, flipped_mask

def vertical_flip(image, mask):
    flipped_image = np.flipud(image)
    flipped_mask = np.flipud(mask)
    return flipped_image, flipped_mask

def rotate_90(image, mask):
    rotated_image = np.rot90(image)
    rotated_mask = np.rot90(mask)
    return rotated_image, rotated_mask

def rotate_270(image, mask):
    rotated_image = np.rot90(image, -1)
    rotated_mask = np.rot90(mask, -1)
    return rotated_image, rotated_mask

augmented_X_test = []
augmented_y_test = []

for i in range(len(X_test)):
    image = X_test[i]
    mask = y_test[i]
    
    augmented_X_test.append(image)
    augmented_y_test.append(mask)

    # Horizontal flip
    h_flip_image, h_flip_mask = horizontal_flip(image, mask)
    augmented_X_test.append(h_flip_image)
    augmented_y_test.append(h_flip_mask)

    # Vertical flip
    v_flip_image, v_flip_mask = vertical_flip(image, mask)
    augmented_X_test.append(v_flip_image)
    augmented_y_test.append(v_flip_mask)

    # Rotate 90 degrees
    rot_90_image, rot_90_mask = rotate_90(image, mask)
    augmented_X_test.append(rot_90_image)
    augmented_y_test.append(rot_90_mask)

    # Rotate 270 degrees
    rot_270_image, rot_270_mask = rotate_270(image, mask)
    augmented_X_test.append(rot_270_image)
    augmented_y_test.append(rot_270_mask)

augmented_X_test = np.array(augmented_X_test)
augmented_y_test = np.array(augmented_y_test)
print("Augmented X_test shape:", augmented_X_test.shape)
print("Augmented y_test shape:", augmented_y_test.shape)

#DATA VISUALIZATION
image = augmented_X_train[52]
mask = augmented_y_train[52]
fig, axes = plt.subplots(1, 2, figsize=(5, 2))
axes[0].imshow(image, cmap='gray')
axes[0].axis('off')
axes[0].set_title('Image')
axes[1].imshow(mask*255, cmap='gray', vmin=0, vmax=1)
axes[1].axis('off')
axes[1].set_title('Mask')
plt.tight_layout()
plt.show()

#METRICS
def dice_score(y_true, y_pred):
    smooth = K.epsilon()
    y_true_flat = K.flatten(K.cast(y_true, 'float32'))
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    score = (2. * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)
    return score

def iou(y_true, y_pred):
    smooth = K.epsilon()
    y_true_flat = K.flatten(K.cast(y_true, 'float32'))
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection + smooth
    iou = (intersection + smooth) / union
    return iou

def sensitivity(y_true, y_pred):
    smooth = K.epsilon()
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_true_flat = K.flatten(K.cast(y_true, 'float32'))
    y_pred_flat = K.flatten(y_pred_pos)
    tp = K.sum(y_true_flat * y_pred_flat)
    fn = K.sum(y_true_flat * (1 - y_pred_flat))
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall

def specificity(y_true, y_pred):
    smooth = K.epsilon()
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_true_flat = K.flatten(K.cast(y_true, 'float32'))
    y_pred_flat = K.flatten(y_pred_pos)
    tn = K.sum((1 - y_true_flat) * (1 - y_pred_flat))
    fp = K.sum((1 - y_true_flat) * y_pred_flat)
    spec = (tn + smooth) / (tn + fp + smooth)
    return spec

def precision(y_true, y_pred):
    smooth = K.epsilon()
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_true_flat = K.flatten(K.cast(y_true, 'float32'))
    y_pred_flat = K.flatten(y_pred_pos)
    tp = K.sum(y_true_flat * y_pred_flat)
    fp = K.sum((1 - y_true_flat) * y_pred_flat)
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision

#LOSS FUNCTIONS
class ReduceMeanLoss(L.Layer):
    def __init__(self, axis, keep_dims):
        super(ReduceMeanLoss, self).__init__()
        self.axis = axis
        self.keep_dims = keep_dims
        
    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis=self.axis, keepdims=self.keep_dims)
    
class Pool(L.Layer):
    def __init__(self):
        super(Pool, self).__init__()
        self.max_pool = MaxPool2D((2, 2))
        
    def call(self, inputs):
        return self.max_pool(inputs)
    
class Up(L.Layer):
    def __init__(self):
        super(Up, self).__init__()
        self.up = UpSampling2D(interpolation="bilinear")
        
    def call(self, inputs):
        return self.up(inputs)

def extract_features(feat, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    fg_feat_map = feat * tf.cast(mask, tf.float32)
    bg_feat_map = feat * tf.cast(1 - mask, tf.float32)
    fg_feat = ReduceMeanLoss(1,False)(fg_feat_map)
    bg_feat = ReduceMeanLoss(1,False)(bg_feat_map)
    fg_feat = ReduceMeanLoss(1,False)(fg_feat)
    bg_feat = ReduceMeanLoss(1,False)(bg_feat)
    return fg_feat, bg_feat

def de_1enc(y_true, y_pred):
    y_pred = Up()(y_pred)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_1dec(y_true, y_pred):
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_2enc(y_true, y_pred):
    y_pred = Up()(y_pred)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_2dec(y_true, y_pred):
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_3enc(y_true, y_pred):
    y_pred = Up()(y_pred)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_3dec(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_4(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def dice_loss(y_true, y_pred):
    loss = 1 - dice_score(y_true, y_pred)
    return loss

def iou_loss(y_true, y_pred):
    loss = 1 - iou(y_true, y_pred)
    return loss
    
def bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    return loss

def combined_loss(y_true, y_pred):
    loss = dice_loss(y_true, y_pred) + iou_loss(y_true, y_pred)
    return loss

class WeightClip(Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, 0.0, 1.0)
    
weight_clip = WeightClip()
weight_1_enc = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_1_dec = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_2_enc = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_2_dec = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_3_enc = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_3_dec = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_4_bot = tf.Variable(0.0, trainable=True, constraint=weight_clip)

def de_1_loss_enc(y_true, y_pred):
    y_pred = Up()(y_pred)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_1_enc*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_1_loss_dec(y_true, y_pred):
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_1_dec*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_2_loss_enc(y_true, y_pred):
    y_pred = Up()(y_pred)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_2_enc*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_2_loss_dec(y_true, y_pred):
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_2_dec*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_3_loss_enc(y_true, y_pred):
    y_pred = Up()(y_pred)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_3_enc*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_3_loss_dec(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_3_dec*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_4_loss(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_4_bot*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

#MODEL
class conv_block(L.Layer):
    def __init__(self, num_filters):
        super(conv_block, self).__init__()
        self.num_filters = num_filters
        self.conv1 = Conv2D(self.num_filters, 3, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2 = Conv2D(self.num_filters, 3, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())
        self.act1 = Activation("relu")
        self.act2 = Activation("relu")
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x
    
class encoder_block(L.Layer):
    def __init__(self, num_filters):
        super(encoder_block, self).__init__()
        self.num_filters = num_filters
        self.conv_block = conv_block(num_filters=self.num_filters)
        self.max_pool = MaxPool2D((2, 2))

    def call(self, x):
        x = self.conv_block(x)
        p = self.max_pool(x)
        
        return x, p
    
class attention_gate(L.Layer):
    def __init__(self, num_filters):
        super(attention_gate, self).__init__()
        self.num_filters = num_filters
        self.conv1 = Conv2D(self.num_filters, 1, padding="same")
        self.conv2 = Conv2D(self.num_filters, 1, padding="same")
        self.conv3 = Conv2D(self.num_filters, 1, padding="same")
        self.act = Activation("relu")
        self.sig = Activation("sigmoid")
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, g, s):
        Wg = self.conv1(g)
        Wg = self.bn1(Wg)

        Ws = self.conv2(s)
        Ws = self.bn2(Ws)

        out = self.act(Wg + Ws)
        out = self.conv3(out)
        out = self.sig(out)
        
        return out * s
    
class decoder_block(L.Layer):
    def __init__(self, num_filters):
        super(decoder_block, self).__init__()
        self.num_filters = num_filters
        self.up = UpSampling2D(interpolation="bilinear")
        self.concat = Concatenate()
        self.conv_block = conv_block(num_filters=self.num_filters)
        self.attention = attention_gate(num_filters=self.num_filters)

    def call(self, x, s):
        x = self.up(x)
        s = self.attention(x, s)
        x = self.concat([x, s])
        x = self.conv_block(x)
        
        return x

def AttnUnet(input_shape, num_classes=1):
    """ Inputs """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(64)(inputs)
    s2, p2 = encoder_block(128)(p1)
    s3, p3 = encoder_block(256)(p2)
    
    """ Bottleneck """
    b1 = conv_block(num_filters=512)(p3)

    """ Decoder """
    d1 = decoder_block(256)(b1, s3)
    d2 = decoder_block(128)(d1, s2) 
    d3 = decoder_block(64)(d2, s1) 

    """ Outputs """
    outputs = Conv2D(num_classes, 1, padding="same", activation="sigmoid", name='OUT')(d3)

    """ Model """
    model = Model(inputs, [outputs, p1, d3, p2, d2, p3, d1, b1], name="Attention-UNET")
    return model

model = AttnUnet((512, 512, 3))
optimizer = AdamW(learning_rate=0.0001)
METRIC = ["accuracy", dice_score, sensitivity, specificity, iou]
loss = [combined_loss,de_1_loss_enc,de_1_loss_dec,de_2_loss_enc,de_2_loss_dec,de_3_loss_enc,de_3_loss_dec,de_4_loss]
metr = [METRIC,de_1enc,de_1dec,de_2enc,de_2dec,de_3enc,de_3dec,de_4]
model.compile(loss=loss, metrics=metr, optimizer=optimizer)
model.summary()

#TRAINING
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='/kaggle/working/model.weights.h5',
    monitor='val_OUT_dice_score',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
    )

log_dir = "/kaggle/working/"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      write_graph=True,)

class LogWeightsCallback(keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(LogWeightsCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for layer in self.model.layers:
                if hasattr(layer, 'weights'):
                    for weight in layer.weights:
                        tf.summary.histogram(weight.name, weight, step=epoch)
                        
callbacks = [model_checkpoint_callback,
             tensorboard_callback,
             LogWeightsCallback(log_dir)]

history = model.fit(augmented_X_train, (augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train),
                    epochs = 50,
                    batch_size = 4,
                    validation_data = (X_val,(y_val,y_val,y_val,y_val,y_val,y_val,y_val,y_val)),
                    verbose = 1,
                    callbacks=callbacks,
                    shuffle = True)

#EVALUATION
model.load_weights("/kaggle/working/model.weights.h5")
model.evaluate(X_test, (y_test,y_test,y_test,y_test,y_test,y_test,y_test,y_test), batch_size = 4, verbose = 1)

modeller = Model(inputs=model.input, outputs=[model.get_layer(name="conv_block_3").output,model.get_layer(name="decoder_block").output,model.get_layer(name="decoder_block_1").output,model.get_layer(name="decoder_block_2").output])

dice_scores = []
lde_enc1 = []
lde_dec1 = []
lde_enc2 = []
lde_dec2 = []
lde_enc3 = []
lde_dec3 = []
lde_bot = []

for z in range(0,25):
    test_image = augmented_X_test[z]  
    test_mask = augmented_y_test[z] 

    test_image = np.reshape(test_image, (1,) + test_image.shape)

    predicted_mask = model.predict(test_image)[0]
    bot, dec1, dec2, dec3 = modeller.predict(test_image)
    loss,acc,dice,sen,spe,iou,lde0,lde1,lde2,lde3,lde4,lde5,lde6 = model.evaluate(test_image, (tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0)), verbose = 0)
    dice_scores.append(dice)
    lde_enc1.append(lde0)
    lde_dec1.append(lde1)
    lde_enc2.append(lde2)
    lde_dec2.append(lde3)
    lde_enc3.append(lde4)
    lde_dec3.append(lde5)
    lde_bot.append(lde6)
    predicted_mask_binary = np.where(predicted_mask > 0.5, 1, 0) * 255

    print("Sample:", z, "-> Loss:", loss, "|| Dice:", dice, "|| Accuracy:", acc)
    
    if z % 5 == 0:
        fig, axes = plt.subplots(1, 6, figsize=(20, 5))
        
        # Plot the test image
        axes[0].imshow(test_image[0], cmap='gray')
        axes[0].set_title('Test Image')
        axes[0].axis('off')

        # Plot the ground truth mask
        axes[1].imshow(test_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        # Plot the binarized predicted mask
        axes[2].imshow(predicted_mask_binary[0], cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')

        axes[3].imshow(tf.math.reduce_mean(tf.math.reduce_mean(bot, axis=-1), axis=0), cmap='jet')
        axes[3].set_title('Bottleneck')
        axes[3].axis('off')
    
        axes[4].imshow(tf.math.reduce_mean(tf.math.reduce_mean(dec1, axis=-1), axis=0), cmap='jet')
        axes[4].set_title('Decoder 1')
        axes[4].axis('off') 
        
        axes[5].imshow(tf.math.reduce_mean(tf.math.reduce_mean(dec2, axis=-1), axis=0), cmap='jet')
        axes[5].set_title('Decoder 2')
        axes[5].axis('off')
        
        axes[5].imshow(tf.math.reduce_mean(tf.math.reduce_mean(dec3, axis=-1), axis=0), cmap='jet')
        axes[5].set_title('Decoder 3')
        axes[5].axis('off')

fig, axes = plt.subplots(2, 4, figsize=(15, 6))  
plt.subplots_adjust(wspace=0.4, hspace=0.4) 

def plot_with_regression(ax, x_data, y_data, label):
    m, b = np.polyfit(x_data, y_data, 1)
    line_x = np.linspace(min(x_data), max(x_data), 100)
    line_y = m * line_x + b
    ax.scatter(x_data, y_data, label=label)
    ax.plot(line_x, line_y, color='red', label='Best Fit Line')
    ax.set_xlabel('Dice Score')
    ax.set_ylabel(label)
    ax.set_title(f'Dice Score vs. {label}')

plot_with_regression(axes[0, 0], dice_scores, lde_enc1, 'Lde Encoder 1')
plot_with_regression(axes[0, 1], dice_scores, lde_enc2, 'Lde Encoder 2')
plot_with_regression(axes[0, 2], dice_scores, lde_enc3, 'Lde Encoder 3')
plot_with_regression(axes[0, 3], dice_scores, lde_dec1, 'Lde Decoder 1')
plot_with_regression(axes[1, 0], dice_scores, lde_dec2, 'Lde Decoder 2')
plot_with_regression(axes[1, 1], dice_scores, lde_dec3, 'Lde Decoder 3')
plot_with_regression(axes[1, 2], dice_scores, lde_bot, 'Lde Bottleneck')

plt.show()