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

def load_data(root_path, size):
    images = []
    masks = []

    image_folder = os.path.join(root_path, 'original')
    mask_folder = os.path.join(root_path, 'GT')

    for image_path in sorted(glob(os.path.join(image_folder, '*png'))):
        img_id = os.path.basename(image_path).split('.')[0]
        mask_path = os.path.join(mask_folder, f'{img_id}.png')

        img = load_image(image_path, size) / 255.0
        mask = load_image(mask_path, size, mask=True) / 255.0

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

size = 256
root_path = '/kaggle/input/udiat-segmentation-dataset/UDIAT_Dataset_B/UDIAT_Dataset_B'
X_train, y_train = load_data(root_path, size)
print(f"X shape: {X_train.shape}     |  y shape: {y_train.shape}")
y_train = np.expand_dims(y_train, -1)
print(f"\nX shape: {X_train.shape}  |  y shape: {y_train.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=35)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=35)
print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_val shape:',X_val.shape)
print('X_val shape:',y_val.shape)
print('X_test shape:',X_test.shape)
print('y_test shape:',y_test.shape)

#DATA AUGMENTATION
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

#DATA VISUALIZATION
image = X_train[52]
mask = y_train[52]
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
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_1dec(y_true, y_pred):
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_2enc(y_true, y_pred):
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

def de_4enc(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_4dec(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = 1.0*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_5(y_true, y_pred):
    y_true = Pool()(y_true)
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
    loss = dice_loss(y_true, y_pred) + bce_loss(y_true, y_pred)
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
weight_4_enc = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_4_dec = tf.Variable(0.0, trainable=True, constraint=weight_clip)
weight_5_bot = tf.Variable(0.0, trainable=True, constraint=weight_clip)

def de_1_loss_enc(y_true, y_pred):
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_1_enc*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_1_loss_dec(y_true, y_pred):
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_1_dec*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_2_loss_enc(y_true, y_pred):
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

def de_4_loss_enc(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_4_enc*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_4_loss_dec(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_4_dec*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

def de_5_loss(y_true, y_pred):
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    y_true = Pool()(y_true)
    fg, bg = extract_features(y_true, y_pred)
    loss = weight_5_bot*ReduceMeanLoss(-1,False)(-tf.math.log(tf.norm(fg - bg, axis=0)))
    return loss

#MODEL
class MSAG(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(MSAG, self).__init__()
        self.channels = channels
        self.pointwise_conv = Conv2D(self.channels, (1, 1), padding='same')
        self.ordinary_conv = Conv2D(self.channels, (3, 3), padding='same')
        self.dilation_conv = Conv2D(self.channels, (3, 3), padding='same', dilation_rate=2)
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        self.vote_conv = Conv2D(self.channels, (1, 1), padding='same', activation='sigmoid')

    def call(self, x):
        x1 = self.pointwise_conv(x)
        x1 = self.batch_norm(x1)
        x2 = self.ordinary_conv(x)
        x2 = self.batch_norm(x2)
        x3 = self.dilation_conv(x)
        x3 = self.batch_norm(x3)
        concatenated = self.relu(tf.concat([x1, x2, x3], axis=-1))
        attention = self.vote_conv(concatenated)
        return x * attention + x

    
class conv_block(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(conv_block, self).__init__()
        self.channels = channels
        self.conv1 = Conv2D(self.channels, (3, 3), padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv2 = Conv2D(self.channels, (3, 3), padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

def up_conv_block(x, filters):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

class up_conv_block(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(up_conv_block, self).__init__()
        self.channels = channels
        self.up = UpSampling2D(size=(2, 2))
        self.conv = Conv2D(self.channels, (3, 3), padding='same')
        self.bn = BatchNormalization()
        self.relu= ReLU()

    def call(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def conv_mixer_block(x, filters, depth, kernel_size):
    for _ in range(depth):
        residual = x
        x = Conv2D(filters, (kernel_size, kernel_size), padding='same', groups=filters)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (1, 1))(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])
    return x

def CMUNet(input_shape, output_channels, l=7, k=7):
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block(64)(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(128)(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(256)(p2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(512)(p3)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = conv_block(1024)(p4)
    cm = conv_mixer_block(c5, 1024, depth=l, kernel_size=k)

    # Decoder with skip connections and MSAG
    msag4 = MSAG(512)
    msag3 = MSAG(256)
    msag2 = MSAG(128)
    msag1 = MSAG(64)

    u5 = up_conv_block(512)(cm)
    u5 = Concatenate()([msag4(c4), u5])
    u5 = conv_block(512)(u5)

    u4 = up_conv_block(256)(u5)
    u4 = Concatenate()([msag3(c3), u4])
    u4 = conv_block(256)(u4)

    u3 = up_conv_block(128)(u4)
    u3 = Concatenate()([msag2(c2), u3])
    u3 = conv_block(128)(u3)

    u2 = up_conv_block(64)(u3)
    u2 = Concatenate()([msag1(c1), u2])
    u2 = conv_block(64)(u2)

    outputs = Conv2D(output_channels, (1, 1), activation='sigmoid', name='OUT')(u2)

    model = Model(inputs, [outputs,c1,u2,c2,u3,c3,u4,c4,u5,cm])
    return model


input_shape = (256, 256, 3)
output_channels = 1
model = CMUNet(input_shape, output_channels)
optimizer = AdamW(learning_rate=0.0001)
METRIC = ["accuracy", dice_score, sensitivity, specificity, iou]
loss = [combined_loss,de_1_loss_enc,de_1_loss_dec,de_2_loss_enc,de_2_loss_dec,de_3_loss_enc,de_3_loss_dec,de_4_loss_enc,de_4_loss_dec,de_5_loss]
metr = [METRIC,de_1enc,de_1dec,de_2enc,de_2dec,de_3enc,de_3dec,de_4enc,de_4dec,de_5]
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

history = model.fit(augmented_X_train, (augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train,augmented_y_train),
                    epochs = 50,
                    batch_size = 4,
                    validation_data = (X_val,(y_val,y_val,y_val,y_val,y_val,y_val,y_val,y_val,y_val,y_val)),
                    verbose = 1,
                    callbacks=callbacks,
                    shuffle = True)

#EVALUATION
model.load_weights("/kaggle/working/model.weights.h5")
model.evaluate(X_test, (y_test,y_test,y_test,y_test,y_test,y_test,y_test,y_test,y_test,y_test), batch_size = 4, verbose = 1)

modeller = Model(inputs=model.input, outputs=[model.get_layer(name="add_6").output,model.get_layer(name="conv_block_5").output,model.get_layer(name="conv_block_6").output,model.get_layer(name="conv_block_8").output])
dice_scores = []
lde_enc1 = []
lde_dec1 = []
lde_enc2 = []
lde_dec2 = []
lde_enc3 = []
lde_dec3 = []
lde_enc4 = []
lde_dec4 = []
lde_bot = []

for z in range(0,85):
    test_image = augmented_X_test[z]  
    test_mask = augmented_y_test[z] 

    test_image = np.reshape(test_image, (1,) + test_image.shape)

    predicted_mask = model.predict(test_image)[0]
    bot, dec1, dec2, dec3 = modeller.predict(test_image)
    loss,acc,dice,sen,spe,iou,lde0,lde1,lde2,lde3,lde4,lde5,lde6,lde7,lde8 = model.evaluate(test_image, (tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0),tf.expand_dims(test_mask, axis=0)), verbose = 0)
    dice_scores.append(dice)
    lde_enc1.append(lde0)
    lde_dec1.append(lde1)
    lde_enc2.append(lde2)
    lde_dec2.append(lde3)
    lde_enc3.append(lde4)
    lde_dec3.append(lde5)
    lde_enc4.append(lde6)
    lde_dec4.append(lde7)
    lde_bot.append(lde8)
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

fig, axes = plt.subplots(3, 4, figsize=(15, 6))  
plt.subplots_adjust(wspace=0.4, hspace=0.7) 

def plot_with_regression(ax, x_data, y_data, label):
    m, b = np.polyfit(x_data, y_data, 1)
    line_x = np.linspace(min(x_data), max(x_data), 100)
    line_y = m * line_x + b
    ax.scatter(x_data, y_data, label=label)
    ax.plot(line_x, line_y, color='red', label='Best Fit Line')
    ax.set_xlabel('Dice Score')
    ax.set_ylabel(label)
    ax.set_title(f'Dice Score vs. {label}')
    ax.legend()

plot_with_regression(axes[0, 0], dice_scores, lde_enc1, 'Lde Encoder 1')
plot_with_regression(axes[0, 1], dice_scores, lde_enc2, 'Lde Encoder 2')
plot_with_regression(axes[0, 2], dice_scores, lde_enc3, 'Lde Encoder 3')
plot_with_regression(axes[0, 3], dice_scores, lde_enc4, 'Lde Encoder 4')
plot_with_regression(axes[1, 0], dice_scores, lde_dec1, 'Lde Decoder 1')
plot_with_regression(axes[1, 1], dice_scores, lde_dec2, 'Lde Decoder 2')
plot_with_regression(axes[1, 2], dice_scores, lde_dec3, 'Lde Decoder 3')
plot_with_regression(axes[1, 3], dice_scores, lde_dec4, 'Lde Decoder 4')
plot_with_regression(axes[2, 0], dice_scores, lde_bot, 'Lde Bottleneck')

plt.show()