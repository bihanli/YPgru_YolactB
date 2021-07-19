import tensorflow as tf
import config as cfg
from anchor import Anchor
from detection import Detect
class PredictionModule(tf.keras.layers.Layer):

  def __init__(self, out_channels, num_anchors, num_class, num_mask):
      super(PredictionModule, self).__init__()
      self.num_anchors = num_anchors
      self.num_class = num_class
      self.num_mask = num_mask

      self.Conv = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same",
                        kernel_initializer=tf.keras.initializers.glorot_uniform(),
                        activation="relu")

      self.classConv = tf.keras.layers.Conv2D(self.num_class * self.num_anchors, (3, 3), 1, padding="same",
                           kernel_initializer=tf.keras.initializers.glorot_uniform())
      self.boxConv = tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform())
      # activation of mask coef is tanh
      self.maskConv = tf.keras.layers.Conv2D(self.num_mask * self.num_anchors, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform())

  def call(self, p):
      p = self.Conv(p)
      pred_class = self.classConv(p)
      pred_box = self.boxConv(p)
      pred_mask = self.maskConv(p)

      # pytorch input  (N,Cin,Hin,Win) 
      # tf input (N,Hin,Win,Cin) 
      # so no need to transpose like (0, 2, 3, 1) as in original yolact code
      # reshape the prediction head result for following loss calculation
      pred_class = tf.reshape(pred_class, [tf.shape(pred_class)[0], -1, self.num_class])
      pred_box = tf.reshape(pred_box, [tf.shape(pred_box)[0], -1, 4])
      pred_mask = tf.reshape(pred_mask, [tf.shape(pred_mask)[0], -1, self.num_mask])

      # add activation for conf and mask coef
      # pred_class = tf.nn.softmax(pred_class, axis=-1)
      pred_mask = tf.keras.activations.tanh(pred_mask)

      return pred_class, pred_box, pred_mask
class ProtoNet(tf.keras.layers.Layer):
  """
      Creating the component of ProtoNet
      Arguments:
      num_prototype
  """

  def __init__(self, num_prototype):
      super(ProtoNet, self).__init__()
      self.Conv1 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")
      self.Conv2 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")
      self.Conv3 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")
      self.upSampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
      self.Conv4 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")

      self.finalConv = tf.keras.layers.Conv2D(num_prototype, (1, 1), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation='relu')

  def call(self, p3):
      # (3,3) convolution * 3
      proto = self.Conv1(p3)
      proto = self.Conv2(proto)
      proto = self.Conv3(proto)

      # upsampling + convolution
      proto = tf.keras.activations.relu(self.upSampling(proto))
      proto = self.Conv4(proto)
       # final convolution
      proto = self.finalConv(proto)
      return proto

class FPN(tf.keras.layers.Layer):
  def __init__(self,num_fpn_filters):
      super(FPN,self).__init__()
      self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

      # no Relu for downsample layer
      # Pytorch and tf differs in conv2d when stride > 1
      # https://dmolony3.github.io/Pytorch-to-Tensorflow.html
      # Hence, manually adding padding
      self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
      self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="valid",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())

      self.pad2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
      self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="valid",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())

      self.lateralCov1 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())
      self.lateralCov2 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())
      self.lateralCov3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())

      # predict layer for FPN
      self.predictP5 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            activation="relu")
      self.predictP4 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            activation="relu")
      self.predictP3 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                             kernel_initializer=tf.keras.initializers.glorot_uniform(),
                             activation="relu")

  def call(self, c3, c4, c5):
      # lateral conv for c3 c4 c5
      # pytorch input  (N,Cin,Hin,Win) 
      # tf input (N,Hin,Win,Cin) 
      p5 = self.lateralCov1(c5)
      # _, h, w, _ = tf.shape(c4)
      p4 = tf.add(tf.image.resize(p5, [tf.shape(c4)[1],tf.shape(c4)[2]]), self.lateralCov2(c4))
      # _, h, w, _ = tf.shape(c3)
      p3 = tf.add(tf.image.resize(p4, [tf.shape(c3)[1],tf.shape(c3)[2]]), self.lateralCov3(c3))
      # print("p3: ", p3.shape)

      # smooth pred layer for p3, p4, p5
      p3 = self.predictP3(p3)
      p4 = self.predictP4(p4)
      p5 = self.predictP5(p5)

      # downsample conv to get p6, p7
      p6 = self.downSample1(self.pad1(p5))
      p7 = self.downSample2(self.pad2(p6))

      return [p3, p4, p5, p6, p7]


  """
    if training:
        pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg,
            'priors': self.priors
        }
        # Following to make both `if` and `else` return structure same
        result = {
            'detection_boxes': tf.zeros((self.max_output_size, 4)),
            'detection_classes': tf.zeros((self.max_output_size)), 
            'detection_scores': tf.zeros((self.max_output_size)), 
            'detection_masks': tf.zeros((self.max_output_size, 30, 30, 1)), 
            'num_detections': tf.constant(0)}
        pred.update(result)
    else:
        pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg,
            'priors': self.priors
        }

        pred.update(self.detect(pred))

    return pred
  """
  