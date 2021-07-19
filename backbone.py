# This code creats NNs.
#from tensorflow.keras.utils import plot_model
import tensorflow as tf
import itertools
from anchor import Anchor
from detection import Detect
import config as cfg
from yolact import (
    PredictionModule,
    ProtoNet,
    FPN
)
import numpy as np
from util import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
block_args,global_params=get_model_params('efficientnet-b2',None)
class MBConvBlock(tf.keras.layers.Layer):
  def __init__(self, block_args, global_params, name=None):
      super().__init__(name=name)
      self._block_args = block_args
      self._bn_mom = global_params.batch_norm_momentum
      self._bn_eps = global_params.batch_norm_epsilon
      #print(self._bn_mom,self._bn_eps)
      self.has_se = None #(self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
      self.id_skip = block_args.id_skip  # skip connection and drop connect
      self._data_format = global_params.data_format
      self._channel_axis = 1 if self._data_format == 'channels_first' else -1
      
      self._relu_fn = tf.keras.layers.ELU()
      self.vars=[]    #test-var
      self._build()
      
  def get_config(self):
      config ={"_block_args":self._block_args,"_bn_mom":self._bn_mom,"_bn_eps":self._bn_eps,"has_se":self.has_se,"id_skip":self.id_skip,"_data_format": self._data_format,"_channel_axis":self._channel_axis}
      base_config = super(MBConvBlock, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))
      # Expansion phase
  def _build(self):
      inp = self._block_args.input_filters  # number of input channels
      oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
      """Builds block according to the arguments."""
      # pylint: disable=g-long-lambda
      bid = itertools.count(0)
      get_bn_name = lambda: 'tpu_batch_normalization' + ('' if not next(
           bid) else '_' + str(next(bid) // 2))
      cid = itertools.count(0)
      get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
           next(cid) // 2))
    # pylint: enable=g-long-lambda
      #self.vars.append(self.add_weight(initializer='random_normal')) #testvar
      if self._block_args.expand_ratio != 1:
         self._expand_conv = tf.keras.layers.Conv2D(
               filters=oup,
               kernel_size=[1, 1],
               strides=[1, 1],
               kernel_initializer='normal',
               padding='same',
               data_format=self._data_format,
               use_bias=False,
               name=get_conv_name())
         self._bn0=tf.keras.layers.BatchNormalization(axis=self._channel_axis,
               momentum=self._bn_mom,
               epsilon=self._bn_eps,
               name=get_bn_name())
      # Depthwise convolution phase
      k = self._block_args.kernel_size
      s = self._block_args.stride  
      if isinstance(s, list):
         s = s[0]
      self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
             kernel_size=[k,k],
             strides=s,
             depthwise_initializer='normal',
             padding='same',
             data_format=self._data_format,
             use_bias=False,
             name='depthwise_conv2d')
      self._bn1 =tf.keras.layers.BatchNormalization(axis=self._channel_axis,
               momentum=self._bn_mom,
               epsilon=self._bn_eps,
               name=get_bn_name())
      if self.has_se:
         self._local_pooling=tf.keras.layers.AveragePooling2D
         num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
         self._se_reduce = tf.keras.layers.Conv2D(
               num_squeezed_channels,
               kernel_size=[1, 1],
               strides=[1, 1],
               kernel_initializer='normal',
               padding='same',
               data_format=self._data_format,
               use_bias=True,
               name='conv2d')
         self._se_expand =self._se_expand = tf.keras.layers.Conv2D(
                oup,
       		kernel_size=[1, 1],
        	strides=[1, 1],
        	kernel_initializer='normal',
        	padding='same',
        	data_format=self._data_format,
        	use_bias=True,
        	name='conv2d_1')
      final_oup = self._block_args.output_filters
      self._project_conv = tf.keras.layers.Conv2D(
        	filters=final_oup,
        	kernel_size=[1, 1],
        	strides=[1, 1],
        	kernel_initializer='normal',
        	padding='same',
        	data_format=self._data_format,
        	use_bias=False,
        	name=get_conv_name())
      self._bn2 =tf.keras.layers.BatchNormalization(axis=self._channel_axis,
               momentum=self._bn_mom,
               epsilon=self._bn_eps,
               name=get_bn_name())

  def call(self, inputs):
      x = inputs
      
      if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._relu_fn(x)
            

      x = self._depthwise_conv(x)
      x = self._bn1(x)
      x = self._relu_fn(x)
#      print(x.shape)
#      c=itertools.count(0)
#      co = lambda:c:next(c)
#      if co==5 or co==8 or co==16:
#        out.append()
      # Squeeze and Excitation
      if self.has_se:
          x_squeezed = self._local_pooling(pool_size=(x.shape[1],x.shape[2]),padding='valid')(x)
          x_squeezed = self._se_reduce(x_squeezed)
          x_squeezed = self._relu_fn(x_squeezed)
          x_squeezed = self._se_expand(x_squeezed)
          
          x = tf.keras.activations.sigmoid(x)

      x = self._project_conv(x)
      x = self._bn2(x)
      
      #x = tf.math.multiply(self.vars[-1],x)  #test_var
      # Skip connection and drop connect
      input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
      if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
        
          x =tf.keras.layers.add([x,inputs])  # skip connection
      return x  

    
class EfficientNet(tf.keras.Model):
  def __init__(self, blocks_args=None, global_params=None,extract_features=False):
    super().__init__()
    assert isinstance(blocks_args, list), 'blocks_args should be a list'
    assert len(blocks_args) > 0, 'block args must be greater than 0'
#    self.net=Yolact(**cfg.model_params)
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._extract_features=extract_features
    self._build()
  def _build(self):
    bn_mom =  self._global_params.batch_norm_momentum
    bn_eps = self._global_params.batch_norm_epsilon
    # Stem
    
    in_channels = 3  # rgb
    out_channels = round_filters(32, self._global_params)  # number of output channels
    self._blocks = []
    self._conv_stem =tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer='normal',
        padding='same',
        data_format=global_params.data_format,
        use_bias=False)
    self._bn0=tf.keras.layers.BatchNormalization(axis=(1 if global_params.data_format == 'channels_first' else -1),
               momentum=bn_mom,
               epsilon=bn_eps,
               )
    # Builds blocks.
    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    for block_args in self._blocks_args:
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
           input_filters=round_filters(block_args.input_filters, self._global_params),
           output_filters=round_filters(block_args.output_filters, self._global_params),
           num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
        self._blocks.append(MBConvBlock(block_args, self._global_params))
        if block_args.num_repeat > 1:
           block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
        for _ in range(block_args.num_repeat - 1):
            self._blocks.append(MBConvBlock(block_args, self._global_params))
    # Head
    in_channels = block_args.output_filters  # output of final block
    out_channels = round_filters(1280, self._global_params)
    self._conv_head=tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format=global_params.data_format,
        use_bias=False)
    self._bn1 =tf.keras.layers.BatchNormalization(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=bn_mom,
        epsilon=bn_eps,
               )
    # Final linear layer
    if  self._extract_features:
      self._conv_extract=tf.keras.layers.Conv2D(
          filters=32,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format=global_params.data_format,
          use_bias=False)
      self._bn2 =tf.keras.layers.BatchNormalization(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=bn_mom,
        epsilon=bn_eps,
                )
    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=global_params.data_format)
    self._dropout = tf.keras.layers.Dropout(global_params.dropout_rate)
    self._fc = tf.keras.layers.Dense(
          101,
          kernel_initializer='normal')     #1111111111111111111111111111111111111111111111111111111111111111111111
    self._fc2=  tf.keras.layers.Dense(
          51,
          kernel_initializer='normal')
    
    self._relu_fn = tf.keras.layers.ELU()
    self._softmax=tf.keras.layers.Softmax()
    self._upsample=tf.image.resize
    
    self.se=[tf.keras.layers.Conv2D(filters=50,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3)]
    
  def call(self,inputs):
    # Stem
    inputs=tf.keras.layers.Lambda(lambda x : x/255.,input_shape=(inputs.shape[1],inputs.shape[2],inputs.shape[3]))(inputs)
    x=self._relu_fn(self._bn0(self._conv_stem(inputs)))
    feature_maps = []
    # Blocks
    for idx, block in enumerate(self._blocks):
            
         
        x = block(x)
        
        if block._depthwise_conv.strides == (2, 2):
           feature_maps.append(last_x)

        elif idx == len(self._blocks) - 1:
           feature_maps.append(x)


        last_x = x
        #print(last_x.shape)
        #for i in last_x:
        #  print(i.shape)
#    for i in feature_maps:
#       print(i.shape)

    return feature_maps[2:]
    
    # Head
    """
    x = self._relu_fn(self._bn1(self._conv_head(x)))
   
    # Pooling and final linear layer
    if not self._extract_features:
      x = self._avg_pooling(x)
      
      x = self._dropout(x)
      y = self._fc2(x)
      x = self._fc(x)
      x = self._softmax(x)
    else:
      x = self._relu_fn(self._bn2(self._conv_extract(x)))
      #print(x.shape)
      
      
      #x = self._upsample(x,[x.shape[1]*2,x.shape[2]*2],method='nearest')
      x = tf.keras.layers.Flatten()(x)
    
    return x
    """

class Yolact(tf.keras.Model):
  def __init__(self,backbone,fpn_channels,num_class,num_mask,anchor_params,detect_params):
    super(Yolact,self).__init__()
    self._block_args,self._global_params=get_model_params(backbone,None)
    self.backbone_net=EfficientNet(self._block_args,self._global_params,extract_features=True)
    self.fpn_channels=fpn_channels
    self.num_class=num_class
    self.num_mask=num_mask
    self.anchor=anchor_params
    self.detect_params=detect_params
    self.protonet_coefficient=32
    #self.aspect_ratio=[1,0.5,2]
    #self.scale=[24,48,96,130,192]
#    self._build()
#  def _build(self):
    self.fpn=FPN(self.fpn_channels)
    self.protonet=ProtoNet(self.protonet_coefficient)
    # semantic segmentation branch to boost feature richness
    self.semantic_segmentation = tf.keras.layers.Conv2D(self.num_class-1, (1, 1), 1, padding="same",
                              kernel_initializer=tf.keras.initializers.glorot_uniform())
    self.num_anchors=Anchor(img_size=self.anchor["img_size"],feature_map_size=self.anchor["feature_map_size"],aspect_ratio=self.anchor["aspect_ratio"],scale=self.anchor["scale"])
    priors=self.num_anchors.get_anchors()
    self.pred=PredictionModule(self.fpn_channels,len(self.anchor["aspect_ratio"]),self.num_class,self.protonet_coefficient)
    #self.detect eval not train
    self.detection = Detect(anchors=priors, **cfg.detection_params)

    self.max_output_size = 300
    
  def call(self,input1):
#(2, 69, 69, 48)(2, 35, 35, 120)(2, 18, 18, 352)    
    feature_maps=self.backbone_net.call(input1)
    c3,c4,c5=feature_maps
    fpn_out=self.fpn.call(c3,c4,c5)
    p3=fpn_out[0]
    protonet_out = self.protonet.call(p3)
    # print("protonet: ", protonet_out.shape)
    # semantic segmentation branch
    seg = self.semantic_segmentation(p3)

    # Prediction Head branch
    pred_cls = []
    pred_offset = []
    pred_mask_coef = []

    # all output from FPN use same prediction head
    for f_map in fpn_out:
        cls, offset, coef = self.pred.call(f_map)
        pred_cls.append(cls)
        pred_offset.append(offset)
        pred_mask_coef.append(coef)
            
    pred_cls = tf.concat(pred_cls, axis=1)
    pred_offset = tf.concat(pred_offset, axis=1)
    pred_mask_coef = tf.concat(pred_mask_coef, axis=1)

    pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg
        }

    #print(pred_cls.shape,pred_offset.shape,pred_mask_coef.shape,protonet_out.shape,seg.shape)
    #for i in pred.values():
    #  print(i.shape)
    return pred

if __name__ == '__main__':
  block_args,global_params=get_model_params('efficientnet-b0',None)    
  #model=YpNet(0)
  #num_channels=112
  conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
  #bifpn=[BiFPN(num_channels,conv_channel_coef[2],first_time=True),BiFPN(num_channels,conv_channel_coef[2]),BiFPN(num_channels,conv_channel_coef[2])]
  #model=EfficientNet(block_args,global_params)
  model=Yolact(**cfg.model_params)
  input1=tf.keras.Input(shape=(550,550,3),name='img')#160,320
  

  output=model.call(input1)
  model =tf.keras.Model(inputs=input1,outputs=output)
  model.save('yolact.h5')    

  #plot_model(mm ,to_file='modell.png',show_shapes=True,show_dtype=True)
         
