import contextlib
import datetime
from data.coco_dataset import ObjectDetectionDataset
from eval import evaluate
from absl import logging
from utils import learning_rate_schedule

import tensorflow as tf
import argparse
import json
import os
import cv2
import numpy as np
from util import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
import config as cfg
from backbone import Yolact
from loss_yolact import YOLACTLoss
#from server import client_generator
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.utils import plot_model

parser = argparse.ArgumentParser(description='Mask model trainer')
parser.add_argument('--batch', type=int, default=1, help='Batch size.')
parser.add_argument('--epoch', type=int, default=30, help='Number of epochs.')
parser.add_argument('--epochsize', type=int, default=1000, help='How many frames per epoch.')
parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
parser.add_argument('--weight_decay', type=int, default=5*1e-4, help='weight_decay')  
parser.add_argument('--momentum', type=int, default=0.9, help='momentum')
parser.add_argument('--print_interval', type=int, default=10, help='number of iteration between printing loss')
parser.add_argument('--save_interval', type=int, default=100, help='number of iteration between saving model(checkpoint)')
parser.add_argument('--weights', type=str, default=None, help='weights path to store weights')
parser.add_argument('--config', default='coco', help='The config object to use.')
parser.add_argument('--tfrecord_dir', type=str, default='/content/drive/MyDrive/YPNet1/yp-Efficient/data/', help='directory of tfrecord')
parser.add_argument('--train_iter', type=int, default=1000, help='train iteration.')

args = parser.parse_args()

@tf.function
def train_step(model,
        loss_fn,
        metrics,
        optimizer,
        image,
        labels,
        num_class):
  # training using tensorflow gradient tape
  with tf.GradientTape() as tape:
      output = model(image, training=True)
      for i,j in labels.items():
        print(i,j.shape)
      loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn.call(output, labels, num_class)
      print(loc_loss, conf_loss, mask_loss, seg_loss, total_loss)
  grads = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  metrics.update_state(total_loss)
  return loc_loss, conf_loss, mask_loss, seg_loss
"""
def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    
    X, Y = tup
    
    Y= Y[:, -1]
    
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    
    yield X, Y
"""
def train():
  model=Yolact(**cfg.model_params)

  

  @contextlib.contextmanager
  def options(opts):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(opts)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)
  

  
  for layer in model.layers:
      if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
          layer.add_loss(lambda: tf.keras.regularizers.l2(args.weight_decay)(layer.kernel))
      if hasattr(layer, 'bias_regularizer') and layer.use_bias:
          layer.add_loss(lambda: tf.keras.regularizers.l2(args.weight_decay)(layer.bias))
  dateset = ObjectDetectionDataset(tfrecord_dir=os.path.join(args.tfrecord_dir,'coco/train/'),
                    anchor_instance=model.num_anchors,
                    **cfg.parser_params)
  train_dataset = dateset.get_dataloader(subset='train', batch_size=args.batch)
  valid_dataset = dateset.get_dataloader(subset='val', batch_size=1)
  num_val = 0
  for _ in valid_dataset:
    num_val += 1
  #-------------------------------------------------------------------
  # Choose the Optimizor, Loss Function, and Metrics, learning rate schedule
  lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(**cfg.LR_STAGE)
  #logging.info("Initiate the Optimizer and Loss function...")
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=args.momentum)
  yolact_loss = YOLACTLoss(**cfg.loss_params)
  train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  loc = tf.keras.metrics.Mean('loc_loss', dtype=tf.float32)
  conf = tf.keras.metrics.Mean('conf_loss', dtype=tf.float32)
  mask = tf.keras.metrics.Mean('mask_loss', dtype=tf.float32)
  seg = tf.keras.metrics.Mean('seg_loss', dtype=tf.float32)
  # -----------------------------------------------------------------
  input1=tf.keras.Input(shape=(160,160,3))
  output1=model.call(input1)
  #print(output)
  model =tf.keras.Model(inputs=input1,outputs=output1)
  model.compile(optimizer=optimizer, loss=YOLACTLoss)
  #tensorboard
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = './logs/gradient_tape/' + current_time + '/train'
  test_log_dir = './logs/gradient_tape/' + current_time + '/test'
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  #-------------------------------------------------------------------
  
  # setup checkpoints manager
  """
  filepath="./saved_model/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
  mode='min')
  callbacks_list = [checkpoint]
  model.summary()
  #plot_model(model ,to_file='model.png',show_shapes=True,show_dtype=True)
  #model.load_weights('./saved_model/eff2_fix1.keras')
  model.compile(optimizer=optimizer, loss=_criterion)
  
  history=model.fit_generator(
    gen(20, args.host, port=args.port),
    #steps_per_epoch=25623,
    steps_per_epoch=2,
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port),
    #validation_steps=5000,verbose=1,callbacks=callbacks_list)
    validation_steps=5,verbose=1,callbacks=callbacks_list)
  np.save('./loss_history/loss',np.array(history.history['loss']))
  np.save('./loss_history/val_loss',np.array(history.history['val_loss']))
  print("Saving model weights and configuration file.") 
  

  
  filepath="./checkpoint/*.ckpt"
  checkpoint_dir=os.path.dirname(filepath)
  checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=filepath,verbose=1,save_weights_only=True,period=1)
  model.fit(train_image,train_labels,epochs=args.epoch,callback=[cp_callback],validation_data=(test_images,test_labels))
  latesr=tf.train.latest_checkpoint(checkpoint_dir)
  model.load_weights(latest)
  """
  checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(
      checkpoint, directory="./checkpoints", max_to_keep=5
  )
  # restore from latest checkpoint and iteration
  status = checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
      logging.info("Restored from {}".format(manager.latest_checkpoint))
  else:
      logging.info("Initializing from scratch.")
  best_masks_map = 0.
  iterations = checkpoint.step.numpy()
  #print(train_dataset)
  x=[]
  loss={"t":[],"l":[],"c"[],"m":[],"s":[]}
  for epochs in range(args.epoch):
    
    for image, labels in train_dataset:
      #print(image,labels)
      # check iteration and change the learning rate
      if iterations > args.train_iter:
          break
      checkpoint.step.assign_add(1)
      
      #with options({'constant_folding': True,
      #              'layout_optimize': True,
      #              'loop_optimization': True,
      #              'arithmetic_optimization': True,
      #              'remapping': True}):
      loc_loss, conf_loss, mask_loss, seg_loss = train_step(model, yolact_loss, train_loss, optimizer, image, labels, cfg.model_params["num_class"])
    for val_image,val_label in valid_dataset:
      loc_loss, conf_loss, mask_loss, seg_loss = train_step(model, yolact_loss, train_loss, optimizer, val_image, val_labels, cfg.model_params["num_class"])
    loc.update_state(loc_loss)
    conf.update_state(conf_loss)
    mask.update_state(mask_loss)
    seg.update_state(seg_loss)
      #with train_summary_writer.as_default():
      #    tf.summary.scalar('Total loss', train_loss.result(), step=iterations)
      #    tf.summary.scalar('Loc loss', loc.result(), step=iterations)
      #    tf.summary.scalar('Conf loss', conf.result(), step=iterations)
      #    tf.summary.scalar('Mask loss', mask.result(), step=iterations)
      #    tf.summary.scalar('Seg loss', seg.result(), step=iterations)
    
    if iterations and (iterations % args.save_interval)==0
      tf.print("Iteration {:5d}, LR: {:9.8f}, Total Loss: {:6.3f}, B: {:6.3f},  C: {:6.3f}, M: {:6.3f}, S:{:6.3f} ".format(
              iterations,
              optimizer._decayed_lr(var_dtype=tf.float32),
              train_loss.result(),
              loc.result(),
              conf.result(),
              mask.result(),
              seg.result()
          ))
      x.append(iteration)
      loss["t"].append(train_loss.result())
      loss["l"].append(loc.result())
      loss["c"].append(conf.result())
      loss["m"].append(mask.result())
      s.append(seg.result())
    #if iterations and iterations % args.save_interval == 0:
      # save checkpoint
      save_path = manager.save()
      logging.info("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
      # validation and print mAP table
      all_map = evaluate(model, valid_dataset, num_val, cfg.model_params["num_class"])
      box_map, mask_map = all_map['box']['all'], all_map['mask']['all']
      tf.print(f"box mAP:{box_map}, mask mAP:{mask_map}")
      # reset the metrics
      train_loss.reset_states()
      loc.reset_states()
      conf.reset_states()
      mask.reset_states()
      seg.reset_states()    
      with test_summary_writer.as_default():
          tf.summary.scalar('Box mAP', box_map, step=iterations)
          tf.summary.scalar('Mask mAP', mask_map, step=iterations)
      print(mask_map,best_masks_map)
      # Saving the weights:
      #if mask_map > best_masks_map:
    if iterations == args.save_interval:
      best_masks_map = mask_map
      model.save_weights('./weights/{str(best_masks_map)+{str(iterations)}.h5')
    for a,i,j in enumerate(loss.items()):
      plt.subplot(2,3,a+1)      
      plt.xlabel("iterations")
      plt.ylabel(str(i)+"loss")
      plt.plot(x,j)
    plt.show()
  model.save_weights('./weights/{str(iterations)}.h5')

  model.save('./weights/'+str(arg.save_interval)+'.h5')
  model.save_weights("./saved_model/mask.keras", True)
  with open('./saved_model/mask.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
  



if __name__=="__main__":
  #gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
  #assert len(gpu) == 1
  #tf.config.experimental.set_memory_growth(gpu[0], True)
  #os.environ["CUDA_VISIBLE_DEVICES"]="0" # 使用编号为1，2号的GPU
  train()
  