from collections import OrderedDict
import argparse
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from backbone import Yolact
from utils.APObject import APObject
from utils.utils import jaccard, mask_iou, postprocess

iou_thresholds = [x / 100 for x in range(50, 100, 5)]


parser = argparse.ArgumentParser(description='Steering angle model trainer')
parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
parser.add_argument('--port', type=int, default=5557, help='Port of server.')
parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
parser.add_argument('--batch', type=int, default=8, help='Batch size.')

args = parser.parse_args()
# for calculating IOU between gt and detection box
# so as to decide the TP, FP, FN
def _bbox_iou(bbox1, bbox2, is_crowd=False):
    ret = jaccard(bbox1, bbox2, is_crowd)
    return ret


# for calculating IOU between gt and detection mask
def _mask_iou(mask1, mask2, is_crowd=False):
    ret = mask_iou(mask1, mask2, is_crowd)
    return ret


# ref from original arthor
def calc_map(ap_data, num_cls):
    tf.print("Calculating mAP...")

    # create empty list of dict for different iou threshold
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    # 　calculate Ap for every classes individually
    for _class in range(num_cls):
        # each class have multiple different iou threshold to calculate
        for iou_idx in range(len(iou_thresholds)):
            # there are 2 type of mAP we want to know (bounding box and mask)
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                # calculate AP if there is detection in certain class
                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            print(mAP)
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


# ref from original arthor
def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    tf.print()
    tf.print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    tf.print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        tf.print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    tf.print(make_sep(len(all_maps['box']) + 1))
    tf.print()


# ref from original arthor
def prep_metrics(ap_data, dets, img, labels, image_id=None):
    """Mainly update the ap_data for validation table"""
    # get the shape of image
    w = tf.shape(img)[1]
    h = tf.shape(img)[2]
    # tf.print(f"img size (w, h):{w}, {h}")

    # Load prediction
    classes, scores, boxes, masks = postprocess(dets, w, h, 0, "bilinear")

    # if no detection or only one detection
    if classes is None:
        return
    if tf.size(scores) == 1:
        scores = tf.expand_dims(scores, axis=0)
        masks = tf.expand_dims(masks, axis=0)
    boxes = tf.expand_dims(boxes, axis=0)
    #
    # tf.print("prep classes", tf.shape(classes))
    # tf.print("prep scores", tf.shape(scores))
    # tf.print("prep boxes", tf.shape(boxes))
    # tf.print("prep masks", tf.shape(masks))

    # Load gt
    gt_bbox = labels['bbox']
    gt_classes = labels['classes']
    gt_masks = labels['mask_target']
    num_crowd = labels['num_crowd']
    num_obj = labels['num_obj']

    # convert to scalar
    num_crowd = num_crowd.numpy()[0]
    num_obj = num_obj.numpy()[0]

    if num_crowd > 0:
        split = lambda x: (x[:, num_obj - num_crowd:num_obj], x[:, :num_obj - num_crowd])
        gt_crowd_boxes, gt_bbox = split(gt_bbox)
        gt_crowd_classes, gt_classes = split(gt_classes)
        gt_crowd_masks, gt_masks = split(gt_masks)
        gt_crowd_classes = list(gt_crowd_classes[0].numpy())

        # tf.print('split gt bbox', tf.shape(gt_bbox))
        # tf.print('split gt classes', tf.shape(gt_classes))
        # tf.print('split gt masks', tf.shape(gt_masks))
        #
        # tf.print("split crowd boxes", tf.shape(gt_crowd_boxes))
        # tf.print("split crowd masks", tf.shape(gt_crowd_masks))
        # tf.print("split crowd classes length", len(gt_crowd_classes))

    else:
        # get rid of the padding
        gt_classes = gt_classes[:, :num_obj]
        gt_bbox = gt_bbox[:, :num_obj]
        gt_masks = gt_masks[:, :num_obj]

    # prepare data
    classes = list(classes.numpy())
    scores = list(scores.numpy())
    box_scores = scores
    mask_scores = scores

    # if output json, add things to detections objects

    # else
    num_pred = len(classes)
    num_gt = num_obj - num_crowd

    # tf.print("num pred", num_pred)
    # tf.print("num gt", num_gt)
    #
    # tf.print('prep gt bbox', tf.shape(gt_bbox))
    # tf.print('prep gt classes', tf.shape(gt_classes))
    # tf.print('prep gt masks', tf.shape(gt_masks))
    # tf.print('prep num crowd', tf.shape(num_crowd))
    # tf.print("prep num obj", tf.shape(num_obj))

    # resize gt mask
    # should be [num_gt, w, h]
    masks_gt = tf.squeeze(tf.image.resize(tf.expand_dims(gt_masks[0], axis=-1), [h, w],
                                          method='bilinear'), axis=-1)

    # calculating the IOU first
    mask_iou_cache = _mask_iou(masks, masks_gt).numpy()
    bbox_iou_cache = tf.squeeze(_bbox_iou(boxes, gt_bbox).numpy(), axis=0)
    # tf.print(tf.shape(boxes))
    # tf.print(tf.shape(gt_bbox))
    # tf.print(tf.shape(bbox_iou_cache))
    # tf.print("non crowd mask iou shape:", tf.shape(mask_iou_cache))
    # tf.print("non crowd bbox iou shape:", tf.shape(bbox_iou_cache))

    # If crowd label included, split it and calculate iou separately from non-crowd label
    if num_crowd > 0:
        # resize gt mask
        # should be [num_crowd, w, h]
        gt_crowd_masks = tf.squeeze(tf.image.resize(tf.expand_dims(gt_crowd_masks[0], axis=-1), [h, w],
                                                    method='bilinear'), axis=-1)
        # tf.print("crowd masks", tf.shape(gt_crowd_masks))
        crowd_mask_iou_cache = _mask_iou(masks, gt_crowd_masks, is_crowd=True).numpy()
        crowd_bbox_iou_cache = tf.squeeze(
            _bbox_iou(boxes, gt_crowd_boxes, is_crowd=True), axis=0).numpy()
        # tf.print("gt crowd mask iou shape:", tf.shape(crowd_mask_iou_cache))
        # tf.print("gt crowd bbox iou shape:", tf.shape(crowd_bbox_iou_cache))
    else:
        crowd_mask_iou_cache = None
        crowd_bbox_iou_cache = None

    # get the sorted index of scores (descending order)
    box_indices = sorted(range(num_pred), key=lambda idx: -box_scores[idx])
    mask_indices = sorted(box_indices, key=lambda idx: -mask_scores[idx])

    # define some useful lambda function for next section
    # avoid writing "bbox_iou_cache[row, col]" too many times, wrap it as a lambda func
    iou_types = [
        ('box', lambda row, col: bbox_iou_cache[row, col],
         lambda row, col: crowd_bbox_iou_cache[row, col],
         lambda idx: box_scores[idx], box_indices),
        ('mask', lambda row, col: mask_iou_cache[row, col],
         lambda row, col: crowd_mask_iou_cache[row, col],
         lambda idx: mask_scores[idx], mask_indices)]

    gt_classes = list(gt_classes[0].numpy())

    # starting to update the ap_data from this batch
    for _class in set(classes + gt_classes):
        # calculating how many labels belong to this class
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            th = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                # get certain APobject
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positive(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue

                    max_iou_found = th
                    max_match_idx = -1

                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                        iou = iou_func(i, j)
                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        matched_crowd = False
                        if num_crowd > 0:
                            for j in range(len(gt_crowd_classes)):
                                if gt_crowd_classes[j] != _class:
                                    continue
                                iou = crowd_func(i, j)

                                if iou > th:
                                    matched_crowd = True
                                    break
                        # for crowd annotation, if no, push as false positive
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)


def prep_benchmarks():
    ...


def prep_display():
    ...


def eval_image():
    ...


def eval_images():
    ...


def eval_video():
    ...


def evaluate(model, dataset, num_val, num_cls):
    # if use fastnms
    # if use cross class nms

    # if eval image
    # if eval images
    # if eval video

    # if not display or benchmark
    # For mAP evaluation, creating AP_Object for every class per iou_threshold
    ap_data = {
        'box': [[APObject() for _ in range(num_cls)] for _ in iou_thresholds],
        'mask': [[APObject() for _ in range(num_cls)] for _ in iou_thresholds]}

    # detection object made from prediction output. for the purpose of creating json
    #detections = Detections()

    # iterate the whole dataset to save TP, FP, FN
    i = 0
    print(num_val)
    print(dataset)
    progbar = Progbar(num_val)
    tf.print("Evaluating...")
    for image, labels in dataset:
        i += 1
        output = model(image, training=False)
        dets = model.detection(output)
        # update ap_data or detection depends if u want to save it to json or just for validation table
        prep_metrics(ap_data, dets, image, labels)
        progbar.update(i)

    # if to json
    # save detection to json

    # Todo if not training, save ap_data, else calc_map
    return calc_map(ap_data, num_cls)


def main(argv):
    pass


if __name__ == '__main__':
  
  print('Loading model...', end='')
  net = Yolact()
  net.load_weights(args.trained_model)
  net.eval()
  print(' Done.')
  if args.cuda:
    net = net.cuda()
  evaluate(net, dataset)
