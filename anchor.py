from itertools import product
from math import sqrt
import tensorflow as tf

class Anchor(object):
    def __init__(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        """
        self.num_anchors, self.priors = self._generate_anchors(img_size, feature_map_size, aspect_ratio, scale)

    def _generate_anchors(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        :return:
        """
        prior_boxes = []
        num_anchors = 0
        for idx, f_size in enumerate(feature_map_size):
            # print("Create priors for f_size:%s", f_size)
            count_anchor = 0
            for j, i in product(range(f_size), range(f_size)):
                x = (i + 0.5) / f_size
                y = (j + 0.5) / f_size
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a / img_size
                    h = scale[idx] / a / img_size

                    # original author make all priors squre
                    h = w

                    # directly use point form here => [ymin, xmin, ymax, xmax]
                    ymin = y - (h / 2.)
                    xmin = x - (w / 2.)
                    ymax = y + (h / 2.)
                    xmax = x + (w / 2.)
                    prior_boxes += [ymin * img_size, xmin * img_size, ymax * img_size, xmax * img_size]
                count_anchor += 1
            num_anchors += count_anchor
            # print(f_size, count_anchor)
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        return num_anchors, output
    
    def _pairwise_intersection(self, gt_bbox):
        """
        ref: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py
        :param gt_bbox: [num_obj, 4]
        :return:
        """

        # unstack the ymin, xmin, ymax, xmax
        ymin_anchor, xmin_anchor, ymax_anchor, xmax_anchor = tf.unstack(self.priors, axis=-1)
        ymin_gt, xmin_gt, ymax_gt, xmax_gt = tf.unstack(gt_bbox, axis=-1)

        # calculate intersection
        all_pairs_max_xmin = tf.math.maximum(tf.expand_dims(xmin_anchor, axis=-1), tf.expand_dims(xmin_gt, axis=0))
        all_pairs_min_xmax = tf.math.minimum(tf.expand_dims(xmax_anchor, axis=-1), tf.expand_dims(xmax_gt, axis=0))
        all_pairs_max_ymin = tf.math.maximum(tf.expand_dims(ymin_anchor, axis=-1), tf.expand_dims(ymin_gt, axis=0))
        all_pairs_min_ymax = tf.math.minimum(tf.expand_dims(ymax_anchor, axis=-1), tf.expand_dims(ymax_gt, axis=0))
        # tf.print("all max xmin", all_pairs_max_xmin)
        # tf.print("all max xmax", all_pairs_min_xmax)
        # tf.print("all max ymin", all_pairs_max_ymin)
        # tf.print("all max ymax", all_pairs_min_ymax)
        intersect_heights = tf.math.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        intersect_widths = tf.math.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        # tf.print("intersect_heights", intersect_heights)
        # tf.print("intersect_widths", intersect_widths)

        return intersect_heights * intersect_widths

    def _pairwise_iou(self, gt_bbox, is_crowd=False):
        """
         ref: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py
        :param gt_bbox: [num_obj, 4]
        :return:
        """
        # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)
        # calculate A ∩ B (pairwise)
        pairwise_inter = self._pairwise_intersection(gt_bbox=gt_bbox)
        # tf.print("pairwaise inter", pairwise_inter)
        # calculate areaA, areaB
        ymin_anchor, xmin_anchor, ymax_anchor, xmax_anchor = tf.unstack(self.priors, axis=-1)
        ymin_gt, xmin_gt, ymax_gt, xmax_gt = tf.unstack(gt_bbox, axis=-1)

        area_anchor = (xmax_anchor - xmin_anchor) * (ymax_anchor - ymin_anchor)
        area_gt = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
        # tf.print("area anchor", area_anchor)
        # tf.print("area gt", area_gt)

        # create same shape of matrix as intersection
        pairwise_area = tf.expand_dims(area_anchor, axis=-1) + tf.expand_dims(area_gt, axis=0)
        # tf.print("pairwise area", pairwise_area)

        # calculate A ∪ B, consider crowd situation
        if is_crowd:
            pairwise_union = tf.expand_dims(area_gt, axis=0)
        else:
            pairwise_union = pairwise_area - pairwise_inter

        # IOU(Jaccard overlap) = intersection / union, there might be possible to have division by 0
        return pairwise_inter / pairwise_union

    def get_anchors(self):
        return self.priors
    def matching(self, gt_bbox, gt_labels, num_crowd=0, threshold_pos=0.5, threshold_neg=0.4, threshold_crowd=0.7):
        """
        :param gt_bbox:
        :param gt_labels:
        :return:
        Args:
            num_crowd:
            threshold_pos:
            threshold_neg:
            threshold_crowd:
            pos_iou_threshold:
            num_crowd:
            neg_iou_threshold:
        """
        if num_crowd > 0:
            # split the gt_bbox
            gt_bbox = gt_bbox[:-num_crowd]
            crowd_gt_bbox = gt_bbox[-num_crowd:]
        else:
            crowd_gt_bbox = tf.zeros_like(gt_bbox)

        # Matching only for non-crowd annotation
        # --------------------------------------------------------------------------------------------------------------
        num_gt = tf.shape(gt_bbox)[0]
        # tf.print("num gt", num_gt)
        # pairwise IoU
        pairwise_iou = self._pairwise_iou(gt_bbox=gt_bbox, is_crowd=False)
        # tf.print("pairwise_iou", tf.shape(pairwise_iou))
        # assign the max overlap gt index for each anchor
        max_iou_for_anchors = tf.reduce_max(pairwise_iou, axis=-1)
        max_id_for_anchors = tf.math.argmax(pairwise_iou, axis=-1)

        # force the anchors which is the best matched of each gt to predict the correspond gt
        forced_update_id = tf.cast(tf.range(0, num_gt), tf.int64)

        # force the iou over threshold for not wasting any training data
        forced_update_iou = tf.reduce_max(pairwise_iou, axis=0)
        # make sure the it won't be filtered even if under negative threshold
        forced_update_iou += (2-forced_update_iou)
        # tf.print("forced_update_iou", forced_update_iou)
        forced_update_indice = tf.expand_dims(tf.math.argmax(pairwise_iou, axis=0), axis=-1)

        # assign the pair (the gt for priors to predict)
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, forced_update_indice, forced_update_iou)
        max_id_for_anchors = tf.tensor_scatter_nd_update(max_id_for_anchors, forced_update_indice, forced_update_id)

        # decide the anchors to be positive or negative based on the IoU and given threshold
        pos_iou = tf.where(max_iou_for_anchors > threshold_pos)
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, pos_iou, tf.ones(tf.size(pos_iou)))
        neg_iou = tf.where(max_iou_for_anchors < threshold_neg)
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, neg_iou, tf.zeros(tf.size(neg_iou)))
        neu_iou = tf.where(
            tf.math.logical_and((max_iou_for_anchors <= threshold_pos), max_iou_for_anchors >= threshold_neg))
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, neu_iou, -1 * tf.ones(tf.size(neu_iou)))

        # deal with crowd annotations, only affect non-positive
        # --------------------------------------------------------------------------------------------------------------
        if num_crowd > 0 and threshold_crowd < 1:
            # crowd pairwise IoU
            crowd_pairwise_iou = self._pairwise_iou(gt_bbox=crowd_gt_bbox, is_crowd=True)

            # assign the max overlap gt index for each anchor
            crowd_max_iou_for_anchors = tf.reduce_max(crowd_pairwise_iou, axis=-1)

            # assign neutral for those neg iou that over crowd threshold
            crowd_neu_iou = tf.where(
                tf.math.logical_and((max_iou_for_anchors <= 0), crowd_max_iou_for_anchors > threshold_crowd))

            # reassigh from negative to neutral
            max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, crowd_neu_iou,
                                                              -1 * tf.ones(tf.size(crowd_neu_iou)))
        match_positiveness = max_iou_for_anchors

        # create class target
        # map idx to label[idx]
        # match_labels = tf.map_fn(lambda x: gt_labels[x], max_id_for_anchors)
        match_labels = tf.gather(gt_labels, max_id_for_anchors)

        """
        element-wise multiplication of label[idx] and positiveness:
        1. positive sample will have correct label
        2. negative sample will have 0 * label[idx] = 0
        3. neural sample will have -1 * label[idx] = -1 * label[idx] 
        it can be useful to distinguish positive sample during loss calculation  
        """
        target_cls = tf.multiply(tf.cast(match_labels, tf.float32), match_positiveness)

        # create loc target
        # map_loc = tf.map_fn(lambda x: gt_bbox[x], max_id_for_anchors, dtype=tf.float32)
        map_loc = tf.gather(gt_bbox, max_id_for_anchors)

        # convert to center form [cx, cy, w, h]
        # center_anchors = tf.map_fn(lambda x: map_to_center_form(x), self.priors)
        h = self.priors[:, 2] - self.priors[:, 0]
        w = self.priors[:, 3] - self.priors[:, 1]
        center_anchors = tf.stack([self.priors[:, 1] + (w / 2), self.priors[:, 0] + (h / 2), w, h], axis=-1)

        # center_gt = tf.map_fn(lambda x: map_to_center_form(x), map_loc)
        h = map_loc[:, 2] - map_loc[:, 0]
        w = map_loc[:, 3] - map_loc[:, 1]
        center_gt = tf.stack([map_loc[:, 1] + (w / 2), map_loc[:, 0] + (h / 2), w, h], axis=-1)
        variances = [0.1, 0.2]

        # calculate offset
        # target_loc = tf.map_fn(lambda x: map_to_offset(x), tf.stack([center_gt, center_anchors], axis=-1))
        g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]) / center_anchors[:, 2] / variances[0]
        g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]) / center_anchors[:, 3] / variances[0]
        tf.debugging.assert_non_negative(center_anchors[:, 2] / center_gt[:, 2])
        tf.debugging.assert_non_negative(center_anchors[:, 3] / center_gt[:, 3])
        g_hat_w = tf.math.log(center_gt[:, 2] / center_anchors[:, 2]) / variances[1]
        g_hat_h = tf.math.log(center_gt[:, 3] / center_anchors[:, 3]) / variances[1]
        target_loc = tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)
        return target_cls, target_loc, max_id_for_anchors, match_positiveness