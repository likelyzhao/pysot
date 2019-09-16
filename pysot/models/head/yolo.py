import torch
import torch.nn as nn
import numpy as np
from .rpn import DepthwiseXCorr

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1)
        class_pred = class_pred[0]
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1,  keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            nms_in = detections_class.cpu().numpy()
            nms_out_index = nms(
                nms_in[:, :4], nms_thre, score=nms_in[:, 4]*nms_in[:, 5])
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output

class YOLOLayer(nn.Module):
    def __init__(self, anchors, stride, in_ch, ignore_thre=0.7):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.anch_mask = [0,1,2]
        self.n_anchors = len(anchors)
        self.n_classes = 1
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.stride = stride
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        #self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
        self.masked_anchors = self.all_anchors_grid
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)

        self.conv = DepthwiseXCorr(in_ch, in_ch, self.n_anchors * (self.n_classes + 5))
        # self.conv_2 = nn.Conv2d(in_channels=in_ch,
        #                       out_channels=self.n_anchors * (self.n_classes + 5),
        #                       kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_f, x_f, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(z_f,x_f)
        #output = self.conv_2(z_f)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if z_f.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # logistic activation for xy, obj, cls
        #output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])
        output_sigmoid = output.clone()
        output_sigmoid[..., np.r_[:2, 4:n_ch]] = self.sigmoid(output[..., np.r_[:2, 4:n_ch]])


        # calculate pred - xywh obj cls

        x_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        #pred = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).type(dtype)
        pred = output_sigmoid.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if labels is None:
            pred[..., :4] *= self.stride
            pred = pred.view(batchsize, -1, n_ch).data
            output  = postprocess(pred, self.n_classes,0.01)
            bboxes = []
            scores = []
            for out in output[0]:
                scores.append(float(out[4].data.item() * out[5].data.item()))
                x1 = float(out[0])
                y1 = float(out[1])
                x2 = float(out[2])
                y2 = float(out[3])
                bboxes.append([x1,y1,x2-x1,y2-y1])


            bboxes = np.array(bboxes).transpose(1,0)

            return scores , bboxes

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).type(dtype)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).type(dtype)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).type(dtype)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).type(dtype)

        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))
            #best_n_mask = [1]
            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            #print(pred_best_iou.max())
            pred_best_iou = (pred_best_iou > self.ignore_thre).type(torch.int)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            #print(pred_best_iou.max())
            obj_mask[b] = 1 - pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        output_mask = output_sigmoid.clone()
        output_mask[..., 4] = output_sigmoid[..., 4] *  obj_mask
        output_mask[..., np.r_[0:4, 5:n_ch]] = output_sigmoid[..., np.r_[0:4, 5:n_ch]] * tgt_mask
        output_mask[..., 2:4] = output_sigmoid[..., 2:4] * tgt_scale

        target[..., 4] = target[..., 4] * obj_mask
        target[..., np.r_[0:4, 5:n_ch]] = target[..., np.r_[0:4, 5:n_ch]] * tgt_mask
        target[..., 2:4] = target[..., 2:4] * tgt_scale

        loss_l2 = self.l2_loss(output_mask, target)
        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale, size_average=False)
        loss_xy = bceloss(output_mask[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output_mask[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output_mask[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output_mask[..., 5:], target[..., 5:])
    

        loss = loss_xy + loss_wh + loss_obj + loss_cls
        #loss = loss_wh 

        return loss, loss_cls, loss_l2
