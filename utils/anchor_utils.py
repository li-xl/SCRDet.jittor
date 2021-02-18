import jittor as jt 


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios,
                 featuremap_height, featuremap_width,
                 stride):
    '''
    :param base_anchor_size:256
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    '''
    base_anchor = jt.array([0, 0, base_anchor_size, base_anchor_size],dtype="float32")  # [x_center, y_center, w, h]

    ws, hs = enum_ratios(enum_scales(base_anchor, anchor_scales),anchor_ratios)  # per locations ws and hs

    x_centers = jt.index((featuremap_width,), dim=0).float32() * stride
    y_centers = jt.index((featuremap_height,), dim=0).float32() * stride

    x_centers, y_centers = jt.meshgrid(x_centers, y_centers)

    ws, x_centers = jt.meshgrid(ws, x_centers)
    hs, y_centers = jt.meshgrid(hs, y_centers)

    anchor_centers = jt.stack([x_centers, y_centers], 2)
    anchor_centers = jt.reshape(anchor_centers, [-1, 2])

    box_sizes = jt.stack([ws, hs], dim=2)
    box_sizes = jt.reshape(box_sizes, [-1, 2])
    anchors = jt.contrib.concat([anchor_centers - 0.5*box_sizes,
                             anchor_centers + 0.5*box_sizes], axis=1)
    return anchors


def enum_scales(base_anchor, anchor_scales):
    anchor_scales = base_anchor * (jt.array(anchor_scales).reshape(-1,1).float32())
    return anchor_scales


def enum_ratios(anchors, anchor_ratios):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    sqrt_ratios = jt.sqrt(jt.array(anchor_ratios)).unsqueeze(1)

    ws = jt.reshape(ws / sqrt_ratios, [-1, 1])
    hs = jt.reshape(hs * sqrt_ratios, [-1, 1])
    return ws, hs


def shift_anchor(anchors, stride):
    shift_delta = [(stride // 2, 0, stride // 2, 0),
                   (0, stride // 2, 0, stride // 2),
                   (stride // 2, stride // 2, stride // 2, stride // 2)]
    coord = []
    for i in range(anchors.shape[1]):
        coord.append(anchors[:,i])
    anchors_shift = [anchors]
    for delta in shift_delta:
        coord_shift = []
        for sd, c in zip(delta, coord):
            coord_shift.append(sd + c)
        tmp = jt.transpose(jt.stack(coord_shift))
        anchors_shift.append(tmp)
    anchors_ = jt.contrib.concat(anchors_shift, dim=1)
    anchors_ = jt.reshape(anchors_, [-1, 4])

    return anchors_