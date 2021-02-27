import tensorflow as tf 
import jittor as jt
import numpy as np 
import pickle

# Batchnorm Beta -> bias,Gamma ->weight,moving_mean->running_mean,moving_variance->running_var
weight_map={
    "biases":"bias",
    "weights":"weight",
    "beta":"bias",
    "gamma":"weight",
    "moving_mean":"running_mean",
    "moving_variance":"running_var"
}
name_map ={
    "Branch_0/conv2d_0a_1x1":"fpn.sfnet.inception_module.branch_0_conv",
    "Branch_1/conv2d_0a_1x1":"fpn.sfnet.inception_module.branch_1_conv1",
    "Branch_1/conv2d_0b_1x7":"fpn.sfnet.inception_module.branch_1_conv2",
    "Branch_1/conv2d_0c_7x1":"fpn.sfnet.inception_module.branch_1_conv3",
    "Branch_2/conv2d_0a_1x1":"fpn.sfnet.inception_module.branch_2_conv1",
    "Branch_2/conv2d_0b_7x1":"fpn.sfnet.inception_module.branch_2_conv2",
    "Branch_2/Conv2d_0c_1x7":"fpn.sfnet.inception_module.branch_2_conv3",
    "Branch_2/conv2d_0d_7x1":"fpn.sfnet.inception_module.branch_2_conv4",
    "Branch_2/conv2d_0e_1x7":"fpn.sfnet.inception_module.branch_2_conv5",
    "Branch_3/conv2d_0b_1x1":"fpn.sfnet.inception_module.branch_3_conv",
    "c3c4/reduce_dim_c3c4":"fpn.sfnet.fusion_conv",
    "build_attention/Branch_0/conv2d_0a_1x1":"fpn.mdanet.inception_attention.inception_module.branch_0_conv",
    "build_attention/Branch_1/conv2d_0a_1x1":"fpn.mdanet.inception_attention.inception_module.branch_1_conv1",
    "build_attention/Branch_1/conv2d_0b_1x7":"fpn.mdanet.inception_attention.inception_module.branch_1_conv2",
    "build_attention/Branch_1/conv2d_0c_7x1":"fpn.mdanet.inception_attention.inception_module.branch_1_conv3",
    "build_attention/Branch_2/conv2d_0a_1x1":"fpn.mdanet.inception_attention.inception_module.branch_2_conv1",
    "build_attention/Branch_2/conv2d_0b_7x1":"fpn.mdanet.inception_attention.inception_module.branch_2_conv2",
    "build_attention/Branch_2/Conv2d_0c_1x7":"fpn.mdanet.inception_attention.inception_module.branch_2_conv3",
    "build_attention/Branch_2/conv2d_0d_7x1":"fpn.mdanet.inception_attention.inception_module.branch_2_conv4",
    "build_attention/Branch_2/conv2d_0e_1x7":"fpn.mdanet.inception_attention.inception_module.branch_2_conv5",
    "build_attention/Branch_3/conv2d_0b_1x1":"fpn.mdanet.inception_attention.inception_module.branch_3_conv",
    "build_attention/inception_attention_out":"fpn.mdanet.inception_attention.inception_attention_conv",
    "build_attention/SE_fully_connected1":"fpn.mdanet.ca_fc1",
    "build_attention/SE_fully_connected2":"fpn.mdanet.ca_fc2",
    "Fast-RCNN/build_fc_layers/fc1":"head.fc1",
    "Fast-RCNN/build_fc_layers/fc2":"head.fc2",
    "Fast-RCNN/horizen_branch/cls_fc_h":"head.cls_score_h",
    "Fast-RCNN/horizen_branch/reg_fc_h":"head.bbox_pred_h",
    "Fast-RCNN/rotation_branch/cls_fc_r":"head.cls_score_r",
    "Fast-RCNN/rotation_branch/reg_fc_r":"head.bbox_pred_r",
    "rpn_conv/3x3":"rpn.conv1",
    "rpn_bbox_pred":"rpn.loc",
    "rpn_cls_score":"rpn.score",
    "resnet50_v1d/C1/conv0/BatchNorm":"backbone.layer0.bn1",
    "resnet50_v1d/C1/conv0":"backbone.layer0.conv1",
    "resnet50_v1d/C1/conv1":"backbone.layer0.conv2",
    "resnet50_v1d/C1/conv1/BatchNorm":"backbone.layer0.bn2",
    "resnet50_v1d/C1/conv2":"backbone.layer0.conv3",
    "resnet50_v1d/C1/conv2/BatchNorm":"backbone.layer0.bn3",
    "resnet50_v1d/C2/bottleneck_0/conv0/BatchNorm":"backbone.layer1.0.bn1",
    "resnet50_v1d/C2/bottleneck_0/conv0":"backbone.layer1.0.conv1",
    "resnet50_v1d/C2/bottleneck_0/conv1/BatchNorm":"backbone.layer1.0.bn2",
    "resnet50_v1d/C2/bottleneck_0/conv1":"backbone.layer1.0.conv2",
    "resnet50_v1d/C2/bottleneck_0/conv2/BatchNorm":"backbone.layer1.0.bn3",
    "resnet50_v1d/C2/bottleneck_0/conv2":"backbone.layer1.0.conv3",
    "resnet50_v1d/C2/bottleneck_0/shortcut/BatchNorm":"backbone.layer1.0.downsample.1",
    "resnet50_v1d/C2/bottleneck_0/shortcut":"backbone.layer1.0.downsample.0",
    "resnet50_v1d/C2/bottleneck_1/conv0/BatchNorm":"backbone.layer1.1.bn1",
    "resnet50_v1d/C2/bottleneck_1/conv0":"backbone.layer1.1.conv1",
    "resnet50_v1d/C2/bottleneck_1/conv1/BatchNorm":"backbone.layer1.1.bn2",
    "resnet50_v1d/C2/bottleneck_1/conv1":"backbone.layer1.1.conv2",
    "resnet50_v1d/C2/bottleneck_1/conv2/BatchNorm":"backbone.layer1.1.bn3",
    "resnet50_v1d/C2/bottleneck_1/conv2":"backbone.layer1.1.conv3",
    "resnet50_v1d/C2/bottleneck_2/conv0/BatchNorm":"backbone.layer1.2.bn1",
    "resnet50_v1d/C2/bottleneck_2/conv0":"backbone.layer1.2.conv1",
    "resnet50_v1d/C2/bottleneck_2/conv1/BatchNorm":"backbone.layer1.2.bn2",
    "resnet50_v1d/C2/bottleneck_2/conv1":"backbone.layer1.2.conv2",
    "resnet50_v1d/C2/bottleneck_2/conv2/BatchNorm":"backbone.layer1.2.bn3",
    "resnet50_v1d/C2/bottleneck_2/conv2":"backbone.layer1.2.conv3",

    "resnet50_v1d/C3/bottleneck_0/conv0/BatchNorm":"backbone.layer2.0.bn1",
    "resnet50_v1d/C3/bottleneck_0/conv0":"backbone.layer2.0.conv1",
    "resnet50_v1d/C3/bottleneck_0/conv1/BatchNorm":"backbone.layer2.0.bn2",
    "resnet50_v1d/C3/bottleneck_0/conv1":"backbone.layer2.0.conv2",
    "resnet50_v1d/C3/bottleneck_0/conv2/BatchNorm":"backbone.layer2.0.bn3",
    "resnet50_v1d/C3/bottleneck_0/conv2":"backbone.layer2.0.conv3",
    "resnet50_v1d/C3/bottleneck_0/shortcut/BatchNorm":"backbone.layer2.0.downsample.1",
    "resnet50_v1d/C3/bottleneck_0/shortcut":"backbone.layer2.0.downsample.0",
    "resnet50_v1d/C3/bottleneck_1/conv0/BatchNorm":"backbone.layer2.1.bn1",
    "resnet50_v1d/C3/bottleneck_1/conv0":"backbone.layer2.1.conv1",
    "resnet50_v1d/C3/bottleneck_1/conv1/BatchNorm":"backbone.layer2.1.bn2",
    "resnet50_v1d/C3/bottleneck_1/conv1":"backbone.layer2.1.conv2",
    "resnet50_v1d/C3/bottleneck_1/conv2/BatchNorm":"backbone.layer2.1.bn3",
    "resnet50_v1d/C3/bottleneck_1/conv2":"backbone.layer2.1.conv3",
    "resnet50_v1d/C3/bottleneck_2/conv0/BatchNorm":"backbone.layer2.2.bn1",
    "resnet50_v1d/C3/bottleneck_2/conv0":"backbone.layer2.2.conv1",
    "resnet50_v1d/C3/bottleneck_2/conv1/BatchNorm":"backbone.layer2.2.bn2",
    "resnet50_v1d/C3/bottleneck_2/conv1":"backbone.layer2.2.conv2",
    "resnet50_v1d/C3/bottleneck_2/conv2/BatchNorm":"backbone.layer2.2.bn3",
    "resnet50_v1d/C3/bottleneck_2/conv2":"backbone.layer2.2.conv3",
    "resnet50_v1d/C3/bottleneck_3/conv0/BatchNorm":"backbone.layer2.3.bn1",
    "resnet50_v1d/C3/bottleneck_3/conv0":"backbone.layer2.3.conv1",
    "resnet50_v1d/C3/bottleneck_3/conv1/BatchNorm":"backbone.layer2.3.bn2",
    "resnet50_v1d/C3/bottleneck_3/conv1":"backbone.layer2.3.conv2",
    "resnet50_v1d/C3/bottleneck_3/conv2/BatchNorm":"backbone.layer2.3.bn3",
    "resnet50_v1d/C3/bottleneck_3/conv2":"backbone.layer2.3.conv3",

    "resnet50_v1d/C4/bottleneck_0/conv0/BatchNorm":"backbone.layer3.0.bn1",
    "resnet50_v1d/C4/bottleneck_0/conv0":"backbone.layer3.0.conv1",
    "resnet50_v1d/C4/bottleneck_0/conv1/BatchNorm":"backbone.layer3.0.bn2",
    "resnet50_v1d/C4/bottleneck_0/conv1":"backbone.layer3.0.conv2",
    "resnet50_v1d/C4/bottleneck_0/conv2/BatchNorm":"backbone.layer3.0.bn3",
    "resnet50_v1d/C4/bottleneck_0/conv2":"backbone.layer3.0.conv3",
    "resnet50_v1d/C4/bottleneck_0/shortcut/BatchNorm":"backbone.layer3.0.downsample.1",
    "resnet50_v1d/C4/bottleneck_0/shortcut":"backbone.layer3.0.downsample.0",
    "resnet50_v1d/C4/bottleneck_1/conv0/BatchNorm":"backbone.layer3.1.bn1",
    "resnet50_v1d/C4/bottleneck_1/conv0":"backbone.layer3.1.conv1",
    "resnet50_v1d/C4/bottleneck_1/conv1/BatchNorm":"backbone.layer3.1.bn2",
    "resnet50_v1d/C4/bottleneck_1/conv1":"backbone.layer3.1.conv2",
    "resnet50_v1d/C4/bottleneck_1/conv2/BatchNorm":"backbone.layer3.1.bn3",
    "resnet50_v1d/C4/bottleneck_1/conv2":"backbone.layer3.1.conv3",
    "resnet50_v1d/C4/bottleneck_2/conv0/BatchNorm":"backbone.layer3.2.bn1",
    "resnet50_v1d/C4/bottleneck_2/conv0":"backbone.layer3.2.conv1",
    "resnet50_v1d/C4/bottleneck_2/conv1/BatchNorm":"backbone.layer3.2.bn2",
    "resnet50_v1d/C4/bottleneck_2/conv1":"backbone.layer3.2.conv2",
    "resnet50_v1d/C4/bottleneck_2/conv2/BatchNorm":"backbone.layer3.2.bn3",
    "resnet50_v1d/C4/bottleneck_2/conv2":"backbone.layer3.2.conv3",
    "resnet50_v1d/C4/bottleneck_3/conv0/BatchNorm":"backbone.layer3.3.bn1",
    "resnet50_v1d/C4/bottleneck_3/conv0":"backbone.layer3.3.conv1",
    "resnet50_v1d/C4/bottleneck_3/conv1/BatchNorm":"backbone.layer3.3.bn2",
    "resnet50_v1d/C4/bottleneck_3/conv1":"backbone.layer3.3.conv2",
    "resnet50_v1d/C4/bottleneck_3/conv2/BatchNorm":"backbone.layer3.3.bn3",
    "resnet50_v1d/C4/bottleneck_3/conv2":"backbone.layer3.3.conv3",

    "resnet50_v1d/C4/bottleneck_4/conv0/BatchNorm":"backbone.layer3.4.bn1",
    "resnet50_v1d/C4/bottleneck_4/conv0":"backbone.layer3.4.conv1",
    "resnet50_v1d/C4/bottleneck_4/conv1/BatchNorm":"backbone.layer3.4.bn2",
    "resnet50_v1d/C4/bottleneck_4/conv1":"backbone.layer3.4.conv2",
    "resnet50_v1d/C4/bottleneck_4/conv2/BatchNorm":"backbone.layer3.4.bn3",
    "resnet50_v1d/C4/bottleneck_4/conv2":"backbone.layer3.4.conv3",

    "resnet50_v1d/C4/bottleneck_5/conv0/BatchNorm":"backbone.layer3.5.bn1",
    "resnet50_v1d/C4/bottleneck_5/conv0":"backbone.layer3.5.conv1",
    "resnet50_v1d/C4/bottleneck_5/conv1/BatchNorm":"backbone.layer3.5.bn2",
    "resnet50_v1d/C4/bottleneck_5/conv1":"backbone.layer3.5.conv2",
    "resnet50_v1d/C4/bottleneck_5/conv2/BatchNorm":"backbone.layer3.5.bn3",
    "resnet50_v1d/C4/bottleneck_5/conv2":"backbone.layer3.5.conv3",
}

def weight_convert(v):
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def convert_name(key):
    weight_names = list(weight_map.keys())
    names = list(name_map.keys())
    find_name = "" 
    for name in names:
        if name in key and len(name)>len(find_name):
            find_name=name
    assert len(find_name)>0,key
    find_weight = ""
    for weight in weight_names:
        if weight in key:
            find_weight = weight
    key = key.replace(find_name,name_map[find_name])
    key = key.replace(find_weight,weight_map[find_weight])
    key = key.replace("/",".")
    return key
    

def read_tf_weights(ckpt):
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    keys = sorted(var_to_shape_map)
    jt_result = {}
    for key in keys:
        shape = var_to_shape_map[key]
        if not ("Momentum" in key or "ExponentialMovingAverage" in key or "C5" in key or "global_step"==key): 
            tensor = reader.get_tensor(key)
            tensor = weight_convert(tensor)
            new_key = convert_name(key)
            # print(key)
            # print(new_key)
            jt_result[new_key]=tensor

    pickle.dump(jt_result,open("test.pkl","wb"))
            

def read_jt_weights(ckpt):
    data = jt.load(ckpt)
    keys = sorted(data)
    for key in keys:
        tensor = data[key]
        print(key,type(tensor))

def compare_model(jt_ckpt,tf_ckpt):
    jt_model = pickle.load(open(jt_ckpt,"rb"))
    tf_model = pickle.load(open(tf_ckpt,"rb"))
    
    for key,tensor in jt_model.items():
        assert key in tf_model,key
        tf_tensor  = tf_model[key]
        if tf_tensor.shape!=tensor.shape:
            print(key,tf_tensor.shape,tensor.shape)
        assert tf_tensor.shape==tensor.shape,key

    for key,tensor in tf_model.items():
        assert key in jt_model,key
        jt_tensor  = jt_model[key]
        assert jt_tensor.shape==tensor.shape,key 
    
def test():
    ckpt = 'tf_weights/FPN_Res50D_DOTA_1x_20201103/DOTA_378000model.ckpt'
    # read_tf_weights(ckpt)
    # ckpt = "/mnt/disk/lxl/SCRDET/checkpoint_3.pkl"
    # # read_jt_weights(ckpt)
    compare_model("jt_model.pkl","test.pkl")

if __name__ == '__main__':
    test()