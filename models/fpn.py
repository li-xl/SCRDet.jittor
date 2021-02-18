import jittor as jt 
from jittor import nn

class InceptionModule(nn.Module):
    def __init__(self,):
        super(Inception,self).__init__()
        self.branch_0_conv = nn.Conv(512,out_channels=384,kernel_size=[1,1],stride=1)

        self.branch_1_conv1 = nn.Conv(512,out_channels=192,kernel_size=[1,1],stride=1)
        self.branch_1_conv2 = nn.Conv(192,out_channels=224,kernel_size= [1, 7],stride=1)
        self.branch_1_conv3 = nn.Conv(224,out_channels=256,kernel_size= [7, 1],stride=1)

        self.branch_2_conv1 = nn.Conv(512,out_channels=192,kernel_size=[1,1],stride=1)
        self.branch_2_conv2 = nn.Conv(192,out_channels=192,kernel_size= [7, 1],stride=1)
        self.branch_2_conv3 = nn.Conv(192,out_channels=224,kernel_size= [1, 7],stride=1)
        self.branch_2_conv4 = nn.Conv(224,out_channels=224,kernel_size= [7, 1],stride=1)
        self.branch_2_conv5 = nn.Conv(224,out_channels=256,kernel_size= [1, 7],stride=1)

        self.branch_3_avgpool = nn.AvgPool2d(kernel_size=[3,3],stride=1)
        self.branch_3_conv = nn.Conv(512,out_channels=128,kernel_size= [1, 1],stride=1)


    def execute(self,x):
        # branch_0
        branch_0 = self.branch_0_conv(x)
        branch_0 = nn.relu(branch_0)

        # branch_1
        branch_1 = self.branch_1_conv1(x)
        branch_1 = nn.relu(branch_1)
        branch_1 = self.branch_1_conv2(branch_1)
        branch_1 = nn.relu(branch_1)
        branch_1 = self.branch_1_conv3(branch_1)
        branch_1 = nn.relu(branch_1)

        # branch_2 
        branch_2 = self.branch_2_conv1(x)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv2(branch_2)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv3(branch_2)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv4(branch_2)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv5(branch_2)
        branch_2 = nn.relu(branch_2)

        # branch_3
        branch_3 = self.branch_3_avgpool(x)
        branch_3 = self.branch_3_conv(branch_3)
        branch_3 = nn.relu(branch_3)

        out = jt.contrib.concat([branch_0, branch_1, branch_2, branch_3],dim=1)
        return out

class InceptionAttention(nn.Module):
    
    def __init__(self,):
        super(InceptionAttention,self).__init__()
        self.inception_module = InceptionModule()
        self.inception_attention_conv = nn.Conv(1024,out_channels=2,kernel_size=[3,3])
    
    def execute(self,x):
        x = self.inception_module(x)
        x = self.inception_attention_conv(x)
        return x 

class SFNet(nn.Module):
    def __init__(self,anchor_stride):
        super(SFNet,self).__init__()
        self.anchor_stride = anchor_stride
        self.inception_module = InceptionModule()
        self.fusion_conv = nn.Conv(512,out_channels=512,kernel_size=[1, 1], stride=1)

    def fusion_two_layer(self, feat1, feat2):
        h, w = feat1.shape[-2], feat1.shape[-1]
        upsample_feat2 = nn.interpolate(feat2,size=(h, w))
        add_f = upsample_feat2 + feat1
        reduce_dim_f = self.fusion_conv(add_f)
        return reduce_dim_f

    def execute(self,C3,C4):
        h, w = C3.shape[-2], C3.shape[-1]
        resize_scale = 8 // self.anchor_stride
        h_resize, w_size = h * resize_scale, w * resize_scale

        upsample_c3 = nn.interpolate(C3,size=[h_resize, w_size])
        upsample_c3 = self.inception_module(upsample_c3)

        out = self.fusion_two_layer(upsample_c3, C4,)
        return out

class MDANet(nn.Module):
    def __init__(self,out_dim,ratio):
        super(MDANet,self).__init__()
        self.out_dim = out_dim
        self.inception_attention = InceptionAttention()
        self.ca_fc1 = nn.Linear(1024,out_dim//ratio)
        self.ca_fc2 = nn.Linear(out_dim//ratio,out_dim)

    def execute(self,x):
        # Channel Attention
        ca = jt.mean(x,dims=[2,3])
        ca = self.ca_fc1(ca)
        ca = nn.relu(ca)
        ca = self.ca_fc2(ca)
        ca = jt.sigmoid(ca)
        ca = jt.reshape(ca, [-1, 1, 1, self.out_dim])

        # Pixel Attention
        pa_mask = self.inception_attention(x)
        pa_mask_softmax = nn.softmax(pa_mask)
        pa = pa_mask_softmax[:, 0:1,:, :]

        x = pa*x
        x = ca*x

        return x,pa_mask

class SCRDetFPN(nn.Module):
    def __init__(self,anchor_stride,out_dim=512,ratio=16):
        super(SCRDetFPN,self).__init__()
        self.sfnet = SFNet(anchor_stride=anchor_stride)
        self.mdanet = MDANet(out_dim,ratio)

    def execute(self, C3,C4):
        sfnet_feat = self.sfnet(C3,C4)
        mdanet_feat, pa_mask = self.mdanet(sfnet_feat)
        return mdanet_feat, pa_mask