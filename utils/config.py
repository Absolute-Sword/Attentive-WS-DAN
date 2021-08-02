# from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/home/runji/Documents/dataset/VOCdevkit/VOC2007/'
    min_size = 1000  # image resize
    max_size = 1000 # image resize
    img_size = 1000
    num_workers = 8
    test_num_workers = 8
#    load_path='checkpoints/fasterrcnn_07050448_0.6861123169931498'
    load_path=None
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.
    scale = 1
    test_scale = 1
    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.6  # 1e-3 -> 1e-4
    lr = 9e-4
    plot_every = 5000

    # preset
    data = 'voc'
    pretrained_model = 'resnet'

    # training
    epoch = 18


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model


    caffe_pretrain = False # use caffe pretrained model instead of torchvision
#     caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        print(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
