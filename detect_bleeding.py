import ultralytics.yolo.v8.detect as P
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='', help='Path to directory of test images')
parser.add_argument('--device', default='cpu', help="cuda device - 'cuda' or 'cpu'")
parser.add_argument('--vid-path', action='store_true', help='if source is path to video file')
opt = parser.parse_args()

opt.task = 'detect'
opt.data = None
opt.model = './pretrained_models/detect_best.pt'
opt.imgsz = 224 #640
opt.save = True
opt.iou = 0.4
opt.max_det = 300
opt.save_txt = True
opt.save_conf = False 
opt.save_crop = False
opt.show_labels = True 
opt.show_conf = True  
opt.conf = 0.25
opt.boxes = True
opt.line_width = None   # line width of the bounding boxes
opt.visualize = False
opt.augment = False 
opt.agnostic_nms = False
opt.classes = None
opt.verbose = True
opt.half = False
opt.dnn = False

args = vars(opt)
if args['vid_path']:
    args['vid_writer'] = True
    args['vid_stride'] = 1
else:
    args['vid_writer'] = False
    args['vid_stride'] = 0

P.predict(args)
