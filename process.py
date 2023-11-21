from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from utils.process_video import infer_skeleton

register_all_modules()

image_path = "./mmpose-estimation/examples/demo2.png"
video_path = "./mmpose-estimation/examples/demo.mp4"
pose_config = 'mmpose-estimation/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'mmpose-estimation/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

pose_model = init_model(pose_config, pose_checkpoint, device="cuda:0")

infer_skeleton(pose_model, video_path)