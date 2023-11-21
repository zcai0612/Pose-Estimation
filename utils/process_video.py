import gradio as gr

import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import *
from mmpose.apis import inference_topdown

def get_frames(video_in):
    frames = []
    #resize the video
    clip = VideoFileClip(video_in)
    start_frame = 0  # 起始帧数
    end_frame = 50  # 结束帧数
    
    if not os.path.exists('./raw_frames'):
        os.makedirs('./raw_frames')
    
    if not os.path.exists('./mmpose_frames'):
        os.makedirs('./mmpose_frames')
    
    #check fps
    if clip.fps > 30:
        print("vide rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=512)
        clip_resized = clip_resized.subclip(start_frame / clip_resized.fps, end_frame / clip_resized.fps) # subclip 2 seconds
        clip_resized.write_videofile("./video_resized.mp4", fps=30)
    else:
        print("video rate is OK")
        clip_resized = clip.resize(height=512)
        clip_resized = clip_resized.subclip(start_frame / clip.fps, end_frame / clip.fps) # subclip 5 seconds
        clip_resized.write_videofile("./video_resized.mp4", fps=clip.fps)
    
    print("video resized to 512 height")
    
    # Opens the Video file with CV2
    cap= cv2.VideoCapture("./video_resized.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('./raw_frames/kang'+str(i)+'.jpg',frame)
        frames.append('./raw_frames/kang'+str(i)+'.jpg')
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print("broke the video into frames")
    
    return frames, fps


def create_video(frames, fps, type):
    print("building video result")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(type + "_result.mp4", fps=fps)
    
    return type + "_result.mp4"


def get_mmpose_filter(mmpose, i):
    pose_results = inference_topdown(mmpose, i)
    img = Image.open(i)
    width, height = img.size
    body_part = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16
    }
    orange=(51,153,255)
    blue=(255,128,0)
    green=(0,255,0)
    black_img = np.zeros((height, width, 3), np.uint8)
    for person in pose_results:
    # get the keypoints for this person
        keypoints = np.squeeze(person.pred_instances['keypoints'], axis=0)
        scores = np.squeeze(person.pred_instances["keypoint_scores"], axis=0)
        # draw lines between keypoints to form a skeleton
        skeleton = [("right_eye", "left_eye", orange),("nose", "left_eye", orange), ("left_eye", "left_ear", orange), ("nose", "right_eye", orange), ("right_eye", "right_ear", orange),
                    ("left_shoulder", "left_ear", orange),("right_shoulder", "right_ear", orange), ("left_shoulder", "right_shoulder", orange), ("left_shoulder", "left_elbow", green), ("right_shoulder", "right_elbow",blue),
                    ("left_elbow", "left_wrist",green), ("right_elbow", "right_wrist",blue), ("left_shoulder", "left_hip",orange),
                    ("right_shoulder", "right_hip", orange), ("left_hip", "right_hip", orange), ("left_hip", "left_knee",green),
                    ("right_hip", "right_knee",blue), ("left_knee", "left_ankle",green), ("right_knee", "right_ankle",blue)]
        for start_part, end_part, color in skeleton:
            start_idx = list(body_part.keys()).index(start_part)
            end_idx = list(body_part.keys()).index(end_part)
            if scores[start_idx] > 0.1 and scores[end_idx] > 0.1:
                pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                print(pt1, pt2)
                cv2.line(black_img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.waitKey(0)
    frame_name = os.path.basename(i)
    mmpose_frame = os.path.join("./mmpose_frames", frame_name)
    cv2.imwrite(mmpose_frame, black_img)
    cv2.destroyAllWindows()

    return mmpose_frame
        

def infer_skeleton(mmpose, video_in):
    # 1. break video into frames and get FPS
    break_vid = get_frames(video_in)
    frames_list= break_vid[0]
    fps = break_vid[1]
    #n_frame = int(trim_value*fps)
    n_frame = len(frames_list)
    
    if n_frame >= len(frames_list):
        print("video is shorter than the cut value")
        n_frame = len(frames_list)
    
    # 2. prepare frames result arrays
    result_frames = []
    print("set stop frames to: " + str(n_frame))
    
    for i in frames_list[0:int(n_frame)]:
        mmpose_frame = get_mmpose_filter(mmpose, i)
        result_frames.append(mmpose_frame)
        print("frame " + i + "/" + str(n_frame) + ": done;")

    
    final_vid = create_video(result_frames, fps, "mmpose")
    files = [final_vid]
    
    return final_vid, files