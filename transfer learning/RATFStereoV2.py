import rospy
import message_filters
import sensor_msgs.msg
from std_msgs.msg import Header
import std_msgs.msg
import trajectory_msgs.msg

import cv2
from cv_bridge import CvBridge,CvBridgeError

import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import time
from queue import Queue
import threading

DEVICE = 'cuda'
global mStart
mStart = False

def load_model(args):
    print("Loading Model")
    global model
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    print("Model Init Success")

def from_ros_load_image(ros_img):
    #Make sure msg is RGB color，If it is an image read using OpenCV, convert it to RGB first.
    img = np.frombuffer(ros_img.data , dtype=np.uint8).reshape(720,1280,-1)
    img = Image.fromarray(img)
    img = np.array(img.convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def shutdown():
    print('ImageToDispNode ShutDown')

def ros_extimate_trajectory_mode(image_left , image_right,trajectory):
    print('Message received')
    global first_frame
    global stamp
    if (not first_frame):
        if(image_left.header.stamp.secs-stamp.secs<10):
            print(image_left.header.stamp)
            print("[PASS] :Still Delay")
            return
    stamp = image_left.header.stamp
    print(image_left.header.stamp)
    pub_recived.publish(image_left)    
    image1 = from_ros_load_image(image_left) 
    image2 = from_ros_load_image(image_right)

    bridge = CvBridge()
    imageleftdata = bridge.imgmsg_to_cv2(image_left)
    first_frame = False
    with torch.no_grad():
        padder = InputPadder(image1.shape, divis_by=32)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        print("START Estimate")
        time_start = time.time()
        _,flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        time_end = time.time()
        time_run = time_end - time_start
        print("RATF-STEREO run time:  ",time_run )
        disp = flow_up.cpu().numpy().squeeze()
        
        for i in range(3):
            disp_msg = bridge.cv2_to_imgmsg(disp)
            image_left_ = bridge.cv2_to_imgmsg(imageleftdata)
            now_stamp =  rospy.Time.from_sec(time.time())
            disp_msg.header.frame_id='map'
            image_left_.header.frame_id = 'map'
            image_left.header.frame_id = 'map'
            disp_msg.header.stamp =  stamp
            image_left_.header.stamp = stamp
            image_left.header.stamp = stamp
            pub_disp.publish(disp_msg)
            pub_img_left.publish( image_left)
            rate.sleep()
        print("published")

def ros_extimate(image_left , image_right,start_bool):
    print('Message received')
    mStart = start_bool.data
    if not mStart:
        return
    global first_frame
    global stamp
    if (not first_frame):
        if(image_left.header.stamp.secs-stamp.secs<10):
            print(image_left.header.stamp)
            print("[PASS] :Still Delay")
            return
    stamp = image_left.header.stamp
    print(image_left.header.stamp)
    pub_recived.publish(image_left)    
    image1 = from_ros_load_image(image_left) 
    image2 = from_ros_load_image(image_right)

    bridge = CvBridge()
    imageleftdata = bridge.imgmsg_to_cv2(image_left)
    first_frame = False
    with torch.no_grad():
        padder = InputPadder(image1.shape, divis_by=32)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        print("START Estimate")
        time_start = time.time()
        _,flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        time_end = time.time()
        time_run = time_end - time_start
        print("RATF-STEREO run time:  ",time_run )
        disp = flow_up.cpu().numpy().squeeze()
        
        for i in range(2):
            disp_msg = bridge.cv2_to_imgmsg(disp)
            image_left_ = bridge.cv2_to_imgmsg(imageleftdata)
            now_stamp =  rospy.Time.from_sec(time.time())
            disp_msg.header.frame_id='map'
            image_left_.header.frame_id = 'map'
            image_left.header.frame_id = 'map'
            disp_msg.header.stamp =  stamp
            image_left_.header.stamp = stamp
            image_left.header.stamp = stamp
            pub_disp.publish(disp_msg)
            pub_img_left.publish( image_left)
            rate.sleep()
        print("published")
        mStart = False

def ros_init():
    rospy.init_node('ImageToDispNode',anonymous=True)
    rospy.on_shutdown(shutdown)
    #message_filters use to subscribe mutiple msg which uesed in one callback
    sub_img_left = message_filters.Subscriber("/zed_node/left/image_rect_color", sensor_msgs.msg.Image)
    sub_img_right = message_filters.Subscriber("/zed_node/right/image_rect_color" , sensor_msgs.msg.Image)
    # sub_img_left = message_filters.Subscriber("imgl_test1", sensor_msgs.msg.Image)
    # sub_img_right = message_filters.Subscriber("imgr_test1" , sensor_msgs.msg.Image)
    sub_bStart= message_filters.Subscriber("bTAStart" , std_msgs.msg.Bool)
    sub_trajectory =  message_filters.Subscriber("sim_trajectory" , trajectory_msgs.msg.JointTrajectory)
    ts = message_filters.ApproximateTimeSynchronizer([sub_img_left , sub_img_right,sub_trajectory] ,1 , 0.1,allow_headerless=True)
    ts.registerCallback(ros_extimate_trajectory_mode)
    # ts = message_filters.ApproximateTimeSynchronizer([sub_img_left , sub_img_right,sub_bStart] ,2 , 0.1,allow_headerless=True)
    # ts.registerCallback(ros_extimate)
    global pub_disp,pub_img_left,pub_recived
    # pub_disp = rospy.Publisher('disp_test', sensor_msgs.msg.Image , queue_size=10,latch=True)
    # pub_img_left = rospy.Publisher('imgl_test', sensor_msgs.msg.Image , queue_size=10,latch=True)
    # pub_recived = rospy.Publisher('imgl_recived', sensor_msgs.msg.Image , queue_size=10,latch=True)
    pub_disp = rospy.Publisher('disp_test', sensor_msgs.msg.Image , queue_size=10)
    pub_img_left = rospy.Publisher('imgl_test', sensor_msgs.msg.Image , queue_size=10)
    pub_recived = rospy.Publisher('imgl_recived', sensor_msgs.msg.Image , queue_size=10)
    global rate
    global first_frame
    global stamp
    first_frame = True
    rate = rospy.Rate(5)
    print('ROS node init succeced')

    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="./models/raftstereo-middlebury.pth")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="alt", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()
    load_model(args)
    #without_ros_estimate()
    ros_init()
    
    
