from common import estimate_pose, draw_humans, read_imgfile
import tensorflow as tf
import numpy as np
import shutil
import time
import os
import re
from PIL import Image, ImageFont, ImageDraw
import cv2



def inference(imgpath, j):
    #input_width = 656
    #input_height = 368
    
    input_width = 352
    input_height = 288
    
    t0 = time.time()

    tf.reset_default_graph()
    
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    # Download model from https://www.dropbox.com/s/2dw1oz9l9hi9avg/optimized_openpose.pb
    with open('models/optimized_openpose.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    t1 = time.time()
    #print(t1 - t0)

    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
    pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')

    t2 = time.time()
    #print(t2 - t1)

    image = read_imgfile(imgpath, input_width, input_height)

    t3 = time.time()
    #print(t3 - t2)
    
    with tf.Session() as sess:
        heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
            inputs: image
        })

        t4 = time.time()
        #print(t4 - t3)

        heatMat, pafMat = heatMat[0], pafMat[0]

        humans = estimate_pose(heatMat, pafMat)
        
        #coordinates for left hand, head, right hand etc. indexes: 0,2,3,4,5,6,7
        head_coco_idx = 0
        right_shoulder_idx = 2
        right_elbow_idx = 3
        right_hand_idx = 4
        left_shoulder_idx = 5
        left_elbow_idx = 6
        left_hand_idx = 7

        for human in humans:
            if human[right_hand_idx] is None:
                break
                
            head_coords = human[head_coco_idx][1]
            right_shoulder_coords = human[right_shoulder_idx][1]
            right_elbow_coords = human[right_elbow_idx][1]
            right_hand_coords = human[right_hand_idx][1]
            left_shoulder_coords = human[left_shoulder_idx][1]
            left_elbow_coords = human[left_elbow_idx][1]
            left_hand_coords = human[left_hand_idx][1]
            
            fields = [head_coords, right_shoulder_coords, right_elbow_coords,right_hand_coords,  left_shoulder_coords, left_elbow_coords, left_hand_coords]
            
        
        #with open(r'test.csv', 'a') as f:
            #writer = csv.writer(f)
            #writer.writerow(fields)
        # end of printing
        
        # display
        image = cv2.imread(imgpath)
        image_h, image_w = image.shape[:2]
        image = draw_humans(image, humans)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
    
        ### Uncomment below to show image with skeleton
        #cv2.imshow('result', image)
        
        ##### save image with coordinates
        cv2.imwrite('./Outputs/openpos/result_'+str(j)+'.png',image)
        
        t5 = time.time()
        #print(t5 - t4)
        cv2.waitKey(0)
        
        return fields
		
def video_to_frames(input_loc, output_loc):

    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
    print(video_length)
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        resized_frame = cv2.resize(frame, (352, 288), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), resized_frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start) + str("\n\n"))
            break
            
    return(video_length)
