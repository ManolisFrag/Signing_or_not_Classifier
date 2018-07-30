import tensorflow as tf
import cv2
import numpy as np
import argparse
import csv
from contextlib import suppress
import shutil
import time
import os
import re
from fnmatch import fnmatch
import sys
from keras.models import load_model
global graph,model
graph = tf.get_default_graph()
import pympi
from PIL import Image, ImageFont, ImageDraw
from helper_functions import inference, video_to_frames

parser = argparse.ArgumentParser(description='Classify signing')
parser.add_argument('-v', '--video', type=str, metavar='', required=True, help='Directory of the video to be analysed')
args = parser.parse_args()

if __name__ == '__main__':
	# check for folders
	directory1 = "./Input/orig_frames"
	if not os.path.exists(directory1):
		os.makedirs(directory1)

	directory2 = "./Outputs/openpos"
	if not os.path.exists(directory2):
		os.makedirs(directory2)
		
	directory3 = "./Outputs/final"
	if not os.path.exists(directory3):
		os.makedirs(directory3)
		
	# extract frames to Input/orig_frames/ folder
	total_frames = video_to_frames(args.video, "./Input/orig_frames/")
	
	# load keras model
	model = load_model('multilayer_perceptron_7_rmsprop_300.h5')
	
	# run predictions
	root2 = './Input/orig_frames/'
	pattern = "*.jpg"
	total_files = 0
	total_scanned_files = 0

	predictions_array = []

	overlay = 'Signing'
	print("Start predicting")
	for root2, dirs, files in os.walk(root2, topdown=False):
		for name in files:
			#total_files = total_files + 1
			if fnmatch(name, pattern):
				#print(os.path.join(root2, name))
				#Running openpose  
				
				j = re.sub('\.jpg$', '', name)
				fpath = str(os.path.join(root2, name))
				print('predicting frame: '+str(fpath)
				#try:
					#print('predicting frame: '+str(fname))
				coord = inference(fpath, j)
				# except TypeError:
					#print("Not a person in frame: "+str(fname))
					# coord = np.zeros((7, 2))
				#except:
					#print("Unexpected error:", sys.exc_info()[0])
					#print("not file found in frame: "+str(fname))
					#coord = np.zeros((7, 2))            
				coord2 = np.asarray(coord).flatten()
				points = coord2.reshape(1,14)
				points2 = np.float64(points)
				with graph.as_default():    
					preds = model.predict_classes(points2)
					print("predicted frame: "+ str(preds))                
					
					if preds==0:
						overlay = 'Not signing'
						predictions_array.append(0)
					else:
						overlay = 'Signing'
						predictions_array.append(1)                      
							
					# font = ImageFont.truetype("Arial.ttf", 25)
					# img = Image.open('./Outputs/openpos/result_'+str(j)+'.png')
					# draw = ImageDraw.Draw(img)
					# draw.text((20,20), str(overlay), (255,255,0), font=font)
					# draw = ImageDraw.Draw(img)
					# img.save('./Outputs/final/'+str(j)+'.png')                   
					# os.remove('./Outputs/openpos/result_'+str(j)+'.png')
						
	print("done predicting")
	len(predictions_array)
	q = np.asarray(predictions_array)
	np.savetxt('pred.txt', q , delimiter=',')
	print("saved predictions in txt")

	# Create video from extracted frames with predictions

	dir_path = './Outputs/openpos/'
	ext = 'png'
	output = 'output_video.mp4'

	images = []
	for f in os.listdir(dir_path):
		if f.endswith(ext):
			images.append(f)

	# Determine the width and height from the first image
	image_path = os.path.join(dir_path, images[0])
	frame = cv2.imread(image_path)
	cv2.imshow('video',frame)
	height, width, channels = frame.shape

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

	for image in images:

		image_path = os.path.join(dir_path, image)
		frame = cv2.imread(image_path)

		out.write(frame) # Write out frame to video

		cv2.imshow('video',frame)
		if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
			break

	# Release everything if job is finished
	out.release()
	cv2.destroyAllWindows()

	print("The output video is {}".format(output))
	
	# Create annotations based on predictions array

	x = pympi.Eaf(author="Manolis")
	x.add_tier(tier_id="tier1")
	p = np.asarray(predictions_array)
	s = np.asarray(p)
	size = s.size
	#print(size)
	#print(size)
	st = 0
	end = 0
	arr = []
	j = 0
	i = 0

	#master_boolean = True
	#print("strarting values: " + str(st))
	while i< size-1:     
		if s[i] == 1:
			j = i + 1
			while s[j] == 1 and j<size-1:
				end = j
				j +=1
			print("the i: "+str(i)+" the j: "+str(j))
			x.add_annotation(id_tier="tier1", start=i*50, end= j*50, value='signing')
			i = j
			#else:            
				#i = j
				
				#break    
		else:
			i+=1

	x.add_linked_file(file_path='output_video.mp4',relpath='output_video.mp4',mimetype='mp4')
	pympi.Elan.to_eaf(file_path="output_video.eaf", eaf_obj=x, pretty=True)
	print("Elan file created")
	
	##Delete folders with pictures
	shutil.rmtree('/Input/orig_frames', ignore_errors=True)
	shutil.rmtree('/Outputs/openpos', ignore_errors=True)
	print("folders deleted")
