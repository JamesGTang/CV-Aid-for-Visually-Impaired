import numpy as np
import cv2
import os

"""
Script written for CV-Aid Design project at McGill
@author jamestang
@version 1.0
exports common image utilities function

"""
class IMAGE_UTIL:
	# convert image of any filetype to .jpg extension
	def convert_to_jpg(self,filepath):
		img = cv2.imread(filepath)
		cv2.imwrite(filepath[:-3] + 'jpg', img)

	# rotate image and replace original image
	def rotate_img(self,filepath,rot_deg):
		img=cv2.imread(filepath)

		# rotate ccw
		out=cv2.transpose(img)
		out=cv2.flip(out,flipCode=rot_deg)

		cv2.imwrite(filepath, out)

	# rename image by removing "preview" from image name
	def rename_img(self,dir_name):
		for filename in os.listdir(dir_name):
			if filename == '.DS_Store':
				# ignore
				pass
			else:
				new_filename = filename.replace('preview','')
				new_filename = new_filename.replace(' ','')
				os.rename(dir_name+"/"+filename, dir_name+"/"+new_filename)

	def detect_img_black(self,filepath):
		img = cv2.imread(filepath)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# print(cv2.countNonZero(img))
		print(cv2.countNonZero(gray)/(img.shape[0]*img.shape[1]))

	# return 0 if the image is normal cam image, return 1 if image is fisheye
	def detect_img_type(self,filepath,h=50,w=50,y=30,x=30,verbose=False):
		img = cv2.imread(filepath)
		height = img.shape[0]
		width = img.shape[1]
		# print("W:",width,"H: ",height)
		upper_left = img[y:y+h, x:x+w]
		lower_left = img[height-y-h:height-y,x:x+w]
		upper_right =img[y:y+h,width-x-w:width-x]
		lower_right = img[height-y-h:height-y, width-x-w:width-x]
		# print("UL mean:",np.mean(upper_left),"LL mean:",np.mean(lower_left))
		# print("UR mean:",np.mean(upper_right),"LR mean:",np.mean(lower_right))
		avg_corner_pixel = (np.mean(upper_left) + np.mean(lower_left) + np.mean(upper_right) + np.mean(lower_right))/4
		if verbose: print(avg_corner_pixel)
		# cv2.imshow("cropped", lower_right)
		# cv2.waitKey(0)
		if  avg_corner_pixel <= 20:
			if verbose: print("360 image detected")
			return 1
		else:
			if verbose: print("Cam image detected")
			return 0

	def process_img(self,filepath,final_filepath):
		self.convert_to_jpg(filepath)
		self.rotate_img(final_filepath,0)

# iu = IMAGE_UTIL()
# print("========= 360 image =========")
# iu.convert_to_jpg('./sample/preview114 .png')
# iu.detect_img_type('./sample/preview114 .jpg')

# iu.convert_to_jpg('./sample/preview125 .png')
# iu.detect_img_type('./sample/preview125 .jpg')

# iu.convert_to_jpg('./sample/preview188 .png')
# iu.detect_img_type('./sample/preview188 .jpg')

# iu.convert_to_jpg('./sample/preview230 .png')
# iu.detect_img_type('./sample/preview230 .jpg')

# print("======= Normal cam image =======")
# iu.detect_img_type('./sample/sample.jpg')

# iu.convert_to_jpg('./sample/11.png')
# iu.detect_img_type('./sample/11.jpg')

# iu.convert_to_jpg('./sample/12.png')
# iu.detect_img_type('./sample/12.jpg')

# convert_to_jpg('./sample/sample.png')
# rotate_img('./sample/sample.jpg',0)
# rename_img('./sample')