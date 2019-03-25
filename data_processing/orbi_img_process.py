import shutil
import fnmatch
import os

ORBI_DATA_ROOT = '../ORBI Prime footage'

class DATA_PIPELINE():
    
    def process_all_img(self):
        # # remove cam directory and recreate
        # if not os.path.exists(PROCESSED_CAM_ROOT):
        #     os.makedirs(PROCESSED_CAM_ROOT)
        # else:  
        #     print(PROCESSED_CAM_ROOT+' exists, will be removed')
        #     shutil.rmtree(PROCESSED_CAM_ROOT, ignore_errors=True)
        #     os.makedirs(PROCESSED_CAM_ROOT)

        # # remove fs directory and recreate
        # if not os.path.exists(PROCESSED_FS_ROOT):
        #     os.makedirs(PROCESSED_FS_ROOT)
        # else:
        #     print(PROCESSED_FS_ROOT+' exists, will be removed')
        #     shutil.rmtree(PROCESSED_FS_ROOT, ignore_errors=True)
        #     os.makedirs(PROCESSED_FS_ROOT)

        # list to store all image data
        matches = []
        for root, dirnames, filenames in os.walk(ORBI_DATA_ROOT):
            for filename in fnmatch.filter(filenames, '*.mp4'):
                matches.append(os.path.join(root, filename))

        print("Total image files to process: ",len(matches))
    #     # print(matches)

    #     # init image utility object
    #     iu = IMAGE_UTIL()

    #     # variable to count image naming
    #     cam_idx = 0
    #     fisheye_idx = 0 
    #     cam_list = [] # list storing all cam images in new directory
    #     fs_list = [] # list storing all fs images in new directory

    #     # loop through all files in original data
    #     for filepath in matches:
    #         # copy to cam category
    #         if iu.detect_img_type(filepath) == 0:
    #             new_file_name = PROCESSED_CAM_ROOT+str(cam_idx)+".png"
    #             final_file_name = PROCESSED_CAM_ROOT+str(cam_idx)+".jpg"
    #             cam_list.append(new_file_name)
    #             shutil.copy2(filepath, new_file_name)
    #             iu.process_img(new_file_name,final_file_name)
    #             os.remove(new_file_name)
    #             cam_idx += 1
    #         # copy to fisheye category
    #         else:
    #             new_file_name = PROCESSED_FS_ROOT+str(fisheye_idx)+".png"
    #             final_file_name = PROCESSED_FS_ROOT+str(fisheye_idx)+".jpg"
    #             fs_list.append(new_file_name)
    #             shutil.copy2(filepath, new_file_name)
    #             iu.process_img(new_file_name,final_file_name)
    #             os.remove(new_file_name)
    #             fisheye_idx += 1

    #     print("Finished image preprocess")

    # def label_all_img(self):
    #     print("start labelling now")
    #     # init image utility object
    #     iu = IMG_LABELLER()
    #     print("labeling cam images")
    #     iu.label_all(PROCESSED_CAM_ROOT)
    #     print("labeling fish images")
    #     iu.label_all(PROCESSED_FS_ROOT)

dp = DATA_PIPELINE()
dp.process_all_img()
# dp.label_all_img()