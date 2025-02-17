import cv2
import glob
import os

# Mapping on keyboard obtained through trial and error. For Mac
actions = {
    '0': 48,
    '1': 49,
    '2': 50,
    '3': 51,
    '4': 52,
    '5': 53,
    '6': 54,
    '7': 55,
    '9': 57,
}

"""
Script modified for CV-Aid Design project at McGill,
original author Roger: https://github.com/roggirg/labelling_street_veer,

This script allows user to label the data interactively by using keyboard.

"""
class IMG_LABELLER():
    def label_all(self,root_path):
        print(root_path)
        DIR_UNLABELLED = root_path
        DIR_ARCHIVE = DIR_UNLABELLED+"/archives/"
        DIR_LABELLED = root_path + "/labelled_data/"
        DISCARD = root_path +"/discard/"
        if not os.path.isdir(DIR_ARCHIVE):
            os.mkdir(DIR_ARCHIVE)
        init_idx = len(glob.glob(DIR_ARCHIVE+"*.jpg"))

        if not os.path.isdir(DIR_LABELLED):
            os.mkdir(DIR_LABELLED)
        
        if not os.path.isdir(DISCARD):
            os.mkdir(DISCARD)
        
        for i in range(len(actions)):
            if not os.path.isdir(DIR_LABELLED+str(i)):
                os.mkdir(DIR_LABELLED+str(i))

        all_fnames = sorted(glob.glob(DIR_UNLABELLED+"/*.jpg"))
        print("Number of Frames: ", len(all_fnames))
        for img_idx, fname in enumerate(all_fnames):
            if img_idx % 100 == 0:
                print("Number of completed images: ", str(init_idx+img_idx+1))
            image = cv2.imread(fname)
            image = cv2.resize(image, (684, 684))

            # make a copy to not save lines
            orig_img = cv2.resize(image.copy(), (224, 224))

            # Put vertical lines and text
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,500)
            fontScale              = 1
            fontColor              = (0,0,255)
            lineType               = 2

            x = image.shape[1]//8
            line = 0
            for i in range(10):
                cv2.line(image, (line, 0), (line, image.shape[0]), (0,0,255), 2)
                cv2.putText(image, 
                    str(i),
                    (line,30), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
                line += int(x)
            cv2.putText(
                image, "If the image is bad, use 9 to discard",
                (150,670), 
                font, 
                0.8,
                fontColor,
                1)
            labelled = 0

            #Positioning frame properly on screen
            windowName = "Centered Window"
            cv2.namedWindow(windowName)
            #Change the x,y coordinate to center the frame on your screen
            #Size that works best for Adeeb's display - (monitor: -1000, -1000. Laptop screen: 200, 30)
            cv2.moveWindow(windowName, 200, 30)
            cv2.imshow(windowName, image)

            while labelled == 0:
                read_key = 0xFF & cv2.waitKey()

                if read_key in actions.values():
                    label = int([key for (key, value) in actions.items() if value == read_key][0])
                    print("Selected: " + str(label));
                    if(label==9):
                        new_fname = DISCARD+str(label)+"/%05d.jpg" % (img_idx+init_idx)
                        cv2.imwrite(new_fname, orig_img)
                        print("discard: "+new_fname)
                        labelled = 1
                    else:
                        new_fname = DIR_LABELLED+str(label)+"/%05d.jpg" % (img_idx+init_idx)
                        cv2.imwrite(new_fname, orig_img)

                        if label != 0 or label != 4:
                            rotated_img = cv2.flip(orig_img, 1)
                            new_label = len(actions) - label
                            new_fname = DIR_LABELLED+str(new_label)+"/%05d.jpg" % (img_idx+init_idx)
                            cv2.imwrite(new_fname, rotated_img)

                    os.rename(fname, DIR_ARCHIVE+"%05d.jpg" % (img_idx+init_idx))
                    labelled = 1

            cv2.destroyAllWindows()
