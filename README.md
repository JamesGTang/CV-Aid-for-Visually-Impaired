> Written with [StackEdit](https://stackedit.io/).

### Project Title: 
360° Camera Scene Understanding for the Visually Impaired (VI)  

### Project description
Despite the popular misconception that deep learning has solved the scene recognition problem, the quality of machine-generated captions for real-world scenes tends to be quite poor. In the context of live captioning for visually impaired users, this problem is exacerbated by the limited field of view of conventional smartphone cameras and camera-equipped eyewear. This results in a narrow view of the world that often misses important details relevant to gaining an understanding of their environment.

This project will make use of a head-mounted 360° camera and a combination of human (crowdsourced) labeling with deep learning to train systems to provide more relevant and accurate scene descriptions, in particular for navigation in indoor environments, and guidance for intersection crossing, improving upon a system recently developed by our lab for this purpose. Results will be compared to those obtained using camera input from a smartphone held in a neck-worn lanyard.
source:  http://srl.mcgill.ca/projects/

### Project Abstract 
The goal of this project is to aid VI people cross an intersection through the use of crowdsourcing and a 360° camera. The visual data acquired from this camera will be compared to a smartphone attached to a lanyard. Through the use of machine learning and computer vision algorithms, the collected data will be used to predict the correct crossing direction and determine the distance needed to cross the intersection.

The motivation for this project is to facilitate crossing an intersection for the VI safely. When crossing an intersection, veering outside the white lines can cause accidents and result in injuries/death. Currently, Professor Jeremy Cooperstock’s lab worked on a method which uses a conventional smartphone camera to detect the white lines on a crossing. The limited field-of-view tends to miss out on relevant information required to understand the surrounding environment. The field-of-view of a 360° camera can acquire this missing information to determine a safer route for the VI.

Our primary objective for this project is to provide feedback for VI people to successfully walk across an intersection using their smartphone. If we are to achieve this successfully, we would like to expand the project scope to alternate scenarios including traffic light state, obstacle detection and guidance for doorways and stairwells.

### Intellectual Property: 
There will be an industrial partner who will lending the 360° camera hardware.  Any intellectual property developed is owned by Adeeb Ibne Amjad, Stefano Commodari, Hakim Amarouche and James Tang. 

### Project Supervisors:
Dr. Jeremy Cooperstock, Centre for Intelligent Machines 
McGill University
jer at cim.mcgill.ca

http://www.cim.mcgill.ca/ 

### Dataset

The original dataset is published for non-commercial purpose, you can use this dataset for research and you must credit, Shared Reality Lab at McGill Center for Intelligent Machines (CIM), http://srl.mcgill.ca/ in your publication. Data colllection credit by 
Adeeb Ibne Amjad, Stefano Commodari, Hakim Amarouche and James Tang. 

#### Image captured using normal phone camera
| File link | Size | 
|--|--|
| https://drive.google.com/open?id=1yaG_2NSW768V_njTIiE01Sc9i23iQW9B | ~ 962.9 MB   |

#### 360° Camera

Uncompressed video footage of intersection crossing, each intersection contained in one folder, the video files named PRIM0001.mp4,PRIM0002.mp4,PRIM0003.mp4,PRIM0004.mp4 in each folder is the same crossing captured from the four lens installed the glass.

| File link | Size | 
|--|--|
| https://drive.google.com/open?id=1nWGju1AHbJoYvNZnnofRb8Y7jbVBeBCE | ~ 15.00 GB   |

| File type | Description | 
|--|--|
| .THM  | Video file thumbnail   |
| .MP4 | Original video file of intersection crossing captured at 30FPS |
| .BIN | Lens configuration binary file| 
| .CFG | Property file containing the configuration properties of the lens |

Video foortages compressed into zip files. 

| File link | Size | 
|--|--|
| https://drive.google.com/open?id=13gjGYKL2uD86O0bYRAj2wlqGXIJE0QW1 | ~ 3.51 GB   |
| https://drive.google.com/open?id=1_eBZ_63d-734eNjh8_7jbAOSO8nCIsd0 | ~ 3.27 GB |
| https://drive.google.com/open?id=14bGY0XIOrh-dWGjtVB6tfMvVB-R6gyhJ | ~ 2.73 GB| 
| https://drive.google.com/open?id=1JXUz4884y5gNmPkgKoIWyBGHVJGg8kf3 | ~ 3.51 GB |


### File Index
| Folder Name| Description | 
|--|--|
| ICA  | Experimental code using Individual Component Analysis |
| camera-data | Image captured using normal phone camera  |
| data_processing | Data preprocessing pipeline written in Python|
| orbi_processed | Stiched images from frames captured by Orbi lenses|
| training | Code for training the CNN model and the saved model |
| model_conversion | Code used to convert saved model for use on mobile devices |
| graphs | Graph resources for presentation|
| report_resource | Resources for report|
| orbi_sample | Sample images from orbi for demontration purposes|


### Credit 
| Name | Email | 
|--|--|
|Stefano Commodari  | stefano.commodari at mail.mcgill.ca  |
| Adeeb Ibne Amjad | adeeb.amjad at mail.mcgill.ca  |
|Hakim Amarouche | hakim.amarouche at mail.mcgill.ca| 
| James Tang | guokai.tang at mail.mcgill.ca |
