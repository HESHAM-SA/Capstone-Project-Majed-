# Majed

## Story

Our project is in the education sector and we named it "Majed," representing the student whom our systems and the education ecosystem serve either directly or indirectly.

## Introduction

Our project includes three systems:

1. Hather
2. Rakez
3. Fahem

### Hather

"Hather" is a system that contributes to automating routine tasks in the education sector. It performs automatic attendance for students; as soon as a student enters the classroom, their face is recognized, and their name is recorded in the attendance log automatically.

#### First Day Setup
- Upload a video of the students to the system which will classify the students' pictures into folders.
- Name the folders with the students' names.

After the initial setup, the administration only needs to upload the video, and the system will handle attendance throughout the year.

#### Technical Part

##### Data

**Data for training YOLO:**
- Images of students were collected by taking video frames of students in Saudi schools.
- Annotations of faces were done using the Roboflow website.

**Data for training Keras:**
1. **Data Collection:**
   - More than 1200 images of a person were collected.
   - They were classified into 20 folders to create a classification among 20 people.
   - The data was cleaned to ensure the images were of the same person.

2. **Data Processing:**
   - The YOLO model was used to detect faces.
   - Only the face images were cropped.
   - Image dimensions were standardized.
   - They were saved in new folders with the persons' names.

Now the data is ready for training.

##### Model

**Fine-tuning YOLO model:**
- A YOLO model trained on more than 33000 face images was fine-tuned with local data to increase accuracy.

**Building a classification model and feature extraction using the Keras library:**
- The model's accuracy was improved and measured in three steps:
  1. Train a baseline model (EfficientNetV2) with supervised learning on the collected data.
  2. Train a self-supervised model by performing augmentation trying to bring feature vectors of the same image closer using cosine similarity and SimCLR loss function.
  3. Fine-tune the previous model with labeled data or supervised learning.

Now we have two ready models:
- A model for face detection.
- A model for classification and face recognition.

#### Results

**Results for the face detection model:**
- Comparison between three different methods for training the classification model.
![Comparison ](D:\SDAIA_BootCamp\Projects\CapSton_Project\Comparison against the baseline keras.png)
<img src= "D:\SDAIA_BootCamp\Projects\CapSton_Project\Comparison against the baseline keras.png"/>

#### Tools Used

- YOLO
- Keras library
- Roboflow website

### Rakez

#### About the System

"Rakez" is designed to improve concentration and productivity, especially in environments where focus is critical, such as classrooms and remote virtual classes. This system uses advanced computer vision techniques to monitor eye movements and calculate a "focus score."

#### How it Works

- Our system uses the computer camera to monitor the user's eye movements, detecting if the user is looking at the screen, to the side, or has closed eyes.
- By analyzing these eye positions, the system calculates the focus score, indicating the user's level of concentration.

#### Main Components and How They Work

- Eye and iris feature detection.
- Detection of eye closure duration.
- Calculation of focus scores.

### Fahem

#### About the System

"Fahem" is designed to enhance interactive learning by enabling students to study materials with their favorite characters both in voice and image. This approach motivates students and makes studying more enjoyable.

---

### Image Formats and Resolutions

- **Image Format:** Portable Network Graphic (PNG)
- **Bits Per Pixel:** 32
- **Color:** Truecolor with alpha
- **Dimensions:** Various resolutions (e.g., 1431x954, 691x470, 727x334)
- **Interlaced:** Yes
- **XResolution:** Various (e.g., 220, 100, 96)
- **YResolution:** Various (e.g., 220, 100, 96)

Software used for generating images:
- **Matplotlib version:** 3.9.0 [Matplotlib](https://matplotlib.org/)
