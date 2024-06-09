# Majed

## Overview

"Majed" is an education sector project designed to enhance the educational experience for students. The project comprises three integrated systems: Hather, Rkkez, and Fahem.

## Systems Overview

## 1. Hather
"Hather" automates student attendance using facial recognition technology. When a student enters the classroom, their face is recognized, and their attendance is recorded automatically.

<img src="https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/19c44f09-d615-423c-9e84-12d36f71efec" alt="Hather system screenshot" width="800">

<br>

### How It Works: 
<img src="https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/deef738f-3af8-404c-bd76-98b72611d837" alt="Hather System Diagram" width="800">

<br>

<br>

### Setup and Usage:
 1. Upload a video of the students to the system, and The system classifies the students' pictures into folders.
 2. Name the folders with the students' names.
After the initial setup, the system will automates student attendance throughout the year.

<br>

<br>

### Technical Details:

 1. **Data Collection and Processing:**
     1.  *YOLO Training Data:*
         - Images were collected from video frames of students in Saudi schools.
         - Face annotations were done using Roboflow.
 
 3. *Keras Training Data:*
    - Collected over 1200 images per person, classified into 20 folders.
    - Data was cleaned to ensure consistency.
    - YOLO detected and cropped face images, which were then standardized and saved.
 

 2. **Model Development:**
  1. *YOLO Fine-tuning:*
     - The model, trained on over 33,000 face images, was fine-tuned with local data.
 
  2. *Keras Classification Model:*
     - Three approaches to enhance accuracy:
       1. Train a baseline model (EfficientNetV2) with supervised learning.
       2. Train a self-supervised model using augmentation techniques.
       3. Fine-tune the model with labeled data.

<br>

### Results:
- **Face Detection Model:**
  
  <img src="https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/3a4fcf6b-5f13-41ca-a32e-11d118bbeaa1" alt="YOLO PR curve" width="800">

- **Classification Model Comparison:**
  
  <img src="https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/cbc9e4f2-6e98-43d0-94ae-ff2715a2a0b7" alt="Model comparison" width="800">

### Tools Used:
  - YOLO
  - Keras
  - Roboflow





### Demo:
https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/6ae48eed-9189-463a-ad4f-24f7d87fc321


---------------------------
## 2. Rkkez

"Rkkez" aims to enhance concentration and productivity, particularly in classrooms and virtual learning environments, by monitoring eye movements and calculating a focus score.

#### How It Works:
- Uses a computer camera to track eye movements.
- Detects if the user is looking at the screen, to the side, or has closed eyes.
- Calculates a focus score based on eye position analysis.

#### Main Components:
- Eye and iris feature detection.
- Eye closure duration detection.
- Focus score calculation.

<img src="https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/8aaf4fdd-4d26-4f6f-a906-94a73ecd5e49" alt="Rkkez System Diagram" width="800">
  
<tap>

**Tools Used:**

<img src="https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/38815154-70b1-4e9f-ae88-44fe99cd842e" alt="Tools used" width="800">

### Demo:
https://github.com/HESHAM-SA/Capstone-Project-Majed-/assets/62900612/51dfa076-8f92-4d93-9ca2-d302e1527ce8

---------------------------------------




## Future Work
### 3. Fahem

"Fahem" enhances interactive learning by allowing students to study materials with their favorite characters in both voice and image formats, making learning more engaging and enjoyable.
