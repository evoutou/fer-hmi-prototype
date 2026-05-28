# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:10:07 2026

@author: elene
"""

from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Folder containing the images
folder_path = r"C:\Users\elene\Desktop\_assignmet_coehlo\images"

# Get all image files from the folder
images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create figure with subplots (one row per image)
fig, axes = plt.subplots(len(images), 2, figsize=(12, len(images) * 4))

for i, img_file in enumerate(images):
    img_path = os.path.join(folder_path, img_file)
    
    # Analyze the emotion in the image using DeepFace
    result = DeepFace.analyze(img_path, actions=['emotion'])
    emotions = result[0]['emotion']
    dominant = result[0]['dominant_emotion']
    
    # Left subplot: display the image with the dominant emotion as title
    img = mpimg.imread(img_path)
    axes[i, 0].imshow(img)
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f"{img_file} → {dominant.upper()}", 
                          fontsize=12, fontweight='bold', color='green')
    
    # Right subplot: horizontal bar chart showing confidence for all emotions
    colors = ['red' if e == dominant else 'steelblue' for e in emotions.keys()]
    axes[i, 1].barh(list(emotions.keys()), list(emotions.values()), color=colors)
    axes[i, 1].set_xlabel('Confidence (%)')
    axes[i, 1].set_title('Emotion Confidence Scores', fontsize=12)

# Adjust layout and save the final figure
plt.tight_layout()
plt.savefig(r"C:\Users\elene\Desktop\_assignmet_coehlo\results.png", 
            dpi=150, bbox_inches='tight')
plt.show()

print("Analysis complete!")