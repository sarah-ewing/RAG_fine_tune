import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])

install('transformers')
# install('flax')

from transformers import pipeline 

# Load the zero-shot classification pipeline using a pre-trained BART model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  

# Example text to classify 
text = "The new iPhone has a great camera and fast processor." 

# Define potential categories 
candidate_labels = ["Academic Programs & Courses",
"Admissions & Application Process",
"Scholarships & Financial Aid",
"Research & Innovation at ASU",
"Student Life & Campus Activities",
"ASU’s Global & Online Education",
"ASU’s Commitment to Sustainability",
"International Student Support",
"ASU’s AI & Tech Initiatives",
"Sun Devil Athletics & Sports",
"ASU’s History & Rankings",
"ASU Library & Research Resources",
"Career Services & Job Support",
"Housing & Campus Life",
"Health, Wellness & Counseling Services"] 

# Perform zero-shot classification
result = classifier(text, candidate_labels) 

# Print the results
print("Text:", text)
for label, score in zip(result["labels"], result["scores"]):
    print(f"{label}: {score}") 