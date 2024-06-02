
import matplotlib.pyplot as plt
import numpy as np


# Visualize emotion/class distribution in AffectNet dataset

numbers = [
    75374,
    134915,
    25959,
    14590,
    6878,
    4303,
    25382,
    4250,
]

emotions = [
    'Neutral',
    'Happy',
    'Sad',
    'Surprise',
    'Fear',
    'Disgust',
    'Anger',
    'Contempt'
]

plt.figure(figsize=(10, 5))

plt.bar(emotions, numbers, color='blue')
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('AffectNet Emotion Distribution', fontsize=14)

# Add the numbers on top of the bars
for i in range(len(numbers)):
    formatted_number = "{:,}".format(numbers[i])
    plt.text(i, numbers[i] + 1000, formatted_number, ha='center', va='bottom')

# Format the y-axis to have commas
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Add more space above top bar
plt.ylim(0, max(numbers) + 20000)

plt.savefig('affectnet_distribution.png')
plt.show()