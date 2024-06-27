from matplotlib import pyplot as plt

counts = [75374, 134915, 25959, 14590, 6878, 4303, 25382, 4250]

emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

def format_number(number):
    return f"{number:,}"

# Add grid behind bars
plt.grid(axis='y',
         linestyle='-',
         zorder=0,
         alpha=0.5
         )

# Bar plot, with the number of images for each emotion
plt.bar(emotions, counts, color='mediumseagreen', zorder=3, width=0.8)

# Add label above each bar with a bit of space
for i in range(len(counts)):
    # add a bit space above the bar
    plt.text(i, counts[i] + 1500, format_number(counts[i]), ha='center', va='bottom')




# Format y axis
plt.yticks(range(0, 160000, 20000), [format_number(i) for i in range(0, 160000, 20000)])

# make y axis up to 150000
plt.ylim(0, 150000)

# add more space for plot
plt.tight_layout()
# Make sure title and labels left and below are visible
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)


plt.ylabel('Number of Samples')
plt.title('AffectNet Emotion Distribution')
plt.xlabel('Emotion')

plt.savefig('affectnet_distribution.pdf')

plt.show()