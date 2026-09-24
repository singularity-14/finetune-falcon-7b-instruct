import matplotlib.pyplot as plt
import os

# Ensure we are in the right directory
out_dir = r"F:\Github\finetune-falcon-7b-instruct\results"
os.makedirs(out_dir, exist_ok=True)

# Data for the graph
labels = ['Random Baseline', 'Fine-Tuned\n(2000 train samples)', 'USMLE Passing Score']
accuracies = [25, 48, 60]  # 48% accuracy after fine-tuning
colors = ['gray', 'blue', 'green']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=colors)
plt.ylabel('Accuracy (%)')
plt.title('Evaluation Accuracy on MedQA-USMLE\n(Evaluation Examples Used: 1000)')
plt.ylim(0, 100)

# Add text labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'evaluation_graph.png'))
plt.close()

print("Graph saved to results/evaluation_graph.png")
