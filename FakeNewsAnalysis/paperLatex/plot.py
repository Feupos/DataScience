import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


data = [['Original Dataset', 'Original Features (Title)', 1],['Original Dataset', 'Expanded Features (Title)',2],
        ['Original Dataset', 'Original Features (Body)', 1],['Original Dataset', 'Expanded Features (Title)',2],
        ['Original Dataset', 'Original Features (Both)', 1],['Original Dataset', 'Expanded Features (Title)',2],
        ['Expanded Dataset', 'Original Features (Title)', 3],['Expanded Dataset', 'Expanded Features (Title)',4],
        ['Expanded Dataset', 'Original Features (Body)', 3],['Expanded Dataset', 'Expanded Features (Title)',4],
        ['Expanded Dataset', 'Original Features (Both)', 3],['Expanded Dataset', 'Expanded Features (Title)',4]]
df = pd.DataFrame(data,columns=['Dataset','Feature Set','Accuracy'],dtype=float)
sns.barplot(x="Accuracy", y="Feature Set", hue="Dataset", data=df);

plt.show()