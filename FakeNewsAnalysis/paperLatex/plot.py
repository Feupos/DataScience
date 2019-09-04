import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = [['Original', 'Original', 'Title', 72.67],['Original', 'Expanded', 'Title',71.33],
        ['Original', 'Original', 'Body', 78.67],['Original', 'Expanded', 'Body',82],
        ['Original', 'Original', 'Combined', 80],['Original', 'Expanded', 'Combined',82],
        ['Expanded', 'Original', 'Title', 56.26],['Expanded', 'Expanded', 'Title', 61.93],
        ['Expanded', 'Original', 'Body', 62.19],['Expanded', 'Expanded', 'Body', 73.93],
        ['Expanded', 'Original', 'Combined', 61.48],['Expanded', 'Expanded', 'Combined', 71.8]]
df = pd.DataFrame(data,columns=['Dataset','Features', 'Feature Set','Accuracy'])
#sns.barplot(x='Accuracy', y='Dataset', hue='Feature Set', data=df, orient='v')
g = sns.catplot(x="Features", y="Accuracy", hue="Feature Set", col="Dataset", data=df, kind="bar", height=4, aspect=.7, palette="Blues")
g.set(ylim=(50, 85))
plt.show()