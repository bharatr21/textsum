import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfavgr = pd.read_excel('AverageRougeFinalResults.xlsx')
dfavgs = pd.read_excel('AveragesumevalFinalResults.xlsx')
dfmaxr = pd.read_excel('MaxRougeFinalResults.xlsx')
dfmaxs = pd.read_excel('MaxsumevalFinalResults.xlsx')


ind = np.arange(len(dfmaxs['Algorithm']))
width = 0.225

plt.figure(1)
ax = plt.gca()
ax.bar(ind - width, dfmaxs['ROUGE-1'], width=width, label='ROUGE-1', color='r')
ax.bar(ind, dfmaxs['ROUGE-2'], width=width, label='ROUGE-2',  color='g')
ax.bar(ind + width, dfmaxs['ROUGE-L'], width=width, label='ROUGE-L', color='b')

plt.title('Maximum ROUGE Scores by sumeval Package')
plt.xlabel('Algorithms')
plt.ylabel('ROUGE Score')
plt.legend()
plt.xticks(ind, list(dfmaxs['Algorithm']), rotation=90)
plt.grid(axis='y')
plt.savefig('Maxsumevalresults.png', bbox_inches='tight')
plt.show()

plt.figure(2)
ax = plt.gca()
ax.bar(ind - width, dfmaxr['ROUGE-1'], width=width, label='ROUGE-1', color='r')
ax.bar(ind, dfmaxr['ROUGE-2'], width=width, label='ROUGE-2',  color='g')
ax.bar(ind + width, dfmaxr['ROUGE-L'], width=width, label='ROUGE-L', color='b')

plt.title('Maximum ROUGE Scores by Rouge Package')
plt.xlabel('Algorithms')
plt.ylabel('ROUGE Score')
plt.legend()
plt.xticks(ind, list(dfmaxr['Algorithm']), rotation=90)
plt.grid(axis='y')
plt.savefig('MaxRougeresults.png', bbox_inches='tight')
plt.show()

plt.figure(3)
ax = plt.gca()
ax.bar(ind - width, dfmaxs['ROUGE-1'], width=width, label='ROUGE-1', color='r')
ax.bar(ind, dfmaxs['ROUGE-2'], width=width, label='ROUGE-2',  color='g')
ax.bar(ind + width, dfmaxs['ROUGE-L'], width=width, label='ROUGE-L', color='b')

plt.title('Average ROUGE Scores by sumeval Package')
plt.xlabel('Algorithms')
plt.ylabel('ROUGE Score')
plt.legend()
plt.xticks(ind, list(dfavgs['Algorithm']), rotation=90)
plt.grid(axis='y')
plt.savefig('Averagesumevalresults.png', bbox_inches='tight')
plt.show()

plt.figure(4)
ax = plt.gca()
ax.bar(ind - width, dfavgr['ROUGE-1'], width=width, label='ROUGE-1', color='r')
ax.bar(ind, dfavgr['ROUGE-2'], width=width, label='ROUGE-2',  color='g')
ax.bar(ind + width, dfavgr['ROUGE-L'], width=width, label='ROUGE-L', color='b')

plt.title('Average ROUGE Scores by Rouge Package')
plt.xlabel('Algorithms')
plt.ylabel('ROUGE Score')
plt.legend()
plt.xticks(ind, list(dfavgr['Algorithm']), rotation=90)
plt.grid(axis='y')
plt.savefig('AverageRougeresults.png', bbox_inches='tight')
plt.show()