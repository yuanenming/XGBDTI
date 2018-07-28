import os  
import re
import numpy as np
import pandas as pd

##################################
# Construct features & data set. #
#                                #  
##################################
protein = open('protein200Refined.txt')
output = open('totalX.txt','w')
k = 0
for i in protein.readlines():
  drug = open('drug50Refined.txt')
  for j in drug.readlines():
      output.write(i.strip()+'\t'+j.strip()+'\n')
exit()
 

def findStringInFile(filename, txt):
    txtParttern = re.compile(txt)
    lines = [] 
    if not os.path.exists(filename):  
        print filename ,'not exist'  
        return  
    fileHandle = open(filename, 'rb')
    nLine = 0  
    for line in fileHandle:  
        search = txtParttern.search(line)  
        if search:
            pass#print filename ,'(',nLine, ')', line
        else:
            lines.append(nLine)
        nLine = nLine + 1  
    fileHandle.close()
    return lines

p = findStringInFile('../Data/InteractionData/mat_protein_drug.txt', '1')
# f = np.loadtxt('../Data/InteractionData/mat_protein_drug.txt')
# np.savetxt('../Data/InteractionData/mat_drug_protein.txt', f.T, '%d', delimiter = ' ')

d = findStringInFile('../Data/InteractionData/mat_drug_protein.txt', '1')

drugFeature = pd.read_table('../feature/drug50.txt',sep = '\t', header = None)
proteinFeature = pd.read_table('../feature/protein200.txt',sep='\t',header = None)
Interaction = pd.read_table('../Data/InteractionData/mat_protein_drug.txt', sep = ' ', header = None)

drugFeature.drop(d, inplace = True)
proteinFeature.drop(p, inplace = True)
Interaction.drop(p, inplace = True)
Interaction.drop(d, inplace = True, axis = 1)

drugFeature.to_csv("drug50Refined.txt", sep = '\t',index=False,float_format = '%.3f',header=False)
proteinFeature.to_csv("protein200Refined.txt", sep = '\t',index=False,float_format = '%.3f',header=False)
Interaction.to_csv("mat_p_d_Refined.txt", sep = '\t',index=False,header=False)
