import pandas as pd
import numpy as np

# Assign spreadsheet filename to `file`
file = 'ann_data.xlsx'

# Load spreadsheet
xl = pd.ExcelFile(file)

# Print the sheet names
print(xl.sheet_names)

# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('ann_data')
z=df1.values
mydata = np.zeros((len(z), 1, len(z[0]), 1), dtype='float32')

for i in range(len(z)):
    mydata[i,0,:,0]=z[i,:]



np.save('traindata', mydata)
#print(temp)
a=1