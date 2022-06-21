import pandas as pd
import sys
import os,glob
#read text 
dst_file = 'F:\\Dataset\\aclImdb\\data.cvs'
if os.path.exists(dst_file):
    os.remove(dst_file)
file_object = open(dst_file, 'a')
file_object.write("text"	"label")
dirs = {"F:\\Dataset\\aclImdb\\train\\pos\\*.txt":"1","F:\\Dataset\\aclImdb\\test\\pos\\*.txt":"1","F:\\Dataset\\aclImdb\\train\\neg\\*.txt":"0","F:\\Dataset\\aclImdb\\test\\neg\\*.txt":"0"}
for dir,label in dirs.items():
    files = glob.glob(dir)
    for filename in files:
        with open(filename,errors="ignore") as file:
            for line in file:
                #srip \t
                line = line.strip(' \n\t')
                line = line.replace("<br />","")
            file_object.write(line)
            file_object.write('\t')
            file_object.write(label)
            file_object.write('\n')
file_object.close()   
data = pd.read_csv(dst_file,sep='\t',encoding = "ISO-8859-1",on_bad_lines='skip') # tsv file
print('done')