Converting *.Xml to *.Csv
5.3.1 Converting *.Xml to *.Csv
To do this we can write a simple script that iterates through all *.xml  files in the Training\Images\Train and Training\Images\Test folders, and generates a *.csv
for each of the two.

import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    print(path)
    xml_list=[]
    for xml_file in glob.glob(path + '/*.xml'):
        tree=ET.parse(xml_file)
        root=tree.getroot()
        for member in root.findall('object'):
            value =(root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            xml_list.append(value)
column_name=['filename','width','height','class','xmin','ymin','xmax','ymax']
    xml_df = pd.DataFrame(xml_list,columns=column_name)
    return xml_df
def main():
     parser= argparse.ArgumentParser(
            description="sample tensorflow xml-to-csv convertor")
    parser.add_argument("-i", "--inputDir",help="path to the folder where the input.xml files are stored",
                        type=str)
      parser.add_argument("-o","--outputFile",help="name of output .csv file(including path)",type=str)
    args=parser.parse_args()
    	print(args)
     if(args.inputDir is None):
        args.inputDir=os.getcwd()
     if(args.outputFile is None):
        args.outputFile=args.inputDir+"/labels.csv"
    assert(os.path.isdir(args.inputDir))
    xml_df=xml_to_csv(args.inputDir)
    xml_df.to_csv(args.outputFile,index=None)
    print('successfully converted xml to csv')
if __name__ =='__main__':
    main()
