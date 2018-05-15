import sys
import os
sys.path.append('C:/Users/samrari/ComputBuffer')
from my_packages.My_Geoprocess import*

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from skimage import exposure, img_as_float, morphology

from osgeo import gdal
import osr
import xml.etree.ElementTree as ET
import os, fnmatch

RePath_xmlDirectory = 'Data_SHOM/Global'

def RecordSurveyAndYear(XML_List, RePath_xmlDirectory, path_filname):
    
    with open(path_filname, 'w') as mon_fichier:
        print('Date',';', 'Lon', ';', 'Lat', ';', 'Depth', file= mon_fichier)

    for xml in XML_List:
        # initilise
        Set_Point = []
        Set_Depth = []
        Set_xml = []


        # read current xml
        DepthPt = pd.read_csv(RePath_xmlDirectory+'/'+xml[:-10]+'ascii', delim_whitespace=True, header=None, names=['Lat','Lon','Depth'])
        # filter by depth
        DepthPt_40m = DepthPt
        # loads points
        current_pt =  DepthPt_40m[['Lon','Lat']].values
        current_depth =  DepthPt_40m['Depth'].values

        # records points
        Set_Point.extend(current_pt[:,:])
        Set_Depth.extend(current_depth[:])
        root = ET.parse(RePath_xmlDirectory+'/'+xml).getroot()
        Set_xml.extend(np.tile((root.find('Attribute[@name="SURSTA"]').find('Value').text),len(current_depth)))
        

        Set =  [np.asarray(Set_xml),  np.asarray(Set_Point), np.asarray(Set_Depth)]

        with open(path_filname, 'a+') as mon_fichier:
            for i in range(len(Set[2])):
                print(Set[0][i], ';', Set[1][i,0], ';', Set[1][i,1], ';', Set[2][i], file= mon_fichier)

Selected_xml = []
for xmlFile in GetFiles("*.xml",directory=RePath_xmlDirectory, TimLim=1900):
    Selected_xml.append(xmlFile)

RecordSurveyAndYear(Selected_xml, RePath_xmlDirectory, 'D:/Data_SHOM/ShomSurveys.csv')
