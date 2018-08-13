# import libraries
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from osgeo import osr
import h5py

def getNcPathL2(NetCdf_data_path):
	return [f for f in os.listdir(NetCdf_data_path) if f.endswith('L2R.nc')]

def getSen2SubDataset(MetaFileLoc):
    DataWithMeta = gdal.Open(MetaFileLoc, gdal.GA_ReadOnly)
    if DataWithMeta == None:
        print('oups')

    SubDataSet_md = DataWithMeta.GetMetadata('SUBDATASETS')
    ds = []
    for i in range(4):
        gdal.ErrorReset()
        ds.append(gdal.Open(SubDataSet_md['SUBDATASET_%d_NAME' % (i+1)]))
        if ds is None or gdal.GetLastErrorMsg() != '':
            print('subdatasets failed to load')
            print(SubDataSet_md['SUBDATASET_%d_NAME' % (i+1)],'failed')
    return ds
# ______________________________________________________________________________

# location to the inputfile
# ______________________________________________________________________________
filePath = "D:/Image/Poe/Acolyte/S2A_MSIL1C_20180305T230901_N0206_R101_T58KEB_20180306T00234_bisAco/"
# ______________________________________________________________________________
NetCdfFilename = getNcPathL2(filePath)
FileLocation = filePath+NetCdfFilename[0]

if os.path.exists(FileLocation[:-3]+'_tif'):
    directoryPath = FileLocation[:-3]+'_tif_bis'
    os.mkdir(directoryPath)
else:
    directoryPath = FileLocation[:-3]+'_tif'
    os.mkdir(directoryPath)

# open the file
Data2Convert = gdal.Open(FileLocation, gdal.GA_ReadOnly)
if Data2Convert == None:
	print('oups')
# ______________________________________________________________________________
MetaFileLoc = 'D:/Image/Poe/S2A_MSIL1C_20180305T230901_N0206_R101_T58KEB_20180306T002341.SAFE/MTD_MSIL1C.xml'
# ______________________________________________________________________________
DataWithMeta = getSen2SubDataset(MetaFileLoc)

 
# set geotif driver
driver = gdal.GetDriverByName( 'GTiff' )

# Get the datasets
for dataset in Data2Convert.GetSubDatasets():
    # read the dataset
    data = gdal.Open(dataset[0],gdal.GA_ReadOnly)
    # check compatibility 
    if DataWithMeta[0].RasterXSize == data.RasterXSize and DataWithMeta[0].RasterYSize == data.RasterYSize:

        # get the data of the precipitation dataset
        dataBand = data.ReadAsArray()
         
        # set output name
        output = directoryPath+'/'+dataset[1].split(' ')[1][2:]+".tif"
         
        # set projection
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
         
        # write dataset to disk
        outputDataset = driver.Create(output, data.RasterXSize,data.RasterYSize, 1 ,gdal.GDT_Float64)
        outputDataset.SetGeoTransform(DataWithMeta[0].GetGeoTransform())
        outputDataset.SetProjection(DataWithMeta[0].GetProjection())
        outputDataset.SetMetadata(data.GetMetadata())
        outputDataset.GetRasterBand(1).WriteArray(dataBand)
        outputDataset = None