# import libraries
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from osgeo import osr
import h5py

'''xdist = (Enx - E0) / float(nx)
ydist = (Nny - N0) / float(ny)
rtx = (Eny - E0) / float(ny)
rty = (Nnx - N0) / float(nx)'''


def getNcPath(NetCdf_data_path):
	return [f for f in os.listdir(NetCdf_data_path) if f.endswith('.nc')]

def getNCGeoTrans2(file, LonS, LatS ):
    '''Extract Geotransform from Longitude and Latitude destination file'''
    
    f = h5py.File(file, 'r')
    lon = f[LonS][:]
    lat = f[LatS][:]

    ny, nx = lon.shape

    E0, Enx, Eny, Enxny = lon[0, 0], lon[0, nx-1], lon[ny-1, 0], lon[ny-1, nx-1]
    N0, Nny, Nnx, Nnxny = lat[0, 0], lat[ny-1, 0], lat[0, nx-1], lat[ny-1, nx-1]

    A = np.array([[1, nx, 0, 0, 0, 0],[1, 0, ny, 0, 0, 0],[0, 0, 0, 1, 0, ny],[0, 0, 0, 1, nx, 0],[1, nx, ny, 0, 0, 0],[0, 0, 0, 1, nx, ny],[1, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0]])
    C = np.array([Enx, Eny, Nny, Nnx, Enxny, Nnxny, E0, N0]).reshape(8,1)
    Gt = np.linalg.solve(np.dot(A.T,A), np.dot(A.T,C))
    return (Gt[0], Gt[1], Gt[2], Gt[3], Gt[4], Gt[5])

# location to the inputfile
filePath = "D:/Image/Poe/Acolyte/S2B_MSIL1C_20180320T230859last/"
NetCdfFilename = getNcPath(filePath)
FileLocation = filePath+NetCdfFilename[0]
'''if os.path.exists(FileLocation[:-3]):
	if os.path.exists(FileLocation[:-3]+'_bis'):
		os.mkdir(FileLocation[:-3]+'_bis_bis')
	else:
		os.mkdir(FileLocation[:-3]+'_bis')
else:
	os.mkdir(FileLocation[:-3])'''

# open the file
dat = gdal.Open(FileLocation, gdal.GA_ReadOnly)
if dat == None:
	print('oups')
 
# Get the Precipitation dataset
dataset = dat.GetSubDatasets()[6]

# read the precipitation dataset
data = gdal.Open(dataset[0],gdal.GA_ReadOnly)
 
# get the data of the precipitation dataset
dataBand = data.ReadAsArray()
 
# get geotransform
'''GeoT = data.GetGeoTransform()'''
LonS = 'lon'
LatS = 'lat'
GeoT = getNCGeoTrans2(FileLocation, LonS, LatS )
print(GeoT)
 
# set geotif driver
driver = gdal.GetDriverByName( 'GTiff' )
 
# get x,y dimensions of the map
RastXsize = data.RasterXSize
RastYsize = data.RasterYSize
 
# set output name
outname = "testFullIm.tif"
 
# set projection
target = osr.SpatialReference()
target.ImportFromEPSG(4326)
 
# write dataset to disk
outputDataset = driver.Create(outname, RastXsize,RastYsize, 1,gdal.GDT_Float32)
outputDataset.SetGeoTransform(GeoT)
outputDataset.SetProjection(target.ExportToWkt())
outputDataset.GetRasterBand(1).WriteArray(dataBand)
outputDataset.GetRasterBand(1).SetNoDataValue(-9999)
outputDataset = None