import sys
import os
import fnmatch
import numpy as np
from osgeo import osr
from osgeo import gdal
from sklearn.metrics import r2_score
import xml.etree.ElementTree as ET
import pandas as pd

##########################################################################
# Files paths definition
def Get_RGBNIR(src_ds):

    NIR = src_ds.GetRasterBand(4).ReadAsArray()
    RGB_img = np.dstack((src_ds.GetRasterBand(1).ReadAsArray(),\
                         src_ds.GetRasterBand(2).ReadAsArray(),\
                         src_ds.GetRasterBand(3).ReadAsArray()))
    better_contrast = np.empty(RGB_img.shape, dtype= RGB_img.dtype)
    return NIR, RGB_img, better_contrast
    
def GetFiles(pattern, directory=os.curdir, TimLim=1964): 
    '''find all files with difined filename pattern in defined directory.'''
    
    files = os.listdir(os.path.abspath(directory))
    for current_file in fnmatch.filter(files, pattern):
        if int(current_file[1:5])>=TimLim:
            yield current_file
            
def GetSimpleFiles(pattern, directory=os.curdir, TimLim=1980): 
    '''find all files with difined filename pattern in defined directory.'''
    
    files = os.listdir(os.path.abspath(directory))
    for current_file in fnmatch.filter(files, pattern):
        yield current_file  

def getExtentFilename(NetCdf_data_path, endswith='.tif'):
    return [f for f in os.listdir(NetCdf_data_path) if f.endswith(endswith)]

#______________________________________________________________________________

def isInsid(E, N, E_range, N_range):
    if (E_range[0] < E < E_range[1]):
        if (N_range[0] < N < N_range[1]):
            return True
        else:
            return False
    else:
        return False

def isInsid2(E, N, E_range, N_range):
    n = len(E)
    return np.logical_and(np.logical_and(np.tile(E_range[0],(n)) < E, E < np.tile(E_range[1], (n)))\
            ,np.logical_and(np.tile(N_range[0],(n)) < N, N < np.tile(N_range[1], (n))))

def GetXml_byLocation(RePath_xmlDirectory, E, N):
    Selected_xml = []
    count = 0
    LoopCount = 0

    for xmlFile in GetFiles("*.xml",directory=RePath_xmlDirectory):
        count = count+1
        root = ET.parse(RePath_xmlDirectory+'/'+xmlFile).getroot()
        E_range = (float(root.find('Attribute[@name="lonmin"]').find('Value').text), \
                   float(root.find('Attribute[@name="lonmax"]').find('Value').text) )
        N_range = (float(root.find('Attribute[@name="latmin"]').find('Value').text), \
                   float(root.find('Attribute[@name="latmax"]').find('Value').text))

        if isInsid(E, N, E_range, N_range):
            Selected_xml.append(xmlFile)
            LoopCount = LoopCount +1
    return Selected_xml, LoopCount, count

def GetXml_byFootprint(RePath_xmlDirectory, E_range, N_range, FromYear=1950):
    Selected_xml = []
    count = 0
    LoopCount = [0,0,0,0]

    for xmlFile in GetFiles("*.xml",directory=RePath_xmlDirectory, TimLim=FromYear):
        count = count+1
        root = ET.parse(RePath_xmlDirectory+'/'+xmlFile).getroot()
        lonmin = float(root.find('Attribute[@name="lonmin"]').find('Value').text)
        latmin = float(root.find('Attribute[@name="latmin"]').find('Value').text)
        lonmax = float(root.find('Attribute[@name="lonmax"]').find('Value').text)
        latmax = float(root.find('Attribute[@name="latmax"]').find('Value').text)
        setIn = False

        if isInsid(lonmin, latmin, E_range, N_range):
            setIn = True      
            LoopCount[0] = LoopCount[0] +1
        elif isInsid(lonmax, latmax,E_range, N_range):
            setIn = True  
            LoopCount[1] = LoopCount[1] +1
        elif isInsid(lonmin, latmax,E_range, N_range):
            setIn = True  
            LoopCount[2] = LoopCount[2] +1
        elif isInsid(lonmax, latmin, E_range, N_range):
            setIn = True  
            LoopCount[3] = LoopCount[3] +1

        if setIn:
            Selected_xml.append(xmlFile)

    return Selected_xml, LoopCount, count

def GetXml_byFootprint2(RePath_xmlDirectory, min_E, max_E, min_N, max_N, FromYear=1950):
    Selected_xml = []
    count = 0
    LoopCount = [0,0,0,0]

    for xmlFile in GetFiles("*.xml",directory=RePath_xmlDirectory,TimLim=FromYear):
        count = count+1
        root = ET.parse(RePath_xmlDirectory+'/'+xmlFile).getroot()

        setIn = False

        if isInsid(float(root.find('Attribute[@name="lonmin"]').find('Value').text),\
                   float(root.find('Attribute[@name="latmin"]').find('Value').text),\
                   (min_E, max_E), (min_N, max_N)):
            setIn = True      
            LoopCount[0] = LoopCount[0] +1
        elif isInsid(float(root.find('Attribute[@name="lonmax"]').find('Value').text),\
                   float(root.find('Attribute[@name="latmax"]').find('Value').text),\
                   (min_E, max_E), (min_N, max_N)):
            setIn = True  
            LoopCount[1] = LoopCount[1] +1
        elif isInsid(float(root.find('Attribute[@name="lonmin"]').find('Value').text),\
                   float(root.find('Attribute[@name="latmax"]').find('Value').text),\
                   (min_E, max_E), (min_N, max_N)):
            setIn = True  
            LoopCount[2] = LoopCount[2] +1
        elif isInsid(float(root.find('Attribute[@name="lonmax"]').find('Value').text),\
                   float(root.find('Attribute[@name="latmin"]').find('Value').text),\
                   (min_E, max_E), (min_N, max_N)):
            setIn = True  
            LoopCount[3] = LoopCount[3] +1

        if setIn:
            Selected_xml.append(xmlFile)
            
    return Selected_xml, LoopCount, count

def GetSurvey_byDepthFP(XML_List, RePath_xmlDirectory, E_range, N_range, max_Depth=40):
    # initilise
    Set_Point = []
    Set_Depth = []
    Set_xml = []
    countValidPT = []
    NoValid_xml = []
    for i, xml in enumerate(XML_List):
        # read current xml
        try:
            DepthPt = pd.read_csv(RePath_xmlDirectory+'/'+xml[:-10]+'ascii', delim_whitespace=True, header=None, names=['Lat','Lon','Depth'])
            # filter by depth
            DepthPt_40m = DepthPt[DepthPt['Depth']<=max_Depth]
            #DepthPt_40m = DepthPt_40m[0<DepthPt_40m['Depth']]
            # loads points
            current_pt =  DepthPt_40m[['Lon','Lat']].values
            current_depth =  DepthPt_40m['Depth'].values
            # looks for points inside footprint
            index = isInsid2(current_pt[:,0], current_pt[:,1], E_range, N_range)
            # records points
            Set_Point.extend(current_pt[index,:])
            Set_Depth.extend(current_depth[index])
            root = ET.parse(RePath_xmlDirectory+'/'+xml).getroot()
            Set_xml.extend(np.tile((xml,root.find('Attribute[@name="SURSTA"]').find('Value').text),(np.sum(index), 1)))
            countValidPT.append(np.sum(index))     
        except:
            # in case of invalid xml path
            print('invalid at row: ', i)
            NoValid_xml.append(xml)
    return [np.asarray(Set_xml),  np.asarray(Set_Point), np.asarray(Set_Depth)], countValidPT, NoValid_xml

def GetSurvey_byDepthFP2(XML_List, RePath_xmlDirectory, E_range, N_range, max_Depth=40):
    # initilise
    Set_Selected = []
    Set_Point = []
    countValidPT = []
    NoValid_xml = []
    for i, xml in enumerate(XML_List):
        # read current xml
        try:
            DepthPt = pd.read_csv(RePath_xmlDirectory+'/'+xml[:-10]+'ascii', delim_whitespace=True, header=None, names=['Lat','Lon','Depth'])
            # filter by depth
            DepthPt_40m = DepthPt[DepthPt['Depth']<=max_Depth]
            # update book of points
            current_pt =  np.asarray(DepthPt_40m[['Lon','Lat','Depth']].values)
            index = isInsid2(current_pt[:,0], current_pt[:,1], E_range, N_range)
            Set_Selected.append(current_pt[index,:])
            Set_Point.extend([[ (pt[0], pt[1]), pt[2], i ] for  pt in current_pt[index,:]])
            countValidPT.append(np.sum(index))
        except:
            print(xml)
            NoValid_xml.append(NoValid_xml)
    return Set_Selected,  np.asarray(Set_Point), countValidPT, NoValid_xml

def recordSurvey(path_filname, Set): 
    with open(path_filname, 'w+') as mon_fichier:
        for i in range(len(Set[2])):
            print( Set[1][i,0], ';', Set[1][i,1], ';', Set[2][i], file= mon_fichier)
            
def recordPosition(path_filname, Set): 
    with open(path_filname, 'w+') as mon_fichier:
        for i in range(len(Set[2])):
            print( Set[1][i,0], ';', Set[1][i,1], file= mon_fichier)
            
def GetLonLat(Pix, src_ds):
    ''' Pix: Nx2 with N number of pixels '''
    GT_ds = src_ds.GetGeoTransform()
    LonLat = np.empty((Pix.shape[0], 2), dtype= np.float)
    LonLatProj= np.empty((Pix.shape[0], 2), dtype= np.float)
    
    LonLat = np.hstack((GT_ds[0] + Pix.dot(np.asarray(GT_ds[1:3]).reshape(2,1)),\
                      GT_ds[3] + Pix.dot(np.asarray(GT_ds[4:6]).reshape(2,1))))

    srs_ds = osr.SpatialReference()
    srs_ds.ImportFromWkt(src_ds.GetProjection())

    srsLatLong = srs_ds.CloneGeogCS()
    ct_ds = osr.CoordinateTransformation(srs_ds,srsLatLong)

    LonLatProj = np.asarray(ct_ds.TransformPoints(LonLat))
    
    return LonLatProj 

def GetPixel(ctInv, Inv_geomat, Coord, integer=True):
    n = Coord.shape[0]
    if integer:
        x = np.zeros((n,), dtype= np.int_)
        y = np.zeros((n,), dtype= np.int_)
        print('with int')
        for i in range(n):
            (X, Y, height) = ctInv.TransformPoint(Coord[i,0], Coord[i,1])
            x[i] = int(Inv_geomat[0] + Inv_geomat[1] * X + Inv_geomat[2] * Y)
            y[i] = int(Inv_geomat[3] + Inv_geomat[4] * X + Inv_geomat[5] * Y)
    elif integer == False :
        x = np.zeros((n,), dtype= np.float_)
        y = np.zeros((n,), dtype= np.float_)
        print('no int')
        for i in range(n):
            (X, Y, height) = ctInv.TransformPoint(Coord[i,0], Coord[i,1])
            x[i] = Inv_geomat[0] + Inv_geomat[1] * X + Inv_geomat[2] * Y
            y[i] = Inv_geomat[3] + Inv_geomat[4] * X + Inv_geomat[5] * Y
    return x, y 
'''
 *pdfGeoX = padfGeoTransform[0] + dfPixel * padfGeoTransform[1]
                                + dfLine  * padfGeoTransform[2];
 *pdfGeoY = padfGeoTransform[3] + dfPixel * padfGeoTransform[4]
                                + dfLine  * padfGeoTransform[5];

def GetPixel2(ctInv, Inv_geomat, Coord):
    x = []
    y = []
    for i in range(Coord.shape[0]):
        (X, Y, height) = ctInv.TransformPoint(Coord[i,0], Coord[i,1])
        x.append(int(inv_geometrix[0] + inv_geometrix[1] * X + inv_geometrix[2] * Y))
        y.append(int(inv_geometrix[3] + inv_geometrix[4] * X + inv_geometrix[5] * Y))
    return x, y 

def GetPixel3(ctInv, Inv_geomat, Coord):

    XY = [ctInv.TransformPoint(Co[0], Co[1])[:2] for Co in Coord]
    x = [int(inv_geometrix[0] + inv_geometrix[1] * xy[0] + inv_geometrix[2] * xy[1]) for xy in XY]
    y = [int(inv_geometrix[3] + inv_geometrix[4] * xy[0] + inv_geometrix[5] * xy[1]) for xy in XY]

    return x, y '''
        
#def GetPixel4(ctInv, Inv_geomat, Coord):
#
#
#    pix = [(int(inv_geometrix[0] + inv_geometrix[1] * xy[0] + inv_geometrix[2] * xy[1])\
#           int(inv_geometrix[3] + inv_geometrix[4] * xy[0] + inv_geometrix[5] * xy[1]))\
#           for xy in XY \
#           for XY in ctInv.TransformPoint(Co[0], Co[1])[:2]\
#           for Co in Coord]
#    return pix

def write_raster(fname, data, geo_transform, projection, DriverName='GTiff'):
    """Create a GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName(DriverName)
    if len(data.shape)>2:
        rows, cols, NBands = data.shape
    else:
        rows, cols = data.shape
        NBands = 1
    dataset = driver.Create(fname, cols, rows,  NBands, gdal.GDT_UInt16 )
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    for b in range(NBands): 
        band = dataset.GetRasterBand(b+1)
        try:
            band.WriteArray(data[:,:,b])
        except IndexError:
            band.WriteArray(data[:,:])
    dataset = None  # Close the file
    

    
def write_raster2(fname, data, geo_transform, projection, DriverName="GTiff", formatMem=gdal.GDT_UInt16, Offset=None ):
    """Create a GeoTIFF file with the given data."""
    
    driver = gdal.GetDriverByName(DriverName)
    if driver is None:
        print('DriverFailed')
    
    if len(data.shape)==3:
        rows, cols, NBands = data.shape
    else:
        rows, cols = data.shape
        NBands = 1

    dataset = driver.Create(fname, cols, rows,  NBands, formatMem )
    if driver is None:
        print('DatasetCreation Failed')

    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    if NBands > 1:
        for b in range(NBands): 
            band = dataset.GetRasterBand(b+1)
            if Offset == None:
                band.WriteArray(data[:,:,b])
            else:
                band.WriteArray(data[:,:,b], xoff=Offset[0], yoff=Offset[1])
    else:
        band = dataset.GetRasterBand(1)
        if Offset == None:
            band.WriteArray(data[:,:])
        else:
            band.WriteArray(data[:,:], xoff=Offset[0], yoff=Offset[1])
        
    if DriverName == 'MEM':
        return dataset
    else:
        dataset = None  # Close the file

    
def Get_LinReg(Samples, NIRSamples, NBand):
    '''Generate Coeff from Linear Regression of Bands containded in Samples on infrared Signal contained in NIRSamples
        return Coeff with format Coeff[f][c][a, b]: f for tile, c for band, a and b for Samples*a + b = NIRSamples'''
    
    Coeff = []
    # case with several bands
    if NBand>1:
        for f in range(len(NIRSamples)):
            N = len(NIRSamples[f])# check sample validity (empty tiled _> unvalid)
            # [MAIN DIFF]
            Coeff.append([])
            if N>0:
                for j in range(NBand):
                    A = np.hstack((NIRSamples[f].reshape(N,1), np.ones((N,1))))
                    Y = Samples[f][j].reshape((N,1))
                    Coeff[f].append(np.linalg.solve(A.T.dot(A), A.T.dot(Y)))
                    
    # case with only one band
    else:
        for f in range(len(NIRSamples)):
            N = len(NIRSamples[f])# check sample validity (empty tiled _> unvalid)
            if N>0:           
                A = np.hstack((NIRSamples[f].reshape(N,1), np.ones((N,1))))
                Y = Samples[f].reshape((N,1))
                Coeff.append(np.linalg.solve(A.T.dot(A), A.T.dot(Y)))
            elif N==0:
                Coeff.append([])
    return Coeff
        

def Get_R2(coeff, Samples, NIRSamples, NBands):
    '''Generate coeff of determination from Linear Regression of Band number c 
        containded in Samples on infrared Signal contained in NIRSamples 
        with regression coefficients contained in coeff.
        
        return R2 with format R2[f][c]: f for tile, c for band'''
    R2 = []
    if NBands>1:
        for f in range(len(NIRSamples)):
            R2.append([])
            N = len(NIRSamples[f])# check sample validity (empty tiled _> unvalid)
            if N>0:
                for c in range(NBands):
                    y_pred = np.hstack((NIRSamples[f].reshape(N,1), np.ones((N,1)))).dot(coeff[f][c])
                    R2[f].append(r2_score(Samples[f][c].reshape(N,1), y_pred))
    else:
        for f in range(len(NIRSamples)):
            N = len(NIRSamples[f])# check sample validity (empty tiled _> unvalid)
            if N>0:
                y_pred = np.hstack((NIRSamples[f].reshape(N,1), np.ones((N,1)))).dot(coeff[f])
                R2.append(r2_score(Samples[f].reshape(N,1), y_pred))
            elif N==0:
                R2.append([])
    return R2

def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1, format=gdal.GDT_UInt16):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize the vectors in the given directory in a single image."""
    labeled_pixels = np.zeros((rows, cols),dtype=np.int8)
    for i, path in enumerate(file_paths):
        label = i+1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1).ReadAsArray()
        index = np.logical_and(band!=0, labeled_pixels!=0)
        labeled_pixels += band
        labeled_pixels[index] = 0
        ds = None
        #print(np.unique(band),'global :',np.unique(labeled_pixels))
    return labeled_pixels

##########################################################################
# Depth invariant index computation    
def get_a(I, J):
    index = np.logical_and(~np.isnan(I), ~np.isnan(J))
    Cov_mat = np.cov(I[index],J[index], bias=1)
    return (Cov_mat[0,0] - Cov_mat[1,1]) / (2.0 * Cov_mat[0,1] )

def get_kikj(I, J):
    a = get_a(I, J)
    return a + (np.sqrt(a**2+1))

def get_dij_map(Xi, Xj, I, J):
    return Xi - (get_kikj(I, J) * Xj)

##########################################################################


def output_multipolygone_from_raster_layer(Classif,raster_NoDataValue,classes,classes_attributes,Output_File):
    """
    Convert an input 2D numpy array to shapefile, merging all pixels with the same value.
    @inspired by : output_ogr_from_2darray() , from : damo_ma
    
    usage :
        Classif_Tiff_File = 'ClassifTestVectoirze.tif'
        Classif = gdal.Open( Classif_Tiff_File)
        classes = np.arange(1,9)
        attributes_n = 5
        attributes_values = np.random.random_sample((len(classes),attributes_n))
        NoDataValue = False

        # assign classes attributes names
        # Important : Driver "ESRI Shapefile" limit field name to 10 characters!!
        classes_attributes = { 'names' : ['attr_'+str(i) for i in range(attributes_n)]}
        # assign classes attributes values
        for i in range(classes.size):
            classes_attributes[classes[i]] = attributes_values[i,:]

        output_file = "test.shp"

        output_ogr_from_raster_band(Classif,False,classes,classes_attributes,output_file)
    """
    ##########################################################################
    # create 1 bands in raster file in Memory

    from osgeo import ogr
    
    raster_band = Classif.GetRasterBand(1) 
    SRS = osr.SpatialReference()
    SRS.ImportFromWkt(Classif.GetProjection())
    
    if raster_NoDataValue != False:
        raster_band.SetNoDataValue(raster_NoDataValue) # set the NoDataValues
    ##########################################################################
    # create a Vector Layer in memory
    drv = ogr.GetDriverByName( 'Memory' )
    ogr_datasource = drv.CreateDataSource( 'out' )

    # create a new layer to accept the ogr.wkbPolygon from gdal.Polygonize
    input_layer = ogr_datasource.CreateLayer('polygonized', SRS, ogr.wkbPolygon )

    # add a field to put the classes in
    # see OGRFieldType > http://www.gdal.org/ogr/ogr__core_8h.html#a787194bea637faf12d61643124a7c9fc
    field_defn = ogr.FieldDefn('class', ogr.OFTInteger) # OFTInteger : Simple 32bit integer
    input_layer.CreateField(field_defn) # add the field to the layer

    # create "vector polygons for all connected regions of pixels in the raster sharing a common pixel value"
    # see documentation : www.gdal.org/gdal_polygonize.html
    gdal.Polygonize( raster_band, None, input_layer,  0)

    print( 'Create a Vector Layer of %s Features in Memory' % (input_layer.GetFeatureCount()))
    print( 10*'=')
    ##########################################################################
    # create 1 bands in raster file

    layerDefinition = input_layer.GetLayerDefn()

    driver = ogr.GetDriverByName("ESRI Shapefile")

    # select the field name to use for merge the polygon from the first and unique field in input_layer
    field_name = layerDefinition.GetFieldDefn(0).GetName()
    print( 'Join classification based on values in field "%s"' % field_name)
    print( 10*'=')
    # Remove output shapefile if it already exists
    if os.path.exists(Output_File):
        driver.DeleteDataSource(Output_File) 
    out_datasource = driver.CreateDataSource( Output_File )
    
    # create a new layer with wkbMultiPolygon, Spatial Reference as middle OGR file = input_layer
    multi_layer = out_datasource.CreateLayer("merged", input_layer.GetSpatialRef(), ogr.wkbMultiPolygon)
    print( 'Create output file in %s' % Output_File)
    print( 10*'=')
    
    # Add the fields we're interested in
    field_field_name = ogr.FieldDefn(field_name, ogr.OFTInteger) # add a Field named field_name = class
    multi_layer.CreateField(field_field_name)
    for attributes_name in classes_attributes['names']:
        print( ' Set feature "%s" in output file' % attributes_name)
        # iteratively define a new file from classes_attributes
        field_defn = ogr.FieldDefn(attributes_name, ogr.OFTReal) # make the type matching input type?
        # field_defn.SetWidth = len(attributes_name) # unusfeul : driver "ESRI Shapefile" limit field name to 10 characters!
        # add new file to actual layer
        multi_layer.CreateField(field_defn)

    # print out the field name defined
    multylayerDefinition = multi_layer.GetLayerDefn()
    print( 10*'=')

    for i in classes:
        # select the features in the middle OGR file with field_name == i
        input_layer.SetAttributeFilter("%s = %s" % (field_name,i))
        print( 'Field %s == %s : %s features ' % (field_name,i,input_layer.GetFeatureCount()))

        # # create a new layer with wkbMultiPolygon : each new layer is a new shp file, se this to create separated file for each class
        # multi_layer = out_datasource.CreateLayer("class %s"  % i, input_layer.GetSpatialRef(), ogr.wkbMultiPolygon) # add a Layer
        # To Do : add field definiton for each layer

        multi_feature = ogr.Feature(multi_layer.GetLayerDefn()) # generate a feature
        multipolygon  = ogr.Geometry(ogr.wkbMultiPolygon) # generate a polygon based on layer unique Geometry definition
        for feature in input_layer:
            multipolygon.AddGeometry(feature.geometry()) #aggregate all the input geometry sharing the class value i
        multi_feature.SetGeometry(multipolygon) # add the merged geoemtry to the current feature
        multi_feature.SetField2(0, i) # set the field of the current feature
        # set all the field
        for k in range(len(classes_attributes[i])):
            multi_feature.SetField(classes_attributes['names'][k],classes_attributes[i][k]) # set the field of the current feature
        multi_layer.CreateFeature(multi_feature) # add the current feature to the layer
        multi_feature.Destroy() # desrtroy the current feature

    gdal_datasource = None
    ogr_datasource  = None
    out_datasource  = None
    print(20*'=')
    