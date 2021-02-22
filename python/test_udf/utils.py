import builtins
from datetime import date as Date
from multiprocessing.pool import Pool
from os import listdir
from os.path import join, dirname, exists, abspath
from warnings import warn

import numpy as np
from osgeo import gdal
from numba import jit, prange, set_num_threads
from qgis._core import QgsRectangle, QgsCoordinateReferenceSystem, QgsRasterLayer

from enmapboxprocessing.driver import Driver
from enmapboxprocessing.rasterreader import RasterReader


class Utils(object):
    TILE_DIRECTORY = r'\\141.20.140.222\dagobah\dc\deu\ard\X0069_Y0043'
    TILE_DIRECTORY = r'C:\Work\data\FORCE\deu\ts\X0069_Y0043'
    TSS_DIRNAME = dirname(__file__)
    BOA_NAMES1 = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']
    BOA_NAMES2 = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
    NO_DATA = -9999

    #CRS = QgsCoordinateReferenceSystem('EPSG:3035')
    #EXTENT = QgsRectangle(
    #    4526026.36304164957255125, 3266319.60796480439603329, 4556026.36304164957255125, 3269319.60796480439603329
    #)
    NPROC = 8

    @classmethod
    def applyPixelNumbaUdf(cls, filename, udf_init, udf_pixel, dateMin, dateMax, sensorFilter=None):
        inarray, dates, sensors = cls.boaBlock(dateMin, dateMax, sensorFilter)
        bandNames = np.array(udf_init(dates, sensors, Utils.BOA_NAMES2))
        outarray = Utils.outBlock(len(bandNames), *inarray.shape[-2:])
        mapPixelNumba(udf_pixel, inarray, outarray, dates, sensors, bandNames, cls.NO_DATA, cls.NPROC)
        cls.writeOutput(outarray, bandNames, filename)
        return outarray

    @classmethod
    def applyPixelSimpleUdf(cls, filename, udf_init, udf_pixel, dateMin, dateMax, sensorFilter=None):
        inarray, dates, sensors = cls.boaBlock(dateMin, dateMax, sensorFilter)
        bandNames = np.array(udf_init(dates, sensors, Utils.BOA_NAMES2))
        outarray = Utils.outBlock(len(bandNames), *inarray.shape[-2:])
        mapPixelSimple(udf_pixel, inarray, outarray, dates, sensors, bandNames, cls.NO_DATA, cls.NPROC)
        cls.writeOutput(outarray, bandNames, filename)
        return outarray

    @classmethod
    def applyBlockUdf(cls, filename, udf_init, udf_block, dateMin, dateMax, sensorFilter=None):
        inarray, dates, sensors = cls.boaBlock(dateMin, dateMax, sensorFilter)
        bandNames = udf_init(dates, sensors, cls.BOA_NAMES2)
        outarray = cls.outBlock(len(bandNames), *inarray.shape[-2:])
        udf_block(inarray, outarray, dates, None, None, cls.NO_DATA, cls.NPROC)
        cls.writeOutput(outarray, bandNames, filename)
        return outarray

    @classmethod
    def applyNdviBlockUdf(cls, filename, udf_init, udf_block, dateMin, dateMax, sensorFilter=None):
        inarray, dates, sensors = cls.boaBlock(dateMin, dateMax, sensorFilter)
        inarray = cls.ndviBlock(inarray)
        bandNames = udf_init(dates, sensors, ['NDVI'])
        outarray = cls.outBlock(len(bandNames), *inarray.shape[-2:])
        udf_block(inarray, outarray, dates, None, None, cls.NO_DATA, cls.NPROC)
        cls.writeOutput(outarray, bandNames, filename)
        return outarray


    @classmethod
    def writeOutput(cls, outBlock, bandNames, filename):
        layer = QgsRasterLayer(join(cls.TSS_DIRNAME, 'BLU_TSS.tif'))
        driver = Driver(filename, 'GTiff', 'INTERLEAVE=BAND COMPRESS=LZW PREDICTOR=2 TILED=YES BIGTIFF=YES'.split())
        writer = driver.createFromArray(outBlock, layer.extent(), layer.crs())
        for bandNo, bandName in enumerate(bandNames, 1):
            writer.setBandName(bandName, bandNo)
        writer.setNoDataValue(cls.NO_DATA)

    @classmethod
    def outBlock(cls, nBands, nY, nX):
        return np.full((nBands, nY, nX), cls.NO_DATA, dtype=np.int16)

    @classmethod
    def ndviBlock(cls, arrayBoa):
        red = arrayBoa[:, 2:3]
        nir = arrayBoa[:, 3:4]
        ndvi = np.clip((nir - red) / (nir + red) * 10000, -10000, 10000)
        ndvi[red == cls.NO_DATA] = cls.NO_DATA
        ndvi = ndvi.astype(np.int16)
        return ndvi

    @classmethod
    def boaBlock(cls, dateMin, dateMax, sensorFilter=None):
        readers = [RasterReader(join(cls.TSS_DIRNAME, name + '_TSS.tif')) for name in cls.BOA_NAMES1]
        array = list()
        dates = list()
        sensors = list()
        for bandNo in range(1, readers[0].bandCount() + 1):
            name = readers[0].bandName(bandNo)
            datestamp = name[:8]
            sensor = name[16:21]
            if dateMin is not None:
                if datestamp < dateMin:
                    continue
            if dateMax is not None:
                if datestamp > dateMax:
                    continue
            if sensorFilter is not None:
                if sensor not in sensorFilter:
                    continue
            daysSinceEpoch = (Date(int(name[:4]), int(name[4:6]), int(name[6:8])) - Date(1970, 1, 1)).days
            dates.append(daysSinceEpoch)
            sensors.append(sensor)
            for reader in readers:
                array.append(reader.array(bandList=[bandNo])[0])
        array = np.array(array).reshape((-1, 6, readers[0].height(), readers[0].width()))
        dates = np.array(dates, np.int16)
        sensors = np.array(sensors)
        return array, dates, sensors

    @classmethod
    def createTestRaster_OLD(cls):
        if exists(cls.ARD_FILENAME):
            warn(f'Creation skipped, ARD file already exists: {cls.ARD_FILENAME}')
            return

        names = list()
        for name in listdir(cls.TILE_DIRECTORY):
            if name.endswith('BOA.tif') and name:
                names.append(name)
            #if len(names) == 3: break

        array = list()
        dates = list()
        sensors = list()
        namesSelected =list()
        for name in sorted(names):
            date = np.datetime64(f'{name[:4]}-{name[4:6]}-{name[6:8]}')
            sensor = name[16:21]
            dates.append(date)
            sensors.append(sensor)
            print('read', name, sensor)
            buf_xsize, buf_ysize = 1000, 100
            if 'LND' in name:
                xoff, yoff = 0, 520
                xsize, ysize = 1000, 100
                bandList = [1, 2, 3, 4, 5, 6]
            elif 'SEN2' in name:
                xoff, yoff = 0 * 3, 520 * 3
                xsize, ysize = 1000 * 3, 100 * 3
                bandList = [1, 2, 3, 8, 9, 10]
            else:
                assert 0

            ds: gdal.Dataset = gdal.Open(join(cls.TILE_DIRECTORY, name.replace('BOA', 'QAI')))
            qaiArray = ds.ReadAsArray(xoff, yoff, xsize, ysize, None, buf_xsize, buf_ysize)

            bit, code, mask = 1, 0, 3  # gives CloudState = clear
            valid = np.right_shift(qaiArray, bit) & mask == code

            bit, code, mask = 3, 0, 1  # gives CloudShadow = no
            valid = np.logical_and(np.right_shift(qaiArray, bit) & mask == code, valid)

            bit, code, mask = 4, 0, 1  # gives Snow = no
            valid = np.logical_and(np.right_shift(qaiArray, bit) & mask == code, valid)

            bit, code, mask = 8, 0, 1  # gives Subzero = no
            valid = np.logical_and(np.right_shift(qaiArray, bit) & mask == code, valid)

            bit, code, mask = 9, 0, 1  # gives Saturation = no
            valid = np.logical_and(np.right_shift(qaiArray, bit) & mask == code, valid)

            validFraction = np.sum(valid) / buf_xsize * buf_ysize
            if validFraction < 0.1:
                print('skip', name)
                continue

            namesSelected.append(name)

            invalid = np.logical_not(valid)

            ds: gdal.Dataset = gdal.Open(join(cls.TILE_DIRECTORY, name))
            for bandNo in bandList:
                rb: gdal.Band = ds.GetRasterBand(bandNo)
                boaBandArray = rb.ReadAsArray(xoff, yoff, xsize, ysize, buf_xsize, buf_ysize)
                boaBandArray[invalid] = cls.NO_DATA
                array.append(boaBandArray)

        driver = Driver(
            cls.ARD_FILENAME, 'GTiff', 'INTERLEAVE=BAND COMPRESS=LZW PREDICTOR=2 TILED=YES BIGTIFF=YES'.split()
        )
        writer = driver.createFromArray(array, cls.EXTENT, cls.CRS)
        writer.setNoDataValue(cls.NO_DATA)
        bandNo = 0
        for name in sorted(namesSelected):
            for boaName in cls.BOA_NAMES:
                bandNo += 1
                writer.setBandName(name.replace('BOA.tif', boaName), bandNo)

    @classmethod
    def createTestRaster(cls):
        if exists(join(cls.TSS_DIRNAME, 'BLU_TSS.tif')):
            warn('Creation skipped, TSS files already exists: '+ join(cls.TSS_DIRNAME, '*_TSS.tif'))
            return
        bandList = list()

        for name in ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']:
            ds: gdal.Dataset = gdal.Open(join(cls.TILE_DIRECTORY, f'1984-2020_001-365_HL_TSA_LNDLG_{name}_TSS.tif'))
            translateOptions = gdal.TranslateOptions(
                format='GTiff', bandList=bandList, srcWin=[0, 520, 1000, 100],
                creationOptions='INTERLEAVE=BAND COMPRESS=LZW PREDICTOR=2 BIGTIFF=YES'.split()
            )
            outds: gdal.Dataset = gdal.Translate(join(cls.TSS_DIRNAME, name + '_TSS.tif'), ds, options=translateOptions)
            outds.SetMetadata(ds.GetMetadata('FORCE'), 'FORCE')
            for bandNo in range(1, ds.RasterCount + 1):
                rb: gdal.Band = ds.GetRasterBand(bandNo)
                outrb: gdal.Band = outds.GetRasterBand(bandNo)
                outrb.SetDescription(rb.GetDescription())
                outrb.SetMetadata(rb.GetMetadata('FORCE'), 'FORCE')


def mapPixelSimple(f, inblock, outblock, dates, sensors, bandnames, nodata, nproc):

    argss = list()
    nDates, nBands, nY, nX = inblock.shape
    for iYX in range(nY * nX):
        iX = iYX % nX
        iY = iYX // nX
        inarray = inblock[:, :, iY: iY + 1, iX:iX + 1]
        outarray = outblock[:, iY, iX]
        f(inarray, outarray, dates, sensors, bandnames, nodata, nproc)


@jit(nopython=True, nogil=True, parallel=True)
def mapPixelNumba(f, inblock, outblock, dates, sensors, bandnames, nodata, nproc):
    set_num_threads(nproc)

    nDates, nBands, nY, nX = inblock.shape
    for iYX in prange(nY * nX):
        iX = iYX % nX
        iY = iYX // nX
        inarray = inblock[:, :, iY: iY + 1, iX:iX + 1]
        outarray = outblock[:, iY, iX]
        f(inarray, outarray, dates, sensors, bandnames, nodata, nproc)
