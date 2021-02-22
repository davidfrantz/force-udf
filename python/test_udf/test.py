from os import listdir
from os.path import join, dirname, exists
from unittest import TestCase
import numpy as np
from osgeo import gdal
from numba import jit, prange, set_num_threads

from ard.medoid.pixel_simple.medoid import forcepy_pixel as ardPixelSimple_Medoid
from ard.medoid.block_numba.medoid import forcepy_block as ardBlockNumba_Medoid
from ts.clear_observation_sequence.clear_observation_sequence \
    import forcepy_pixel as tsPixelNumba_ClearObservationSequence

from enmapboxprocessing.driver import Driver
from enmapboxprocessing.rasterreader import RasterReader
from enmapboxprocessing.utils import Utils as EnmapboxProcessingUtils


class TestPixelSimpleMedoid(TestCase):

    def test_handmadePixel(self):
        nodata = -9999
        inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
        inarray = np.concatenate([inarray, inarray * 0 + nodata, inarray * 0 + nodata])  # add a lot of no data pixel
        outarray = np.array([nodata, nodata], dtype=np.int16)
        ardPixelSimple_Medoid(inarray, outarray, None, None, None, nodata, 1)
        assert np.all(outarray == (5, 5))

    def test_realPixel(self):
        nodata = -9999
        inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
        inarray = np.concatenate([inarray, inarray * 0 + nodata, inarray * 0 + nodata])  # add a lot of no data pixel
        outarray = np.array([nodata, nodata], dtype=np.int16)
        ardPixelSimple_Medoid(inarray, outarray, None, None, None, nodata, 1)
        assert np.all(outarray == (5, 5))
        print(Utils.ardBlock())


class TestBlockNumbaMedoid(TestCase):

    def test_handmadeBlock(self):
        nodata = -9999
        inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
        inarray = np.concatenate([inarray * 0 + nodata, inarray, inarray * 0 + nodata])  # add no data pixel
        nDates, nBands = inarray.shape
        nY, nX = 300, 300
        inarray = inarray.reshape((nDates, nBands, 1, 1)) * np.ones((nDates, nBands, nY, nX), dtype=np.int16)
        outarray = np.full((nBands, nY, nX), nodata, dtype=np.int16)
        nproc = 4
        ardBlockNumba_Medoid(inarray, outarray, None, None, None, nodata, nproc)
        print('Done', outarray)
        assert np.all(outarray == np.full_like(outarray, 5))


class TestCompleteTile(TestCase):

    def setUp(self):
        self.nproc = 8
        self.nodata = -9999
        self.inarrayBoa, self.dates, self.sensors, self.extent, self.crs = Utils.boaBlock()
        self.inarrayNdvi = Utils.ndviBlock(self.inarrayBoa)
        self.nDates, self.nBands, self.nY, self.nX = self.inarrayBoa.shape

    def test_blockNumba_Medoid(self):

        self.outarray = np.full((self.nBands, self.nY, self.nX), self.nodata, dtype=np.int16)

        ardBlockNumba_Medoid(self.inarrayBoa, self.outarray, self.dates, None, None, self.nodata, self.nproc)
        self.assertEqual(65343043, self.outarray.flatten()[::100].sum())
        self.filename = 'c:/vsimem/blockNumba_Medoid.tif'

    def test_pixelNumba_ClearObservationSequence(self):
        self.outarray = np.full((self.nDates * 5, self.nY, self.nX), self.nodata, dtype=np.int16)
        self.bandNames = list()
        for date, sensor in zip(self.dates, self.sensors):
            datestamp = str(np.datetime64('1970-01-01') + np.timedelta64(date, 'D')).replace('-', '')
            for name in ['COS-1', 'COS0', 'COS+1', 'COS-1_DTIME', 'COS1_DTIME']:
                self.bandNames.append(f'{name}_NDV_{datestamp}_{sensor}')
        [print(n) for n in self.bandNames]
        utilsMapPixelNumba(
            tsPixelNumba_ClearObservationSequence, self.inarrayNdvi, self.outarray, self.dates, None, None, self.nodata,
            self.nproc
        )
        #self.assertEqual(1, self.outarray.flatten()[::100].sum())
        self.filename = 'c:/vsimem/pixelNumba_ClearObservationSequence.tif'

    def tearDown(self):
        print('write data')
        print(self.outarray.shape)
        writer = Driver(self.filename).createFromArray(self.outarray, extent=self.extent, crs=self.crs)
        for bandNo, bandName in enumerate(self.bandNames, 1):
            writer.setBandName(bandName, bandNo)
        writer.setNoDataValue(-9999)
        print('created', self.filename)


class Utils(object):

    @classmethod
    def ndviBlock(cls, arrayBoa):
        red = arrayBoa[:, 2:3]
        nir = arrayBoa[:, 3:4]
        ndvi = np.clip((nir - red) / (nir + red) * 10000, -10000, 10000)
        ndvi[red == -9999] = -9999
        ndvi = ndvi.astype(np.int16)
        return ndvi

    @classmethod
    def boaBlock(cls):

        pickleFilename = join(dirname(__file__), 'ard_block.dat')
        print('read data')
        if not exists(pickleFilename):
            noData = -9999
            xsize, ysize = 1000, 100
            # test qai masking
            # reader = RasterReader('C:/Work/data/FORCE/deu/ard/X0069_Y0043/20190102_LEVEL2_SEN2B_QAI.tif')
            # qai = reader.gdalDataset.ReadAsArray(buf_xsize=100, buf_ysize=100)
            # Driver('c:/vsimem/mask.tif').createFromArray(mask[None], extent=reader.extent(), crs=reader.crs())

            rootArd = r'C:\Work\data\FORCE\deu\ard\X0069_Y0043'
            array = list()
            dates = list()
            sensors = list()
            for name in listdir(rootArd):
                if name.endswith('BOA.tif') and name:
                    date = np.datetime64(f'{name[:4]}-{name[4:6]}-{name[6:8]}')
                    sensor = name[16:21]
                    dates.append(date)
                    sensors.append(sensor)
                    print('read', name, sensor)
                    if 'LND' in name:
                        bandList = [1, 2, 3, 4, 5, 6]
                    elif 'SEN2' in name:
                        bandList = [1, 2, 3, 8, 9, 10]
                    else:
                        assert 0

                    ds: gdal.Dataset = gdal.Open(join(rootArd, name.replace('BOA', 'QAI')))
                    qaiArray = ds.ReadAsArray(buf_xsize=xsize, buf_ysize=ysize)
                    bit, code, mask = 1, 0, 3  # gives CloudState = clear
                    cloudfree = np.right_shift(qaiArray, bit) & mask == code
                    bit, code, mask = 3, 0, 1  # gives CloudShadow = no
                    noshadow = np.right_shift(qaiArray, bit) & mask == code
                    invalid = np.logical_not(np.logical_and(cloudfree, noshadow))

                    ds: gdal.Dataset = gdal.Open(join(rootArd, name))
                    boaArray = list()
                    for bandNo in bandList:
                        boaBandArray = ds.GetRasterBand(bandNo).ReadAsArray(buf_xsize=xsize, buf_ysize=ysize)
                        boaBandArray[invalid] = noData
                        boaArray.append(boaBandArray)
                    array.append(boaArray)
                    #if len(dates) == 5: break
            array = np.array(array, dtype=np.int16)
            assert dates == sorted(dates)
            dates = np.array([(date - np.datetime64('1970-01-01')).item().days for date in dates])
            sensors = np.array(sensors)
            anyInputFilename = join(rootArd, name)
            EnmapboxProcessingUtils.pickleDump((array, dates, sensors, anyInputFilename), filename=pickleFilename)
        else:
            array, dates, sensors, anyInputFilename = EnmapboxProcessingUtils.pickleLoad(filename=pickleFilename)

        print(array.shape)
        reader = RasterReader(anyInputFilename)
        return array, dates, sensors, reader.extent(), reader.crs()


@jit(nopython=True, nogil=True, parallel=True)
def utilsMapPixelNumba(f, inblock, outblock, dates, sensors, bandnames, nodata, nproc):
    set_num_threads(nproc)

    nDates, nBands, nY, nX = inblock.shape
    for iYX in prange(nY * nX):
        iX = iYX % nX
        iY = iYX // nX
        inarray = inblock[:, :, iY, iX]
        outarray = outblock[:, iY, iX]
        f(inarray, outarray, dates, sensors, bandnames, nodata, nproc)
