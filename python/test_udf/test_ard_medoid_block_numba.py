from unittest import TestCase
import numpy as np

from .utils import Utils

from ard.medoid.block_numba.medoid import forcepy_block, forcepy_init

nodata = Utils.NO_DATA


class TestArdBlockNumba_Medoid(TestCase):

    def test_handmade(self):
        inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
        inarray = np.concatenate([inarray * 0 + nodata, inarray, inarray * 0 + nodata])  # add no data pixel
        nDates, nBands = inarray.shape
        nY, nX = 300, 300
        inarray = inarray.reshape((nDates, nBands, 1, 1)) * np.ones((nDates, nBands, nY, nX), dtype=np.int16)
        outarray = np.full((nBands, nY, nX), nodata, dtype=np.int16)
        nproc = 4
        forcepy_block(inarray, outarray, None, None, None, nodata, nproc)
        self.assertTrue(np.all(outarray == np.full_like(outarray, 5)))

    def test_noData(self):
        inarray = np.array([[nodata, nodata]], dtype=np.int16).reshape((1, 2, 1, 1))
        outarray = np.array([nodata, nodata], dtype=np.int16).reshape((2, 1, 1))
        forcepy_block(inarray, outarray, None, None, None, nodata, 1)
        self.assertTrue(np.all(outarray == nodata))

    def test_applyToRaster(self):
        outarray = Utils.applyBlockUdf(
            'c:/vsimem/blockNumba_Medoid.tif', forcepy_init, forcepy_block, '20190101', '20191231'
        )
        self.assertEqual(631050686, outarray.flatten().sum())
