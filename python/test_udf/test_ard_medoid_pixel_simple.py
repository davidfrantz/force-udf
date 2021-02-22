from unittest import TestCase
import numpy as np

from .utils import Utils

from ard.medoid.pixel_simple.medoid import forcepy_pixel, forcepy_init


class TestArdPixelSimple_Medoid(TestCase):

    def test_handmade(self):
        nodata = -9999
        inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
        nDates, nBands = inarray.shape
        inarray = inarray.reshape((nDates, nBands, 1, 1))
        outarray = np.full((nBands), nodata, dtype=np.int16)
        forcepy_pixel(inarray, outarray, None, None, None, nodata, None)
        self.assertTrue(np.all(outarray == np.full_like(outarray, 5)))

    def test_applyToRaster(self):
        outarray = Utils.applyPixelSimpleUdf(
            'c:/vsimem/pixelSimple_Medoid.tif', forcepy_init, forcepy_pixel, '20190101', '20191231'
        )
        self.assertEqual(648523542, outarray.flatten().sum())
