from unittest import TestCase

from test_udf.utils import Utils


class TestUtils(TestCase):

    def test_createTestRaster(self):
        Utils.createTestRaster()

    def test_boaBlock(self):
        boaArray, dates, sensors, bandNames = Utils.boaBlock(dateMin='19810101', dateMax='19841231')
        self.assertEqual(4, boaArray.ndim)
        self.assertEqual((14, 6, 100, 1000), boaArray.shape)

    def test_ndviBlock(self):
        boaArray, dates, sensors, bandNames = Utils.boaBlock(dateMin='19810101', dateMax='19841231')
        ndviArray = Utils.ndviBlock(boaArray)
        self.assertEqual((14, 1, 100, 1000), ndviArray.shape)
