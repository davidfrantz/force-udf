from unittest import TestCase
import numpy as np

from .utils import Utils

from ts.clear_observation_sequence.clear_observation_sequence import forcepy_block, forcepy_init

nodata = Utils.NO_DATA


class TestTsBlockNumba_ClearObservationSequence(TestCase):

    def test_handmade(self):
        assert 0

    def test_applyToRaster(self):
        outarray = Utils.applyNdviBlockUdf(
            'c:/vsimem/blockNumba_ClearObservationSequence.tif', forcepy_init, forcepy_block, '20190101', '20191231'
        )
        self.assertEqual(631050686, outarray.flatten().sum())
