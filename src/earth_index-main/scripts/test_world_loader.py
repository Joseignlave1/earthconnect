import hashlib
import struct
import unittest

import numpy as np

from load_world import SRID, check_wkb


class WorldRasterWkbTests(unittest.TestCase):
    def fixture(self, order='<'):
        codes = np.array([[0,1,65535],[3,41,12]], dtype=order+'u2')
        layer = dict(grid_definition=dict(shape=[2,3],transform=[4000.,0.,-6000.,0.,-4000.,4000.]),
                     codes_sha256=hashlib.sha256(codes.astype('<u2').tobytes()).hexdigest())
        payload = struct.pack(order+'BHHddddddiHH',int(order=='<'),0,1,4000.,-4000.,-6000.,4000.,0.,0.,SRID,3,2)
        return payload+bytes([0x46])+struct.pack(order+'H',0)+codes.tobytes(),layer

    def test_postgis_u16_transport_in_both_byte_orders(self):
        for order in ('<','>'):
            payload,layer = self.fixture(order)
            check_wkb(payload,layer)

    def test_changed_pixel_is_detected(self):
        payload,layer = self.fixture()
        with self.assertRaisesRegex(ValueError,'roundtrip'):
            check_wkb(payload[:-1]+bytes([7]),layer)

    def test_wrong_grid_and_nodata_are_rejected(self):
        payload,layer = self.fixture()
        layer['grid_definition']['shape']=[3,2]
        with self.assertRaisesRegex(ValueError,'header'):
            check_wkb(payload,layer)
        payload,layer=self.fixture()
        with self.assertRaisesRegex(ValueError,'NODATA'):
            check_wkb(payload[:62]+struct.pack('<H',1)+payload[64:],layer)


if __name__=='__main__':
    unittest.main()
