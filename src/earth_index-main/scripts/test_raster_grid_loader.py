"""Grid declarations are checked against bytes before PostgreSQL is touched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
import load_raster_pilot as loader

class GridLoaderTest(unittest.TestCase):
    def fixture(self,root):
        processor=root/'processor.py';processor.write_text('# fixture\n')
        site=dict(key='fixture',crs_wkt=CRS.from_epsg(32619).to_wkt(),srid=32619)
        a=np.array([[1,0,2],[3,4,5]],dtype='<u2');path=root/'tile.tif'
        with rasterio.open(path,'w',driver='GTiff',height=2,width=3,count=1,dtype='uint16',nodata=0,crs=site['crs_wkt'],transform=from_origin(-6000,4000,4000,4000)) as dst:dst.write(a,1)
        layer=dict(site_key='fixture',path='tile.tif',file_sha256=loader.sha(path.read_bytes()),codes_sha256=loader.sha(a.tobytes()),valid_cells=5,scale=1,max_rounding_error=0,
            grid_definition=dict(shape=[2,3],transform=[4000,0,-6000,0,-4000,4000],crs_wkt=site['crs_wkt']))
        return dict(processor_path='processor.py',script_sha256=loader.sha(processor.read_bytes()),sites=[site],layers=[layer])
    def test_nonsquare_4km_and_false_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root)
            with patch.object(loader,'ROOT',root):
                (root/'manifest.json').write_text(json.dumps(data));loader.manifest(root)
                data['layers'][0]['grid_definition']['transform'][2]=-50000
                (root/'manifest.json').write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'declared grid'):loader.manifest(root)
    def test_false_valid_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root);data['layers'][0]['valid_cells']=6
            (root/'manifest.json').write_text(json.dumps(data))
            with patch.object(loader,'ROOT',root),self.assertRaisesRegex(ValueError,'Valid-cell'):loader.manifest(root)

if __name__=='__main__':unittest.main()
