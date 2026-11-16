# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""End-to-end QC, annual quantization, tile writing, cache and resume fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
import thermal_import as t
from raster_source_cache import SourceCache, asset_path


class ImportTests(unittest.TestCase):
    def fixture(self, cache):
        site=dict(key='fixture',crs_wkt=t.p.CRS_WKT)
        items=[]
        for month in range(1,13):
            assets={}
            for band in t.reference.BANDS:
                href=f'https://fixture.invalid/{month}-{band}.tif';path=asset_path(cache,href)
                qc=band.startswith('QC')
                raw=np.zeros((4,4),dtype='uint8') if qc else np.full((4,4),14000+month*20,dtype='uint16')
                if band=='QC_Day':raw[month%4,0]=1
                if band=='QC_Night':raw[(month+1)%4,1]=2
                if band=='LST_Night_1km':raw[2,2]=0
                with rasterio.open(path,'w',driver='GTiff',height=4,width=4,count=1,dtype=raw.dtype,
                                   crs=t.p.CRS_WKT,transform=from_origin(-2000,2000,1000,1000),nodata=0) as dst:
                    dst.write(raw,1)
                    if not qc:dst.scales=(.02,);dst.offsets=(0.,)
                assets[band]=dict(href=href)
            items.append(dict(id=f'fixture-{month}',assets=assets,properties={'start_datetime':f'2023-{month:02d}-01T00:00:00Z'}))
        return site,items

    def test_reuses_plan_and_preserves_yearly_codes_at_both_resolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            site,items=self.fixture(tmp)
            for cell_m in (1000,4000):
                fast=t.compute_tile(site,items,tmp,cell_m=cell_m,side_km=4,verify=True)
                slow=t.compute_tile(site,items,tmp,cell_m=cell_m,side_km=4,engine='reference')
                self.assertEqual(fast['stats']['plan_builds'],1)
                self.assertEqual(fast['stats']['reference_checks'],36)
                for (key,a,step,offset),(_,b,_,_) in zip(fast['layers'],slow['layers']):
                    np.testing.assert_array_equal(t.p.quantize(a,step,offset)[0],t.p.quantize(b,step,offset)[0],err_msg=key)

    def test_new_grid_geotiffs_and_resume_hash_validation(self):
        with tempfile.TemporaryDirectory(dir=t.ROOT/'research',prefix='test-thermal-import-') as tmp:
            output=Path(tmp)
            for child in ('tiles','sources','checkpoints','cache'):(output/child).mkdir()
            site,items=self.fixture(output/'cache')
            config=dict(output=tmp,cell_m=4000,side_km=4,verify=False,identity='fixture')
            result=t.write_tile((site,items,str(output/'cache'),config))
            self.assertEqual(len(result['layers']),9)
            layer=result['layers'][0]
            with rasterio.open(t.ROOT/layer['path']) as src:
                self.assertEqual(src.shape,(1,1));self.assertEqual(src.transform.a,4000)
                self.assertEqual(src.nodata,0);self.assertEqual(src.dtypes,('uint16',))
            checkpoint=output/'checkpoints/fixture.json'
            self.assertIsNotNone(t.valid_checkpoint(checkpoint,'fixture'))
            self.assertIsNone(t.valid_checkpoint(checkpoint,'other'))
            sources=output/'sources/fixture-windows.json';original=sources.read_bytes();sources.write_bytes(b'corrupted')
            self.assertIsNone(t.valid_checkpoint(checkpoint,'fixture'));sources.write_bytes(original)
            (t.ROOT/layer['path']).write_bytes(b'corrupt')
            self.assertIsNone(t.valid_checkpoint(checkpoint,'fixture'))

    def test_two_native_tiles_are_mosaicked_before_annual_support_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            site,whole=self.fixture(tmp);items=[]
            for item in whole:
                for part in range(2):
                    assets={}
                    for band,asset in item['assets'].items():
                        href=asset['href']+f'-part-{part}'
                        with rasterio.open(asset_path(tmp,asset['href'])) as src:
                            window=Window(part*2,0,2,4)
                            data=src.read(1,window=window)
                            profile=src.profile|dict(width=2,transform=src.window_transform(window))
                            scales,offsets=src.scales,src.offsets
                        with rasterio.open(asset_path(tmp,href),'w',**profile) as dst:
                            dst.write(data,1);dst.scales=scales;dst.offsets=offsets
                        assets[band]=dict(href=href)
                    items.append(item|dict(id=item['id']+f'-part-{part}',assets=assets))
            fast=t.compute_tile(site,items[::-1],tmp,cell_m=4000,side_km=4)
            expected=t.compute_tile(site,items,tmp,cell_m=4000,side_km=4,engine='reference')
            whole_result=t.compute_tile(site,whole,tmp,cell_m=4000,side_km=4,engine='reference')
            self.assertEqual(fast['stats']['plan_builds'],2)
            self.assertEqual(fast['stats']['source_granules'],24)
            self.assertEqual(fast['stats']['composites'],12)
            for (key,a,step,offset),(_,b,_,_) in zip(fast['layers'],expected['layers']):
                np.testing.assert_allclose(a,b,rtol=2e-14,atol=1e-10,equal_nan=True)
                np.testing.assert_array_equal(t.p.quantize(a,step,offset)[0],t.p.quantize(b,step,offset)[0],err_msg=key)
            for (_,a,_,_),(_,b,_,_) in zip(fast['layers'],whole_result['layers']):
                np.testing.assert_allclose(a,b,rtol=2e-14,atol=1e-10,equal_nan=True)

    def test_download_deduplication_and_corrupt_cache_recovery(self):
        href='https://fixture.invalid/source.tif';payload=b'II*\x00fixture'
        with tempfile.TemporaryDirectory() as tmp,patch.object(SourceCache,'_sign',return_value='unused'),patch.object(SourceCache,'_fetch',return_value=(payload,dict(seconds=.1,retries=0))) as fetch:
            result=SourceCache(tmp).fetch_all([href,href],workers=2)
            self.assertEqual(fetch.call_count,1);self.assertEqual(len(result),1)
            result=SourceCache(tmp).fetch_all([href],workers=1)
            self.assertTrue(result[0]['cached']);self.assertEqual(fetch.call_count,1)
            asset_path(tmp,href).write_bytes(b'corrupted')
            result=SourceCache(tmp).fetch_all([href],workers=1)
            self.assertFalse(result[0]['cached']);self.assertEqual(fetch.call_count,2)

    def test_empty_catalog_is_missing_not_zero_temperature(self):
        site=dict(key='empty',crs_wkt=t.p.CRS_WKT)
        result=t.compute_tile(site,[],'/unused',cell_m=4000,side_km=4)
        for key,array,_,_ in result['layers']:
            if key.endswith('_valid_fraction'):self.assertTrue((array==0).all())
            else:self.assertTrue(np.isnan(array).all())


if __name__=='__main__':unittest.main()
