# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Scientific fixtures for the current-grid clear-sky thermal summaries."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import thermal_pilot as t

class ThermalTests(unittest.TestCase):
    def test_requires_four_seasons_and_twelve_usable_composites(self):
        dates=[f'2023-{m:02d}-{day:02d}' for m in range(1,13) for day in (1,9)]
        stack=np.zeros((24,2,2));fraction=np.ones_like(stack)
        stack[:,0,1]=-20;fraction[:,1,0]=.1;fraction[:18,1,1]=0
        mean,low,high,coverage=t.summarize(stack,fraction,dates,[8]*24)
        self.assertEqual(mean[0,0],0);self.assertEqual(mean[0,1],-20)
        self.assertTrue(np.isnan(mean[1]).all())
        self.assertTrue(np.allclose([low[0,0],high[0,0]],[0,0]))
        self.assertAlmostEqual(coverage[0,0],192/365)
    def test_duration_and_accepted_area_weight_the_mean(self):
        dates=[f'2023-{m:02d}-01' for m in range(1,13)]
        stack=np.full((12,1,1),20.);stack[-1]=0
        fractions=np.ones_like(stack);fractions[-1]=.5
        result=t.summarize(stack,fractions,dates,[8]*11+[4])
        self.assertAlmostEqual(result[0][0,0],20*88/90)
        self.assertEqual(result[1][0,0],0)
    def test_quality_zero_is_good_and_pairing_requires_both(self):
        with tempfile.TemporaryDirectory() as directory:
            assets={}
            for band in t.BANDS:
                path=Path(directory)/(band+'.tif');qa=band.startswith('QC')
                data=np.zeros((3,3),dtype='uint8') if qa else np.full((3,3),15000,dtype='uint16')
                if band=='QC_Day': data[0,0]=1
                if band=='LST_Night_1km': data[1,1]=0
                with rasterio.open(path,'w',driver='GTiff',height=3,width=3,count=1,dtype=data.dtype,crs=t.p.CRS_WKT,transform=from_origin(0,3000,1000,1000),nodata=0) as dst:
                    dst.write(data,1)
                    if not qa:dst.scales=(.02,);dst.offsets=(0.,)
                assets[band]={'href':str(path)}
            class IdentitySigner:
                def sign(self,href):return href
            (day,night,paired),_,_=t.decode_window({'id':'fixture','assets':assets},(0,0,3000,3000),IdentitySigner())
            self.assertAlmostEqual(day[2,2],26.85)
            self.assertTrue(np.isnan(day[0,0]));self.assertTrue(np.isnan(night[1,1]))
            self.assertTrue(np.isnan(paired[0,0]));self.assertTrue(np.isnan(paired[1,1]))
            self.assertAlmostEqual(paired[2,2],26.85)
    def test_quantization_keeps_negative_celsius_and_missing_separate(self):
        a=np.array([[-123.15,0,1037.55,np.nan]])
        codes,error=t.p.quantize(a,.02,-150.02)
        self.assertEqual(codes[0,-1],0)
        self.assertTrue(np.all(codes[0,:3]>0));self.assertLessEqual(error,.01000001)

if __name__=='__main__':unittest.main()
