import unittest
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box,mapping
from expansion_features import reduce4,contexts
from raster_overlap import OverlapPlan

class ExpansionFeaturesTest(unittest.TestCase):
    def test_conservative_reduction(self):
        a=np.arange(64,dtype=float).reshape(8,8);w=np.ones_like(a);w[:4,:2]=2
        result=reduce4(a,w)
        self.assertAlmostEqual(result[0,0],np.average(a[:4,:4],weights=w[:4,:4]))
        a[:4,:3]=np.nan
        self.assertTrue(np.isnan(reduce4(a,minimum=.5)[0,0]))
        self.assertTrue(np.isfinite(reduce4(a)[0,0]))
    def test_physical_context_before_coarsening(self):
        yy,xx=np.indices((28,28));height=xx.astype(float)*10
        result=contexts(height,8)
        np.testing.assert_allclose(result['relief_5km_m'],40)
        np.testing.assert_allclose(result['relief_21km_m'],200)
        np.testing.assert_allclose(result['tpi_21km_m'],0)
    def test_missing_domain_is_not_full_support(self):
        a=np.array([[np.nan,4.],[np.nan,4.]])
        f=[dict(type='Feature',properties={'i':0},geometry=mapping(box(0,0,2,2)))]
        plan=OverlapPlan.build(a.shape,from_origin(0,2,1,1),'EPSG:4326',f,geographic=True)
        mean,valid,total=plan.aggregate(a)
        self.assertAlmostEqual(mean[0,0],4)
        self.assertAlmostEqual((valid/total)[0,0],.5)

if __name__=='__main__':unittest.main()
