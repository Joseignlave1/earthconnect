# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Compare reusable weights to independent reference extraction and edge cases."""
import unittest
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box, mapping
import priority_pilot as p
from raster_overlap import OverlapPlan


def feature(i, geometry):
    return dict(type='Feature', properties={'i': i}, geometry=mapping(geometry))


class OverlapTests(unittest.TestCase):
    def compare(self, transform, features, geographic=False):
        rng = np.random.default_rng(42)
        plan = OverlapPlan.build((9,11), transform, p.CRS_WKT, features, geographic=geographic)
        for missing in (0, .3, 1):
            values = rng.uniform(-120,80,(9,11))
            values[rng.random(values.shape) < missing] = np.nan
            actual = plan.aggregate(values)
            expected = p.aggregate(values,transform,p.CRS_WKT,features,geographic)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b,rtol=2e-14,atol=2e-8,equal_nan=True)
            if missing < 1:
                np.testing.assert_array_equal(p.quantize(actual[0],.02,-150.02)[0],p.quantize(expected[0],.02,-150.02)[0])
        return plan

    def test_fractional_rotated_holey_outside_and_reordered_cells(self):
        outer = [(1,1),(5,2),(4,8),(0,7),(1,1)]
        hole = [(2,3),(3,3),(3,4),(2,4),(2,3)]
        fs = [feature(3,box(40,40,50,50)),feature(1,box(5.5,-2,12,4.5)),
              feature(0,Polygon(outer,[hole])),feature(2,box(0,0,11,9))]
        self.compare(from_origin(0,9,1,1),fs)

    def test_cartesian_metre_area(self):
        fs = [feature(i,box(x,y,x+3500,y+3500)) for i,(x,y) in enumerate([(100,200),(4000,200),(100,4000),(4000,4000)])]
        self.compare(from_origin(0,9000,1000,1000),fs)

    def test_spherical_area_at_high_latitude(self):
        fs = [feature(i,box(x,y,x+.04,y+.04)) for i,(x,y) in enumerate([(10,75),(10.04,75),(10,75.04),(10.04,75.04)])]
        self.compare(from_origin(10,75.09,.01,.01),fs,True)

    def test_invalid_source_shape_and_target_ids(self):
        fs = [feature(0,box(0,0,1,1))]
        plan = OverlapPlan.build((2,2),from_origin(0,2,1,1),p.CRS_WKT,fs)
        with self.assertRaises(ValueError): plan.aggregate(np.ones((3,3)))
        with self.assertRaises(ValueError): plan.aggregate(np.full((2,2),np.inf))
        with self.assertRaises(ValueError): OverlapPlan.build((2,2),from_origin(0,2,1,1),p.CRS_WKT,fs*4)
        with self.assertRaises(ValueError): plan.weights[0]=1

    def test_source_tiles_combine_by_area_not_mean_of_means(self):
        fs=[feature(0,box(.25,0,3.5,2))]
        left=OverlapPlan.build((2,2),from_origin(0,2,1,1),p.CRS_WKT,fs)
        right=OverlapPlan.build((2,2),from_origin(2,2,1,1),p.CRS_WKT,fs)
        a=np.array([[0.,np.nan],[0,0]]); b=np.full((2,2),20.)
        ma,va,_=left.aggregate(a); mb,vb,_=right.aggregate(b)
        combined=(np.nan_to_num(ma)*va+np.nan_to_num(mb)*vb)/(va+vb)
        whole=np.concatenate([a,b],axis=1)
        expected=p.aggregate(whole,from_origin(0,2,1,1),p.CRS_WKT,fs)[0]
        np.testing.assert_allclose(combined,expected,rtol=1e-14)


if __name__=='__main__':unittest.main()
