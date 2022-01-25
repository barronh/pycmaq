import numpy as np
import pandas as pd
import pytest
from ..utils import shapes
from .helper import has_shapely, gettest


def test_shapes_wholedomainlonlat():
    if not has_shapely:
        pytest.skip('requires shapely')

    ds = gettest(True)
    shape = shapes.wholedomain(
        ds, lonlat=True, proj=ds.cmaq.pyproj
    )
    x, y = shape.exterior.xy
    refx = np.array([
        -119.45922366147477, -113.6158552249225, -107.50188966618343,
        -101.17368717523799, -94.7021077178194, -88.16857260610836,
        -81.6592852630197, -75.25856165429865, -69.04251626207342,
        -69.04251626207342, -67.27351421727766, -65.18879023327482,
        -62.69930055121067, -62.69930055121067, -70.63756080979887,
        -78.98252797131047, -87.5745658946059, -96.22262391418992,
        -104.72815634815119, -112.91141417357606, -120.6319015779604,
        -127.79792422662393, -127.79792422662393, -124.56911453553859,
        -121.82111949208092, -119.45922366147477
    ], dtype='d')
    assert(np.allclose(x, refx))
    refy = np.array([
        28.527355597838845, 30.27990734103588, 31.656989996287628,
        32.62735786394558, 33.16811103583171, 33.26615849349422,
        32.91910075164628, 32.13539540071818, 30.93377882327522,
        30.93377882327522, 36.20203031357923, 41.50850206139401,
        46.8008806229395, 46.8008806229395, 48.407443166778094,
        49.46101516662424, 49.92910736482153, 49.79676928118771,
        49.06826110036393, 47.76656813511721, 45.93090542174311,
        43.61291784560312, 43.61291784560312, 38.601290746794724,
        33.554854309332484, 28.527355597838845
    ], dtype='d')
    assert(np.allclose(y, refy))


def test_util_to_grid():
    from shapely.geometry import Polygon
    if not has_shapely:
        pytest.skip("requires shapely")
    gf = gettest(True)
    xy = np.array([(0, 0), (8, 0), (8, 3), (0, 3), (0, 0)], dtype='d')
    x, y = xy.T
    refxy = pd.DataFrame(dict(x=x, y=y)).sort_values(by=['x', 'y'])
    mysll = Polygon([
        gf.cmaq.pyproj(x, y, inverse=True)
        for x, y in xy
    ])
    srcshapes = [mysll]

    destshapes = shapes.togrid(gf, srcshapes, srcproj=4326, clip=True)
    x, y = np.asarray(destshapes[0].exterior.xy)
    chkxydf = pd.DataFrame(dict(x=x, y=y)).round(6)
    chkxy = chkxydf.sort_values(by=['x', 'y'], ignore_index=True)
    assert(np.allclose(chkxy.values, refxy.values))
    destshapes = shapes.togrid(gf, srcshapes, srcproj=4326, clip=False)
    x, y = np.asarray(destshapes[0].exterior.xy)
    chkxydf = pd.DataFrame(dict(x=x, y=y)).round(6)
    chkxy = chkxydf.sort_values(by=['x', 'y'], ignore_index=True)
    assert(np.allclose(chkxy.values, refxy.values))
