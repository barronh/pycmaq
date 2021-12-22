import numpy as np


def cellpolygons(NROWS, NCOLS):
    """
    Retrieve cell polygons as a 2d array.

    Arguments
    ---------
    NROWS : int
        number of rows
    NCOLS : int
        number of cols

    Returns
    -------
    gridpolys : np.array
        Array of shapely.Polygon objects with NROWS x NCOLS shape
    """
    from shapely.geometry import Polygon

    # First create a series of polygons representing each grid cell
    # Then, create a tree for fast subsetting based on the ocean
    # segment.
    def cell(i, j):
        return Polygon(
            [
                [i + 0, j + 0],
                [i + 1, j + 0],
                [i + 1, j + 1],
                [i + 0, j + 1],
                [i + 0, j + 0],
            ]
        )

    gridpolys = np.array([
        [cell(i, j) for i in range(NCOLS)] for j in range(NROWS)
    ], dtype='object')
    return gridpolys


def wholedomain(NROWS, NCOLS, lonlat=False, proj=None):
    """
    Return a wholedomain polygon (optionally in lonlat coordinates)

    Arguments
    ---------
    NROWS : int
        number of rows
    NCOLS : int
        number of cols
    lonlat : bool
        If True, return domain in lon/lat
    proj : pyproj.Proj
        Only required if lonlat is True

    Returns
    -------
    celltree : shapely.strtree.STRtree
        tree of cell polygons. Good for fast querying
    """
    from shapely.geometry import Polygon

    sx = np.arange(NCOLS + 1)
    sy = np.zeros_like(sx)
    ey = np.arange(NROWS + 1)
    ex = np.ones_like(ey) * sx[-1]
    nx = sx[::-1]
    ny = np.ones_like(nx) * ey[-1]
    wy = ey[::-1]
    wx = np.zeros_like(wy)
    x = np.concatenate([sx, ex, nx, wx])
    y = np.concatenate([sy, ey, ny, wy])
    wholedomain = Polygon(np.array([x, y]).T)
    if lonlat:
        if proj is None:
            raise ValueError('When lonlat=True, proj is required')
        return aslonlat(wholedomain, proj)
    else:
        return wholedomain


def aslonlat(poly, proj):
    from shapely.ops import transform
    return transform(lambda x, y: proj(x, y, inverse=True), poly)


def togrid(gf, srcshapes, srcproj=4326, clip=True):
    """
    Convert srcshapes to gf grid projection

    Arguments
    ---------
    gf : xr.Dataset
        Dataset with cmaq accessor
    srcshapes : list
        List of shapely Shapes to project.
    srcproj : str or int
        Must work with pyproj.CRS
    clip : bool
        Perform clipping in srcproj before regridding. Useful for ill-defined
        shapes in the destination coordinates.

    Returns
    -------
    outshapes : list
        List of shapes in grid projection
    """
    from shapely.ops import transform
    from shapely.geometry import Polygon
    from pyproj import Transformer

    if clip:
        domain_raw = wholedomain(gf.NROWS, gf.NCOLS)
        domain = transform(
            Transformer.from_crs(
                gf.cmaq.proj4, srcproj, always_xy=True
            ).transform,
            domain_raw.buffer(1)
        )
        if gf.GDTYP == 6:
            # The polar stereographic domain is ill-defined in lonlat
            # Use the minimum y to create a full domain
            # miny = min(domain.exterior.xy[1]) - 1
            miny = -30
            domain = Polygon(
                [
                    [-180, miny], [180, miny],
                    [180, 90], [-180, 90],
                    [-180, miny]
                ]
            )

    ll2xy = Transformer.from_crs(
        srcproj, gf.cmaq.proj4, always_xy=True
    ).transform
    shapes_polys_proj = []
    for shapei in srcshapes:
        if clip:
            cshapei = shapei.intersection(domain)
        else:
            cshapei = shapei
        cshp = transform(ll2xy, cshapei)
        if hasattr(shapei, 'record'):
            setattr(cshp, 'record', getattr(shapei, 'record'))
        shapes_polys_proj.append(cshp)

    return shapes_polys_proj


def gridfraction_frompoly(
    gf, shapes, srcproj=None, propname='area', verbose=0
):
    """
    Arguments
    ---------
    gf : xr.Dataset
        Must have the cmaq accessor
    shapes : list
        List of polygons in projected space
    srcproj : str, int
        Must define the projection of the shapes. If None, the shapes are
        assumed to be in the grid projection already.
    propname : str
        Property to add. Usually area, but length can be useful.

    Returns
    -------
    out : xr.DataArray
        Sum of propname (usually area) from shapes in each grid cell
    """
    import xarray as xr
    from shapely.prepared import prep
    from shapely.strtree import STRtree

    tree = STRtree(cellpolygons(gf.NROWS, gf.NCOLS).ravel())
    afrac = np.zeros((gf.NROWS, gf.NCOLS), dtype='f')
    nl = len(shapes)
    if srcproj is None:
        gshapes = shapes
    else:
        gshapes = togrid(gf, shapes, srcproj=srcproj)
    if verbose > 0:
        print('Progress\n NP   POLY  NGCELL  GCELL')
    for li, poly in enumerate(gshapes):
        if not poly.is_valid:
            if verbose > 0:
                print('Original poly is invalid, buffering to 0')
            poly = poly.buffer(0)
            if not poly.is_valid:
                print('Buffered poly is still invalid, skipping', li)
                continue
        gcells = tree.query(poly)
        ng = len(gcells)
        if ng > 20:
            ppoly = prep(poly)
            contains = ppoly.contains
        else:
            contains = poly.contains

        intersection = poly.intersection

        for gi, gcell in enumerate(gcells):
            if verbose > 0:
                print(
                    f'\r{nl:3d} {(li)/nl:6.1%} {ng:-7d} {(gi)/ng:6.1%}',
                    end=''
                )
            j = int(gcell.exterior.coords[0][1])
            i = int(gcell.exterior.coords[0][0])
            if contains(gcell):
                intxarea = 1
            else:
                intxarea = getattr(intersection(gcell), propname)

            afrac[j, i] += intxarea

    if verbose > 0:
        print(f'\r{nl:3d} 100.0% {ng:-7d} 100.0%', flush=True)

    out = xr.DataArray(
        afrac, dims=('ROW', 'COL'),
        coords=dict(
            ROW=np.arange(gf.NROWS) + 0.5,
            COL=np.arange(gf.NCOLS) + 0.5
        )
    )
    return out
