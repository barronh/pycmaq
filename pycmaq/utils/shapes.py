import numpy as np


def cellcenters(gf):
    """
    Retrieve cell centroids as a 2d array.

    Arguments
    ---------
    gf : xr.Dataset
        Must have ROW and COL coordinates

    Returns
    -------
    gridpolys : np.array
        Array of shapely.Polygon objects with NROWS x NCOLS shape
    """
    from shapely.geometry import MultiPoint
    row = gf['ROW'].values
    col = gf['COL'].values

    COL, ROW = np.meshgrid(col, row)
    mp = MultiPoint(np.array([COL.ravel(), ROW.ravel()]).T)
    out = np.array(mp.geoms, dtype='object').reshape(*COL.shape)
    return out


def cellpolygons(gf):
    """
    Retrieve cell polygons as a 2d array.

    Arguments
    ---------
    gf : xr.Dataset
        Must have GDTYP, XCELL, YCELL attributes and ROW and COL coordinates

    Returns
    -------
    gridpolys : np.array
        Array of shapely.Polygon objects with NROWS x NCOLS shape
    """
    from shapely.geometry import Polygon

    if gf.attrs['GDTYP'] == 1:
        dx = gf.attrs['XCELL'] / 2
        dy = gf.attrs['YCELL'] / 2
    else:
        dx = dy = 0.5

    # First create a series of polygons representing each grid cell
    # Then, create a tree for fast subsetting based on the ocean
    # segment.
    def cell(i, j):
        return Polygon(
            [
                [i - dx, j - dy],
                [i + dx, j - dy],
                [i + dx, j + dx],
                [i - dx, j + dx],
                [i - dx, j - dy],
            ]
        )

    gridpolys = np.array([
        [cell(i, j) for i in gf['COL']] for j in gf['ROW']
    ], dtype='object')
    return gridpolys


def wholedomain(gf, lonlat=False, proj=None):
    """
    Return a wholedomain polygon (optionally in lonlat coordinates)

    Arguments
    ---------
    gf : xr.Dataset
        Must have GDTYP, XCELL, YCELL attributes and ROW and COL coordinates
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
    ROW = gf['ROW'].values
    COL = gf['COL'].values
    dx = np.diff(COL).mean() / 2
    dy = np.diff(ROW).mean() / 2
    EROW = np.append(ROW - dy, ROW[-1] + dy)
    ECOL = np.append(COL - dx, COL[-1] + dx)
    sx = ECOL
    sy = np.ones_like(sx) * EROW[0]
    ey = EROW
    ex = np.ones_like(ey) * ECOL[-1]
    nx = sx[::-1]
    ny = np.ones_like(nx) * EROW[-1]
    wy = ey[::-1]
    wx = np.ones_like(wy) * ECOL[0]
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
    import warnings

    if srcproj is None and clip:
        warnings.warn(
            'clip with srcproj=None is not implemented;'
            + 'unclipped shapes returned'
        )

    if srcproj is None:
        return srcshapes

    if clip:
        domain_raw = wholedomain(gf)
        # Transform with shape complexity to capture distortion
        # Then, create an envelope and buffer
        # This ensures that distortion does not lose data
        envelope = transform(
            Transformer.from_crs(
                gf.cmaq.proj4, srcproj, always_xy=True
            ).transform,
            domain_raw
        ).envelope
        bufflen = 1
        domain = envelope.buffer(bufflen).envelope
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
    gf, shapes, srcproj=None, clip=True, propname='area', dataframe=False,
    verbose=0
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
    clip : bool
        Passed to togrid if srcproj is not None
    propname : str
        Property to add. Usually area, but length can be useful.
    dataframe : bool
        If True, output a sparse weighting pd.DataFrame

    Returns
    -------
    out : xr.DataArray
        Sum of propname (usually area) from shapes in each grid cell
    """
    import xarray as xr
    import pandas as pd
    from shapely.prepared import prep
    from shapely.strtree import STRtree

    tree = STRtree(cellpolygons(gf).ravel())
    if dataframe:
        afrac = []
    else:
        afrac = np.zeros((gf.NROWS, gf.NCOLS), dtype='f')
    nl = len(shapes)
    if srcproj is None:
        gshapes = shapes
    else:
        gshapes = togrid(gf, shapes, srcproj=srcproj, clip=clip)
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

            if dataframe:
                afrac.append((j + 0.5, i + 0.5, li, intxarea))
            else:
                afrac[j, i] += intxarea

    if verbose > 0:
        print(f'\r{nl:3d} 100.0% {ng:-7d} 100.0%', flush=True)

    if dataframe:
        out = pd.DataFrame(afrac, columns=['ROW', 'COL', 'shapeidx', 'weight'])
        out.set_index(list(out.columns[:-1]), inplace=True)
    else:
        out = xr.DataArray(
            afrac, dims=('ROW', 'COL'),
            coords=dict(
                ROW=np.arange(gf.NROWS) + 0.5,
                COL=np.arange(gf.NCOLS) + 0.5
            )
        )
    return out


def shp_weights_centroid(
    gf, shapes, srcproj=None, clip=True, verbose=0
):
    """

    Arguments
    ---------

    Returns
    -------
    """
    from shapely.strtree import STRtree
    from shapely.prepared import prep

    tree = STRtree(cellcenters(gf).ravel())

    if srcproj is None:
        gshapes = shapes
    else:
        gshapes = togrid(gf, shapes, srcproj=srcproj, clip=clip)

    nshapes = len(gshapes)
    for shpi, poly in enumerate(gshapes):
        if verbose > 0:
            print(f'\r{shpi/nshapes:.1%}', end='')
        shppts = tree.query(poly)
        npts = len(shppts)
        if npts > 20:
            ppoly = prep(poly)
            intersects = ppoly.intersects
        else:
            intersects = poly.intersects

        for p in shppts:
            if intersects(p):
                yield (p.y, p.x, shpi, 1)

    if verbose > 0:
        print()


def attr_from_shapefile_areafractions(
    gf, shapepath, key, srcproj=None, clip=True, verbose=0
):
    """
    This function returns the area fractions for each unique value of key in
    the shapefile at shapepath

    See attr_from_shapefile for description of keywords
    """

    import xarray as xr
    from shapely.geometry import shape
    from collections import OrderedDict

    shpf = _openshapefile(shapepath)
    attrdf = _getattrdf(shpf)

    vals = OrderedDict()
    groups = attrdf.groupby([key])
    ngroups = len(groups)
    for gi, (val, rows) in enumerate(groups):
        if verbose > 0:
            print(f'\r{gi/ngroups:.1%}', end='')
        shapes = [shape(shpf.shape(i)) for i in rows.index]
        gfrac = gf.cmaq.gridfraction(shapes=shapes, srcproj=srcproj, clip=clip)

        valkey = val
        # Adding TSTEP and LAY dimensions
        vals[valkey] = gfrac.expand_dims(TSTEP=1, LAY=1)

    if verbose > 0:
        print()

    valkeys = sorted(vals)
    valfracs = xr.concat([vals[valkey] for valkey in valkeys], dim='TSTEP')
    return valkeys, valfracs


def attr_from_shapefile_largestareafraction(
    gf, shapepath, key, srcproj=None, clip=True, verbose=0
):
    """
    Returns the attribute value that has the largest area fraction
    covering each cell.

    See attr_from_shapefile for description of keywords
    """
    import xarray as xr

    valkeys, valfracs = attr_from_shapefile_areafractions(
        gf, shapepath, key, srcproj=srcproj, clip=clip, verbose=verbose
    )
    outvals = xr.DataArray([
        valkey for valkey in valkeys
    ], dims=('TSTEP',))[valfracs.argmax('TSTEP')].where(
        valfracs.max('TSTEP') > 0
    )
    return outvals.expand_dims(TSTEP=1)


def attr_from_shapefile_areaweighted(
    gf, shapepath, key, srcproj=None, clip=True, verbose=0
):
    """
    Returns the area weighted attribute value from the area fractions
    of each unique value of key covering each cell.

    See attr_from_shapefile for description of keywords
    """
    import xarray as xr

    valkeys, valfracs = attr_from_shapefile_areafractions(
        gf, shapepath, key, srcproj=srcproj, clip=clip, verbose=verbose
    )
    outvals = (
        xr.DataArray(valkeys, dims=('TSTEP',)) * valfracs
    ).sum('TSTEP') / valfracs.sum('TSTEP')
    return outvals.expand_dims(TSTEP=1)


def attr_from_shapefile_centroid(
    gf, shapepath, key, srcproj=None, clip=True, verbose=0
):
    """
    Returns the attribute value of the shape that the grid cell centroid
    intersects.

    See attr_from_shapefile for description of keywords
    """
    import xarray as xr
    import numpy as np
    from shapely.geometry import shape

    shpf = _openshapefile(shapepath)
    attrdf = _getattrdf(shpf)
    out = xr.DataArray(
        np.ma.masked_all((1, 1, gf.NROWS, gf.NCOLS), dtype='f'),
        dims=('TSTEP', 'LAY', 'ROW', 'COL'),
        coords=gf.coords
    )
    shapes = [shape(s) for s in shpf.iterShapes()]
    wgtiter = shp_weights_centroid(
        gf, shapes, srcproj=srcproj, clip=clip, verbose=verbose
    )
    for row, col, shapei, wgt in wgtiter:
        out.sel(ROW=row, COL=col)[:] = attrdf.loc[shapei, key]

    return out


def attr_from_shapefile(
    gf, shapepath, key, method='laf', srcproj=None, clip=True, units='unknown',
    verbose=0
):
    """
    Calculate timezone offset based on largest area overlap of time zone
    polygons from a shapefile. The offset is returned as a variable.

    Arguments
    ---------
    gf : xr.Dataset
        Must support the cmaq accessor function gridfraction
    shapepath : str
        Path to shapefile with timezone as a UTC offset
    key : str
        Attribute field name for UTC offset
    method : str
        Method for selecting shape for each grid cell. Currently supports
        laf, areaweigthed, centroid
    srcproj : scalar
        Projection definition in terms that pyproj can interpret
    units : str
        Unit for output variable
    verbose : int
        verbosity level.

    Returns
    -------
    tzvar : xr.DataArray
        Array (TSTEP, LAY, ROW, COL) with UTC offsets as 32-bit floats
    """
    import os
    try:
        shapename = os.path.basename(shapepath)
    except Exception:
        shapename = 'unknown'

    attfunc = {
        'laf': attr_from_shapefile_largestareafraction,
        'largestareafraction': attr_from_shapefile_largestareafraction,
        'areaweighted': attr_from_shapefile_areaweighted,
        'centroid': attr_from_shapefile_centroid,
    }[method]
    outvar = attfunc(
        gf, shapepath, key, srcproj=srcproj, clip=clip, verbose=verbose
    )
    outvar.attrs.update(dict(
        units=units.ljust(16),
        long_name=key.ljust(16),
        var_desc=f'{key} using {method} from {shapename}'.ljust(80)[:80],
    ))
    return outvar


def _openshapefile(shapepath):
    import shapefile
    if isinstance(shapepath, shapefile.Reader):
        return shapepath
    else:
        return shapefile.Reader(shapepath)


def _getattrdf(shpf):
    import pandas as pd
    attrdf = pd.DataFrame.from_records(
        shpf.iterRecords(),
        columns=[v[0] for v in shpf.fields[1:]]
    )
    return attrdf
