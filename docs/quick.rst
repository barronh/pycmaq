.. Quick

Quick Start
-----------

Fpir simple examples are shown below. The first plots a time subset
for layer one. The second calculates daily average PM25 and MDA8 ozone
in local solar time as shown below.

Example 1
~~~~~~~~~

.. code-block:: python

  import pycmaq as pq

  aconcpath = 'ACONC_20160101'
  qf = pq.open_dataset(aconcpath)

  qf.O3.sel(TSTEP=slice('2016-01-01 17', '2016-01-01 23')).mean('TSTEP').isel(LAY=0).plot()
  qf.cmaq.cnostates()


Example 2
~~~~~~~~~

.. code-block:: python

  import pycmaq as pq

  g2path = 'GRIDCRO2D_20160101'
  g2f = pq.open_dataset(g2path)

  # Estimate the timezone (as integer UTC offset),
  # removing the TSTEP and LAY dimensions for broadcasting
  tzone = (g2f.LON.isel(TSTEP=0, LAY=0) / 15).round(0).astype('i')

  combinepath = 'COMBINE_201601'
  qf = pq.open_dataset(combinepath)

  # Subset species
  qsf = qf[['O3', 'PM25_FRM']]

  # Speed up by loading to memory
  qsf.O3.load()
  qsf.PM25_FRM.load()

  # Convert to LST
  qlstf = qsf.cmaq.to_lst(tzone)

  # For PM25, Group by Daily resolution and average
  qa24f = qlstf[['PM25_FRM']].groupby(
      qlstf.TSTEP.astype('datetime64[D]')
  ).mean('TSTEP')

  # For O3, calculate the MDA8 with 17 valid hours
  # 1) apply an 8h moving average
  # 2) mask out hours greater than 17
  # 3) group by Daily resolution
  # 4) take the max
  #
  # This takes time.
  qmda8f = qlstf[['O3']].rolling(
      TSTEP=8, min_periods=6
  ).mean('TSTEP').where(
      qlstf['TSTEP'].dt.hour < 17
  ).groupby(
      qlstf.TSTEP.astype('datetime64[D]')
  ).max('TSTEP')


Example 3
~~~~~~~~~~

Assuming that you have completed Example 2, you can do a quick evaluation
for MDA8 ozone.


.. code-block:: python

  aqsdailypath = 'daily_44201_2016.zip'

  rawaqs = pq.pd.read_csv(aqsdailypath).query('`Event Type` in ("None", "Included")')
  rawaqs['TSTEP'] = pq.pd.to_datetime(
      rawaqs['Date Local'] + 'T' + rawaqs['1st Max Hour'].astype(str).str.rjust(2, '0')
  )
  aqs = rawaqs.query(
          'TSTEP >= "2016-01-01" and TSTEP < "2016-02-01" and `1st Max Value` > 0'
  ).copy()
  aqs['COL'], aqs['ROW'] = qf.cmaq.pyproj(aqs['Longitude'].values, aqs['Latitude'].values)
  aqs['COL'] = aqs['COL'].round(0) + 0.5
  aqs['ROW'] = aqs['ROW'].round(0) + 0.5
  aqs['LAY'] = 0

  qaqs = pq.xr.Dataset(aqs)
  ataqs = qmda8f.interp(
      TSTEP=qaqs['TSTEP'], ROW=qaqs['ROW'], COL=qaqs['COL']
  ).isel(LAY=0)

  evaldf = aqs.join(ataqs.to_dataframe(), rsuffix='cmaq')

  r = evaldf[['1st Max Value', 'O3']].corr().loc['O3', '1st Max Value']

  fig, ax = pq.plt.subplots(1, 1)
  ax.hexbin(evaldf['1st Max Value'] * 1000, evaldf['O3'], mincnt=1)
  ax.set(
      xlabel='AQS ppb', ylabel='CMAQ ppb',
      title=f'Jan 17h MDA8 Ozone (r={r:.02f})',
      xlim=(0, None), ylim=(0, None),
  )
  pq.plt.colorbar(ax.collections[0], label='count')
  x = pq.np.array(ax.get_ylim())
  ax.set_aspect('equal', 'box')
  ax.plot(x, x, 'k-')
  ax.plot(x / 2, x, 'k--')
  ax.plot(x, x / 2, 'k--')


Example 4
~~~~~~~~~

Created gridded fractional area from a Shapefile. In this case, we'll use
the Natural Earlth 110m Countries shapefile (at https://www.naturalearthdata.com/downloads/110m-cultural-vectors/).


.. code-block:: python

  import pycmaq as pq


  shppath = 'ne_110m_admin_0_countries.shp'

  gf = pq.open_griddesc(None, GDNAM='1188NHEMI2')
  valkeys, valfracs = pycmaq.utils.shapes.attr_from_shapefile_areafractions(
      gf, shppath, 'REGION_UN', srcproj=4326, clip=True, verbose=1
  )
  for i, valkey in enumerate(valkeys):
      gf[valkey] = valfracs.sel(TSTEP=i).expand_dims(TSTEP=1)

  # Optionally plot the Americas
  # gf['Americas'][0, 0].plot.pcolormesh()
  # gf.cmaq.cnocountries()

  gf[valkeys].cmaq.to_ioapi('REGION_UN.nc')

## Notes

* Large CMAQ files in NETCDF3_CLASSIC are more efficiently read with 
  the NC_SHARE option. This is the default with `pycmaq.open_dataset`,
  but not with `pycmaq.xr.open_dataset`. The `xr` module is just a
  pointer to the `xarray`.
* LON-based timezone is solar time, not standard time used by
  observations. Replace the LON-based timezone with one based on
  timezones to make more consistent with observations.
  Local Standard Time.
