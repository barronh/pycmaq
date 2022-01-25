__all__ = ['default_griddesc', 'open_default_griddesc']

# The default domains are arbitrarily based on what I commonly use.
# The open_griddesc method supplies easy access to your own domains,
# so these values are for convenience only.
default_griddesc = """' '
'LamCon_40N_97W'
 2         33.000        45.000       -97.000      -97.000         40.000
'LamCon_63N_155W'
 2         60.000        70.000      -155.000      -155.000        63.000
'LamCon_21N_157W'
 2         19.000        22.000      -157.000      -157.000        21.000
'LamCon_18N_66W'
 2         17.000        19.000       -66.000       -66.000        18.000
'POLSTE_HEMI'
 6          1.000        45.000       -98.000       -98.000        90.000
'LATLON'
 1          0.000         0.000         0.000         0.000         0.000
'LAM_34N110E'
 2         25.000        40.000       110.000       110.000        34.000
'LamCon_40N_90W'
 2         30.000        60.000       -90.000      -90.000         40.000
'COLOMBIA'
  7         0.000         0.000       -98.000       -98.000         0.000
' '
'Pt1dUS1'
'LATLON'               -140.0        20.0      0.1      0.1  900  400 1
'global_1'
'LATLON'               -180.0       -90.0      1.0      1.0  360  180 1
'global_0pt1'
'LATLON'               -180.0       -90.0      0.1      0.1 3600 1800 1
'global_2x2.5'
'LATLON'               -181.5       -89.0      2.5      2.0  144   89 1
'global_4x5'
'LATLON'               -182.5       -88.0      5.0      4.0   72   44 1
'108NHEMI2'
'POLSTE_HEMI'     -10098000.0 -10098000.0 108000.0 108000.0  187  187 1
'HEMIS'
'POLSTE_HEMI'     -10098000.0 -10098000.0 108000.0 108000.0  187  187 1
'12US1'
'LamCon_40N_97W'   -2556000.0  -1728000.0  12000.0  12000.0  459  299 1
'12US2'
'LamCon_40N_97W'   -2412000.0  -1620000.0  12000.0  12000.0  396  246 1
'36US3'
'LamCon_40N_97W'   -2952000.0  -2772000.0  36000.0  36000.0  172  148 1
'108US1'
'LamCon_40N_97W'   -2952000.0  -2772000.0 108000.0  08000.0   60   50 1
'4LISTOS1'
'LamCon_40N_97W'    1140000.0   -456000.0   4000.0   4000.0  300  315 1
'27AK1'
'LamCon_63N_155W'  -1971000.0  -1701000.0  27000.0  27000.0  146  126 1
'9AK1'
'LamCon_63N_155W'  -1107000.0  -1134000.0   9000.0   9000.0  312  252 1
'27HI1'
'LamCon_21N_157W'  -1012500.0  -1012500.0  27000.0  27000.0   75   75 1
'3HI1'
'LamCon_21N_157W'   -391500.0   -346500.0   3000.0   3000.0  225  201 1
'9HI1'
'LamCon_21N_157W'   -517500.0   -490500.0   9000.0   9000.0  100  100 1
'27PR1'
'LamCon_18N_66W'   -1012500.0  -1012500.0  27000.0  27000.0   75   75 1
'3PR1'
'LamCon_18N_66W'    -274500.0   -202500.0   3000.0   3000.0  150  150 1
'9PR1'
'LamCon_18N_66W'    -517500.0   -436500.0   9000.0   9000.0  100  100 1
'M_27_08CHINA'
'LAM_34N110E'      -3132000.0  -2457000.0  27000.0  27000.0  232  182 1
'FourCornerX'
'LamCon_40N_90W'   -2831931.0   -730076.0 600000.0 600000.0    8    3 1
'CMAQNORTHSA'
'COLOMBIA'          251759.25  -1578187.0  27000.0  27000.0  179  154 1
'CMAQCOLOMBIA'
'COLOMBIA'         2123759.00  158812.062   9000.0   9000.0   94   76 1
'CMAQCUNDINAMARCA'
'COLOMBIA'         2522759.50  413813.469   3000.0   3000.0   82   67 1
'CMAQBOGOTA'
'COLOMBIA'         2622759.25  483812.062   1000.0   1000.0   64   64 1
' '"""


def open_default_griddesc(GDNAM, **kwds):
    if kwds.get('help', False):
        print('Choose domain from')
        print(default_griddesc)
        return
    from . import open_griddesc
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w') as gdf:
        gdpath = gdf.name
        gdf.write(default_griddesc)
        gdf.flush()
        return open_griddesc(gdpath, GDNAM=GDNAM, **kwds)
