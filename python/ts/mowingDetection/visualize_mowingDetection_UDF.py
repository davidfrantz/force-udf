import enmapbox.qgispluginsupport.qps.pyqtgraph.pyqtgraph as pg
from enmapbox.qgispluginsupport.qps.plotstyling.plotstyling import PlotStyle, MarkerSymbol
from profileanalyticsapp.profileanalyticsdockwidget import Profile
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from enmapboxprocessing.utils import Utils
from FORCE_UFD_mowingDetection import *
import numpy as np
from typing import List


def updatePlot(profile: Profile, profiles: List[Profile], plotWidget: pg.PlotItem):

    # get x (decimal dates) and y (vegetation index values from QGIS)
    xValues = np.array(profile.xValues)
    yValues = np.array(profile.yValues)

    # default values for running mowingDetection_UDF.py
    bandnames = forcepy_init(None, None, None)
    sensors = None
    dates = xValues
    inarray = yValues
    nodata = -9999
    nproc = 1
    outarray = np.ones(len(bandnames))
    mowingEvents, mowing_doy_out, xPeak, yPeak, xPol, yPol = forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc)

    # plot indentified verticies (Start, end and peak values)
    style = PlotStyle()
    style.setMarkerSymbol(MarkerSymbol.Cross)
    style.markerBrush.setColor(QColor('#ff0000'))
    style.markerSize = 15
    plotDataItem = plotWidget.plot(xPeak, [i * 10000 for i in yPeak], name='Vertices')
    style.apply(plotDataItem)

    # plot the interpolated "convex hull"
    style = PlotStyle()
    style.setMarkerSymbol(MarkerSymbol.No_Symbol)  # options: Circle, Triangle_Down, Triangle_Up, Triangle_Right, Triangle_Left, Pentagon, Hexagon, Square, Star, Plus, Diamond, Cross, ArrowUp, ArrowRight, ArrowDown, ArrowLeft, No_Symbol
    style.linePen.setColor(QColor('#0000ff'))
    style.linePen.setWidth(2)
    style.linePen.setStyle(Qt.SolidLine)
    plotDataItem = plotWidget.plot(xPol, [i * 10000 for i in yPol], name='Interpolation')
    style.apply(plotDataItem)

    # plot identified mowing events
    style = PlotStyle()
    style.setMarkerSymbol(MarkerSymbol.No_Symbol)
    style.linePen.setColor(QColor('#00ff00'))
    style.linePen.setWidth(2)
    style.linePen.setStyle(Qt.DashLine)

    dateTimes = [Utils.parseDateTime(mowingEvent) for mowingEvent in mowingEvents]
    dyears = [Utils.dateTimeToDecimalYear(dateTime) for dateTime in dateTimes]

    for i, dyear in enumerate(dyears,1):
        plotDataItem = plotWidget.plot(x=[dyear, dyear], y=[0,10000], name=f'Mowing event {i}')
        style.apply(plotDataItem)

    # print dates in pyhton console
    print(dyears)
