# -*- coding: utf-8 -*-
"""
@author: Jason Pruitt
"""

import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
import sys

## Takes data from file and returns as vars
def read_data(filename):
    data = np.genfromtxt(fname = filename, delimiter = ',', \
                         dtype = 'unicode',skip_header = 1)
    refTemp = np.genfromtxt(fname = filename, delimiter = ',', \
                            dtype = 'unicode',skip_footer = 1)
    data = data.astype(np.float)
    bData = data[0::2]
    vData = data[1::2]

    return bData,vData,refTemp

## Corrects for offset with zero field measurements
def correct_offset(bData,vData):
    bCorr = []
    vCorr = []

    for index,val in enumerate(zip(bData,vData)):
        if val[0] != 0.000:
            correction = (val[1]) - (vData[index+1]) + (vData[index-1])/2
            vCorr.append(correction)
            bCorr.append(val[0])

    # print(bCorr)
    # print(vCorr)
    # print(len(vCorr))
    return bCorr,vCorr

## Subtracts out the magneto-resistance
def mag_res(bData,vData):
    magrV = []
    posB = []
    posV = []
    negV = []

    for b,v in zip(bData,vData):
        if b > 0:
            posB.append(b)
            posV.append(v)
        elif b < 0:
            negV.append(v)

    for ii in range(0,len(negV)):
        vCorr = (posV[ii] - negV[ii]) / 2
        magrV.append(vCorr)

    return posB,magrV

def plot_data(x1,y1,x2,y2,x3,y3,temp):
    x1 = np.array(x1)
    y2 = np.array(y2)

    x2col = np.array(x2).reshape((-1,1))
    x3col = np.array(x3).reshape((-1,1))

    model1 = np.polyfit(x1,y1,1)
    p1 = np.poly1d(model1)

    xinterp1 = np.linspace(np.sort(x1)[0],np.sort(x1)[-1],100)
    xinterp2 = np.linspace(np.sort(x2)[0],np.sort(x2)[-1],100)
    xinterp3 = np.linspace(np.sort(x3)[0],np.sort(x3)[-1],100)

    slope1,int1,r_value1,p_value1,std_err1 = stats.linregress(x1,y1)

    model2 = LinearRegression(fit_intercept = False).fit(x2col,y2)
    slope2 = model2.coef_
    slope2 = float(slope2)
    int2 = model2.intercept_
    r_value2 = model2.score(x2col,y2)

    model3 = LinearRegression(fit_intercept = False).fit(x3col,y3)
    slope3 = model3.coef_
    slope3 = float(slope3)
    int3 = model3.intercept_
    r_value3 = model3.score(x3col,y3)

    p2 = np.poly1d(np.array([model2.coef_,int2]))
    p3 = np.poly1d(np.array([model3.coef_,int3]))

    # plt.figure(figsize = (12,9))
    # plt.subplot(1,3,1)
    # plt.plot(x1,y1,'b.',markersize = 14)
    # plt.plot(xinterp1,p1(xinterp1),'--')
    # plt.title('Raw Data',fontsize = 16)
    # plt.xlabel('B-Field[T]', fontsize = 16)
    # plt.ylabel('Voltage[V]', fontsize = 16)
    # plt.grid()
    # # plt.show()
    #
    # # plt.figure(figsize = (12,9))
    # plt.subplot(1,3,2)
    # plt.plot(x2,y2,'b.',markersize = 14)
    # plt.plot(xinterp2,p2(xinterp2),'--')
    # plt.title('Corrected',fontsize = 16)
    # plt.xlabel('B-Field[T]', fontsize = 16)
    # plt.ylabel('Voltage[V]', fontsize = 16)
    # plt.grid()
    # # plt.show()
    #
    # # plt.figure(figsize = (12,9))
    # plt.subplot(1,3,3)
    # plt.plot(x3,y3,'b.',markersize = 14)
    # plt.plot(xinterp3,p3(xinterp3),'--')
    # plt.title('Magneto-Resistance',fontsize = 16)
    # plt.xlabel('Abs-Val B-Field[T]', fontsize = 16)
    # plt.ylabel('Voltage[V]', fontsize = 16)
    # plt.grid()
    # plt.show()

    return slope1,int1,r_value1,slope2,int2,r_value2,slope3,int3,r_value3

def calc(slope):
    l = 1.01
    hallCoeff = slope*l
    hallcm = hallCoeff*1e6 # cm^3 / C
    electronC = -1.60217662e-19 # C
    conc = (1/(hallcm*electronC)) # cm^-3
    # print(float(hallCoeff),float(conc))
    return hallcm,conc

def show_data(hallcm,conc,hc_offset,conc_offset,slope1,int1,r_value1,slope2,
              int2,r_value2,slope3,int3,r_value3,refTemp):

    print('\n\tReference Temperature:',refTemp)
    print('\nHall Coefficient:\t\t   ',hallcm,'[cm^3/C]')
    print('Carrier Concentration:\t\t   ',conc,'[cm^-3]')
    print('Hall Coeff: (Adj for Offset):\t',hallcm,'[cm^3/C]')
    print('Carrier Conc: (Adj for Offset):\t',conc,'[cm^-3]')

    print('\n\t\t Raw Data')
    print('\nslope1:','\t\t\t','  ',slope1)
    print('r-squared1:','\t\t\t','   ', r_value1**2)
    # print('Uncertainty in slopeCu:','\t','   ', std_err1)
    print('\n\t\t Corrected')
    print('slope2:','\t\t\t','  ',float(slope2))
    print('r-squared1:','\t\t\t','   ', r_value2**2)
    # print('Uncertainty in slopeCu:','\t','   ', std_err2)
    print('\n\t\t Magneto-Resistance')
    print('\nslope3:','\t\t\t','  ',float(slope3))
    print('r-squared3:','\t\t\t','   ',r_value3**2)
    print('intercept:','\t\t\t','   ', int3)

def grab_and_plot(path):
    temp = []
    hc = []
    hc_offset = []
    conc = []
    conc_offset = []
    rawslope = []
    magResSlope = []
    magResOffsetSlope = []

    for filename in glob.glob(path):
        with open(filename,'r') as method:
            header = method.readline()
            info = header.split(',')
            line = method.readline()
            while line:
                data = line.split(',')
                temp.append(float(data[0]))
                hc.append(float(data[1]))
                hc_offset.append(float(data[2]))
                conc.append(float(data[3]))
                conc_offset.append(float(data[4]))
                rawslope.append(float(data[5]))
                magResSlope.append(float(data[6]))
                magResOffsetSlope.append(float(data[7]))

                line = method.readline()
        ## Plot the data
    plt.figure(figsize = (12,9))
    plt.subplot(1,2,1)
    plt.plot(temp,np.abs(hc),'k.',markersize = 14)
    plt.title('Hall Coefficient vs Temperature',fontsize = 16)
    plt.xlabel('Temperature Range',fontsize = 14)
    plt.ylabel('Hall Coefficient',fontsize = 14)
    plt.grid()

    plt.subplot(1,2,2)
    # plt.figure(figsize = (12,9))
    plt.plot(temp,conc,'k.',markersize = 14)
    plt.title('Carrier Concentration vs Temperature')
    plt.xlabel('Temperature Range',fontsize = 14)
    plt.ylabel('Carrier Concentration',fontsize = 14)
    plt.grid()
    plt.savefig(os.path.join('images','NISTHall.png'),dpi=300)
    plt.show()

    return os.path.join('images','NISTHall.png',dpi=300)

def main(args):
    print('Hall Data')
    # hallDates = os.path.join('HallData','NIST_Run1','*')
    # # For a given day...
    # for day in glob.glob(hallDates):
    #     temps = os.path.join(day,'*')
    #     # ... at a given temp...
    #     for temp in glob.glob(temps):
    #         namedTemp = temp[-4:]
    #         data = os.path.join(temp,'*')
    #         # ... there is a measurement
    #         for meas in glob.glob(data):
    #             # print(meas)
    #             raw_B,raw_V,refTemp = read_data(meas)
    #             bCorr,vCorr = correct_offset(raw_B,raw_V)
    #             posB1,mrV1 = mag_res(raw_B,raw_V)
    #             posB2,mrV2 = mag_res(bCorr,vCorr)
    #
    #             slope1,int1,r_value1,slope2,int2,r_value2,slope3,int3,r_value3 \
    #             = plot_data(raw_B,raw_V,posB1,mrV1,posB2,mrV2,refTemp)
    #
    #
    #             hallCoeff,conc = calc(slope2)
    #             hc_offset,conc_offset = calc(slope3)
    #             show_data(hallCoeff,conc,hc_offset,conc_offset,slope1,int1,
    #                       r_value1,slope2,int2,r_value2,slope3,int3,r_value3,
    #                       refTemp)
    #
    #             header = ['Temp','HallCoeff','HallCoeffOffset,',
    #             'CarrierConc','CarrierConcOffset','RawSlope','MagResSlope',
    #             'MagResOffsetSlope']
    #             row = [refTemp,hallCoeff,hc_offset,conc,conc_offset,slope1,
    #                    slope2,slope3]
    #
    #             with open(f'HallNIST_{namedTemp}.csv','w+',newline = '') as csvfile:
    #                 writer = csv.writer(csvfile)
    #                 writer.writerow(header)
    #                 writer.writerow(row)
    #
    # currDir = os.getcwd()
    # currDirCSV = os.path.join(str(currDir),'HallNIST*.csv')
    # grab_and_plot(currDirCSV)

if __name__ == '__main__':
    exit(main(sys.argv))
