# -*- coding: utf-8 -*-
"""
@author: Jason Pruitt
"""

import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import stats
import sys

## Reads and parses file, gives back all data from file for later use as vars
def read_data(filename):
    Voltage = np.genfromtxt(fname = filename, delimiter = ',',dtype =
    'unicode', skip_header = 1)
    Voltage = Voltage.astype(np.float)

    Ref_TempK = np.genfromtxt(fname = filename, delimiter = ',', dtype =
    'unicode', skip_footer = 1)
    Ref_Temp = float(Ref_TempK) - 273.15

    TempVoltage_Hot = Voltage[0::4]
    TempVoltage_Cold = Voltage[1::4]
    Cu_Voltage = Voltage[2::4]
    Const_Voltage = Voltage[3::4]

    if Ref_Temp < 0:
        Vor = 0 + 3.8748106364e1*Ref_Temp + 4.4194434347e-2*Ref_Temp**2 + \
        1.1844323105e-4*Ref_Temp**3 + 2.0032973554e-5*Ref_Temp**4 + \
        9.0138019559e-7*Ref_Temp**5 + 2.2651156593e-8*Ref_Temp**6 + \
        3.6071154205e-10*Ref_Temp**7 + 3.8493939883e-12*Ref_Temp**8 + \
        2.8213521925e-14*Ref_Temp**9 + 1.4251594779e-16*Ref_Temp**10 + \
        4.8768662286e-19*Ref_Temp**11 + 1.0795539270e-21*Ref_Temp**12 + \
        1.3945027062e-24*Ref_Temp**13 + 7.9795153927e-28*Ref_Temp**14
    else:
        Vor = 0 + 3.8748106364e1*Ref_Temp + 3.3292227880e-2*Ref_Temp**2 + \
        2.0618243404e-4*Ref_Temp**3 + -2.1882256846e-6*Ref_Temp**4 + \
        1.0996880928e-8*Ref_Temp**5 + -3.0815758772e-11*Ref_Temp**6 + \
        4.5479135290e-14*Ref_Temp**7 + -2.7512901673e-17*Ref_Temp**8

    Voj_Hot = Vor + TempVoltage_Hot*(1e6)
    Voj_Cold = Vor + TempVoltage_Cold*(1e6)

    TC_Hot = []
    TC_Cold = []
    for ii in range(0,len(Voj_Hot)):
        if Ref_Temp < 0:
            Temp_Hot = 0 + 2.5949192e-2*Voj_Hot[ii] +\
            -2.1316967e-7*Voj_Hot[ii]**2 + 7.9018692e-10*Voj_Hot[ii]**3 +\
            4.2527777e-13*Voj_Hot[ii]**4 + 1.3304473e-16*Voj_Hot[ii]**5 + \
            2.0241446e-20*Voj_Hot[ii]**6 + 1.2668171e-24*Voj_Hot[ii]**7
            Temp_Cold = 0 + 2.5949192e-2*Voj_Cold[ii] + \
            -2.1316967e-7*Voj_Cold[ii]**2 + 7.9018692e-10*Voj_Cold[ii]**3 + \
            4.2527777e-13*Voj_Cold[ii]**4 + 1.3304473e-16*Voj_Cold[ii]**5 + \
            2.0241446e-20*Voj_Cold[ii]**6 + 1.2668171e-24*Voj_Cold[ii]**7

            Temp_Hot_K = Temp_Hot + 273.15
            Temp_Cold_K = Temp_Cold + 273.15
            TC_Hot.append(Temp_Hot_K)
            TC_Cold.append(Temp_Cold_K)
        else:
            Temp_Hot = 0 + 2.592800e-2*Voj_Hot[ii] +\
            -7.602961e-7*Voj_Hot[ii]**2 + 4.637791e-11*Voj_Hot[ii]**3 +\
            -2.165394e-15*Voj_Hot[ii]**4 + 6.048144e-20*Voj_Hot[ii]**5 +\
            -7.293422e-25*Voj_Hot[ii]**6
            Temp_Cold = 0 + 2.592800e-2*Voj_Cold[ii] +\
            -7.602961e-7*Voj_Cold[ii]**2 + 4.637791e-11*Voj_Cold[ii]**3 +\
            -2.165394e-15*Voj_Cold[ii]**4 + 6.048144e-20*Voj_Cold[ii]**5 +\
            -7.293422e-25*Voj_Cold[ii]**6

            Temp_Hot_K = Temp_Hot + 273.15
            Temp_Cold_K = Temp_Cold + 273.15
            TC_Hot.append(Temp_Hot_K)
            TC_Cold.append(Temp_Cold_K)

    TC_Hot = np.array(TC_Hot)
    TC_Cold = np.array(TC_Cold)

    DeltaT = TC_Hot - TC_Cold
    avgDeltaT = np.mean(DeltaT)

    Cu_Voltage = np.array(Cu_Voltage)
    avgCu = np.mean(Cu_Voltage)
    Const_Voltage = np.array(Const_Voltage)
    avgConst = np.mean(Const_Voltage)

    Ref_TempK = float(Ref_TempK)

    Seebeck_Cu = 0.041*Ref_TempK*(np.exp(-Ref_TempK/93) + 0.123 -\
                0.442/(1 + (Ref_TempK/172.4)**3) ) + 0.804

    Seebeck_CuConst = 4.37184 + 0.1676*(Ref_TempK) + \
    -1.84371e-4*(Ref_TempK**2) + 1.2244e-7*(Ref_TempK**3) +\
    -4.47618e-11*(Ref_TempK**4)

    Seebeck_Const = Seebeck_Cu - Seebeck_CuConst

    return (Ref_TempK,TC_Hot,TC_Cold,DeltaT,avgDeltaT,Cu_Voltage,Const_Voltage,
        avgCu,avgConst,Seebeck_Cu,Seebeck_CuConst,Seebeck_Const)

## Plotting functions are optional but only from command line arg, do not
## comment out to suppress since vars returned here are important for analysis
def plot_data_QS(choice,temp,DeltaT,Cu_Voltage,Const_Voltage,TC_Hot,TC_Cold):

    slopeCu,intercept1,r_value1,p_value1,std_err1 = stats.linregress(DeltaT,
                                                                    Cu_Voltage)
    slopeConst,intercept2,r_value2,p_value2,std_err2 = stats.linregress(DeltaT,
                                                                Const_Voltage)
    slope3,intercept3,r_value3,p_value3,std_err3 = stats.linregress(
                                                    Const_Voltage,Cu_Voltage)

    curve1 = np.polyfit(DeltaT,Cu_Voltage,1)
    curve2 = np.polyfit(DeltaT,Const_Voltage,1)
    curve3 = np.polyfit(Const_Voltage,Cu_Voltage,1)

    p1 = np.poly1d(curve1)
    p2 = np.poly1d(curve2)
    p3 = np.poly1d(curve3)

    xinterp1 = np.linspace(DeltaT[0],DeltaT[-1],100)
    xinterp2 = np.linspace(Const_Voltage[0],Const_Voltage[-1],100)

    if choice == 'y':
        plt.figure(1,figsize = (12,9))
        plt.subplot(1,2,1)
        plt.gcf().set_size_inches((12,9))
        plt.plot(TC_Hot,'r.')
        plt.plot(TC_Cold,'b.')
        plt.xlabel('Scan Count')
        plt.ylabel('Temperature [K]')
        plt.title('Temperature Gradient')
        plt.grid()

        plt.subplot(1,2,2)
        plt.plot(DeltaT,'r.')
        plt.ylabel('\u0394 Temperature')
        plt.grid()
        plt.show()

        plt.figure(2,figsize = (12,9))
        plt.subplot(1,2,1)
        plt.plot(DeltaT,Cu_Voltage,'r.')
        plt.plot(xinterp1,p1(xinterp1),'--', label = 'fit')
        plt.ylabel('Cu Voltage [V]')
        plt.xlabel('\u0394 Temperature')
        plt.title('Cu Voltage vs. \u0394 Temperature')
        plt.grid()

        plt.subplot(1,2,2)
        plt.plot(DeltaT,Const_Voltage,'b.')
        plt.plot(xinterp1,p2(xinterp1),'--', label = 'fit')
        plt.ylabel('Constantan Voltage [V]')
        plt.xlabel('\u0394 Temperature')
        plt.title('Constantan Voltage vs. \u0394 Temperature')
        plt.grid()
        plt.savefig(F'QuasiSteady{temp}K.png',dpi = 300)
        plt.show()

        plt.figure(3,figsize = (12,9))
        plt.subplot(1,1,1)
        plt.plot(Const_Voltage,Cu_Voltage,'k.')
        plt.plot(xinterp2,p3(xinterp2),'--',label = 'fit')
        plt.title('Copper vs Constantan Voltages')
        plt.xlabel('Constantan Voltage[$\mu$V]')
        plt.ylabel('Copper Voltage[$\mu$V]')
        plt.grid()
        plt.show()

    return(slopeCu,r_value1,std_err1,slopeConst,r_value2,std_err2,slope3,
        r_value3,std_err3)

## Because of the sequence of events, write_SS will be within show_dataSS
def write_SS(date,namedTemp,temp,copperSlope,uncertaintyCuSlope,constSlope,
uncertaintyConstSlope,copperSeebeck,constSeebeck,platwireCopper,platwireConst,
empiricalCheck):

    with open(os.path.join('SeebeckData','NIST_Run1_CSV',
            F'Steady_NIST_{namedTemp}K.csv','w+',newline='') as csvfile:
        fieldnames = ['Date','Temp','CopperSlope','uncertaintyCuSlope',
        'ConstSlope','uncertaintyConstSlope','SeebeckCu','SeebeckConst',
        'PlatWireCopper','PlatWireConst','Empirical Check']
        writer = csv.DictWriter(csvfile,fieldnames = fieldnames)

        writer.writeheader()
        writer.writerow({'Date':date,'Temp':temp,'CopperSlope':copperSlope,
        'uncertaintyCuSlope':uncertaintyCuSlope,'ConstSlope':constSlope,
        'uncertaintyConstSlope':uncertaintyConstSlope,'SeebeckCu':copperSeebeck,
        'SeebeckConst':constSeebeck,'PlatWireCopper':platwireCopper,
        'PlatWireConst':platwireConst,'Empirical Check':empiricalCheck})


def plot_data_SS(choice,temp,DeltaT,Cu_Voltage,Const_Voltage,TC_Hot,TC_Cold,
tData,cuData,coData,counterSS):

    slopeCu,intercept1,r_value1,p_value1,std_err1 = stats.linregress(tData,
                                                                        cuData)
    slopeConst,intercept2,r_value2,p_value2,std_err2 = stats.linregress(tData,
                                                                        coData)
    slope3,intercept3,r_value3,p_value3,std_err3 = stats.linregress(
                                                    Const_Voltage,Cu_Voltage)

    if counterSS >=4:
        curve1 = np.polyfit(tData,cuData,1)
        curve2 = np.polyfit(tData,coData,1)
        curve3 = np.polyfit(Const_Voltage,Cu_Voltage,1)

        p1 = np.poly1d(curve1)
        p2 = np.poly1d(curve2)
        p3 = np.poly1d(curve3)

        xinterp = np.linspace(np.sort(tData)[0],np.sort(tData)[-1],100)
        xinterp2 = np.linspace(np.sort(Const_Voltage)[0],
                    np.sort(Const_Voltage)[-1],100)

        if choice == 'y':
            plt.figure(1,figsize = (12,9))
            plt.subplot(1,2,1)
            plt.plot(tData,cuData,'b.',markersize = 15)
            plt.plot(xinterp,p1(xinterp),'--',label = 'fit')
            plt.grid()
            plt.title(F'Cu V vs deltaT at {temp}K')
            plt.xlabel('DeltaT')
            plt.ylabel('Avg Cu Voltages [$\mu$V]')

            plt.subplot(1,2,2)
            plt.plot(tData,coData,'r.',markersize = 15)
            plt.plot(xinterp,p2(xinterp),'--',label = 'fit')
            plt.grid()
            plt.title(F'Constantan V vs deltaT at {temp}K')
            plt.xlabel('DeltaT')
            plt.ylabel('Avg Const Voltages [$\mu$V]')
            plt.savefig(F'SteadyState{temp}K.png',dpi=300)
            plt.show()

            plt.figure(2,figsize = (12,9))
            plt.subplot(2,2,1)
            plt.plot(Cu_Voltage,'b.')
            plt.title('Cu Voltage vs Counts')
            plt.xlabel('Counts')
            plt.ylabel('Voltage')

            plt.subplot(2,2,2)
            plt.plot(Const_Voltage,'r.')
            plt.title('Const Voltage vs Counts')
            plt.xlabel('Counts')
            plt.ylabel('Voltage')

            plt.subplot(2,2,3)
            plt.plot(DeltaT,Cu_Voltage,'b.')
            plt.title('Cu Voltage vs DeltaT')
            plt.xlabel('DeltaT')
            plt.ylabel('Voltage')

            plt.subplot(2,2,4)
            plt.plot(DeltaT,Const_Voltage,'r.')
            plt.title('Const Voltage vs DeltaT')
            plt.xlabel('DeltaT[K]')
            plt.ylabel('Voltage')
            plt.show()

            plt.figure(3,figsize = (12,9))
            plt.subplot(1,1,1)
            plt.plot(Const_Voltage,Cu_Voltage,'k.')
            plt.plot(xinterp2,p3(xinterp2),'--',label = 'fit')
            plt.title('Copper vs Constantan Voltages')
            plt.xlabel('Constantan Voltage[$\mu$V]')
            plt.ylabel('Copper Voltage[$\mu$V]')
            plt.grid()
            plt.show()

    return(slopeCu,r_value1,std_err1,slopeConst,r_value2,std_err2,slope3,
        r_value3,std_err3)

## Displays the data for QS but also returns three vars important for later,
## best not to comment out
def show_dataQS(temp,copperSlope,rCopper,copperUncertainty,constSlope,
rConst,constUncertainty,cuconstSlope,rCuConst,cuconstUncertainty,
copperSeebeck,constSeebeck):

    platwireCopper = -copperSlope*10**6 + copperSeebeck
    platwireConst = -constSlope*10**6 + constSeebeck

    a = -2.2040e-1
    b = 3.9706e-3
    c = 7.2922e-6
    d = -1.0864e-9
    sA = -230.03
    A = 295

    empiricalCheck = sA + a*temp*(1-A/temp) + b*temp**2*(1-A/temp)**2 + \
    c*temp**3*(1-A/temp)**3 + d*temp**4*(1-A/temp)**4

    print(F'\n\tReference Temperature: {temp}K')
    print('\nslopeCu:','\t\t\t','   ',copperSlope)
    print('r-squared1:','\t\t\t','   ', rCopper**2)
    print('Uncertainty in slopeCu:','\t','   ', copperUncertainty)
    print('\nslopeConst:','\t\t\t','  ',constSlope)
    print('r-squared2:','\t\t\t','   ', rConst**2)
    print('Uncertainty in slopeConst:','\t','   ',constUncertainty)
    print('\nSlope of Cu vs Const:','\t\t','  ',cuconstSlope)
    print('r-squared3:','\t\t\t','   ', rCuConst**2)
    print('Uncertainty in slope:','\t\t','   ', cuconstUncertainty)
    print('\nSeebeck Coefficient for Copper:\t','   ', copperSeebeck)
    print('Seebeck Coefficient for Constantan:',constSeebeck)
    print('\nSeebeck of Platinum wire for Cu:','  ',platwireCopper)
    print('Seebeck of Platinum wire for Const:',platwireConst)
    print('Empirical Check:\t\t',' ','',empiricalCheck,'\n')

    return(platwireCopper,platwireConst,empiricalCheck)


def show_dataSS(temp,copperSlope,rCopper,copperUncertainty,constSlope,
rConst,constUncertainty,cuconstSlope,rCuConst,cuconstUncertainty,
copperSeebeck,constSeebeck,counterSS,date,namedTemp):
    platwireCopper = -copperSlope*10**6 + copperSeebeck
    platwireConst = -constSlope*10**6 + constSeebeck

    a = -2.2040e-1
    b = 3.9706e-3
    c = 7.2922e-6
    d = -1.0864e-9
    sA = -230.03
    A = 295

    empiricalCheck = sA + a*temp*(1-A/temp) + b*temp**2*(1-A/temp)**2 + \
    c*temp**3*(1-A/temp)**3 + d*temp**4*(1-A/temp)**4

    if counterSS >=4:
        print(F'\n\tReference Temperature: {temp}K')
        print('\nslopeCu:','\t\t\t','  ',copperSlope)
        print('r-squared1:','\t\t\t','   ', rCopper**2)
        print('Uncertainty in slopeCu:','\t','   ', copperUncertainty)
        print('\nslopeConst:','\t\t\t','  ',constSlope)
        print('r-squared2:','\t\t\t','   ', rConst**2)
        print('Uncertainty in slopeConst:','\t','   ',constUncertainty)
        print('\nSlope of Cu vs Const:','\t\t','   ',cuconstSlope)
        print('r-squared3:','\t\t\t','   ', rCuConst**2)
        print('Uncertainty in slope:','\t\t','   ', cuconstUncertainty)
        print('\nSeebeck Coefficient for Copper:\t','   ', copperSeebeck)
        print('Seebeck Coefficient for Const:','\t',' ',constSeebeck)
        print('\nSeebeck of Platinum wire for Cu:','   ',platwireCopper)
        print('Seebeck of Platinum wire for Const:','',platwireConst)
        print('Empirical Check:\t\t',' ',' ',empiricalCheck,'\n')

    write_SS(date,namedTemp,temp,copperSlope,copperUncertainty,constSlope,
    constUncertainty,copperSeebeck,copperSeebeck,platwireCopper,platwireConst,
    empiricalCheck)

    return(platwireCopper,platwireConst,empiricalCheck)


## Uses csv.DictWriter to write header, then another function to write without
## (I have since found a better way to write to csv with a header but have
## left this way in the code)
def writeheader(date,namedTemp,temp,copperSlope,uncertaintyCuSlope,constSlope,
uncertaintyConstSlope,copperSeebeck,constSeebeck,platwireCopper,platwireConst,
empiricalCheck):

    loc = os.path.join('SeebeckData','NIST_Run1_CSV')

    with open(os.path.join('SeebeckData','NIST_Run1_CSV',
            F'Quasi_NIST_{namedTemp}K.csv'),'a',newline='') as csvfile:
        fieldnames = ['Date','Temp','CopperSlope','uncertaintyCuSlope',
        'ConstSlope','uncertaintyConstSlope','SeebeckCu','SeebeckConst',
        'PlatWireCopper','PlatWireConst','Empirical Check']
        writer = csv.DictWriter(csvfile,fieldnames = fieldnames)

        writer.writeheader()
        writer.writerow({'Date':date,'Temp':temp,'CopperSlope':copperSlope,
        'uncertaintyCuSlope':uncertaintyCuSlope,'ConstSlope':constSlope,
        'uncertaintyConstSlope':uncertaintyConstSlope,'SeebeckCu':copperSeebeck,
        'SeebeckConst':constSeebeck,'PlatWireCopper':platwireCopper,
        'PlatWireConst':platwireConst,'Empirical Check':empiricalCheck})


def writenoheader(date,namedTemp,temp,copperSlope,uncertaintyCuSlope,constSlope,
uncertaintyConstSlope,copperSeebeck,constSeebeck,platwireCopper,platwireConst,
empiricalCheck):

    with open(os.path.join('SeebeckData','NIST_Run1_CSV',
            F'Quasi_NIST_{namedTemp}K.csv'),'a',newline='') as csvfile:
        fieldnames = ['Date','Temp','CopperSlope','uncertaintyCuSlope',
        'ConstSlope','uncertaintyConstSlope','SeebeckCu','SeebeckConst',
        'PlatWireCopper','PlatWireConst','Empirical Check']
        writer = csv.DictWriter(csvfile,fieldnames = fieldnames)

        # writer.writeheader()
        writer.writerow({'Date':date,'Temp':temp,'CopperSlope':copperSlope,
        'uncertaintyCuSlope':uncertaintyCuSlope,'ConstSlope':constSlope,
        'uncertaintyConstSlope':uncertaintyConstSlope,'SeebeckCu':copperSeebeck,
        'SeebeckConst':constSeebeck,'PlatWireCopper':platwireCopper,
        'PlatWireConst':platwireConst,'Empirical Check':empiricalCheck})


## Since append does not truncate, clean_QS will reset the data file each time
## as if it did, however if the program is run for just one file then only one
## data set will be present
def clean_QS(namedTemp):

    if os.path.exists(os.path.join('SeebeckData','NIST_Run1_CSV',
            F'Quasi_NIST_{namedTemp}K.csv')):
        os.remove(os.path.join('SeebeckData','NIST_Run1_CSV',
                F'Quasi_NIST_{namedTemp}K.csv'))


## QuasiSteady can run after each file is read in its loop
def quasi_steady_state(filename,choice):
    (Ref_TempK,TC_Hot,TC_Cold,DeltaT,avgDeltaT,Cu_Voltage,Const_Voltage,avgCu,
    avgConst,Seebeck_Cu,Seebeck_CuConst,Seebeck_Const) = read_data(filename)

    (slopeCu,r_value1,std_err1,slopeConst,r_value2,std_err2,slope3,r_value3,
    std_err3) = plot_data_QS(choice,Ref_TempK,DeltaT,Cu_Voltage,Const_Voltage,
    TC_Hot,TC_Cold)

    (platwireCopper,platwireConst,empiricalCheck) = show_dataQS(Ref_TempK,
    slopeCu,r_value1,std_err1,slopeConst,r_value2,std_err2,slope3,r_value3,
    std_err3,Seebeck_Cu,Seebeck_Const)

    return(Ref_TempK,slopeCu,std_err1,slopeConst,std_err2,Seebeck_Cu,
    Seebeck_Const,platwireCopper,platwireConst,empiricalCheck)


## Steady will be anatomically different than QuasiSteady because it needs
## to read multiple files before finally executing
def steady_state(filename,choice,tData,cuData,coData,counterSS,date,namedTemp):
    (Ref_TempK,TC_Hot,TC_Cold,DeltaT,avgDeltaT,Cu_Voltage,Const_Voltage,avgCu,
    avgConst,Seebeck_Cu,Seebeck_CuConst,Seebeck_Const) = read_data(filename)
    tData.append(avgDeltaT)
    cuData.append(avgCu)
    coData.append(avgConst)

    # Need minimum of three data points to fit a line to
    if counterSS >= 2:
        (slopeCu,r_value1,std_err1,slopeConst,r_value2,std_err2,slope3,
        r_value3,std_err3) = plot_data_SS(choice,Ref_TempK,DeltaT,Cu_Voltage,
        Const_Voltage,TC_Hot,TC_Cold,tData,cuData,coData,counterSS)

        (platwireCopper,platwireConst,empiricalCheck) = show_dataSS(Ref_TempK,
        slopeCu,r_value1,std_err1,slopeConst,r_value2,std_err2,slope3,r_value3,
        std_err3,Seebeck_Cu,Seebeck_Const,counterSS,date,namedTemp)

## Func to grab all the CSVs that were just created
def grab_and_plot(path,method,title):
    temp = []
    copperSlope = []
    cuSlopeUnc = []
    constSlope = []
    constSlopeUnc = []
    SeebeckCu = []
    SeebeckConst = []
    copperleads = []
    constleads = []
    empiricalCheck = []
    avgPtCu = []
    avgPtConst = []

    for filename in glob.glob(path):
        with open(filename,'r') as method:
            header = method.readline()
            info = header.split(',')
            line = method.readline()
            while line:
                data = line.split(',')
                temp.append(float(data[1]))
                copperSlope.append(float(data[2]))
                cuSlopeUnc.append(float(data[3]))
                constSlope.append(float(data[4]))
                constSlopeUnc.append(float(data[5]))
                SeebeckCu.append(float(data[6]))
                SeebeckConst.append(float(data[7]))
                copperleads.append(float(data[8]))
                constleads.append(float(data[9]))
                empiricalCheck.append(float(data[10]))
                line = method.readline()

    plt.figure(figsize = (12,9))
    plt.plot(temp,empiricalCheck,'k--',label = 'empiricalCheck',markersize = 14)
    plt.plot(temp,copperleads,'bo',label = 'copperleads',markersize = 10)
    plt.plot(temp,constleads,'r.',label = 'constleads',markersize = 10)
    plt.title(F'NIST {title}',fontsize = 16)
    plt.xlabel('Temperature Range',fontsize = 14)
    plt.ylabel('Seebeck[$\mu$V]',fontsize = 14)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('images',F'NISTSeebeck{title}.png'),dpi=300)

    figLoc = os.path.join('images',F'NISTSeebeck{title}.png')

    return figLoc

## Main will switch between either Steady or QuasiSteady on its own
def main(args):
    choice = 'n'
    # Path to date
    day = os.path.join('SeebeckData','NIST_Run1','*')
    for filename in glob.glob(day):
        date = filename[-7:]
        print(date)

        # path to QS or SS measurement folder
        QS = os.path.join(filename,'QS','*')
        SS = os.path.join(filename,'SS','*')

        # Procedure for QuasiSteady
        for filename in glob.glob(QS):

            # Path to a given temperature
            tempQS = os.path.join(filename,'*')
            namedTemp = filename[-4:-1]
            print(namedTemp)
            clean_QS(namedTemp)

            # QS does the same operation for each file at a given temperature
            for index,filename in enumerate(glob.glob(tempQS)):
                (Ref_TempK,slopeCu,uncertaintyCuSlope,slopeConst,
                uncertaintyConstSlope,Seebeck_Cu,Seebeck_Const,platwireCopper,
                platwireConst,empiricalCheck) = quasi_steady_state(filename,
                choice)

                if index < 1:
                    writeheader(date,namedTemp,Ref_TempK,slopeCu,
                    uncertaintyCuSlope,slopeConst,uncertaintyConstSlope,
                    Seebeck_Cu,Seebeck_Const,platwireCopper,platwireConst,
                    empiricalCheck)

                    continue

                else:
                    writenoheader(date,namedTemp,Ref_TempK,slopeCu,
                    uncertaintyCuSlope,slopeConst,uncertaintyConstSlope,
                    Seebeck_Cu,Seebeck_Const,platwireCopper,platwireConst,
                    empiricalCheck)

    #     # Procedure for SteadyState
    #     for filename in glob.glob(SS):
    #
    #         # Path to a given temperature
    #         tempSS = os.path.join(filename,'*')
    #         namedTemp = filename[-4:-1]
    #         tData = []
    #         cuData = []
    #         coData = []
    #
    #         # SS must acquire 3 or more files before performing analysis
    #         for index,filename in enumerate(glob.glob(tempSS)):
    #
    #             steady_state(filename,choice,tData,cuData,coData,index,date,
    #             namedTemp)
    #
    # # The CSVs were created in the current directory
    dirQS = os.path.join('SeebeckData','NIST_Run1_CSV','Quasi*.csv')
    dirSS = os.path.join('SeebeckData','NIST_Run1_CSV','Steady*.csv')
    # currDirQS = os.path.join(str(currDir),'Quasi*.csv')
    # currDirSS = os.path.join(str(currDir),'Steady*.csv')
    qsFig = grab_and_plot(dirQS,'qs','QuasiSteady')
    # ssFig = grab_and_plot(dirSS,'ss','SteadyState')
    print('QuasiSteadyFig:',qsFig)
    # print('SteadyStateFig:',ssFig)
    # # The CSVs can be safely deleted or moved after running


if __name__ == '__main__':
    # Stay Frosty
    exit(main(sys.argv))
