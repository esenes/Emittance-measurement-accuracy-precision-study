
print('Version 0.03. This is the latest version.')
print('Please help me to improve it reporting bugs to guido.sterbini@cern.ch.')
# Fixes with respect to v0.2.
# unixtime2datetimeVectorize modified len with np.size
# Fixes with respect to v0.1.
# Fixed the bug of the local/UTC time in importing the matlab files
# Avoid importing multiple time the same variable from CALS or TIMBER
# Fixes with respect to v0.00.
# Thanks to Panos who spotted an error in the vectorization of the time stamp
get_ipython().magic('matplotlib inline')
import os
import glob
import scipy.io
import datetime   
import pickle     
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import matplotlib.dates as md
import matplotlib
import pandas as pnd
from pandas.tseries import converter
converter.register() 
import platform 
import math
import sys
import time
import csv
from scipy.constants import c
from IPython.display import Image, display, HTML
from scipy.optimize import curve_fit
#!git clone https://github.com/rdemaria/pytimber.git
#sys.path.insert(0,'/eos/user/s/sterbini/MD_ANALYSIS/public/pytimber/pytimber')
import pytimber; 
#pytimber.__file__
sys.path.append('/eos/project/l/liu/Instrumentation/Scripts')
import metaclass

sys.path.append('/eos/project/l/liu/LEIR/data/2016/TuneSpreadEvolution_2016/')
from TuneDiagram.tune_diagram import resonance_lines
from tunespread import *
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon

try:
    import seaborn as sns
except:
    print('If you want to use "seaborn" package, install it from a SWAN terminal "pip install --user seaborn"')

    
os.environ['LD_LIBRARY_PATH']=os.environ['LD_LIBRARY_PATH'] +':/eos/user/s/sterbini/MD_ANALYSIS/public/sharedLibraries/'
os.environ['PATH']=os.environ['PATH']+':/eos/user/s/sterbini/MD_ANALYSIS/public/'
    
plt.rcParams['axes.grid']=False
plt.rcParams['axes.facecolor']='none'
plt.rcParams['axes.grid.axis']='both'
plt.rcParams['axes.spines.bottom']=True
plt.rcParams['axes.spines.left']=True
plt.rcParams['axes.spines.right']=True
plt.rcParams['axes.spines.top']=True

plt.rcParams
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("paper")
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=plt.rcParams['xtick.labelsize']
plt.rcParams['axes.titlesize']=plt.rcParams['xtick.labelsize']

plt.rcParams['axes.labelsize']=plt.rcParams['xtick.labelsize']
plt.rcParams['legend.fontsize']=plt.rcParams['xtick.labelsize']
matplotlib.rcParams.update({'font.size': 8*2})
matplotlib.rc('font',**{'family':'serif'})
plt.rcParams['figure.figsize']=cm2inch(7.5*2.5,4.7*2.5) #cm2inch(7.5,4.7)
    
#matplotlib.rcParams['figure.figsize']=(15,7.5)
#matplotlib.rcParams.update({'font.size': 15})

#%config InlineBackend.figure_format = 'retina'

import matplotlib.dates as mdates

def check_ping(hostname):
    response = os.system("ping -c1 -W1 " + hostname)
    # and then check the response...
    if response == 0:
        pingstatus = True
    else:
        pingstatus = False

    return pingstatus
def convertUnixTime(a):
    return  719163+a/3600/24

def tagIt(a):
    """Use the tag_it(\'tag1, tag2\') function to tag the notebook. It will be useful for sorting them with a grep. """
    print('TAGS: '+ a)

def whereamI():
    import socket
    return socket.gethostbyname(socket.gethostname())


generalInfo={'myIP': whereamI(), 'myPWD': os.getcwd(), 'platform': platform.platform()}
    
#plt.rcParams['figure.figsize'] = (10, 8)
print('Your platform is ' + generalInfo['platform'])
print('Your folder is ' + generalInfo['myPWD'])
print('Your IP is ' + generalInfo['myIP'])
print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))



class dotdict(dict):
    '''A dict with dot access and autocompletion.
    
    The idea and most of the code was taken from 
    http://stackoverflow.com/a/23689767,
    http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
    http://stackoverflow.com/questions/2390827/how-to-properly-subclass-dict-and-override-get-set
    '''
    
    def __init__(self,*a,**kw):
        dict.__init__(self)
        self.update(*a, **kw)
        self.__dict__ = self
    
    def __setattr__(self, key, value):
        if key in dict.__dict__:
            raise AttributeError('This key is reserved for the dict methods.')
        dict.__setattr__(self, key, value)
    
    def __setitem__(self, key, value):
        if key in dict.__dict__:
            raise AttributeError('This key is reserved for the dict methods.')
        dict.__setitem__(self, key, value)
        
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
        
    def __getstate__(self):
        return self
 
    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


# In[ ]:




# In[1]:


if 'log' not in locals():
    log=pytimber.LoggingDB()

class myToolbox:
    import functools

    speedOfLight=299792458;

    
    @staticmethod    
    def plotSamplerFromObject(myobject, scale=1):
        info=myobject
        unitFactor=info.Samples.value.timeUnitFactor
        firstSampleTime=info.Samples.value.firstSampleTime
        samplingTrain=info.Samples.value.samplingTrain
        data=info.Samples.value.samples*scale
        x=np.arange(firstSampleTime*unitFactor,samplingTrain*unitFactor*len(data)+
                    firstSampleTime*unitFactor,
                    samplingTrain*unitFactor)
        plt.plot(x,data);
        plt.grid()
        plt.xlabel('t [s]')
        myCycleStamp=np.array(myToolbox.unixtime2datetime(myobject.Samples.cycleStamp/1e9))
        myCycleStamp=myCycleStamp.all()
        plt.title(myobject.Samples.cycleName + ', ' + myCycleStamp.ctime() )
        
    @staticmethod            
    def plotOasisFromObject(myobject,timeFactor=1, myLabel=''):
        myScope=myobject
        offset=myScope.Acquisition.value.offset
        mySignal=myScope.Acquisition.value.value*myScope.Acquisition.value.sensitivity+offset
        myTime=np.arange(0,(len(mySignal)),1)*myScope.Acquisition.value.sampleInterval*1e-9*timeFactor
        plt.plot(myTime,mySignal, label= myLabel)
        plt.grid('on')
        plt.xlabel('t [s/' + str(timeFactor) + ']')
        plt.ylabel('[V]')
        
    @staticmethod    
    def plotSuperSamplerFromObject(myobject):
        info=myobject
        unitFactor=info.SuperSamples.value.superTimeUnitFactor
        firstSampleTime=info.SuperSamples.value.firstSuperSampleTime
        samplingTrain=info.SuperSamples.value.superSamplingTrain
        data=info.SuperSamples.value.superSamples
        x=np.arange(firstSampleTime*unitFactor,samplingTrain*unitFactor*len(data)+
                    firstSampleTime*unitFactor,
                    samplingTrain*unitFactor)
        plt.plot(x,data);
        plt.grid()
        plt.xlabel('t [s]')
        myCycleStamp=np.array(myToolbox.unixtime2datetime(myobject.SuperSamples.value.superCycleStamp/1e9))
        myCycleStamp=myCycleStamp.all()
        plt.title(myobject.SuperSamples.cycleName + ', ' + myCycleStamp.ctime() )
    
    @staticmethod    
    def plotSamplerFromDataFrame(dateframeColumn, index, dataFormatField):
        info=dataFormatField
        aux=dateframeColumn
        unitFactor=info.Samples.value.timeUnitFactor
        firstSampleTime=info.Samples.value.firstSampleTime
        samplingTrain=info.Samples.value.samplingTrain
        x=np.arange(firstSampleTime*unitFactor,samplingTrain*unitFactor*len(aux.get_values()[index])+
                    firstSampleTime*unitFactor,
                    samplingTrain*unitFactor)
        plt.plot(x,aux.get_values()[index]);
        plt.grid()
        plt.xlabel('t [s]')

    @staticmethod    
    def addToDataFrameFromCALS(myDataFrame, variables,offset_second=0, verbose=False):
        #variables=['CPS.TGM:USER']
        variables=list(set(variables))
        cycleStamps=myDataFrame['cycleStamp'].get_values()
        # select the time interval
        for j in variables:
                aux=j.replace('.','_');
                aux=aux.replace(':','_')
                aux=aux.replace(' ','_')

                exec(aux+'=[]')

        for i in cycleStamps:
            ts1=datetime.datetime.utcfromtimestamp(i/1000000000.-.5+offset_second)
            ts2=datetime.datetime.utcfromtimestamp(i/1000000000.+.5+offset_second)
            if verbose:
                print(ts1)
        
            DATA=log.get(variables,ts1,ts2)
            for j in variables:
                aux=j.replace('.','_');
                aux=aux.replace(':','_')
                aux=aux.replace(' ','_')

                exec('myToolbox.test=len(DATA[\'' + j + '\'][1])')
                if myToolbox.test:
                    exec(aux + '.append(DATA[\'' + j + '\'][1][0])')
                else:
                    exec(aux + '.append(np.nan)')
        if offset_second>0:
            myString='_positiveOffset_'+str(offset_second)+'_s'
        
        if offset_second<0:
            myString='_negativeOffset_'+ str(-1*offset_second)+'_s'
        
        if offset_second==0:
            myString=''
        
        for j in variables:
                aux=j.replace('.','_');
                aux=aux.replace(':','_')
                aux=aux.replace(' ','_')
                exec('myDataFrame[\'' + j + myString + '\']=pnd.Series(' +aux+ ',myDataFrame.index)') 
    
    @staticmethod    
    def addSingleVariableFromMatlab(myInput, myVariable):
        data=myToolbox.japcMatlabImport(myInput);
        myDataFrame=pnd.DataFrame({})
        a=[]
        if hasattr(data,myVariable.split('.')[0]):
            exec('a.append(data.' + myVariable +')')
        else:
            exec('a.append(np.nan)')
        return a[0]
        
    @staticmethod     
    def addToDataFrameFromMatlab(myDataFrame, listOfVariableToAdd):
        listOfVariableToAdd=list(set(listOfVariableToAdd))
        for j in listOfVariableToAdd:
            myDataFrame[j]= myDataFrame['matlabFilePath'].apply(lambda myInput: myToolbox.addSingleVariableFromMatlab(myInput,j))
            
        
    @staticmethod    
    def fromMatlabToDataFrame(listing, listOfVariableToAdd, verbose=False, matlabFullInfo=False):
        listOfVariableToAdd=list(set(listOfVariableToAdd))
        myDataFrame=pnd.DataFrame({})
        cycleStamp=[]
        cycleStampHuman=[]
        PLS_matlab=[]
        matlabObject=[]
        matlabFilePath=[]
        for j in listOfVariableToAdd:
            exec(j.replace('.','_')+'=[]')
        for i in listing:
            if verbose:
                print(i)
            data=myToolbox.japcMatlabImport(i);
            if matlabFullInfo:
                matlabObject.append(data)
            #to correct
            localCycleStamp=np.max(data.headerCycleStamps[1:]);
            deltaLocal_UTC=datetime.datetime.fromtimestamp(localCycleStamp/1e9)-datetime.datetime.utcfromtimestamp(localCycleStamp/1e9)
            utcCycleStamp=localCycleStamp+deltaLocal_UTC.total_seconds()*1e9
            cycleStamp.append(utcCycleStamp)
            aux=myToolbox.unixtime2datetimeVectorize(np.max(data.headerCycleStamps[1:])/1e9)
            cycleStampHuman.append(aux.tolist())
            PLS_matlab.append(data.cycleName)
            matlabFilePath.append(os.path.abspath(i))
            for j in listOfVariableToAdd:
                if hasattr(data,j.split('.')[0]):
                    exec(j.replace('.','_') + '.append(data.' + j + ')')
                else:
                    exec(j.replace('.','_') + '.append(np.nan)')
        myDataFrame['cycleStamp']=pnd.Series(cycleStamp,cycleStampHuman)
        myDataFrame['matlabPLS']=pnd.Series(PLS_matlab,cycleStampHuman)
        myDataFrame['matlabFilePath']=pnd.Series(matlabFilePath,cycleStampHuman)
        if matlabFullInfo:
            myDataFrame['matlabFullInfo']=pnd.Series(matlabObject,cycleStampHuman)
        for j in listOfVariableToAdd:
            exec('myDataFrame[\'' + j + '\']=pnd.Series(' +j.replace('.','_')+ ',cycleStampHuman)')    #myDataFrame=pnd.DataFrame({j:aux,
        return myDataFrame
    
    @staticmethod    
    def fromTimberToDataFrame(listOfVariableToAdd,t1,t2,verbose=False, fundamental=''):
        listOfVariableToAdd=list(set(listOfVariableToAdd))
        #ts1=datetime.datetime(2016,7,1)
        #ts2=datetime.datetime(2016,7,2)
        #listOfVariableToAdd=['CPS.LSA:CYCLE','CPS.TGM:DEST']
        if fundamental=='':
            DATA=log.get(listOfVariableToAdd,t1,t2 )
        else:
            DATA=log.get(listOfVariableToAdd,t1,t2,fundamental)
        myDataFrame=pnd.DataFrame({})
        if DATA!={}:
            for i in listOfVariableToAdd:
                myDataFrame[i]=pnd.Series(DATA[i][1].tolist(),myToolbox.unixtime2datetimeVectorize(DATA[i][0]))
            myDataFrame['cycleStamp']=np.double(myDataFrame.index.astype(np.int64))
        return myDataFrame

    
    @staticmethod
    def japcMatlabImport(myfile):
        """Import the in a python structure a matlab file""" 
        myDataStruct = scipy.io.loadmat(myfile,squeeze_me=True, struct_as_record=False)
        return myDataStruct['myDataStruct']


    @staticmethod
    def unixtime2datetime(x):
        return datetime.datetime.fromtimestamp(x)
    
    @staticmethod
    def unixtime2datetimeVectorize(x):
        """Transform unixtime in python datetime"""
        aux=np.vectorize(myToolbox.unixtime2datetime)
        if np.size(x)!=0:
            return aux(x)
        else:
            return []
    
    @staticmethod
    def unixtime2utcdatetime(x):
        return datetime.datetime.utcfromtimestamp(x)
    
    @staticmethod
    def unixtime2utcdatetimeVectorize(x):
        """Transform unixtime in python datetime"""
        aux=np.vectorize(myToolbox.unixtime2utcdatetime)
        return aux(x)

    @staticmethod    
    def gaussian(x, mu, sig):
        """gaussian(x, mu, sig)"""
        return 1/np.sqrt(2*np.pi)/sig*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    @staticmethod    
    def gaussian_5_parameters(x, c, m, A, mu, sig):
        """gaussian_5_parameter(x, c, m, A, mu, sig)"""
        return c+m*x+A/np.sqrt(2*np.pi)/sig*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    @staticmethod
    def FilterSpill(mySpill): 
        baseline=0
        k=0
        myNewSpill=[]
        for i in np.arange(0,len(mySpill)):
            x = mySpill[i] - baseline;
            y = x + k * 3.2e-5;
            k += x;       
            if i<2000 or i > 23120:
                baseline += y / 5.;
            myNewSpill.append(y)
        return myNewSpill
    
    @staticmethod
    def MTE_efficiencyReduced(mySpill):
        if not (np.isnan(mySpill)).any():
            myNewSpill=myToolbox.FilterSpill(mySpill)
            b1_idx=2066
            b2_idx=6267
            b3_idx=10468
            b4_idx=14669
            b5_idx=18871
            b6_idx=23072
            is1=np.mean(myNewSpill[b1_idx:b2_idx])
            is2=np.mean(myNewSpill[b2_idx:b3_idx])
            is3=np.mean(myNewSpill[b3_idx:b4_idx])
            is4=np.mean(myNewSpill[b4_idx:b5_idx])
            core=np.mean(myNewSpill[b5_idx:b6_idx])
            mySum=(is1+is2+is3+is4+core);
            MTE_efficiency=np.mean([is1,is2,is3,is4])/mySum;
        else:
            MTE_efficiency=np.nan
        return MTE_efficiency
    
    @staticmethod    
    def MTE_efficiency(mySpill):
        if not (np.isnan(mySpill)).any():
            myNewSpill=myToolbox.FilterSpill(mySpill)
            b1_idx=2066
            b2_idx=6267
            b3_idx=10468
            b4_idx=14669
            b5_idx=18871
            b6_idx=23072
            is1=np.mean(myNewSpill[b1_idx:b2_idx])
            is2=np.mean(myNewSpill[b2_idx:b3_idx])
            is3=np.mean(myNewSpill[b3_idx:b4_idx])
            is4=np.mean(myNewSpill[b4_idx:b5_idx])
            core=np.mean(myNewSpill[b5_idx:b6_idx])
            mySum=(is1+is2+is3+is4+core);
            MTE_efficiency=np.mean([is1,is2,is3,is4])/mySum;
            myFFT=np.fft.fft(myNewSpill);
            FFTabs=np.abs(myFFT[3500:4500])     # 175:225 MHz
            FFTphase=np.angle(myFFT[3500:4500]) # 175:225 MHz
            return {"MTE_efficiency": MTE_efficiency,\
                    "Island1": is1/mySum,\
                    "Island2": is2/mySum,\
                    "Island3": is3/mySum,\
                    "Island4": is4/mySum,\
                    "Core"   : core/mySum,\
                    "FFT_abs"    : FFTabs,\
                    "FFT_phase"    : FFTphase,\
                    "mySum": mySum}
        else:
             return {"MTE_efficiency": np.nan, \
                     "Island1": np.nan,\
                     "Island2": np.nan,\
                     "Island3": np.nan,\
                     "Island4": np.nan,\
                     "Core"   : np.nan,\
                     "FFT_abs"    : np.nan,\
                     "FFT_phase"    : np.nan, 
                     "mySum": np.nan}
   
    @staticmethod        
    def myFirst(x):
        return x[0]

    @staticmethod    
    def mySecond(x):
        return x[1]
    
    @staticmethod    
    def first(df):
        return np.array(list(map(myToolbox.myFirst,df.get_values())))

    @staticmethod    
    def second(df):
        return np.array(list(map(myToolbox.mySecond,df.get_values())))
    
    @staticmethod    
    def line():
        return '<hr style="border-top-width: 4px; border-top-color: #34609b;">'

    @staticmethod        
    def computeTransverseEmittance(WS_position_um,WS_profile_arb_unit,off_momentum_distribution_arb_unit,deltaP_P,betaGammaRelativistic,betaOptical_m,Dispersion_m):
        x_inj=WS_position_um/1000;
        y_inj=WS_profile_arb_unit;
        
        
        popt,pcov = myToolbox.makeGaussianFit_5_parameters(x_inj,y_inj)
        #y_inj_1=myToolbox.gaussian_5_parameters(x_inj,popt[0],popt[1],popt[2],popt[3],popt[4])
        y_inj_1=myToolbox.gaussian_5_parameters(x_inj,popt[0],popt[1],popt[2],popt[3],popt[4])
        y_inj_2=y_inj-popt[0]-popt[1]*x_inj
        x_inj_2=x_inj-popt[3]
        x_inj_3=np.linspace(-40,40,1000);
        y_inj_3=scipy.interpolate.interp1d(x_inj_2,y_inj_2)(x_inj_3)
        y_inj_4=y_inj_3/np.trapz(y_inj_3,x_inj_3)
        y_inj_5=(y_inj_4+y_inj_4[::-1])/2

        WS_profile_step1_5GaussianFit=y_inj_1
        WS_profile_step2_dropping_baseline=y_inj_2
        WS_profile_step3_interpolation=y_inj_3
        WS_profile_step4_normalization=y_inj_4
        WS_profile_step5_symmetric=y_inj_5
        WS_position_step1_centering_mm=x_inj_2;
        WS_position_step2_interpolation_mm=x_inj_3;
        Dispersion_mm=Dispersion_m*1000

        Dispersive_position_step1_mm=deltaP_P*Dispersion_mm
        Dispersive_profile_step1_normalized=off_momentum_distribution_arb_unit/np.trapz(off_momentum_distribution_arb_unit,Dispersive_position_step1_mm)
        Dispersive_position_step2_mm=WS_position_step2_interpolation_mm
        Dispersive_step2_interpolation=scipy.interpolate.interp1d(Dispersive_position_step1_mm,Dispersive_profile_step1_normalized,bounds_error=0,fill_value=0)(Dispersive_position_step2_mm)
        Dispersive_step3_symmetric=(Dispersive_step2_interpolation+Dispersive_step2_interpolation[::-1])/2

        def myConvolution(WS_position_step2_interpolation_mm,sigma):
            myConv=np.convolve(Dispersive_step3_symmetric, myToolbox.gaussian(WS_position_step2_interpolation_mm,0,sigma), 'same')
            myConv/=np.trapz(myConv,WS_position_step2_interpolation_mm)
            return myConv

        def myError(sigma):
            myConv=np.convolve(Dispersive_step3_symmetric, myToolbox.gaussian(WS_position_step2_interpolation_mm,0,sigma), 'same')
            myConv/=np.trapz(myConv,WS_position_step2_interpolation_mm)
            aux=myConv-WS_profile_step5_symmetric
            return np.std(aux), aux, myConv

        popt,pcov = curve_fit(myConvolution,WS_position_step2_interpolation_mm,WS_profile_step5_symmetric,p0=[1])
        sigma=popt;
        emittance=sigma**2/betaOptical_m*betaGammaRelativistic
        return {'emittance_um':emittance,'sigma_mm':sigma,'WS_position_mm':WS_position_step2_interpolation_mm, 'WS_profile': WS_profile_step5_symmetric, 'Dispersive_position_mm':Dispersive_position_step2_mm, 'Dispersive_profile':Dispersive_step3_symmetric,
               'convolutionBackComputed':myConvolution(WS_position_step2_interpolation_mm,sigma),
               'betatronicProfile':myToolbox.gaussian(WS_position_step2_interpolation_mm,0,sigma)
               }
    
    @staticmethod        
    #thanks to Hannes
    def makeGaussianFit_5_parameters(X,Y):     
        i = np.where( (X>min(X)+1e-3) & (X<max(X)-1e-3) )
        X = X[i]
        Y = Y[i]

        indx_max = np.argmax(Y)
        mu0 = X[indx_max]
        window = 2*100
        x_tmp = X[indx_max-window:indx_max+window]
        y_tmp = Y[indx_max-window:indx_max+window]
        offs0 = min(y_tmp)
        ampl = max(y_tmp)-offs0
        x1 = x_tmp[np.searchsorted(y_tmp[:window], offs0+ampl/2)]
        x2 = x_tmp[np.searchsorted(-y_tmp[window:], -offs0+ampl/2)]
        FWHM = x2-x1
        sigma0 = np.abs(2*FWHM/2.355)
        ampl *= np.sqrt(2*np.pi)*sigma0
        slope = 0
        popt,pcov = curve_fit(myToolbox.gaussian_5_parameters,X,Y,p0=[offs0,slope,ampl,mu0,sigma0])
        return popt,pcov

    @staticmethod
    def linear_model(x, a, b):
        return a*x+b
    
    @staticmethod        
    def linear_fit(X, Y, a0 = 1e-3, b0 = 0.5):
        popt,pcov = curve_fit(myToolbox.linear_model,X,Y,p0=[a0, b0])
        return popt,pcov
    
    @staticmethod        
    def tricks():
        print("=== Highlight a region of a plot ===")
        print("""plt.gca().fill_between([startBLU, endBLU], [-.015,-.015], [.015,.015],color='g', alpha=.1)""")
        
        print("====")
        print("""from IPython.display import HTML
        HTML('''<script>
        code_show=true; 
        function code_toggle() {
        if (code_show){
        $('div.input').hide();
        } else {
        $('div.input').show();
        }
        code_show = !code_show
        } 
        $( document ).ready(code_toggle);
        </script>
        <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code
        </form>''')""")
        
        print('=== set x-axis date ===')
        print("""import matplotlib.dates as mdates
            myFig=plt.figure()
            ax=myFig.add_subplot(1,1,1)
            plt.plot([datetime.datetime(2016,9,23),datetime.datetime(2016,9,28,3)],[1,2])
            plt.xlim([datetime.datetime(2016,9,23),datetime.datetime(2016,9,28,3)])
            plt.ylim([.195,.218])
            plt.ylabel('MTE efficiency')
            t = np.arange(datetime.datetime(2016,9,23), datetime.datetime(2016,9,29), datetime.timedelta(hours=24)).astype(datetime.datetime)
            plt.xticks(t);
            myFmt =mdates.DateFormatter('%m/%d')
            ax.xaxis.set_major_formatter(myFmt);""")
        
        print('=== RETINA ===')
        print('% config InlineBackend.figure_format = \'retina\'')
        
        print('===list matlab attributes===')
        print("""aux=myDataFormat.PR_BQS72.SamplerAcquisition.value.__dict__
                for i in aux['_fieldnames']:
                    print(i+ ': ' + str(aux[i]))""")
        
        print('===SHOW matlab file===')
        print('''
            aux=myDataFormat.PR_BQS72.Acquisition.value.__dict__
            for i in aux['_fieldnames']:
                print(i+ ': ' + str(aux[i]))
        ''')
    @staticmethod        
    def hideCode():
        HTML('''<script>
        code_show=true; 
        function code_toggle() {
        if (code_show){
        $('div.input').hide();
        } else {
        $('div.input').show();
        }
        code_show = !code_show
        } 
        $( document ).ready(code_toggle);
        </script>
        <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code
        </form>''')
    
    @staticmethod        
    def E0_GeV(particle='proton'):
        #TODO
        return 0.9382723 

    @staticmethod        
    def Ek_GeVToP_GeV(Ek_GeV):
        T=Ek_GeV
        E0=myToolbox.E0_GeV()
        return np.sqrt(T*(2*E0+T))
    
    @staticmethod        
    def P_GeVToEk_GeV(cp):
        return np.sqrt(myToolbox.E0_GeV()**2+cp**2)-myToolbox.E0_GeV()

    @staticmethod        
    def P_GeVToEtot_GeV(cp):
        return np.sqrt(myToolbox.E0_GeV()**2+cp**2)
    
    @staticmethod        
    def P_GeVToBeta(cp):
        return cp*1.0/myToolbox.P_GeVToEtot_GeV(cp)
    
    @staticmethod        
    def P_GeVToGamma(cp):
        return myToolbox.P_GeVToEtot_GeV(cp)/myToolbox.E0_GeV()
    
    @staticmethod        
    def PS_circumference_m():
        return 100.*2*np.pi
    
    @staticmethod        
    def PS_frev_kHz(cp):
        return myToolbox.speedOfLight*myToolbox.P_GeVToBeta(cp)/myToolbox.PS_circunference_m()/1000.

    @staticmethod        
    def P_GeVandRelativisticParameters(cp):
        return {'Ek':myToolbox.P_GeVToEk_GeV(cp),
                'Etot':myToolbox.P_GeVToEtot_GeV(cp),
                'beta':myToolbox.P_GeVToBeta(cp),
                'gamma':myToolbox.P_GeVToGamma(cp),
                'PS_frev_kHz':myToolbox.PS_frev_kHz(cp),
                'PS_Trev_us':1000./myToolbox.PS_frev_kHz(cp)}

    @staticmethod        
    def fromTFStoDF(file):
        import metaclass as metaclass
        a=metaclass.twiss(file);
        aux=[]
        aux1=[]

        for i in dir(a):
            if not i[0]=='_':
                if type(getattr(a,i)) is float:
                    #print(i + ":"+ str(type(getattr(a,i))))
                    aux.append(i)
                    aux1.append(getattr(a,i))
                if type(getattr(a,i)) is str:
                    #print(i + ":"+ str(type(getattr(a,i))))
                    aux.append(i)
                    aux1.append(getattr(a,i))

        myList=[]
        myColumns=[]
        for i in a.keys:
            myContainer=getattr(a, i)
            if len(myContainer)==0:
                print("The column "+ i + ' is empty.')
            else:
                myColumns.append(i)
                myList.append(myContainer)
                
        optics=pnd.DataFrame(np.transpose(myList), index=a.S,columns=myColumns)

        for i in optics.columns:
            aux3= optics.iloc[0][i]
            if type(aux3) is str:
                aux3=str.replace(aux3, '+', '')
                aux3=str.replace(aux3, '-', '')
                aux3=str.replace(aux3, '.', '')
                aux3=str.replace(aux3, 'e', '')
                aux3=str.replace(aux3, 'E', '')


                if aux3.isdigit():
                    optics[i]=optics[i].apply(np.double)

        aux.append('FILE_NAME')
        aux1.append(os.path.abspath(file))

        aux.append('TABLE')
        aux1.append(optics)

        globalDF=pnd.DataFrame([aux1], columns=aux)

        return globalDF 
    
    @staticmethod        
    def plotLattice(ax,DF, height=1., v_offset=0., color='r',alpha=0.5):
        for i in range(len(DF)):
            aux=DF.iloc[i]
            ax.add_patch(
            patches.Rectangle(
                (aux.S-aux.L, v_offset-height/2.),   # (x,y)
                aux.L,          # width
                height,          # height
                color=color, alpha=alpha
            )
            )
        return;

    @staticmethod
    def latexIt(a):
        from matplotlib import rc
        rc('text', usetex=a)
    
    @staticmethod
    def cals2pnd(listOfVariableToAdd,t1,t2,verbose=False, fundamental=''):
        '''
        cals2pnd(listOfVariableToAdd,t1,t2,verbose=False, fundamental='')
        This function return a PANDAS dataframe of the listOfVariableToAdd within the interval t1-t2.
        It can be used in the verbose mode if the corresponding flag is True.
        It can be used to filter fundamentals.
        '''
        listOfVariableToAdd=list(set(listOfVariableToAdd))
        if fundamental=='':
            if verbose:
                print('No fundamental filter.')
            DATA=log.get(listOfVariableToAdd,t1,t2 )
        else:
            DATA=log.get(listOfVariableToAdd,t1,t2,fundamental)
        myDataFrame=pnd.DataFrame({})

        if DATA!={}:
            for i in listOfVariableToAdd:
                if verbose:
                        print('Eleaborating variable: '+ i)
                auxDataFrame=pnd.DataFrame({})
                auxDataFrame[i]=pnd.Series(DATA[i][1].tolist(),myToolbox.unixtime2datetimeVectorize(DATA[i][0]))
                myDataFrame=pnd.merge(myDataFrame,auxDataFrame, how='outer',left_index=True,right_index=True)
        return myDataFrame
    
    @staticmethod
    def LHCFillsByTime2pnd(t1,t2):
        '''
        LHCFillsByTime2pnd(t1,t2)
        This function return two PANDAS dataframes.
        The first dataframe contains the fills in the specified time interval t1-t2.
        The second dataframe contains the fill modes in the specified time interval t1-t2.
        '''
        DATA=log.getLHCFillsByTime(t1,t2)
        fillNumerList=[]
        startTimeList=[]
        endTimeList=[]
        beamModesList=[]
        ST=[]
        ET=[]
        FN=[]
        for i in DATA:
            FN.append(i['fillNumber'])
            ST.append(i['startTime'])
            ET.append(i['endTime'])
            for j in i['beamModes']:
                beamModesList.append(j['mode'])
                fillNumerList.append(i['fillNumber'])
                startTimeList.append(j['startTime'])
                endTimeList.append(j['endTime'])
        auxDataFrame=pnd.DataFrame({})
        auxDataFrame['mode']=pnd.Series(beamModesList, fillNumerList)
        auxDataFrame['startTime']=pnd.Series(myToolbox.unixtime2datetimeVectorize(startTimeList), fillNumerList)
        if endTimeList[-1:]==[None]:
            endTimeList[-1:]=[time.time()]
            print('One fill is not yet ended. We replaced the "None" in endTime with the "now" time (GVA time).')
            
        if ET[-1:]==[None]:
            ET[-1:]=[time.time()]
            print('One fill is not yet ended. We replaced the "None" in ET with the "now" time (GVA time).')
        auxDataFrame['endTime']=pnd.Series(myToolbox.unixtime2datetimeVectorize(endTimeList), fillNumerList)
        auxDataFrame['duration']=auxDataFrame['endTime']-auxDataFrame['startTime']

        aux=pnd.DataFrame({})
        aux['startTime']=pnd.Series(myToolbox.unixtime2datetimeVectorize(ST), FN)
        aux['endTime']=pnd.Series(myToolbox.unixtime2datetimeVectorize(ET), FN)
        aux['duration']=aux['endTime']-aux['startTime']
        return aux, auxDataFrame

    @staticmethod    
    def addRowsFromCals(myDF, deltaTime=datetime.timedelta(minutes=2)):
        """This method extend in time a pandas dataframe using the CALS database.
        It returns a new pandas dataframe. It hase 2 arguments:
        myDF: the initial dataframe
        deltaTime=datetime.timedelta(minutes=2):  the delta of time to apply"""
        aux=myToolbox.cals2pnd(list(myDF),myDF.index[-1],myDF.index[-1]+deltaTime )
        myDF=pnd.concat([myDF,aux])
        return myDF
    
    @staticmethod    
    def addColumnsFromCals(myDF, listOfVariables):
        """This method add a list of variables to a pandas dataframe using the CALS database.
        It returns a new pandas dataframe. It hase 2 arguments:
        myDF: the initial dataframe
        listOfVariable:  the list of variables to add"""
        aux=myToolbox.cals2pnd(listOfVariables,myDF.index[0],myDF.index[-1])
        myDF=pnd.concat([myDF,aux])
        return myDF

    @staticmethod    
    def time_now():
        return datetime.datetime.now()

    @staticmethod    
    def time_1_week_ago(weeks=1):
        fromTime=datetime.datetime.now()
        return fromTime-datetime.timedelta(weeks=weeks)
    
    @staticmethod    
    def time_1_day_ago(days=1):
        fromTime=datetime.datetime.now()
        return fromTime-datetime.timedelta(days=days)
    
    @staticmethod    
    def time_5_minutes_ago(minutes=5):
        fromTime=datetime.datetime.now()
        return fromTime-datetime.timedelta(minutes=minutes)
    
    @staticmethod    
    def time_1_hour_ago(hours=1):
        fromTime=datetime.datetime.now()
        return fromTime-datetime.timedelta(hours=hours)

    @staticmethod    
    def setXlabel(ax, hours=1., myFormat='%H:%M', startDatetime=datetime.datetime.utcfromtimestamp(0)):
        """
        setXlabel(ax=plt.gca(), hours=1., myFormat='%H:%M', startDatetime=datetime.datetime.utcfromtimestamp(0))
        ax: specify the axis
        hours: specify the interval 
        myFormat: specify the format
        startDatetime: specify the starting time (to have round captions)
        """
        aux=ax.get_xlim()
        serial = aux[0]
        if startDatetime ==datetime.datetime.utcfromtimestamp(0):
            seconds = (serial - 719163.0) * 86400.0
            startDatetime=datetime.datetime.utcfromtimestamp(seconds)
        serial = aux[1]
        seconds = (serial - 719163.0) * 86400.0
        date_end=datetime.datetime.utcfromtimestamp(seconds)
        t = np.arange(startDatetime, date_end, datetime.timedelta(hours=hours)).astype(datetime.datetime)
        ax.set_xticks(t);
        myFmt =mdates.DateFormatter(myFormat)
        ax.xaxis.set_major_formatter(myFmt);
        return startDatetime
    
    @staticmethod
    def setArrowLabel(ax, label='myLabel',arrowPosition=(0,0),labelPosition=(0,0), myColor='k', arrowArc_rad=-0.2):
        return ax.annotate(label,
                      xy=arrowPosition, xycoords='data',
                      xytext=labelPosition, textcoords='data',
                      size=10, color=myColor,va="center", ha="center",
                      bbox=dict(boxstyle="round4", fc="w",color=myColor,lw=2),
                      arrowprops=dict(arrowstyle="-|>",
                                      connectionstyle="arc3,rad="+str(arrowArc_rad),
                                      fc="w", color=myColor,lw=2), 
                      )
    @staticmethod
    def setShadedRegion(ax,color='g' ,xLimit=[0,1],alpha=.1):
        """
        setHighlightedRegion(ax,color='g' ,xLimit=[0,1],alpha=.1)
        ax: plot axis to use
        color: color of the shaded region
        xLimit: vector with two scalars, the start and the end point
        alpha: transparency settings
        """
        aux=ax.get_ylim()
        plt.gca().fill_between(xLimit, 
                           [aux[0],aux[0]],  [aux[1],aux[1]],color=color, alpha=alpha)
        
    @staticmethod        
    def mergeDF(df1,df2):
        """
        It returns a new dataframe obtained by merging df1 and df2 with no duplicated columns.
         
        mergeDF(df1,df2)
            return pnd.merge(df1,df2[df2.columns.difference(df1.columns)],
                     left_index=True, 
                     right_index=True, 
                      how='outer')
        """
    
        return pnd.merge(df1,df2[df2.columns.difference(df1.columns)],
                     left_index=True, 
                     right_index=True, 
                     how='outer')
    
    @staticmethod        
    def concatDF(df1,df2):
        """
        It returns a new dataframe that is the concatation of df1 and df2.
         
        def concatDF(df1,df2):
            aux=pnd.concat([df1,df2]).sort_index()
            return aux.groupby(aux.index).first() #to avoid duplicate
        """      
        aux=pnd.concat([df1,df2]).sort_index()
        return aux.groupby(aux.index).first()
    
    @staticmethod
    def indexesFromCALS(indexesList, variables,verbose=False):
        myDF=pnd.DataFrame()
        for i in indexesList:
            if verbose:
                print(i)
            aux=myToolbox.cals2pnd(variables,i,i)
            myDF=myToolbox.concatDF(myDF,aux)
        return myDF

    @staticmethod
    def betarel(gamma):
        return np.sqrt(1-1/gamma**2)

    @staticmethod
    def bunchlength_m(sigt, gamma):
        'Computes the bunch length in meters based on the relativistiv gamma and the bunch length in ns.'
        return myToolbox.betarel(gamma)*c*sigt    

    @staticmethod
    def plotPSaperture(section = [0], plane = 0):
        '''Plot of the PS aperture based on the Excel aperture data from 2013.
        The keyword section corresponds to the straight sections between which the plot should be limited:
        [0]: whole machine
        [1-100]: plot of the aperture in the desired straight section
        [1,10]: plot of the aperture between the two indicated sections
        plane = 0 for horizontal, plane = 1 for vertical aperture.
        Function returns longitudinal position as well as the aperture for the corresponding plane.'''
    
        filename = '/eos/project/l/liu/PS/Aperture/Aperture2013_for_Py.mat'
        aperture = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
        
        if plane == 0:
            aperExt = aperture['HorExt']
            aperInt = -aperture['HorInt']
            plt.fill(aperture['s'], aperExt,'grey')
            plt.fill(aperture['s'], aperInt,'grey')
            plt.ylabel('x [mm]') 
            plt.ylim(-200,200)
        elif plane == 1:
            aperExt = aperture['Ver']
            aperInt = -aperture['Ver']
            plt.fill(aperture['s'], aperExt,'grey')
            plt.fill(aperture['s'], aperInt,'grey')
            plt.ylabel('y [mm]')    
            plt.ylim(-100,100)
        if (len(section) == 1) & (section[0] == 0):
            plt.xlim(0,2*np.pi*100)
        elif (len(section) == 1) & (section[0] != 0):
            plt.xlim(aperture['SSlim'][section[0]-1][0],aperture['SSlim'][section[0]-1][1])
        elif (len(section) == 2):
            plt.xlim(aperture['SSlim'][section[0]-1][0],aperture['SSlim'][section[1]-1][1])    
        plt.xlabel('s [m]')  
        
        return [aperture['s'], aperExt, aperInt]
        
    @staticmethod
    def extractProfile(fileName):
        '''Give me the tomoscope input file (.dat) and I will give you 2 outputs:
            the deltaP_P and the the momentum distribution profile.
        ''' 
    
        print('Treating file ' + fileName)
        a=os.system('/eos/project/l/liu/Instrumentation/tomography/runofflinetomo ' + fileName)
        
        data_file = glob.glob(fileName[0:-4] + '/d*.data')[0]       
        convergence = np.loadtxt(data_file)
        
        image_file = glob.glob(fileName[0:-4] + '/image*.data')[0]
        image = np.loadtxt(image_file)
        
        profile_file = glob.glob(fileName[0:-4] + '/*_profiledata.dat')[0]
        profiles = np.loadtxt(profile_file)
        
        with open(fileName[0:-4] + '/input_v2.dat','r') as stream: tomofileheader = stream.readlines()[:98]
    
        myInput={'PLSUSER': tomofileheader[0][0:-1],
                 'timestamp': myToolbox.unixtime2datetime(np.int(tomofileheader[1][0:-1])),
                'NumberOfframes': np.int(tomofileheader[16][0:-1]),
                'NumberOfbins': np.int(tomofileheader[20][0:-1]),
                'dtprofile': np.double(tomofileheader[22][0:-1]),
                'DeltaTurns': np.int(tomofileheader[24][0:-1]),
                'B_T': np.double(tomofileheader[75][0:-1]),
                'Bdot_T_per_s': np.double(tomofileheader[77][0:-1]),
                'machineRadius_m': np.double(tomofileheader[79][0:-1]),
                'bendingRadius_m': np.double(tomofileheader[81][0:-1]),
                'particleMass_eV': np.double(tomofileheader[85][0:-1]),
                'h': np.int(np.double(tomofileheader[69][0:-1])),
                'ctime': np.int(tomofileheader[2])
        }
    
        with open(fileName[0:-4] + '/plotinfo.data','r') as stream: tomofileheader = stream.readlines()
    
        myInput['profilecount']=np.int(str.split(tomofileheader[1])[2])
        myInput['profilelength']=np.int(str.split(tomofileheader[3])[2])
            
        myInput['dtbin']=np.double(str.split(tomofileheader[5])[2])
        myInput['dEbin']=np.double(str.split(tomofileheader[7])[2])
        myInput['xat0']=np.double(str.split(tomofileheader[11])[2])
        myInput['yat0']=np.double(str.split(tomofileheader[12])[2])
        myInput['eperimage']=np.double(str.split(tomofileheader[9])[2])        
    
        myInput['momentum_eV']=myInput['B_T']*myInput['bendingRadius_m']/(10/c*1e8)*1.e9
        myInput['totalEnergy_eV']=np.sqrt(myInput['momentum_eV']**2+myInput['particleMass_eV']**2)
        myInput['gamma']=myInput['totalEnergy_eV']/myInput['particleMass_eV']
        myInput['beta']=np.sqrt(1-1/myInput['gamma']**2)
        myInput['frev'] = myInput['beta']*c/(2 * np.pi * myInput['machineRadius_m'])
        myInput['trev'] = 1/myInput['frev']
    
        image=image*myInput['eperimage']/myInput['dtbin']/myInput['dEbin']
    
        halfProfileLength=myInput['profilelength']/2.
    
        Toffset= (myInput['xat0']-halfProfileLength)*myInput['dtbin']*1e9
        Eoffset= (myInput['yat0']-halfProfileLength)*myInput['dEbin']/1e6
    
        t=np.arange(-halfProfileLength,halfProfileLength)*myInput['dtbin']*1e9-Toffset
        E=np.arange(-halfProfileLength,halfProfileLength)*myInput['dEbin']/1e6-Eoffset
    
        deltaP_P= 1/myInput['beta']**2*E*1e6/myInput['totalEnergy_eV']
        myProfile=np.sum(np.reshape(image, [myInput['profilelength'], myInput['profilelength']]),0)
        myProfile=myProfile/np.trapz(myProfile,deltaP_P)
        myInput['deltaP_P']=deltaP_P;
        myInput['myProfile']=myProfile;
        myInput['E_MeV']=E
        myInput['t_ns']=t
        myInput['phaseSpace_e_per_eVs']=np.reshape(image, [myInput['profilelength'], myInput['profilelength']]);
        
        myInput['time'] = np.arange(0, myInput['NumberOfframes'], 1) * myInput['trev'] * myInput['DeltaTurns']*1e3
        myInput['dt'] = np.arange(0, myInput['NumberOfbins'], 1) * myInput['dtprofile']*1e9
        myInput['profiles'] = np.reshape(profiles, [myInput['profilecount'], myInput['NumberOfbins']])
        
        myProfileNormalized=myProfile/np.trapz(myProfile,deltaP_P)
        
        #rms emittance calculation
        xbar = 0.
        xms = 0.
        ybar = 0.
        yms = 0.
        xybar = 0.
    
        phaseSpaceNorm = myInput['phaseSpace_e_per_eVs'].T/np.sum(np.sum(myInput['phaseSpace_e_per_eVs'].T))
    
        for i in xrange(0,myInput['profilelength']):
            for j in xrange(0,myInput['profilelength']):
                xbar += phaseSpaceNorm[i,j]*i
                xms += phaseSpaceNorm[i,j]*i**2
                ybar += phaseSpaceNorm[i,j]*j
                yms += phaseSpaceNorm[i,j]*j**2
                xybar += phaseSpaceNorm[i,j]*i*j
        
        myInput['rmsemittance'] = np.pi * myInput['dtbin'] * myInput['dEbin'] * np.sqrt((xms - xbar**2)*(yms - ybar**2) - (xybar -xbar*ybar)**2)    
    
        myMean=np.trapz(deltaP_P*myProfileNormalized,deltaP_P)   
        myRMS=np.sqrt(np.trapz((deltaP_P-myMean)**2*myProfileNormalized,deltaP_P))
        myInput['deltaP_P_RMS']=myRMS
            
        return myInput
        
    @staticmethod
    def extract_TT2BPM_position(x):
        return x[0:7,0]
    
    @staticmethod
    def import_chromaticity(filename):
        Qx = None
        idx = 0
    
        with open(filename, 'rb') as csvfile:
            data = csv.reader(csvfile);
            ## search time index
            items = [];
    
            for i, row in enumerate(data): 
                # define horizontal and vertical tune dataframes
                try:
                    if row[0].rfind('Value') >= 0:
                        if Qx is None:    
                            Qx = pnd.DataFrame(columns = row)
                            Qxdp = pnd.DataFrame(columns = row)
                        else:
                            idx = i
                            Qy = pnd.DataFrame(columns = row)
                            Qydp = pnd.DataFrame(columns = row)
                except IndexError:
                    pass
    
                # import tune and dp/p data
                try:
                    if row[0].rfind('QH') >= 0:
                        Qx = Qx.append([pnd.Series(row, Qx.columns)], ignore_index=True)
                    if row[0].rfind('QV') >= 0:
                        Qy = Qy.append([pnd.Series(row, Qy.columns)], ignore_index=True)
    
                    if (row[0].rfind('Dp/p') >= 0) & (idx == 0):
                        Qxdp = Qxdp.append([pnd.Series(row, Qx.columns)], ignore_index=True)
                    elif (row[0].rfind('Dp/p') >= 0) & (idx > 0):
                        Qydp = Qydp.append([pnd.Series(row, Qy.columns)], ignore_index=True)
    
                except IndexError:
                    pass
    
        Qx = Qx.apply(pnd.to_numeric, errors='ignore').dropna(axis = 0, thresh = 10)
        Qxdp = Qxdp.apply(pnd.to_numeric, errors='ignore').dropna(axis = 0, thresh = 10)
        
        Qy = Qy.apply(pnd.to_numeric, errors='ignore').dropna(axis = 0, thresh = 10)
        Qydp = Qydp.apply(pnd.to_numeric, errors='ignore').dropna(axis = 0, thresh = 10)
        
        return Qxdp, Qx, Qydp, Qy
    
    @staticmethod
    def fit_chromaticity(QdpDF, QDF, degree):
    
        Q = pnd.DataFrame(columns = ['time', 'coefficients'])
    
        for col in QDF.columns[2:]:
            try:    
                Q = Q.append(pnd.Series([np.float(col), (np.polyfit(QdpDF[col][QdpDF[col].notna()], QDF[col][QdpDF[col].notna()], degree))], Q.columns), ignore_index = True)
            except ValueError:
                pass
        
        Q['tune'] = Q['coefficients'].apply(lambda x: x[-1])
        Q['chromaticity'] = Q['coefficients'].apply(lambda x: x[-2])
        try: 
            Q['nl_chromaticity'] = Q['coefficients'].apply(lambda x: x[-3])
        except:
            pass
        
        return Q
        
class SEM_grids:
    
    @staticmethod
    def extract_optics(filename, device):
        '''Extracts the optics parameters at a given device from the given MAD-X table. It is sufficient to provide a substring such as 'MSFHV' as device name.'''
    
        device_dict = {} 
    
        optics = metaclass.twiss(filename)
    
        for i,elem in enumerate(optics.NAME):
            for d in device:
                if d in elem:
    
                    if elem[0:3] == 'ETL':
                        elem = elem.replace('.MSFHV','_MSF')
                    elif elem[0:3] == 'F16':
                        elem = elem.replace('.BSF','_BSF') 
                        elem = elem.replace('.BSG','_BSG')                
    
                    device_dict[elem] = {}
                    device_dict[elem]['indx'] = i
                    device_dict[elem]['bet'] = {}
                    device_dict[elem]['alf'] = {}
                    device_dict[elem]['disp'] = {}
                    device_dict[elem]['mu'] = {}
                    device_dict[elem]['C'] = {}
                    device_dict[elem]['S'] = {}
                    device_dict[elem]['delta'] = {}            
    
                    device_dict[elem]['bet']['X'] = optics.BETA11[i]
                    device_dict[elem]['bet']['Y'] = optics.BETA22[i]
                    device_dict[elem]['alf']['X'] = optics.ALFA11[i]
                    device_dict[elem]['alf']['Y'] = optics.ALFA22[i]
                    device_dict[elem]['disp']['X'] = optics.DISP1[i]
                    try:
                        device_dict[elem]['disp']['Y'] = optics.DISP2[i] 
                    except AttributeError:
                        device_dict[elem]['disp']['Y'] = 0.0
                    device_dict[elem]['mu']['X'] = 2*np.pi*optics.MU1[i]
                    device_dict[elem]['mu']['Y'] = 2*np.pi*optics.MU2[i]  
        
        print('\nThe following devices were found:')
        for k in device_dict.keys():
            print(k)
            
        return device_dict

    @staticmethod
    def pseudo_trigonom(device, reference):
        '''Calculate the cos-like and sin-like function with respect to the given reference point.'''
    
        for p in ['X','Y']:
            c = np.zeros(1)
            s = np.zeros(1)
            
            for i,elem in enumerate(device.keys()):
                dmu = device[elem]['mu'][p] - reference['mu'][p]
    
                c = np.sqrt(device[elem]['bet'][p]/reference['bet'][p]) * (np.cos(dmu) + reference['alf'][p] * np.sin(dmu))
                s = np.sqrt(device[elem]['bet'][p] * reference['bet'][p]) * np.sin(dmu)  
            
                device[elem]['C'][p] = c
                device[elem]['S'][p] = s
        
        return device
        
    @staticmethod
    def delta(device):
        '''Prepared to be used for more than 3 monitors.'''
        for p in ['X','Y']:
            for i,elem in enumerate(device.keys()):
                device[elem]['delta'][p] = 1
                
        return device
    
    @staticmethod
    def sigma_measured(device, path, plane):
        '''Extracts the measurement data in the given path and fits the beam sizes. Plane should be 'H' or 'V'.'''
        
        folders=sorted(glob.glob(path))
        files = []
        for folder in folders:
            print(folder + '\n')
            files += sorted(glob.glob(folder + '/2*.mat'))
        
        parameters = []
    
        for k in device.keys():
            parameters += ([k + '.Acquisition.value.projPositionSet1', k + '.Acquisition.value.projDataSet1', k + '.Acquisition.value.planeSet1'])
        
        print('Importing data into Dataframe.')  
        myDataFrame=myToolbox.fromMatlabToDataFrame(files, parameters)
        
        indx = {'H': [], 'V': []}
        
        for i,elem in enumerate(myDataFrame[device.keys()[0] + '.Acquisition.value.planeSet1'].iloc[0:]):
            if elem == 'HORIZONTAL':
                indx['H'].append(i)
            elif elem == 'VERTICAL':
                indx['V'].append(i)

        print('Masking Faulty wires.')
        myDataFrame = SEM_grids.mask_wires(myDataFrame, indx, plane)
        
        myDataFrame['plane']=pnd.Series()
        myDataFrame['plane'].iloc[indx['H']] = 'X'    
        myDataFrame['plane'].iloc[indx['V']] = 'Y'    
        
        for k in device.keys():
            myDataFrame[k + '.Acquisition.value.projDataSet1'] = myDataFrame[k + '.Acquisition.value.projDataSet1'].apply(lambda x: np.abs(x))  
        
        print('Peforming fit.')
        for k in device.keys():
            myDataFrame[k + '_fit'] = myDataFrame[[k + '.Acquisition.value.projPositionSet1', k + '.Acquisition.value.projDataSet1']].apply(lambda x: SEM_grids.GaussianFit(*x), axis=1)  
            myDataFrame[k + '_sigma'] = myDataFrame[k + '_fit'].apply(lambda x: x[0][2]*1e-3)
            myDataFrame[k + '_mean'] = myDataFrame[k + '_fit'].apply(lambda x: x[0][1]*1e-3)
    
        print('Finished.')
        return myDataFrame

    @staticmethod
    def mask_wires(myDataFrame, indx, plane):
        '''Mask the dead wires in the respective SEM grid or fil. Plane has to be either 'H' or 'V'.'''
        devices = ['ETL_MSF20', 'ETL_MSF30', 'F16_BSG258', 'F16_BSG268', 'F16_BSG278', 'F16_BSF257']
        
        # mask_indx = {'ETL_MSF20': {'H': 28}, 
        #              'ETL_MSF30': {'H': 18}, 
        #              'F16_BSG258': {'H': 15}, 
        #              'F16_BSG268': {'H': 3}, 
        #              'F16_BSG278': {'H': 15}, 
        #              'F16_BSF257': {'H': 18}}
        
        mask_indx = {'H': {}, 'V': {}}
        mask_indx['H']['ETL_MSF20'] = 28
        mask_indx['H']['ETL_MSF30'] = 18
        
        mask_indx['H']['F16_BSG258'] = 15
        mask_indx['V']['F16_BSG258'] = [6,15]
        
        mask_indx['H']['F16_BSG268'] = 3
        mask_indx['V']['F16_BSG268'] = [1,4,13,15]
        
        mask_indx['H']['F16_BSG278'] = 15
        
        mask_indx['H']['F16_BSF257'] = [41,43]
        mask_indx['V']['F16_BSF257'] = [32,33,34,35]

        for k in devices:
            try: 
                myDataFrame[k + '.Acquisition.value.projPositionSet1'].iloc[indx[plane]] = myDataFrame[k + '.Acquisition.value.projPositionSet1'].iloc[indx[plane]].apply(lambda x: np.delete(x, mask_indx[plane][k]))
                
                myDataFrame[k + '.Acquisition.value.projDataSet1'].iloc[indx[plane]] = myDataFrame[k + '.Acquisition.value.projDataSet1'].iloc[indx[plane]].apply(lambda x: np.delete(x, mask_indx[plane][k]))

                myDataFrame[k + '.Acquisition.value.projDataSet1'] = myDataFrame[k + '.Acquisition.value.projDataSet1'].apply(SEM_grids.set_zero)   
                print('\nMeasurements with SEM grid ' + k + ' found. Measurement plane: ' + plane)
                print('Signals from faulty wires have been removed.\n')
            except:
                pass
            
        return myDataFrame
    
    @staticmethod
    def sigma2betatronic(device, params):
        '''Calculates the betatronic beam size.'''
        
        sigma_meas = params[0]
        disp = device['disp'][params[1]]
        dp = params[2]
    
        sigma_bet = sigma_meas**2 - disp**2 * dp**2
    #     sigma_bet = np.sqrt(sigma_meas**2 - disp**2 * dp**2)
        
        return sigma_bet
    
    @staticmethod
    def emittance(device, params):
        '''Returns the reconstructed emittance as well as the measured beta and alpha functions at the reference point.'''
        plane = params[0]
        
        a = b = c = d = e = f = g = h = i = 0
        M = {}
        v = {}
        sol = {}
        emit = {}
        
        for elem in device.keys():
            # params[elem +  '_sigma_bet'] are the betatronic emittances
    
            # matrix elements
            b += device[elem]['C'][plane]**4 / device[elem]['delta'][plane]**2
            c += - 2 * device[elem]['C'][plane]**3 * device[elem]['S'][plane] / device[elem]['delta'][plane]**2
            d += device[elem]['C'][plane]**2 * device[elem]['S'][plane]**2 / device[elem]['delta'][plane]**2
            f += 4 * device[elem]['C'][plane]**2 * device[elem]['S'][plane]**2 / device[elem]['delta'][plane]**2
            g += - 2 * device[elem]['C'][plane] * device[elem]['S'][plane]**3 / device[elem]['delta'][plane]**2
            i += device[elem]['S'][plane]**4 / device[elem]['delta'][plane]**2
    
            # vector elements
            a += device[elem]['C'][plane]**2 * params[elem + '_sigma_bet2'] / device[elem]['delta'][plane]**2
            e += -2 * device[elem]['C'][plane] * device[elem]['S'][plane] * params[elem +  '_sigma_bet2'] / device[elem]['delta'][plane]**2
            h += device[elem]['S'][plane]**2 * params[elem +  '_sigma_bet2'] / device[elem]['delta'][plane]**2
    
        M[plane] = np.matrix([[b, c, d], [c, f, g], [d, g, i]])
        v[plane] = np.matrix([[a], [e], [h]]) 
        sol[plane] = M[plane].I * v[plane]
        emit= np.sqrt(sol[plane][0] * sol[plane][2] - sol[plane][1]**2)
        beta = sol[plane][0]/emit.item(0)
        alpha = sol[plane][2]/emit.item(0)
        
        return emit.item(0), beta.item(0), alpha.item(0)

    @staticmethod
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    @staticmethod
    def GaussianFit(X,Y):     
        mu0 = sum(X * Y) / sum(Y)
        sigma0 = np.sqrt(sum(Y * (X - mu0)**2) / sum(Y))
        ampl = np.amax(Y)
        popt,pcov = curve_fit(SEM_grids.gaus,X,Y,p0=[ampl,mu0,sigma0])
        return popt,pcov

    @staticmethod
    def set_zero(value):
        for i,elem in enumerate(value):
            if elem < 0.:
                value[i] = 0.
        return value
    
    
class TuneDiagram:
    
    @staticmethod
    def tunediagram(ax, Qx = (5.99, 6.5), Qy = (5.99, 6.5), order = (1,2,3,4), sym = 10):
        'Plot of the tune diagram for the given input parameters.'
        resonances = resonance_lines(Qx, Qy, order, sym)

        ax.set_xlim(resonances.Qx_min, resonances.Qx_max)
        ax.set_ylim(resonances.Qy_min, resonances.Qy_max)

        ax.set_xlabel('$Q_x$', fontsize=15)
        ax.set_ylabel('$Q_y$', fontsize=15)

        # plot tune diagram
        for res in resonances.resonance_list:
            nx = res[0]
            ny = res[1]
            for res_sum in res[2]:
                if ny:
                    line, = ax.plot([resonances.Qx_min, resonances.Qx_max], [(res_sum-nx*resonances.Qx_min)/ny, (res_sum-nx*resonances.Qx_max)/ny])
                else:
                    line, = ax.plot([np.float(res_sum)/nx, np.float(res_sum)/nx],[resonances.Qy_min, resonances.Qy_max])
                if ny%2:
                    plt.setp(line, linestyle='--') # for skew resonances
                if res_sum%resonances.periodicity:
                    plt.setp(line, color='royalblue')	# non-systematic resonances
                else:
                    plt.setp(line, color='firebrick', linewidth=2.0) # systematic resonances
                    
    
    @staticmethod
    def calculate_sc(gamma, intensity, emittance_norm, sigz, deltap = 1e-3, twissfile = '/eos/user/a/ahuschau/MD_analysis/Tune_spread_computation/LHC_injection_Qx_24_Qy_245.tfs'): 
        'Computes the maximum direct space charge tune shift based on A. Oeftigers library. The normalized emittance is expected in units of mm.mrad. All quantities are one sigma.'

        mp = 0.938272046

        emittance_geom = [elem*1E-6/myToolbox.betarel(gamma)/gamma for elem in emittance_norm] 

        params = {'n_part': intensity, 
                  'emit_geom_x': emittance_geom[0], 
                  'emit_geom_y': emittance_geom[1], 
                  'gamma': gamma, 
                  'beta':myToolbox.betarel(gamma), 
                  'deltap': deltap, 
                  'mass': mp, 
                  'n_charges_per_part': 1, 
                  'sig_z': sigz, 
                  'coasting': False}

        optics = metaclass.twiss(twissfile)

        twiss = {}
        twiss['beta_x'] = optics.BETA11
        twiss['beta_y'] = optics.BETA22
        twiss['d_x'] = optics.DISP1
        twiss['d_y'] = optics.DISP1 * 0
        twiss['s'] = optics.S

        return calc_tune_spread(twiss, params)
    
    @staticmethod
    def plot_necktie(ax, Q0, dQ, col):
        radius = 0.5 * np.sqrt(dQ[0]**2 + dQ[1]**2)
        angle  = np.arctan(dQ[1]/dQ[0])
        angle_offset = 18 * np.pi / 180                  

        Qx = Q0[0]-dQ[0]
        Qy = Q0[1]-dQ[1]
        poly = ([Qx, Qy], [Qx+radius*np.cos(angle-angle_offset), Qy+radius*np.sin(angle-angle_offset)],
                [Q0[0], Q0[1]], [Qx+radius*np.cos(angle+angle_offset), Qy+radius*np.sin(angle+angle_offset)])

        ax.plot(Q0[0], Q0[1], '*', color = 'k', ms=13)
        patch = PatchCollection([Polygon(poly, True)], color=col)#, alpha=0.3)
        spread = ax.add_collection(patch)