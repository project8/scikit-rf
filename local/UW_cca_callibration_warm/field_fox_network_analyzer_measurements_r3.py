# =============================================================================
# Keysight Fieldfox N9952A Network Analyzer Remote Control 
#
# Author: Jonathan Tedeschi
#
# Notes: 
# 1) install python Anaconda
# 2) required python modules pip install: scikit-rf, PyVisa
#
# =============================================================================
import pyvisa as visa
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import skrf as rf

class str2(str):
    def __repr__(self):
        return ''.join(('"', super().__repr__()[1:-1], '"'))
    
#%% connecting to field fox
def connect_to_FieldFox_VNA():
    rm=visa.ResourceManager()  
    myFieldFox = rm.open_resource("TCPIP0::192.168.1.206::inst0::INSTR")   
    time.sleep(1)
    myFieldFox.query("*IDN?")
    myFieldFox.write("*RST")
    myFieldFox.write(":INST:SEL 'NA'")
    myFieldFox.write(":INIT:CONT 1")# sets VNA into hold mode (not continuously sweeping)

    return(myFieldFox)

def recal_cal_state(myFieldFox,cal_fname, window_format = 'D4'):
    '''
    window_format = 'D4' (all 4 traces in 1 window), or 'D12_34' (each trace with it's own window')
    '''
    myFieldFox.write(str2('MMEMory:LOAD:STAte '+"'"+cal_fname+"'"))    


    myFieldFox.write("CALC:PAR:COUN 4") 
    myFieldFox.write("DISP:WIND:SPL "+window_format)
    myFieldFox.write(":CALC:PAR1:DEF S11")
    myFieldFox.write(":CALC:PAR2:DEF S21")
    myFieldFox.write(":CALC:PAR3:DEF S12")
    myFieldFox.write(":CALC:PAR4:DEF S22")

    time.sleep(3)
    
def plot_data(path,file_name):
    data_raw=rf.Network(path+file_name+'.s2p',f_unit='Hz')

    plt.figure()
    plt.plot(data_raw.f/1e9,20*np.log10(np.abs(data_raw.s[:,0,0])),label='S11')
    plt.plot(data_raw.f/1e9,20*np.log10(np.abs(data_raw.s[:,1,0])),label='S21')
    plt.plot(data_raw.f/1e9,20*np.log10(np.abs(data_raw.s[:,0,1])),label='S12')
    plt.plot(data_raw.f/1e9,20*np.log10(np.abs(data_raw.s[:,1,1])),label='S22')
    plt.grid()
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.title(data_raw.name,fontsize=16)
    plt.legend(bbox_to_anchor=(1, 0.8))
    plt.xlabel('Frequency (GHz)',fontsize=12)
    plt.ylabel('Magnitude (dB)',fontsize=12)
    plt.tight_layout()
    plt.savefig(path+file_name+'.png',bbox_inches='tight',dpi=200)


def s2p_measurement(myFieldFox,path,fname):

    myFieldFox.write(":INIT:CONT 0")# sets VNA into hold mode (not continuously sweeping)
    myFieldFox.query("*OPC?")#wait/ready command

    #initial formatting
    myFieldFox.write(":INITiate:IMMediate")# que a single sweep
    
    myFieldFox.query("*OPC?")
    time.sleep(2)
    
    '''
    Saving s2p file and transfering it over to host PC
    '''

    myFieldFox.write(":MMEMory:STORe:SNP:DATA 'temp.s2p'")# save s2p file
    temp=myFieldFox.query(":MMEMory:DATA? 'temp.s2p'")# save s2p file
    
    s2p_file = open(path+fname+'.s2p', 'w')

    for line in temp[7:]: # temp[7:] inserted to solve an illegal frequency prompt in s2p file reading. 
        newline = line.rstrip('\n')
        s2p_file.write(newline)
    s2p_file.close()     

    '''
    Plotting data
    '''
    plot_data(path,fname)

    myFieldFox.write(":INIT:CONT 1")# sets VNA into continuous sweep

def save_power_trace_csv(myFieldFox,path,fname,trace_num):
    save_all_trace_data=[]

    myFieldFox.write(":INIT:CONT 0")# sets VNA into hold mode (not continuously sweeping)
    myFieldFox.query("*OPC?")#wait/ready command

    #initial formatting
    myFieldFox.write(":INITiate:IMMediate")# que a single sweep
    myFieldFox.query("*OPC?")#wait/ready command

    f_start=np.float(myFieldFox.query("SENSe:FREQuency:STARt?")[:-1])
    #myFieldFox.query("*OPC?")#wait/ready command

    f_stop=np.float(myFieldFox.query("SENSe:FREQuency:STOP?")[:-1])
    #myFieldFox.query("*OPC?")#wait/ready command

    pts=np.float(myFieldFox.query("SENSe:SWEep:POINts?"))
    #myFieldFox.query("*OPC?")#wait/ready command

    f_array=np.linspace(f_start,f_stop,int(pts))

    save_all_trace_data.append(f_array)    


    myFieldFox.write(str2(":CALCulate:PARameter"+str(trace_num)+":SELect"))
    #myFieldFox.query("*OPC?")#wait/ready command

    
    data=np.float_(list(csv.reader([myFieldFox.query(str2("CALCulate:SELected:DATA:SDATa?"))]))[0])
    myFieldFox.query("*OPC?")#wait/ready command

    data2=np.reshape(data,[int(len(data)/2),2])
    data_complex=data2[:,0]+1j*data2[:,1]
    data_mag=20*np.log10(np.abs(data_complex))
    # data_phase=np.angle(data_complex)*180/np.pi
    save_all_trace_data.append(data_mag)
    # save_all_trace_data.append(data_phase) # power traces do not have phase information

    formatted_data=np.flipud(np.rot90(np.array(save_all_trace_data)))

    comment = 'Hz, dBm'
    np.savetxt(path+fname+'.csv',formatted_data,delimiter=',',header='Power Trace Measurement\n Trace Measured: '+str(trace_num)+'\n'+ comment)

    return(formatted_data)

#%%
if __name__ =='__main__':
# =============================================================================
# Initializing Connection with VNA
# =============================================================================
    file_path='\\\pnl\\projects\\project8\\people\\tedeschi\\Dec2024_measurements\\data\\'   
    #file_path='C:\\Users\\D3X730\\OneDrive - PNNL\\Desktop\\Test\\'
    
    init_fname='sandbox'
    pre_text=''
    post_text=''
    fname=pre_text+init_fname+post_text
    FF=connect_to_FieldFox_VNA()
    
    
    #%%
# =============================================================================
# Recalling VNA calibration and state    
# =============================================================================
    cal_fname='P8_CAL.sta'
    recal_cal_state(FF,cal_fname,'D12_34')
#%%
# =============================================================================
# Collecting data and creating plot
# =============================================================================
    s2p_measurement(FF,file_path,fname)

