# Maser Simulation
# Vineeth Thalakottoor (vineethfrancis.physics@gmail.com)

# ---------- Package
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import time
from scipy.integrate import solve_ivp
import os
import sys
path = os.getcwd()
# ---------- Constants
counter = 0

# ---------- Frequency Analysis: Analyze data 
class Fourier:
    def __init__(self,Mx,My,ax,fig,line1,line2,line3,line4,vline1,vline2,vline3,vline4,text1,text2):
        self.x1, self.y1 = line1.get_data()
        self.x2, self.y2 = line2.get_data()
        self.x3, self.y3 = line3.get_data()
        self.x4, self.y4 = line4.get_data()
        self.dt = self.x1[1] - self.x1[0]
        self.fs = 1.0/self.dt
        self.ax = ax
        self.fig = fig
        self.vline1 = vline1
        self.vline2 = vline2
        self.text1 = text1
        self.vline3 = vline3
        self.vline4 = vline4 
        self.text2 = text2
        self.Mx = Mx
        self.My = My
        self.Mt = Mx + 1j * My

    def button_press(self,event):
        if event.inaxes is ax[0,0]:
            x1, y1 = event.xdata, event.ydata
            global x1in
            x1in = min(np.searchsorted(self.x1, x1), len(self.x1) - 1)
	    
        if event.inaxes is ax[1,0]:
            x3, y3 = event.xdata, event.ydata
            global x3in
            x3in = min(np.searchsorted(self.x3, x3), len(self.x3) - 1)

        if event.inaxes is ax[0,1]:
            x2, y2 = event.xdata, event.ydata
            global x2in
            x2in = x2
            self.vline1.set_xdata([x2in])
            plt.draw()

        if event.inaxes is ax[1,1]:
            x4, y4 = event.xdata, event.ydata
            global x4in
            x4in = x4
            self.vline3.set_xdata([x4in])
            plt.draw()

    def button_release(self,event):
        if event.inaxes is ax[0,0]:
            x1, y1 = event.xdata, event.ydata
            global x1fi
            x1fi = min(np.searchsorted(self.x1, x1), len(self.x1) - 1)
	        
            spectrum = np.fft.fft(self.Mt[x1in:x1fi])
            spectrum = np.fft.fftshift(spectrum)
            freq = np.linspace(-self.fs/2,self.fs/2,spectrum.shape[-1])
            la = ax[0,1].get_lines()
            la[-1].remove()
            line2, = self.ax[0,1].plot(self.x2,np.absolute(self.y2),"-", color='blue')
            #line, = self.ax[0,1].plot(freq,spectrum,"-", color='red')
            line, = self.ax[0,1].plot(freq,np.absolute(spectrum),"-", color='red')
            plt.draw()

        if event.inaxes is ax[1,0]:
            x3, y3 = event.xdata, event.ydata
            global x3fi
            x3fi = min(np.searchsorted(self.x3, x3), len(self.x3) - 1)
            y3 = self.y3
            print(y3.shape)
            window = np.zeros((y3.shape[-1]))
            window[x3in:x3fi] = 1.0
            sig = np.fft.ifftshift(y3*window)
            sig = np.fft.ifft(sig)
            t = np.linspace(0,self.dt*y3.shape[-1],y3.shape[-1])
            lb = ax[1,1].get_lines()
            lb[-1].remove()
            line4, = self.ax[1,1].plot(self.x4,self.y4,'-', color='blue')
            line, = self.ax[1,1].plot(t,sig,"-", color='red')
            plt.draw()

        if event.inaxes is ax[0,1]:
            x2, y2 = event.xdata, event.ydata
            global x2fi
            x2fi = x2
            self.vline2.set_xdata([x2fi])
            self.text1.set_text(f'Freq={abs(x2fi-x2in):1.5f} Hz')
            plt.draw()

        if event.inaxes is ax[1,1]:
            x4, y4 = event.xdata, event.ydata
            global x4fi
            x4fi = x4
            self.vline4.set_xdata([x4fi])
            self.text2.set_text(f'Time={abs(x4fi-x4in):1.5f} s')
            plt.draw()


# ---------- Maxwell Bloch Equation
def MDOT(t,M,Nspins,No_Chem_Peaks,Wx,Wy,Wz,R1,R2,Mo,Rd_Cont,Rd_Phase,FB_Gain,FB_Phase,leakage_amp,leakage_freq,leakage_phase):
    global counter

    Mx = M[0::3]
    My = M[1::3]
    Mz = M[2::3]
	
    # Reshaping Mx,My according to the chemical shift	
    Mxp = np.reshape(Mx,(No_Chem_Peaks,Nspins))
    Myp = np.reshape(My,(No_Chem_Peaks,Nspins))
    Mxaverage = 0
    Myaverage = 0
    
    # Volume Average of Nspins of individual chemical shift
    for i in range(No_Chem_Peaks):
        Mxaverage = Mxaverage + np.average(Mxp[i])
        Myaverage = Myaverage + np.average(Myp[i])
	
    omega_x = Wx
    omega_y = Wy
    omega_z = Wz

    Xi = Rd_Cont

    omega_RD = np.zeros((Nspins*No_Chem_Peaks))
    omega_RD_FB = np.zeros((Nspins*No_Chem_Peaks))
    
    #omega_RD = 1j * Xi * (np.average(Mx) + 1j * np.average(My)) # old (averaging all the transverse components)
    omega_RD = 1j * Xi * (Mxaverage + 1j * Myaverage) # new (averaging transverse components of corresponding chemical shifts)
    

    omega_RD_FB = omega_RD * FB_Gain * np.exp(-1j * FB_Phase)

    
    W_leakage = leakage_amp * np.exp(1j * (leakage_freq * t + leakage_phase)) 

    omega_x = omega_x + omega_RD_FB.real + W_leakage.real
    omega_y = omega_y + omega_RD_FB.imag + W_leakage.imag
    
    # Equation 13 of https://doi.org/10.1063/1.470468
    Mdot = np.zeros((3*Nspins*No_Chem_Peaks))
    Mdot[0::3] = -R2 * Mx - omega_z * My - omega_y * Mz
    Mdot[1::3] = omega_z * Mx - R2 * My + omega_x * Mz
    Mdot[2::3] = omega_y * Mx - omega_x * My - R1 * Mz + R1 * Mo

    counter = counter + 1
    if counter == 10:
        pass
        #print("Mxp shape",Mxp.shape)

    return Mdot

# ---------- Automatic Folder Generator
folder_number = 1 # First folder generated to save data
while True:
    try:
        folder = path + "/" +str(folder_number) # path is current directory
        os.mkdir(folder)
        break
    except OSError as err:
        folder_number = folder_number + 1

save_file = open(folder + "/info.txt", "a")

print(f"folder = {folder}")
save_file.write(f"folder = {folder}")

# ---------- Parameters
T1 = 5.0 # Longitudinal relaxation
T2 = 0.2 # Transverse relaxation
R1 = 1.0/T1
R2 = 1.0/T2 
Mo_all = 0.004 * np.asarray([1.0,1.0]) # Equlibrium magnetization


print(f"T1 = {T1:0.2f}, T2 = {T2:0.2f}, R1 = {R1:0.2f}, R2 = {R2:0.2f}, Mo = {Mo_all}")
save_file.write(f"\nT1 = {T1:0.2f}, T2 = {T2:0.2f}, R1 = {R1:0.2f}, R2 = {R2:0.2f}, Mo = {Mo_all}")

Nspins = 1 #  if Nspins = 1, single spin at chemical shift frequency. If Nspins = 100, 100 spins at chemical shift frequency with inhomogenety. Inhomogenety is defined by parameter "Freq_bin".

RD_rate = 7.0 # Rd rate, Eq 18 of https://doi.org/10.1063/1.470468
Rd_Cont = RD_rate/Mo_all[0] # Rd_Cont is same as Xi in jeener'paper (https://doi.org/10.1063/1.470468), Eq 12
Rd_Phase = (np.pi/180.0) * 0.0 # phase corresponding to rotating spins towards +z 
FB_Gain = 1.0 # Feedback gain of RDCU (Radiation Damping Control Unit) which is set to 1.0 
FB_Phase = (np.pi/180.0) * 180.0 # phase corresponding to rotating spins towards -z with the help of RDCU

print(f"RD_rate = {RD_rate}, Rd_Cont (chi) = {Rd_Cont}, Rd_Phase = {Rd_Phase}, FB_Gain = {FB_Gain}, FB_Phase = {FB_Phase}")
save_file.write(f"\nRD_rate = {RD_rate}, Rd_Cont (chi) = {Rd_Cont}, Rd_Phase = {Rd_Phase}, FB_Gain = {FB_Gain}, FB_Phase = {FB_Phase}")

flip_theta = (np.pi/180.0) * 10.0 # angle between +z and M vector
flip_phi = (np.pi/180.0) * 0.0 # angle between +x and projection of M vector in XY plane

print(f"flip_theta (deg) = {flip_theta*180/np.pi}, flip_phi (deg) = {flip_phi*180/np.pi}")
save_file.write(f"\nflip_theta (deg) = {flip_theta*180/np.pi}, flip_phi (deg) = {flip_phi*180/np.pi}")

Wo_x = 2.0 * np.pi * 0.0 # gamma omega_x, gamma is the gyromagnetic ratio.
Wo_y = 2.0 * np.pi * 0.0 # gamma omega_y
if False:  # gamma omega_z
    # Automation using bifurcation.sh
    Wo_z = 2.0 * np.pi * np.asarray([10,float(sys.argv[1])/100.0]) # The chemical shift frequencies in Hz. Second element in the array can be replaced by chemical shift frequency. float(sys.argv[1])/100.0] will automatically take the frequency from the bash script, if run the file bifurcation.sh, frequency range: 10.5 Hz to 11.7 Hz
else:
    # OR use this for individual simulation
    Wo_z = 2.0 * np.pi * np.asarray([10,10.5])

print(f"Wo (Hz) = {Wo_z/(2.0*np.pi)}")
save_file.write(f"\nWo (Hz) = {Wo_z/(2.0*np.pi)}")

No_Chem_Peaks = Wo_z.shape[-1] 

print(f"No_Chem_Peaks = {No_Chem_Peaks}, No_isochromats = {Nspins}")
save_file.write(f"\nNo_Chem_Peaks = {No_Chem_Peaks}, No_isochromats = {Nspins}")

# Acquisition time
if False:
    if np.max(Wo_z) == 0:
        AQ_fs = 1000.0
        AQ_dt = 1.0/ AQ_fs
        AQ_time = 0.0625
        AQ_points = int(AQ_time/AQ_dt)
    else:
        AQ_fs = 10.0 * np.max(Wo_z)/(2.0*np.pi)
        AQ_dt = 1.0/ AQ_fs
        AQ_time = 0.0625
        AQ_points = int(AQ_time/AQ_dt)
else:
    AQ_points = 400000
    AQ_dt = 0.00025
    #AQ_points = 800000
    #AQ_dt = 0.001
    AQ_time = AQ_points * AQ_dt 

print(f"AQ_points = {AQ_points}, AQ_dt = {AQ_dt}, AQ_time = {AQ_time}")
save_file.write(f"\nAQ_points = {AQ_points}, AQ_dt = {AQ_dt}, AQ_time = {AQ_time}")

leakage_amp = 2.0*np.pi * 0.0 # B1 amplitude (leakage amplitude)
leakage_freq = 2 * np.pi * 0.0 # B1 frequency (leakage frequency)
leakage_phase = (np.pi/180.0) * 0.0 # B1 phase (leakage phase)

print(f"leakage_amp = {leakage_amp}, leakage_freq = {leakage_freq}, leakage_phase = {leakage_phase} ")
save_file.write(f"\nleakage_amp = {leakage_amp}, leakage_freq = {leakage_freq}, leakage_phase = {leakage_phase} ")

Freq_bin = 2.0 * np.pi * 0.05 # Frequency Bin of spins corresponds to Nspins for a given chemical shift frequency.
Wz_band = np.zeros((No_Chem_Peaks,Nspins))

print(f"Freq_bin = {Freq_bin/(2.0*np.pi)}")
save_file.write(f"\nFreq_bin = {Freq_bin/(2.0*np.pi)}")

# Generate Wz inhomogenty frequencies
for i in range(No_Chem_Peaks):
    if (Nspins%2) == 0: # Nspins is even
        Nm = int(Nspins/2)
        Wz_band[i] = np.linspace(Wo_z[i] - Nm * Freq_bin, Wo_z[i] + Nm * Freq_bin, Nspins, endpoint=False, dtype= np.float64)
    else:
        Nm = int((Nspins-1)/2) # Nspins is odd
        Wz_band[i] = np.linspace(Wo_z[i] - Nm * Freq_bin, Wo_z[i] + Nm * Freq_bin, Nspins, endpoint=True, dtype= np.float64)

print(f"Wz_band_shape = {Wz_band.shape}")
save_file.write(f"\nWz_band_shape = {Wz_band.shape}")
print(f"Wz_band = {Wz_band/(2.0*np.pi)}")
save_file.write(f"\nWz_band = {Wz_band/(2.0*np.pi)}")
print("Band Widht = ", (Wz_band[0,-1]-Wz_band[0,0])/(2.0*np.pi))
save_file.write(f"\nBand Widht = {(Wz_band[0,-1]-Wz_band[0,0])/(2.0*np.pi)}")

Wz_band = np.reshape(Wz_band,Nspins*No_Chem_Peaks)

Wx = Wo_x 
Wy = Wo_y
Wz = Wz_band

M = np.zeros((No_Chem_Peaks,3*Nspins))
Mo = np.zeros((No_Chem_Peaks,Nspins))
for i in range(No_Chem_Peaks):
    M[i, 0::3] = np.absolute(Mo_all[i]) * np.sin(flip_theta) * np.cos(flip_phi) 
    M[i, 1::3] = np.absolute(Mo_all[i]) * np.sin(flip_theta) * np.sin(flip_phi) 
    M[i, 2::3] = np.absolute(Mo_all[i]) * np.cos(flip_theta) 
    Mo[i] = Mo_all[i]

M = np.reshape(M,3*Nspins*No_Chem_Peaks)
Mo = np.reshape(Mo,Nspins*No_Chem_Peaks)
print(M.shape)
print(Mo.shape)
tpoints = np.linspace(0.0, AQ_time, AQ_points, endpoint=True)

# python ODE solver

Method = 'DOP853' # or 'LSODA'

print(f"Solve IVP method = {Method}")
save_file.write(f"\nSolve IVP method = {Method}")

start_time = time.time()
Msol = solve_ivp(MDOT,[0,AQ_time],M,method=Method,t_eval=tpoints,args=(Nspins,No_Chem_Peaks,Wx,Wy,Wz,R1,R2,Mo,Rd_Cont,Rd_Phase,FB_Gain,FB_Phase,leakage_amp,leakage_freq,leakage_phase),atol = 1e-10, rtol = 1e-10)
end_time = time.time()
timetaken = end_time - start_time
print(f"Total time = {timetaken}")
save_file.write(f"\nTotal time = {timetaken}")

save_file.close()

tpoints, Mpoints = Msol.t, Msol.y

Mx = np.sum(Mpoints[0::3,:], axis=0) # Adding all Mx components of all spins
My = np.sum(Mpoints[1::3,:], axis=0) # Adding all My components of all spins
Mz = np.sum(Mpoints[2::3,:], axis=0) # Adding all Mz components of all spins
Mabs = np.sqrt(Mx**2 + My**2)

if True: # Save average data
	np.savetxt(folder + '/datat.txt',tpoints, fmt='%1.10e')
	np.savetxt(folder + '/dataMx.txt',Mx, fmt='%1.10e')
	np.savetxt(folder + '/dataMy.txt',My, fmt='%1.10e')
	np.savetxt(folder + '/dataMz.txt',Mz, fmt='%1.10e')
if False: # save data of Spin 1
	np.savetxt(folder + '/datat1.txt',tpoints, fmt='%1.10e')
	np.savetxt(folder + '/dataMx1.txt',Mpoints[0,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMy1.txt',Mpoints[1,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMz1.txt',Mpoints[2,:], fmt='%1.10e')
if False: # save data of Spin 2
	np.savetxt(folder + '/datat2.txt',tpoints, fmt='%1.10e')
	np.savetxt(folder + '/dataMx2.txt',Mpoints[3,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMy2.txt',Mpoints[4,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMz2.txt',Mpoints[5,:], fmt='%1.10e')
if False: # save data of Spin 3
	np.savetxt(folder + '/datat3.txt',tpoints, fmt='%1.10e')
	np.savetxt(folder + '/dataMx3.txt',Mpoints[6,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMy3.txt',Mpoints[7,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMz3.txt',Mpoints[8,:], fmt='%1.10e')
if False: # save data of Spin 4
	np.savetxt(folder + '/datat4.txt',tpoints, fmt='%1.10e')
	np.savetxt(folder + '/dataMx4.txt',Mpoints[9,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMy4.txt',Mpoints[10,:], fmt='%1.10e')
	np.savetxt(folder + '/dataMz4.txt',Mpoints[11,:], fmt='%1.10e')
	
dt = tpoints[1] - tpoints[0]
fs = 1.0/dt
signal = Mx + 1j * My

spectrum = np.fft.fft(signal)
spectrum = np.fft.fftshift(spectrum)
freq = np.linspace(-fs/2,fs/2,signal.shape[-1])

# --------- Plotting time and Mx/My/Mz
if True:
	rc('font', weight='bold')
	fig = plt.figure(1,constrained_layout=True, figsize=(15, 5))
	spec = fig.add_gridspec(1, 1)

	ax1 = fig.add_subplot(spec[0, 0])

	ax1.plot(tpoints,Mx,linewidth=3.0,color='blue',label = "Mx")
	ax1.plot(tpoints,My,linewidth=3.0,color='green',label = "My")

	ax1.set_xlabel(r'Time (s)', fontsize=25, color='black',fontweight='bold')
	ax1.set_ylabel(r'$M_{T}$ (AU)', fontsize=25, color='blue',fontweight='bold')
	ax1.legend(fontsize=25,frameon=False)
	ax1.tick_params(axis='both',labelsize=14)
	ax1.grid(True, linestyle='-.')
	#ax1.set_xlim(0,0.02)
	#ax1.text(0.05, 200000, '(a)', ha='center', fontsize=25, color='black',fontweight='bold')

	ax10 = ax1.twinx()
	ax10.plot(tpoints,Mz,linewidth=3.0,color='red',label = "Mz")
	ax10.set_xlabel(r'Time (s)', fontsize=30, color='black',fontweight='bold')
	ax10.set_ylabel(r'$M_{Z}$ (AU)', fontsize=30, color='red',fontweight='bold')
	ax10.legend(fontsize=30,frameon=False)
	ax10.tick_params(axis='both',labelsize=20)
	#ax10.grid(False, linestyle='-.')
	#ax10.set_xlim(0,0.02)
	# plt.savefig('block_relax_hypo_2b.pdf',bbox_inches='tight')
	plt.savefig(folder + '/pic1.pdf',bbox_inches='tight')

# --------- Plotting time and Mabs/Mz
if True:
	fig = plt.figure(2,constrained_layout=True, figsize=(15, 5))
	spec = fig.add_gridspec(1, 1)

	ax1 = fig.add_subplot(spec[0, 0])

	ax1.plot(tpoints,Mabs,linewidth=3.0,color='black',label = "Mabs" )
	ax1.set_xlabel(r'Time (s)', fontsize=25, color='green',fontweight='bold')
	ax1.set_ylabel(r'$M_{T}$ (AU)', fontsize=25, color='black',fontweight='bold')
	ax1.legend(fontsize=25,frameon=False)
	ax1.tick_params(axis='both',labelsize=14)
	ax1.grid(True, linestyle='-.')
	#ax1.set_xlim(0,0.02)
	#ax1.text(0.05, 200000, '(a)', ha='center', fontsize=25, color='black',fontweight='bold')

	ax10 = ax1.twinx()
	ax10.plot(tpoints,Mz,linewidth=3.0,color='red',label = "Mz")
	ax10.set_xlabel(r'Time (s)', fontsize=30, color='green',fontweight='bold')
	ax10.set_ylabel(r'$M_{Z}$ (AU)', fontsize=30, color='red',fontweight='bold')
	ax10.legend(fontsize=30,frameon=False)
	ax10.tick_params(axis='both',labelsize=20)
	#ax10.grid(False, linestyle='-.')
	#ax10.set_xlim(0,0.02)
	# plt.savefig('block_relax_hypo_2b.pdf',bbox_inches='tight')
	plt.savefig(folder + '/pic2.pdf',bbox_inches='tight')

# --------- Plotting Spectrum
if True:
	fig = plt.figure(3,constrained_layout=True, figsize=(15, 5))
	spec = fig.add_gridspec(1, 1)

	ax1 = fig.add_subplot(spec[0, 0])

	ax1.plot(freq,spectrum,linewidth=3.0,color='black',label = "Mabs" )
	ax1.set_xlabel(r'Frequency (Hz)', fontsize=25, color='green',fontweight='bold')
	ax1.set_ylabel(r'Spectrum (AU)', fontsize=25, color='black',fontweight='bold')
	ax1.legend(fontsize=25,frameon=False)
	ax1.tick_params(axis='both',labelsize=14)
	ax1.grid(True, linestyle='-.')
	ax1.set_xlim(8,13)
	#ax1.text(0.05, 200000, '(a)', ha='center', fontsize=25, color='black',fontweight='bold')
	plt.savefig(folder + '/pic4.pdf',bbox_inches='tight')
	
# --------- Fourier Plotting
if True:
	fig, ax = plt.subplots(2,2)
	plt.figure(4,figsize=(20, 20))

	line1, = ax[0,0].plot(tpoints,signal,"-", color='green')
	ax[0,0].set_xlabel("time [s]")
	ax[0,0].set_ylabel("signal" )
	ax[0,0].grid()

	vline1 = ax[0,1].axvline(color='k', lw=0.8, ls='--')
	vline2 = ax[0,1].axvline(color='k', lw=0.8, ls='--')
	text1 = ax[0,1].text(0.0, 0.0, '', transform=ax[0,1].transAxes)
	line2, = ax[0,1].plot(freq,spectrum,"-", color='green')
	ax[0,1].set_xlabel("Frequency [Hz]")
	ax[0,1].set_ylabel("spectrum" )
	#ax[0,1].set_xlim(-40,40)
	ax[0,1].grid()

	line3, = ax[1,0].plot(freq,spectrum,"-", color='green')
	ax[1,0].set_xlabel("Frequency [Hz]")
	ax[1,0].set_ylabel("spectrum" )
	#ax[1,0].set_xlim(-40,40)
	ax[1,0].grid()

	vline3 = ax[1,1].axvline(color='k', lw=0.8, ls='--')
	vline4 = ax[1,1].axvline(color='k', lw=0.8, ls='--')
	text2 = ax[1,1].text(0.0, 0.0, '', transform=ax[1,1].transAxes)
	line4, = ax[1,1].plot(tpoints,signal,"-", color='green')
	ax[1,1].set_xlabel("time [s]")
	ax[1,1].set_ylabel("signal" )
	ax[1,1].grid()
	plt.savefig(folder + '/pic3.pdf',bbox_inches='tight')

	fourier = Fourier(Mx,My,ax,fig,line1,line2,line3,line4,vline1,vline2,vline3,vline4,text1,text2)
	fig.canvas.mpl_connect("button_press_event",fourier.button_press)
	fig.canvas.mpl_connect("button_release_event",fourier.button_release)

plt.show()

# --------- Bloch Sphere
if True:
	S_phi = np.linspace(0, np.pi, 20)
	S_theta = np.linspace(0, 2*np.pi, 20)
	S_phi, S_theta = np.meshgrid(S_phi, S_theta)
	S_x = np.sum(Mo_all) * np.sin(S_phi) * np.cos(S_theta)
	S_y = np.sum(Mo_all) * np.sin(S_phi) * np.sin(S_theta)
	S_z = np.sum(Mo_all) * np.cos(S_phi)

	tlim1 = 0 #-10000
	tlim2 = -1
	ax = plt.figure(5,figsize=(30,30)).add_subplot(projection='3d')
	ax.plot_wireframe(S_x,S_y,S_z, color="cyan",linewidth=1.0)

	ax.plot(Mx[tlim1:tlim2],My[tlim1:tlim2],Mz[tlim1:tlim2], color="black",linewidth=1.0)
	#ax.plot(Mpoints[0,tlim1:tlim2],Mpoints[1,tlim1:tlim2],Mpoints[2,tlim1:tlim2], color="green",linewidth=1.0) 
	#ax.plot(Mpoints[3,tlim1:tlim2],Mpoints[4,tlim1:tlim2],Mpoints[5,tlim1:tlim2], color="blue",linewidth=1.0) 
	ax.view_init(10, 20)
	ax.set_xlabel(r'My', fontsize=14, color='black',fontweight='bold')
	ax.set_ylabel(r'Mx', fontsize=14, color='black',fontweight='bold')
	ax.set_zlabel(r'Mz', fontsize=14, color='black',fontweight='bold')
	ax.tick_params(axis='both',labelsize=10)
	ax.grid(True, linestyle='-.')
	plt.savefig(folder + '/bloch.png',transparent=True,bbox_inches='tight')

	plt.show()
