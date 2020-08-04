import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(r"/Users/yavas/xrt-master")
#sys.path.append(r"/Users/yavas/xrt-1.3.2")
sys.path.append(r"/Users/yavas/xrt-1.3.0/simulations")

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from numpy import pi, sin, cos, tan, arctan, degrees, radians
from xrt.backends.raycing.oes import OE, ParabolicalMirrorParam
from xrt.backends.raycing.apertures import RectangularAperture
from xrt.backends.raycing.materials import Material, CrystalFromCell, CrystalSi
from xrt.backends.raycing.screens import Screen

from spectrograph4500.elements import LMultilayer, PMirror, LSource
import CommonModules.materials as materials

# details
fig_dir = './fig/'
dat_dir = './dat/'
title = '6Bmono'
nrays = 1E5
repeats = 30

#plot spectrometer in 3D
glow3D = False

#just show angle calculations
just_calc = False


# source 
E0 = 17795.0
dE = 0.02
bsize = (0.015, 0.4)  
bdiv = (1.2E-6, 6.5E-8)

enstep = 0 # gaussian profile same as GeometricSource
#enstep = 0.003  # stepwidth for lines within dE range
#enstep = [0] # single value
#enstep = [0,0.005] # specific values

ensig = 0 # sigma for lines, 1.277 --> 3meV
enmix = 0.5 # basic pv mixing 0-gaussian, 1-lorentzian


# Detector / figures

#ss = ['S0','S1','S2','S3','S4','S5',S6']
#mode = ['X','E','Zp','EZp','D'] #XvsZ, EvsZ, Z'vsZ, EvsZ', detector 

ss = ['S0','S1','S2','S3','S4','S5','S6']  # screen number after the number of reflections
mode = ['X','E','EZp']
fig_bin = 256 #for standard figures

##Andor iKon L
#det_pix = 512
#det_pitch = 50 * 1E-3
#det_dim = det_pitch * det_pix/2


##### crystals
c_1 = materials.Si400_125K
c_2 = materials.Si400_125K
c_3 = materials.Si1284_125K
c_4 = materials.Si1284_125K
c_5 = materials.Si400_125K
c_6 = materials.Si400_125K

#### first pair
alpha_1 = radians(-9.34)
bragg_1 = c_1.get_Bragg_angle(E0) - c_1.get_dtheta(E0, alpha_1)
b_1 = -sin(bragg_1+alpha_1) / sin(bragg_1-alpha_1)


alpha_2 = alpha_1
bragg_2 = bragg_1
b_2 = b_1

##### second pair
alpha_3 = radians(70.0)
bragg_3 = c_3.get_Bragg_angle(E0) - c_3.get_dtheta(E0, alpha_3)
b_3 = -sin(bragg_3+alpha_3) / sin(bragg_3-alpha_3)

alpha_4 = -alpha_3
bragg_4 = c_4.get_Bragg_angle(E0) - c_4.get_dtheta(E0, alpha_4)
b_4 = -sin(bragg_4+alpha_4) / sin(bragg_4-alpha_4)

##### third pair
alpha_5 = -alpha_2
bragg_5 = c_5.get_Bragg_angle(E0) - c_5.get_dtheta(E0, alpha_5)
b_5 = -sin(bragg_5+alpha_5) / sin(bragg_5-alpha_5)

alpha_6 = -alpha_1
bragg_6 = c_6.get_Bragg_angle(E0) - c_6.get_dtheta(E0, alpha_6)
b_6 = -sin(bragg_6+alpha_6) / sin(bragg_6-alpha_6)


#first-1 center: [-0.0006015607504430965, 320000, -5.496318098810899e-05]
#first-1 pitch: 0.09656113002046415
#first-2 center: [-0.0006029475008335704, 320500, 285.7083968739564]
#first-2 pitch: -0.42257620988089417
#second-1 center: [-0.00204656894543015, 322000, 285.70225931776326]
#second-1 pitch: 2.5098499292900245
#second-2 center: [-0.0009856608292031913, 321000, -348.69616231656335]
#second-2 pitch: -2.5098498597105476
#third-1 center: [0.0012078289657950097, 323000, -348.69805306920455]
#third-1 pitch: 0.42257628709586226
#third-2 center: [0.0012052903927464204, 323500, -62.989543490428844]
#third-2 pitch: -0.09656120142949756

##### angular alignment
c1_pitch = bragg_1 + alpha_1
c1_out = bragg_1*2 + c_1.get_dtheta(E0, alpha_1)
c1_pitch += 0.0
c1_out += 0.0 #optimize manually

c2_pitch = -1*(c1_out - bragg_2 -  alpha_2)
c2_out = c1_out - bragg_2*2 - c_2.get_dtheta(E0, alpha_2)
c2_pitch += 1E-6
#c2_pitch += 0.05E-6
c2_out += 0.0 #optimize manually

c3_pitch = -1*(c2_out - bragg_3 -  alpha_3)
c3_out = c2_out - bragg_3*2 - c_3.get_dtheta(E0, alpha_3)
c3_pitch += 5.987E-6
c3_pitch += 0.1E-6  # misalignment
c3_out += -3E-5 #optimize manually

c4_pitch = c3_out + bragg_4 + alpha_4
c4_out = c3_out + bragg_4*2 + c_4.get_dtheta(E0, alpha_4)
c4_pitch += -4.09E-6
#c4_pitch += 0.1E-6  # misalignment
c4_out += 0.6E-5 #optimize manually

c5_pitch = c4_out + bragg_5 + alpha_5
c5_out = c4_out + bragg_5*2 + c_5.get_dtheta(E0, alpha_5)
c5_pitch += -7.4e-6
#c5_pitch += 0.05E-6  # misalignment
c5_out += 0.0 #optimize manually

c6_pitch = -1*(c5_out - bragg_6 -  alpha_6)
c6_out = c5_out - bragg_6*2 - c_6.get_dtheta(E0, alpha_6)
c5_pitch += -1e-6
c6_out += 0.0 #optimize manually

# element spacing
c1_h = 300000
c2_v = 100
c3_h = 1000
c4_v = -200
c5_h = 500
c6_v = 100

#position of every elements should be calculated manually

c1_pos = [0, c1_h, 0]
c2_pos = [0, c1_pos[1] + c2_v/tan(c1_out), c2_v]
c3_pos = [0, c2_pos[1] + c3_h, c2_v]
c4_pos = [0, c3_pos[1] + c4_v/tan(c3_out) , c2_pos[2] + c4_v]
c5_pos = [0, c4_pos[1] + c5_h, c4_pos[2]]
c6_pos = [0, c5_pos[1] + c6_v/tan(c5_out), c5_pos[2] + c6_v]


## Setup simulation

def build_beamline():

    ## beamline
    BL = raycing.BeamLine()

    BL.source = LSource(BL, nrays=nrays, energies=[E0, dE],
                        enstep=enstep, ensig=ensig, enmix=enmix,
                        distx='flat', dx=[-bsize[0]/2,bsize[0]/2],
                        distz='flat', dz=[-bsize[1]/2,bsize[1]/2],
                        distxprime='flat', dxprime=[-bdiv[0],bdiv[0]],
                        distzprime='flat', dzprime=[-bdiv[1],bdiv[1]])

    BL.c1 = OE(BL, center=c1_pos, material=c_1,
                  pitch=c1_pitch, alpha=alpha_1)
    BL.c2 = OE(BL, center=c2_pos, material=c_2, positionRoll=pi,
                  pitch=c2_pitch, alpha=alpha_2)
    BL.c3 = OE(BL, center=c3_pos, material=c_3, positionRoll=pi,
                  pitch=c3_pitch, alpha=alpha_3)
    BL.c4 = OE(BL, center=c4_pos, material=c_4,
                  pitch=c4_pitch, alpha=alpha_4)
    BL.c5 = OE(BL, center=c5_pos, material=c_5,
                  pitch=c5_pitch, alpha=alpha_5)
    BL.c6 = OE(BL, center=c6_pos, material=c_6, positionRoll=pi,
                  pitch=c6_pitch, alpha=alpha_6)

    #BL.slt1 = RectangularAperture(BL, center=slit_pos,
    #                              opening=[-0.2, 0.2, -1, 1])
    
    ## virtual screens
    BL.s0 = Screen(BL, center=c1_pos)
    BL.s1 = Screen(BL, center=c2_pos, z=[0, -sin(c1_out), cos(c1_out)])
    BL.s2 = Screen(BL, center=c3_pos, z=[0, -sin(c2_out), cos(c2_out)])
    BL.s3 = Screen(BL, center=c4_pos, z=[0, -sin(c3_out), cos(c3_out)])
    BL.s4 = Screen(BL, center=c5_pos, z=[0, -sin(c4_out), cos(c5_out)])
    BL.s5 = Screen(BL, center=c6_pos, z=[0, -sin(c5_out), cos(c6_out)])
    BL.s6 = Screen(BL, center=c6_pos + [0,1000,0], z=[0, -sin(c6_out), cos(c6_out)])

    return BL


def run_process(BL):

    beam_r0 = BL.source.shine()
    beam_r1 = BL.c1.reflect(beam_r0)[0]  
    beam_r2 = BL.c2.reflect(beam_r1)[0]
    beam_r3 = BL.c3.reflect(beam_r2)[0]
    beam_r4 = BL.c4.reflect(beam_r3)[0]
    beam_r5 = BL.c5.reflect(beam_r4)[0]
    beam_r6 = BL.c6.reflect(beam_r5)[0]
#

    outDict = {
               'S0': BL.s0.expose(beam_r0),
               'S1': BL.s1.expose(beam_r1),
               'S2': BL.s2.expose(beam_r2),
               'S3': BL.s3.expose(beam_r3),
               'S4': BL.s4.expose(beam_r4),
               'S5': BL.s4.expose(beam_r5),
               'S6': BL.s4.expose(beam_r6),
              }
    
    if glow3D:
        BL.prepare_flow()

    return outDict


def beamline_stats(BL):
    
    c1_y, c1_z = BL.c1.center[1], BL.c1.center[2]
    c2_y, c2_z = BL.c2.center[1], BL.c2.center[2]
    c3_y, c3_z = BL.c3.center[1], BL.c3.center[2]
    c4_y, c4_z = BL.c4.center[1], BL.c4.center[2]
    c5_y, c5_z = BL.c5.center[1], BL.c5.center[2]
    c6_y, c6_z = BL.c6.center[1], BL.c6.center[2]

    c1_a, c2_a = BL.c1.pitch, BL.c2.pitch
    c3_a, c4_a = BL.c3.pitch, BL.c4.pitch
    c5_a, c6_a = BL.c5.pitch, BL.c6.pitch

    disp = [f'c1_y:    {c1_y:12.6f}   c1_z:    {c1_z:12.6f}',
            f'c2_y:    {c2_y:12.6f}   c2_z:    {c2_z:12.6f}',
            f'c3_y:    {c3_y:12.6f}   c3_z:    {c3_z:12.6f}',
            f'c4_y:    {c4_y:12.6f}   c4_z:    {c4_z:12.6f}',
            f'c5_y:    {c5_y:12.6f}   c5_z:    {c5_z:12.6f}',
            f'c6_y:    {c6_y:12.6f}   c6_z:    {c6_z:12.6f}',
             '',
            f'c1_pitch:{degrees(c1_a):12.8f}   c1_out:  {degrees(c1_out):12.8f}',
            f'c2_pitch:{degrees(c2_a):12.8f}   c2_out:  {degrees(c2_out):12.8f}',
            f'c3_pitch:{degrees(c3_a):12.8f}   c3_out:  {degrees(c3_out):12.8f}',
            f'c4_pitch:{degrees(c4_a):12.8f}   c4_out:  {degrees(c4_out):12.8f}',
            f'c5_pitch:{degrees(c5_a):12.8f}   c5_out:  {degrees(c5_out):12.8f}',
            f'c6_pitch:{degrees(c6_a):12.8f}   c6_out:  {degrees(c6_out):12.8f}',
             '',
            f'c1_bragg:{degrees(bragg_1):12.8f}   c2_bragg:{degrees(bragg_2):12.8f}',
            f'c1_alpha:{degrees(alpha_1):12.8f}   c2_alpha:{degrees(alpha_2):12.8f}',
            f'c1_b:    {b_1:12.8f}   c2_b:    {b_2:10.8f}'
             '',            
            f'c3_bragg:{degrees(bragg_3):12.8f}   c4_bragg:{degrees(bragg_4):12.8f}',
            f'c3_alpha:{degrees(alpha_3):12.8f}   c4_alpha:{degrees(alpha_4):12.8f}',
            f'c3_b:    {b_3:12.8f}   c4_b:    {b_4:10.8f}'
                         '',            
            f'c5_bragg:{degrees(bragg_5):12.8f}   c6_bragg:{degrees(bragg_6):12.8f}',
            f'c5_alpha:{degrees(alpha_5):12.8f}   c6_alpha:{degrees(alpha_6):12.8f}',
            f'c5_b:    {b_5:12.8f}   c6_b:    {b_6:10.8f}']
    disp = '\n'.join(disp)

    print(disp+'\n')
    return disp


def define_plots(screen_names, mode=['X','E','Zp','EZp'],
                 step=None, step2=None):

    plots = []
    if step is not None and step2 is not None:
        save_ext = f'_{step:.3f}_{step2:.3f}.png'
    elif step is not None:
        save_ext = f'_{step:.3f}.png'
    else:
        save_ext = '.png'

    for od in screen_names:
        if 'X' in mode:
            plots.append(xrtplot.XYCPlot(od,aspect='auto',
                         yaxis=xrtplot.XYCAxis("z", "mm", bins=fig_bin),
                         xaxis=xrtplot.XYCAxis("x", "mm", bins=fig_bin), 
                         saveName=f'{fig_dir}{title}_X_{od}{save_ext}'))
        if 'E' in mode:
            plots.append(xrtplot.XYCPlot(od,aspect='auto',
                         yaxis=xrtplot.XYCAxis("z", "mm", bins=fig_bin),
                         xaxis=xrtplot.XYCAxis("energy", "eV", bins=fig_bin), 
                         saveName=f'{fig_dir}{title}_E_{od}{save_ext}'))

        if 'Zp' in mode:
            plots.append(xrtplot.XYCPlot(od,aspect='auto',
                         yaxis=xrtplot.XYCAxis("z", "mm", bins=fig_bin),
                         xaxis=xrtplot.XYCAxis("z'", "µrad", bins=fig_bin), 
                         saveName=f'{fig_dir}{title}_Zp_{od}{save_ext}'))

        if 'EZp' in mode:
            plots.append(xrtplot.XYCPlot(od,aspect='auto',
                         yaxis=xrtplot.XYCAxis("energy", "eV", bins=fig_bin),
                         xaxis=xrtplot.XYCAxis("z'", "µrad", bins=fig_bin), 
                         saveName=f'{fig_dir}{title}_EZp_{od}{save_ext}'))
            
    for plot in plots:
        
        sn = plot.saveName[:-4].split('/')[-1]
        od = sn.split('_')[2]
        pm = sn.split('_')[1]           
        
        if od == 'S0':
            if pm == 'E':
                plot.xaxis.limits = [E0-2*dE,E0+2*dE]
                plot.xaxis.offset = E0
        if od == 'S2':
            if pm == 'EZp':
                plot.yaxis.limits = [E0-2*dE,E0+2*dE]
#                plot.xaxis.limits = [-800, 800]
            else:
                plot.yaxis.limits = [-10, 10]
            if pm == 'X':
               plot.xaxis.limits = [-1, 1]
            if pm == 'E':
                plot.xaxis.limits = [E0-2*dE,E0+2*dE]
            if pm == 'Zp':
#                plot.xaxis.limits = [-800, 800]
                pass

        if od == 'S3':
            if pm == 'EZp':
                plot.yaxis.limits = [E0-0.003,E0+0.003]
#                plot.xaxis.limits = [-1200, 1200]                
            else:
                plot.yaxis.limits = [-3.5, 3.5]
            if pm == 'X':
               plot.xaxis.limits = [-7, 7]
            if pm == 'E':
                plot.xaxis.limits = [E0-0.003,E0+0.003]
            if pm == 'Zp':
#                plot.xaxis.limits = [-1200, 1200]
                pass

        if od == 'S4':
            if pm == 'EZp':
                plot.yaxis.limits = [E0-0.003,E0+0.003]
                #plot.xaxis.limits = [-20000, 20000]
            if pm == 'X':
                pass
#               plot.xaxis.limits = [-3.5, 3.5]      
            if pm == 'E':
                plot.xaxis.limits = [E0-0.003,E0+0.003]
                plot.xaxis.offset = E0
#            if pm == 'Zp':
#                plot.xaxis.limits = [-20000, 20000]

        if od == 'S5':
            if pm == 'EZp':
                plot.yaxis.limits = [E0-0.003,E0+0.003]
                #plot.xaxis.limits = [-20000, 20000]
            if pm == 'X':
                pass
#               plot.xaxis.limits = [-3.5, 3.5]      
            if pm == 'E':
                plot.xaxis.limits = [E0-0.003,E0+0.003]
                plot.xaxis.offset = E0
            if pm == 'Zp':
#                plot.xaxis.limits = [-20000, 20000]
                pass

        if od == 'S6':
            if pm == 'EZp':
                plot.yaxis.limits = [E0-0.003,E0+0.003]
                #plot.xaxis.limits = [-20000, 20000]
            if pm == 'X':
                pass
#               plot.xaxis.limits = [-3.5, 3.5]      
            if pm == 'E':
                plot.xaxis.limits = [E0-0.003,E0+0.003]
                plot.xaxis.offset = E0
#            if pm == 'Zp':
#                plot.xaxis.limits = [-20000, 20000]

        if od in ['S0','S1']:
            plot.caxis.limits = [E0-2*dE,E0+2*dE]
        if od in ['S2']:
            plot.caxis.limits = [E0-2*dE,E0+2*dE]
        if od in ['S3','S4']:
            ftr = 1
            #plot.caxis.limits = [(E0-dE)*ftr,(E0+dE)*ftr]
            plot.caxis.limits = [E0-2*dE,E0+2*dE]
            plot.caxis.factor = ftr
            plot.caxis.unit = r"eV"
            plot.caxis.fwhmFormatStr = '%1.4f'
            plot.caxis.label = r"energy"
        plot.caxis.offset = E0
        plot.xaxis.fwhmFormatStr = '%1.6f'
        plot.yaxis.fwhmFormatStr = '%1.6f'
#        plot.caxis.fwhmFormatStr = '%1.6f'
 
#    Plot02 = xrtplot.XYCPlot(
#        beam=r"screen01beamLocal01",
#        xaxis=xrtplot.XYCAxis(
#            label=r"x",
#            fwhmFormatStr=r"%.3f"),
#        yaxis=xrtplot.XYCAxis(
#            label=r"z",
#            fwhmFormatStr=r"%.3f"),
#        caxis=xrtplot.XYCAxis(
#            label=r"energy",
#            unit=r"eV",
#            fwhmFormatStr=r"%.3f"),
#        title=r"Screen between the two crystal")
    
    return plots

# important!
raycing.run.run_process = run_process



if __name__ == '__main__':

    try:
        step = float(sys.argv[1])
        det_ang_offset = radians(step)
    except:
        step = None
    
    try:
        step2 = float(sys.argv[2])
        Estep = step2
    except:
        step2 = None

    BL = build_beamline()
    BL.alignE = E0
    if glow3D:
        BL.glow(scale=[10, 1, 1],centerAt='Screen5')
        sys.exit()
    
    stats = beamline_stats(BL)
    print(stats)
    
    if just_calc:
        sys.exit()

    plots = define_plots(ss, mode, step, step2)
    xrtrun.run_ray_tracing(beamLine=BL, repeats=repeats, plots=plots)

#    if step:
#        fout = f'{dat_dir}{title}_{step:.3f}'
#        if step2 is not None:
#            fout += f'_{step2:.3f}'
#        for plot in plots:
#            sn = plot.saveName[:-4].split('/')[-1]
#            od = sn.split('_')[2]
#            pm = sn.split('_')[1]
#            if od == 'S4' and pm == 'D':
#                y = plot.yaxis.total1D
#                x = plot.yaxis.binEdges[:-1]
#                np.savetxt(fout+'.dat', np.array([x,y]).T, header=stats)
    
    fout = f'{dat_dir}{title}_{abs(degrees(det_ang_offset)):.1f}'
    fout += f'_{enstep:.3f}'
    for plot in plots:
        sn = plot.saveName[:-4].split('/')[-1]
        od = sn.split('_')[2]
        pm = sn.split('_')[1]
        if od == 'S4' and pm == 'D':
            y = plot.yaxis.total1D
            x = plot.yaxis.binEdges[:-1]
            np.savetxt(fout+'.dat', np.array([x,y]).T, header=stats)

