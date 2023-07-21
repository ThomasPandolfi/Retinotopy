#sample R2 data convolved with Polar/Eccen/X and Y position
#Displays significant voxels in a plot
import os
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nb
import sys
from scipy.special import rel_entr
markerSize = 16

#Working directory with nii result files
wdir = os.environ['niiLocation']
which_plots = [[0,1,2,3], [4,5,6,7]]
runFolders = os.environ['pyR2'][1:-1].split(' ')
print(runFolders)
sysColorMap = os.environ['sysColorMap']
import matplotlib
import seaborn as sns
#sysColorMap='
sns.set_theme(style="dark")
#os.environ['GauSigSpat']
#run_extension =  os.environ['GauSigSpat'] + '_' + os.environ['GauSigTemp'] + '_' + os.environ['LFcut'] + '_' + os.environ['HFcut'] + '_' + os.environ['HighResDimension']

nbins = 20
frac_width = 0.8
width = 360 * frac_width / nbins
	
doweshow = 1	

#analysis_combinations = [x.split('_') for x in os.listdir() if '_' in x]

#run_extension = analysis_combinations[jkl][
run_extension = sys.argv[3]

try:

	if sys.argv[3]:
		run_extension = sys.argv[3]
except:
	print('Using default OS variables for R2 analysis')

#/Desktop/SUBJECT01/MRI/Post/analysis/main/pyprf/00cwexp/6_55_0.005_0.05_300
#base file name of the nii files
#usually in the form of ______eccentricity.nii.gz or ________R2.nii.gz
#	Whatever the _____ is

base_fname = 'results_'
import sys
#R2 significance. Only take R2 above a certain amount
significance = 0.3


try:
	significance = float(sys.argv[1])
	basic=float(sys.argv[2])
	
except:
	print('At least one sys argument not provided, defaulting on just displaying normal data')
	

	
R2means = np.zeros((8,2))

XYCALL = np.zeros((23, 23, 8))

isPolar = 0
isEccen = 0
isSD = 0

#plt.figure(5)
cc = 0

for figureSeparation in range(0,len(which_plots)):
	
	plt.figure(figureSeparation)
	#fig, axes = plt.subplots(2, 4)
	for i, plotNumber in enumerate(which_plots[figureSeparation]):
		
		plt.figure(figureSeparation)
		cc += 1
		specificFolder = runFolders[plotNumber]
		sns.set_theme(style="dark")
		R2 = nb.load(wdir + '/pyprf/' + specificFolder + '/' + run_extension + '/' + base_fname + 'R2.nii.gz').get_fdata()
		Xpos = nb.load(wdir + '/pyprf/' + specificFolder + '/' + run_extension + '/' + base_fname + 'x_pos.nii.gz').get_fdata()
		Ypos = nb.load(wdir + '/pyprf/' + specificFolder + '/' + run_extension + '/' + base_fname + 'y_pos.nii.gz').get_fdata()
		Polar = nb.load(wdir + '/pyprf/' + specificFolder + '/' + run_extension + '/' + base_fname + 'polar_angle.nii.gz').get_fdata()
		SD = nb.load(wdir + '/pyprf/' + specificFolder + '/' + run_extension + '/' + base_fname + 'SD.nii.gz').get_fdata()
		Eccen = nb.load(wdir + '/pyprf/' + specificFolder + '/' + run_extension + '/' + base_fname + 'eccentricity.nii.gz').get_fdata()
		affine = nb.load(wdir+ '/pyprf/' + specificFolder + '/' + run_extension + '/' + base_fname + 'R2.nii.gz').affine

		R2means[cc - 1, 0] = np.sum((R2 > significance) * 1)
		R2means[cc - 1, 1] = np.mean(R2[R2 > significance])
		
		
		if specificFolder == '009':
			pol1 = Polar[R2 > significance]
		elif specificFolder ==  '011':
			pol2 = Polar[R2 > significance]
		elif specificFolder == '013':
			ecen1 = Eccen[R2 > significance]
		elif specificFolder == '015':
			ecen2 = Eccen[R2 > significance]
			
		elif specificFolder == '00rings':
			ecen3 = Eccen[R2 > significance]
		elif specificFolder == '00wedges':
			pol3 = Polar[R2 > significance]

		elif specificFolder == '00cwexp':
			pol4 = Polar[R2 > significance]
			ecen4 = Eccen[R2 > significance]
		elif specificFolder == 'comb':
			pol5 = Polar[R2 > significance]
			ecen5 = Eccen[R2 > significance]
		
		
		#%%
		#All nii are the same size, so simply use np indexing
		X = Xpos[R2>significance]
		Y = Ypos[R2>significance]
		SZ = SD[R2 > significance]
		SZColora = R2[R2 > significance]
		SZColoration = np.transpose(np.tile(SZColora,(3,1)))

        
        
        
        
		CombinedXY = np.transpose(np.vstack((X,Y)))
		xy_pair_dict = dict()
		Cmapper = np.array([])
		tup = dict()
		for x in CombinedXY:
			if tuple(x) in tup:
				tup[tuple(x)] += 1
			else:
				tup[tuple(x)] = 1
#		print(tup)

		#x values, yvalues, and colormap values
		
		Cmapper = np.array(list(tup.keys()))
		XYC = np.transpose(np.vstack((Cmapper[:,0],Cmapper[:,1], np.array(list(tup.values())))))
		#plt.figure(5)
		#plt.figure(0)
		
		
		plt.subplot(2,len(which_plots[figureSeparation]),i + 1)
		#First color plot
		#sysColorMap='seismic'
		plt.scatter(XYC[:,0], XYC[:,1], c=XYC[:,2], norm=matplotlib.colors.LogNorm(), cmap=sysColorMap, s=markerSize)
		#plt.scatter(XYC[:,0], XYC[:,1], c=XYC[:,2], cmap=sysColorMap, s=markerSize)
		plt.ylim([-12,12])
		plt.xlim([-12,12])
		plt.title(specificFolder)
		#smf=plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(sysColorMap))
		#sm.set_clim(vmin=0, vmax=1)
		#plt.colorbar(smf)
		
		
		#plt.colorbar()
		#sns.histplot(x=XYC[:,0], y=XYC[:,1], bins=15, pthresh=.02, cmap="mako")
		#sns.kdeplot(x=XYC[:,0], y=XYC[:,1], levels=10, color="b", linewidths=1)
		#print(XYC[:,0], XYC[:,1])
		#sns.JointGrid(x=XYC[:,0], y=XYC[:,1], space=0)
		#sns.kdeplot(x=XYC[:,0], y=XYC[:,1], fill=True, clip=((-11, 11), (-11, 11)), thresh=0, levels=100, cmap="rocket")
		#sns.histplot(x=XYC[:,0], y=XYC[:,1], color="#03051A", alpha=1, bins=25)
		#plt.colorbar()
		#sns.jointplot(x=XYC[:,0], y=XYC[:,1], kind="hex", color="#4CB391")
		#sns.histplot(x=XYC[:,0], y=XYC[:,1], binwidth=(1, 1), cbar=True, ax=axes[0,i])
		#plt.clim(1,200)  
		
		
		
            
            
            
            
		
		#plt.xlabel(['X Coor'])
		#plt.ylabel(['Y Coord, degrees'])
		

		####### Undersampling the same image
		#if you have 23 points
		for xcoord in range(-11, 1 + 11):
			for ycoord in range(-11, 11 + 1):
				try:
					if tup[(xcoord, ycoord)] != 0:
						pass
				except:
					tup[(xcoord, ycoord)] = 1
					
		GRID = np.zeros((23,23))
		for x in range(-11, 12):
			for y in range(-11, 12):
				GRID[y + 11][x + 11] = tup[(x, y)]
		extent = [-11, 11, -11, 11]

		methods = ['bicubic']

	   	
	   	
		for x in methods:
			plt.subplot(2,len(which_plots[figureSeparation]),i + len(which_plots[figureSeparation]) + 1)
			plt.imshow(GRID, interpolation=x, extent=extent, origin="lower", cmap=sysColorMap, norm=colors.LogNorm())	
			#plt.clim(1,200)
			
		
		plt.suptitle(sys.argv[3])
		
			
		#print(GRID.shape)	
		XYCALL[:,:,i - 1] = GRID	
			
		#trying out some seaborn stuff
			
		#plt.figure(2020 + figureSeparation)
		#plt.subplot(1,4,i+1)
		#sns.relplot(x=XYC[:,0], y=XYC[:,1], hue=XYC[:,2], sizes=(40, 40), alpha=.5, palette="muted", height=6)
		
		
		
		#################################
		#### FOR THE BLACK OVERALAY
		#################################
		
		#plt.subplot(3,len(which_plots[figureSeparation]),i + len(which_plots[figureSeparation]) + 1 + len(which_plots[figureSeparation]))
		
		
		#plt.scatter(Eccen[R2 > significance] * np.cos(Polar[R2>significance]),Eccen[R2 > significance] * np.sin(Polar[R2 > significance]), s=20, alpha=0.3, edgecolors='black', facecolors='none')
		
		
		
		
		#print(SZColora)
		#print(SZColoration)
		
		
		
		
		#For the signifiance heatmaps
		sns.set_theme(style="white")
		
		plt.figure((figureSeparation+1) * 22)
		plt.subplot(1,4, i + 1)
		c = plt.cm.plasma_r(SZColora)
		BNM = plt.scatter(X,Y, s=(2 * SZ)**2, facecolors="None", edgecolors=c, lw=SZColora * 2, alpha=1)	
		
		sm=plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('plasma_r'))
		sm.set_clim(vmin=0, vmax=1)
		
		plt.colorbar(sm)
		plt.title(specificFolder)
			#
		plt.ylim([-11,11])
		plt.xlim([-11,11])	
							 
			
	plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)
	
	try:
	
		plt.suptitle(sys.argv[3])
	except:
		print('nothing for arg 3')
		
	#plt.figure(89)
	#plt.hist(R2[R2 > significance], bins = nbins, align='mid', width=width)
	#plt.title(str(sum(R2 > significance)))
	
	

#plt.show()
print(XYCALL)
print(XYCALL.shape)
	#%%
	#unique, counts = np.unique(CombinedXY, return_counts=True)
	#Cmapper = dict(zip(unique, counts))
plt.figure(92929)
plt.scatter(X,Y, s=(4*SZ)**2, alpha=0.3, edgecolors='black', facecolors='none')
plt.ylim([-11,11])
plt.xlim([-11,11])
plt.xlabel('Degrees (horizontal)')
plt.ylabel('Degrees (vertical)')
 



plt.figure(92930)
#print(pol5)
#print(ecen5)
plt.scatter(ecen5 * np.cos(pol5) ,ecen5 * np.sin(pol5), s=20, alpha=0.3, edgecolors='black', facecolors='none')
plt.ylim([-11,11])
plt.xlim([-11,11])
#for combo in range(0, 8):
#	paircombination = np.array([])
#	for x in range(0, 23):
#		for y in range(0, 23):
#			if XYCALL[:,:,combo] == 0:
							
	
	
sns.set_theme(style="whitegrid")


#cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
#g = sns.relplot(data=planets,
    #x="distance", y="orbital_period", hue="year", palette=cmap, sizes=(10, 200))
#g.set(xscale="log", yscale="log")
#g.ax.xaxis.grid(True, "minor", linewidth=.25)
#g.ax.yaxis.grid(True, "minor", linewidth=.25)
#g.despine(left=True, bottom=True)	
	
#numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)
#plt.scatter(wines['fixed acidity'], wines['alcohol'], s=wines['residual sugar']*25, 
 #           alpha=0.4, edgecolors='w')

#plt.xlabel('Fixed Acidity')
#plt.ylabel('Alcohol')
#plt.title('Wine Alcohol Content - Fixed Acidity - Residual Sugar',y=1.05)

	
	
plt.figure(67)
plt.subplot(2,1,1)
angles = (pol1 * 180 / 3.1415) + 180	
#plt.subplot(1,3,1)
plt.hist(angles, bins = nbins, alpha=0.5, align='mid', width=width, color='blue', label='009')

A = np.histogram(angles, bins=18, range=(0,360))
Abins = A[1]
Anums = A[0] / sum(A[0])
#print(Anums)
#print(A)
plt.title('Polar Angles 009')

angles = (pol2 * 180 / 3.1415) + 180
#plt.subplot(1,3,2)
plt.hist(angles, bins = nbins, alpha=0.5, align='mid', width=width, color='green', label='011')
B = np.histogram(angles, bins=18, range=(0,360))
Bbins = B[1]
Bnums = B[0] / sum(B[0])


plt.title('Polar Angles 011')

angles = (pol3 * 180 / 3.1415) + 180
#plt.subplot(1,3,3)
plt.hist(angles, bins = nbins, alpha=0.5, align='mid', width=width, color='red', label='combined')
plt.title('Polar Angles wedges')
C = np.histogram(angles, bins=18, range=(0,360))
Cbins = C[1]
Cnums = C[0] / sum(C[0])

plt.legend(loc='upper right')






'''
##################




ECCENTRICITY






####################3
'''









width = 15 * frac_width / nbins

plt.subplot(2,1,2)	
plt.hist(ecen1, bins = nbins, alpha=0.5, align='mid', width=width, color='blue', label='013')
plt.title('Eccentricity (degrees) 013')
#print(ecen1)
D = np.histogram(ecen1, bins=18, range=(0,16))
Dbins = D[1]
Dnums = D[0] / sum(D[0])




#plt.subplot(1,3,2)	
plt.hist(ecen2, bins = nbins, alpha=0.5, align='mid', width=width, color='green', label='015')
plt.title('Eccentricity (degrees) 015')
E = np.histogram(ecen2, bins=18, range=(0,16))
Ebins = E[1]
Enums = E[0] / sum(E[0])

	
#plt.subplot(1,3,3)	
plt.hist(ecen3, bins = nbins, alpha=0.5, align='mid', width=width, color='red', label='combined')
plt.title('Eccentricity (degrees) rings')	
F = np.histogram(ecen3, bins=18, range=(0,16))
Fbins = F[1]
Fnums = F[0] / sum(F[0])


	
Gr = np.histogram(ecen4, bins=18, range=(0,16))
Grbins = Gr[1]
Grnums = Gr[0] / sum(Gr[0])

angles = (pol4 * 180 / 3.1415) + 180
Gw = np.histogram(angles, bins=18, range=(0,360))
#print(Gw)
Gwbins =Gw[1]
Gwnums = Gw[0] / sum(Gw[0])

Hr = np.histogram(ecen5, bins=18, range=(0,16))
Hrbins = Hr[1]
Hrnums = Hr[0] / sum(Hr[0])

angles = (pol5 * 180 / 3.1415) + 180
Hw = np.histogram(angles, bins=18, range=(0,360))
Hwbins = Hw[1]
Hwnums = Hw[0] / sum(Hw[0])





#Compute KL divergence
#Q will be the standard rings or wedges
#P will  be the individual runs
KL_A2C = 0
KL_B2C = 0
KL_D2F = 0
KL_E2F = 0
KL_G2H = 0
KL_Gw2Hw = 0
KL_Gr2Hr = 0

#A B D E are the wedges and rings respectively
# C and F are the combined wedges and rings respectively
#G is the CWEXP
#H is the combined
#print(Gwnums, Hwnums)
for i in range(0, 18):
	KL_A2C += Anums[i] * np.log(Anums[i] / Cnums[i])
	KL_B2C += Bnums[i] * np.log(Bnums[i] / Cnums[i])

	KL_D2F += Dnums[i] * np.log(Dnums[i] / Fnums[i])
	KL_E2F += Enums[i] * np.log(Enums[i] / Fnums[i])
	
	KL_Gw2Hw += Gwnums[i] * np.log(Gwnums[i] / Hwnums[i])
	KL_Gr2Hr += Grnums[i] * np.log(Grnums[i] / Hrnums[i])
	
	#KL_E2F += Enums[i] * np.log(Enums[i] / Fnums[i])


#print([KL_A2C, KL_B2C, KL_D2F, KL_E2F, KL_Gw2Hw, KL_Gr2Hr]) 

#print(KL_A2C, KL_B2C, KL_D2F, KL_E2F)

#Also, Q will be the Combined runs
#P will be the cwexpf
#('cw' 'ccw' 'exp' 'cont' 'combined' 'cwccw' 'expcont' 'cwexp')

print(R2.shape)

print('Where > 0.3')
print(np.sum(R2>0.3))


print('Where > 0.5')
print(np.sum(R2 > 0.5))
	

if doweshow == 1:

	plt.show()	

print(R2means)

