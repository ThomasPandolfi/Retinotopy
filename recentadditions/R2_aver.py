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
wdir = '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/cwexp_demeanedfirst/001/sm2_pyprf/results_rh'
wdir_RH = ['/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/RHCWEXP/sm0', '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/RHCCWCONT/sm0']
wdir_LH = ['/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/LHCWEXP/sm0', '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/LHCCWCONT/sm0']

base_fname = 'results_'

def aver_results(hemi_dir, hemi):
	NM = ['R2', 'x_pos', 'y_pos', 'polar_angle', 'SD', 'eccentricity']
	
	for x in NM:
	 	
		A = nb.load(hemi_dir[0] + '/' + base_fname + x + '.nii.gz').get_fdata()
		B = nb.load(hemi_dir[1] + '/' + base_fname + x + '.nii.gz').get_fdata()
		C = nb.load(hemi_dir[0] + '/' + base_fname + 'R2.nii.gz').affine
		AR2 = nb.load(hemi_dir[0] + '/' + base_fname + 'R2.nii.gz').get_fdata()
		BR2 = nb.load(hemi_dir[1] + '/' + base_fname + 'R2.nii.gz').get_fdata()
		#av = np.mean((A,B), axis=0)
		av = np.sum(((A * [AR2 >= BR2]) + (B * [BR2 > AR2])), axis=0)
		im = nb.Nifti1Image(av, C)
		nb.save(im, 'aver_' + hemi + x + '.nii.gz')
		
	
aver_results(wdir_RH, 'rh')
aver_results(wdir_LH, 'lh')
which_plots = [[0,1,2,3], [4,5,6,7]]
which_plots = [[0]]
#runFolders = os.environ['pyR2'][1:-1].split(' ')
runFolders = 'cwexp'
print(runFolders)
sysColorMap = 'rainbow'
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
		specificFolder = 'new'
		sns.set_theme(style="dark")
		

		
		
		
		R2LH = nb.load(wdir_LH + '/' + base_fname + 'R2.nii.gz').get_fdata()
		XPOSLH = nb.load(wdir_LH +  '/' + base_fname + 'x_pos.nii.gz').get_fdata()
		YPOSLH = nb.load(wdir_LH + '/' + base_fname + 'y_pos.nii.gz').get_fdata()
		POLARLH = nb.load(wdir_LH + '/' + base_fname + 'polar_angle.nii.gz').get_fdata()
		#POLARLH = nb.load(wdir_LH + '/' + base_fname + 'polar_angle_VOLlh.nii.gz').get_fdata()
		SDLH = nb.load(wdir_LH + '/' + base_fname + 'SD.nii.gz').get_fdata()
		ECCENLH = nb.load(wdir_LH + '/' + base_fname + 'eccentricity.nii.gz').get_fdata()
		AFFINELH = nb.load(wdir_LH+ '/' + base_fname + 'R2.nii.gz').affine
		
	
		
		R2RH = nb.load(wdir_RH + '/' + base_fname + 'R2.nii.gz').get_fdata()
		XPOSRH = nb.load(wdir_RH +  '/' + base_fname + 'x_pos.nii.gz').get_fdata()
		YPOSRH = nb.load(wdir_RH + '/' + base_fname + 'y_pos.nii.gz').get_fdata()
		POLARRH = nb.load(wdir_RH + '/' + base_fname + 'polar_angle.nii.gz').get_fdata()
		#POLARRH = nb.load(wdir_RH + '/' + base_fname + 'polar_angle_VOLrh.nii.gz').get_fdata()
		SDRH = nb.load(wdir_RH + '/' + base_fname + 'SD.nii.gz').get_fdata()
		ECCENRH = nb.load(wdir_RH + '/' + base_fname + 'eccentricity.nii.gz').get_fdata()
		AFFINERH = nb.load(wdir_RH + '/' + base_fname + 'R2.nii.gz').affine
		
		
		
		
		
		R2 = np.vstack((R2LH, R2RH))
		Xpos = np.vstack((XPOSLH, XPOSRH))
		Ypos = np.vstack((YPOSLH, YPOSRH))
		Polar = np.vstack((POLARLH,POLARRH))
		SD = np.vstack((SDLH, SDRH))
		Eccen = np.vstack((ECCENLH,ECCENRH))
		
		
		
		
		
		
		
		#R2 = nb.load(wdir + '/' + base_fname + 'R2.nii.gz').get_fdata()
		#Xpos = nb.load(wdir +  '/' + base_fname + 'x_pos.nii.gz').get_fdata()
		#Ypos = nb.load(wdir + '/' + base_fname + 'y_pos.nii.gz').get_fdata()
		#Polar = nb.load(wdir + '/' + base_fname + 'polar_angle.nii.gz').get_fdata()
		#SD = nb.load(wdir + '/' + base_fname + 'SD.nii.gz').get_fdata()
		#Eccen = nb.load(wdir + '/' + base_fname + 'eccentricity.nii.gz').get_fdata()
		#affine = nb.load(wdir+ '/' + base_fname + 'R2.nii.gz').affine

		R2means[cc - 1, 0] = np.sum((R2 > significance) * 1)
		R2means[cc - 1, 1] = np.mean(R2[R2 > significance])
		
		
		
		
		#%%
		#All nii are the same size, so simply use np indexing
		X = Xpos[R2>significance]
		Y = Ypos[R2>significance]
		SZ = SD[R2 > significance]
		#SZ = R2 > 0.3
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
		
		
		plt.subplot(2,1,1)
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
			plt.subplot(2,1,2)
			plt.imshow(GRID, interpolation=x, extent=extent, origin="lower", cmap=sysColorMap, norm=colors.LogNorm())	
			#plt.clim(1,200)
			
		
		
		
			
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

print(R2.shape)

print('Where > 0.3')
print(np.sum(R2>0.3))


print('Where > 0.5')
print(np.sum(R2 > 0.5))
	
	

if doweshow == 1:

	plt.show()	

print(R2means)

