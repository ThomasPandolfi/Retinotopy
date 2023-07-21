from sklearn import preprocessing
import nibabel as nb
from scipy import signal
import numpy as np
from scipy.ndimage import gaussian_filter
import os

#sigma for scipy filter gaussian
# sigma = cutoff(seconds) / sqqrt( 8 * ln(2)) * TR(seconds)
# For 100 seconds cutoff, sigma = 65
# for 90 seconds cutoff, sigma = 59
#

#cutoff frequency of 

stem = os.environ['niiLocation'] + '/'
GauSigTemp = float(os.environ['GauSigTemp'])

if os.environ['upsampled'] == '1':
	loadprefix = 'rUS_'
	TR=0.325
	GauSigTemp = GauSigTemp * 2
else:
	loadprefix = 'r'
	TR=0.65

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return(sos)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.sosfiltfilt(sos, data)
        return(y)

#y = signal.filtfilt(b, a, x, padlen=150)

#import data

filenames = ['comb', '009', '011', '013', '015', '00cwexp', '00wedges', '00rings']#['009f', '011f', '013f', '015f']

filenames = [];

if int(os.environ['isIndividual']) == 1:
	filenames.extend(['009', '011', '013', '015'])
		

if int(os.environ['isCombined']) == 1:
	filenames.append('comb')
	
if int(os.environ['otherCombinations']) == 1:
	filenames.extend(['00cwexp', '00wedges', '00rings'])



bound1 = (24,74)
bound2 = (58,86)
bound3 = (20,56)

bound1 = (0,92)
bound2 = (0,92)
bound3 = (0,92)


saveprefix = 'DF_r'
savesuffix = 'f.nii'

for x in filenames:

	loadsuffix = 'f.nii'
	savesuffix = 'f.nii'
	
		
	currentim = nb.load(stem + x + '/' + loadprefix + x + loadsuffix)
	currentim_shape = currentim.shape
	currentim_affine = currentim.affine
	currentim_header = currentim.header
	currentim = currentim.get_data()#dtype='float32')
	#currentim = np.resize(currentim.get_fdata(), (458,1))
	
	#currentim = peprocessing.StandardScaler().fit(currentim)
	print(3 * '\n')
	print('Filtering: ', x)
	print('Slice out of 92')
	
	for slice1 in range(0, currentim_shape[0]):
		print(slice1, end=" ", flush=True)
		for slice2 in range(0, currentim_shape[1]):
			for slice3 in range(0, currentim_shape[2]):
				oneslice = currentim[slice1,slice2,slice3,:]
				#np.resize(oneslice, (458,1))
				
		
				
				if (slice1 > bound1[0]) and (slice1 < bound1[1]):
					if (slice2 > bound2[0]) and (slice2 < bound2[1]):
						if (slice3 > bound3[0]) and (slice3 < bound3[1]):
			
							#data = np.reshape(oneslice, (568,))
							#data = signal.detrend(oneslice)
							#changed to this on12/03/2022
							data = oneslice
							filt = gaussian_filter(data, GauSigTemp)
							data = data - filt			
							data = butter_bandpass_filter(data, float(os.environ['LFcut']), float(os.environ['HFcut']), 1/TR)
							currentim[slice1,slice2,slice3,:] = data
							
							#data = np.reshape(data, (-1,1))
							#T = preprocessing.StandardScaler().fit(data)
							#print(T)
							#data = np.reshape(data, (-1,1))
							#normalized = preprocessing.normalize([data])
							
				
				
				
	normNii = nb.Nifti1Image(currentim, currentim_affine, currentim_header)
	nb.save(normNii, stem + x + '/' + saveprefix+x+savesuffix)		
	
