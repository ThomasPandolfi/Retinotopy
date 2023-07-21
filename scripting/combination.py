#python
import nibabel as nb
import gc
import numpy as np
import os
#file_names = ['cont', 'exp', 'ccw', 'cw']
file_names = os.environ['actualOrderPY'].split(',')



if os.environ['upsampled'] == 1:

	file_numbers_prefix = 'Us_'#norm_detrend_'
else:
	file_numbers_prefix = ''
	
file_numbers = ['009f.nii', '011f.nii', '013f.nii', '015f.nii']#, '017f.nii']


#m = 0
#if len(file_numbers) == 5:#
#	m = 1


#%%
if file_numbers_prefix != '':
	files = [file_numbers_prefix + file_numbers[x] for x in range(0,len(file_numbers))]
else:
	
	files = file_numbers

#%%
#stem_folder = '/home/thomaszeph/Desktop/Useful_Functions/raw_nii/US/'
stem_folder = os.environ['niiLocation']



ims = []
affines = []
m=0

#%%
#keep memory tight


if int(os.environ['isCombined']) == 1:

	print('################## Making overall combination ####################')
	A = nb.load(stem_folder + '/' + files[0][0:3] + '/' + files[0])
	shape = A.shape
	affine = A.affine
	header = A.header


	A = A.get_data()
	
	if m ==1:
	
		preallo = np.zeros((shape[0],shape[1],shape[2],5*shape[3]), dtype='int16')
	else:
		preallo = np.zeros((shape[0],shape[1],shape[2],4*shape[3]), dtype='int16')
		
	print('Preallocated zero array created')	
	preallo[:,:,:,0:shape[3]] = A
	del A

	
	B = nb.load(stem_folder + '/' + files[1][0:3] + '/' + files[1]).get_data()#dtype='float32')
	preallo[:,:,:,shape[3] :2 * shape[3]] = B
	del B
	
	print('Second block added')
	C = nb.load(stem_folder + '/' + files[2][0:3] + '/' + files[2]).get_data()#dtype='float32')
	preallo[:,:,:,2 * shape[3] :3 * shape[3]] = C
	del C

	print('Third block added')
	D = nb.load(stem_folder + '/' + files[3][0:3] + '/' + files[3]).get_data()#dtype='float32')
	preallo[:,:,:,3 * shape[3]:4 * shape[3]] = D
	del D

	print('Fourth block added')
	
	


	print('C and D')
	V = nb.Nifti1Image(preallo, affine)#, header)
	nb.save(V, stem_folder+ '/comb/' + file_numbers_prefix + 'combf.nii')
	del V
	print(4 * '\n')
	
	
if int(os.environ['otherCombinations']) == 1:

#Create the CWCCW file
	print('################## Creating Wedges combination ##################')
	print('Loading Clockwise')
	A = nb.load(stem_folder + '/' + files[0][0:3] + '/' + files[0])
	A = A.get_data()
	preallo = np.zeros((shape[0],shape[1],shape[2],2*shape[3]), dtype='int16')
	preallo[:,:,:,0:shape[3]] = A
	del A
	
	print('Loading CounterClockwise')
	B = nb.load(stem_folder + '/' + files[1][0:3] + '/' + files[1]).get_data()#dtype='float32')
	preallo[:,:,:,shape[3] :2 * shape[3]] = B
	del B
	
	print('Saving CWCCW')
	V = nb.Nifti1Image(preallo, affine)#, header)
	nb.save(V, stem_folder+ '/00wedges/' + file_numbers_prefix + '00wedgesf.nii')
	del V
	print('###### Done ######')
	print(4 * '\n')
	
	
	
	
	
	
#Create the EXPCONT file
	print('##################Creating Ring combination file##############')
	print('Loading Expanding')
	A = nb.load(stem_folder + '/' + files[2][0:3] + '/' + files[2])
	A = A.get_data()
	preallo = np.zeros((shape[0],shape[1],shape[2],2*shape[3]), dtype='int16')
	preallo[:,:,:,0:shape[3]] = A
	del A
	
	print('Loading Contracting')
	B = nb.load(stem_folder + '/' + files[3][0:3] + '/' + files[3]).get_data()#dtype='float32')
	preallo[:,:,:,shape[3] :2 * shape[3]] = B
	del B
	
	print('Saving expcont')
	V = nb.Nifti1Image(preallo, affine)#, header)
	nb.save(V, stem_folder+ '/00rings/' + file_numbers_prefix + '00ringsf.nii')
	del V
	print('###### Done ######')
	print(4 * '\n')
	
	
	
	
# Create the CWEXP file
	print('#################Creating CW and EXP file ################3')
	print('Loading CW')
	A = nb.load(stem_folder + '/' + files[0][0:3] + '/' + files[0])
	A = A.get_data()
	preallo = np.zeros((shape[0],shape[1],shape[2],2*shape[3]), dtype='int16')
	preallo[:,:,:,0:shape[3]] = A
	del A
	
	print('Loading expanding')
	B = nb.load(stem_folder + '/' + files[2][0:3] + '/' + files[2]).get_data()#dtype='float32')
	preallo[:,:,:,shape[3] :2 * shape[3]] = B
	del B
	
	print('Saving cwexp')
	V = nb.Nifti1Image(preallo, affine)#, header)
	nb.save(V, stem_folder+ '/00cwexp/' + file_numbers_prefix + '00cwexpf.nii')
	del V
	print('###### Done ######')
	print(4 * '\n')	






#E = nb.concat_images([A, B, C,D], False, 3)

#AB = np.concatenate((A,B), axis=3)
#del A
#del B
#CD = np.concatenate((C,D), axis = 3)
#del C
#del D
#ABCD = np.concatenate((AB, CD), axis = 3)
#del CD
#del AB
print('All images loaded')
#%%
#for x in files:
	#ims.append = nb.load(stem_folder + x)
	#A = nb.concat_images([A,B], True, 3)

	
#%%	
#affines = [x.affine for x in ims]

#A = nb.concat_images(ims, True, 3)
#B = nb.concat_images(ims[2:4], True, 3)

#V = nb.Nifti1Image(preallo, affine, header)
#nb.save(V, 'ND_CONTEXPCCWCW_combined.nii')
