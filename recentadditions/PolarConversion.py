import nibabel as nb
import numpy as np

stem_folder = './rh_highestsig/'

filename= 'aver_rhpolar_angle.nii.gz'

polar = nb.load(stem_folder + filename)
affine = polar.affine
data = polar.get_data()
header = polar.header


adjusted = np.where(data<0, data*-1, data)
adjusted = np.mod((data + 3.1415/2),3.1415)


VLH = nb.Nifti1Image(adjusted, affine)#, header)
nb.save(VLH, stem_folder+'results_polar_angle_adjusted.nii.gz')
