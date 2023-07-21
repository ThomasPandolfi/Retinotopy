
#export FREESURFER_HOME=/usr/local/freesurfer/7.3.2
#export SUBJECTS_DIR=/usr/local/freesurfer/7.3.2/subjects
#source $FREESURFER_HOME/SetUpFreeSurfer.sh
#sudo chmod -R a+w $SUBJECTS_DIR

sub_append=${fresurName}
#sub_append=DB00
stimtype=00rings
stimtype=${actualNumbering[${value}]}
#params=0_0_0.001_0.3_300
params=${GauSigSpat}_${GauSigTemp}_${LFcut}_${HFcut}_${HighResDimension}
#filedir=/home/thomaszeph/Desktop/Deborah_MRI/firsttime/analysis/main/pyprf/${stimtype}/${params}
filedir=${niiLocation}/pyprf/${stimtype}/${params}

cd ${filedir}
mkdir surf_projection

#cp results_eccentricity.nii.gz 
#cp results_polar_angle.nii.gz
cp results_R2.nii.gz surf_projection/${sub_append}${stimtype}-vexpl.nii.gz
cp results_SD.nii.gz surf_projection/${sub_append}${stimtype}-sigma.nii.gz
cp results_x_pos.nii.gz surf_projection/${sub_append}${stimtype}-xcrds.nii.gz
cp results_y_pos.nii.gz surf_projection/${sub_append}${stimtype}-ycrds.nii.gz

cd surf_projection
python3 /home/thomaszeph/Desktop/noah_ret.py ${sub_append} ./
