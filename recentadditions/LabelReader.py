import nibabel as nb
import numpy as np
label_dir = '/usr/local/freesurfer/7.3.2/subjects/DB00/label/lh.V1_exvivo.label'


def read_label(label_dir, thresh):
    
    #Takes each line in label file
    #splits string into list
    #maps the float argument for each value
    
    with open(label_dir) as f:
        f.readline()
        f.readline()
        contents = [list(map(float, line.rstrip().split())) for line in f.readlines()]
    
    contents = np.array(contents)
    viableVertex = contents[:,0][contents[:,4] >= thresh]
    return(viableVertex)


c = read_label(label_dir, 0.1)
Vertices = np.array(list(map(int, c)))

locations = np.zeros((140000,1))
locations[Vertices - 1] = 1


