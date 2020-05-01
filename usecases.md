# Some use cases and notes

## NeuroCAAS integration

#### Motion correction example

This runs motion correction on a file and returns ``Y`` motion corrected array.
Use ``mmap = True`` to avoid loading the whole file to memory and overwrite on disk.

Saves a file to with the motion correction (x,y) shifts. Use `outputdir` to control where that gets written to.

```python
dat_path = '/mnt/dual/temp_folder/CSP23_20200226/frames_2_540_640_uint16.dat'

# This will load a file to memory, and return motion corrected data
from wfield.ncaas import load_and_motion_correct 
Y = load_and_motion_correct(dat_path,
                            chunksize = 1048,     # increase:use more memory
                            mmap = False,         # true: overwrite raw
                            flatten_frames=False) # true: return frames and channels as single dimension
```

### Hemodynamics correction example

This performs hemodynamics correction on widefield data collected with 2 excitation wavelengths (470 and 405 nm). Result is saved in `SVTcorr.npy`.

```python

U = np.load('U.npy')
SVT = np.load('SVT.npy')

frame_rate = 30.        # acquisition rate (2 channels)
output_folder = None    # write to current directory

from wfield.ncaas import dual_color_hemodymamic_correction

SVTcorr = dual_color_hemodymamic_correction(U,SVT,
                                            frame_rate = frame_rate, 
                                            output_folder = output_folder);
					    
```