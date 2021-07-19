# Code: /home/jinn/openpilot/tools/lib/hevctoh5a.py 
# Input: /home/jinn/data1/8bfda98c9c9e4291|2020-05-11--03-00-57--61/fcamera.hevc
# Output: /home/jinn/data1/8bfda98c9c9e4291|2020-05-11--03-00-57--61/camera.h5
import os
import cv2
import h5py
import matplotlib.pyplot as plt
from tools.lib.logreader import LogReader
from tools.lib.framereader import FrameReader

dirs=os.listdir('/home/jinn/data1')
dirs=['/home/jinn/data1/'+i +'/' for i in dirs]
print(dirs)

path_all=[]
for di1 in dirs:
  dir1=os.listdir(di1)
  path=[di1 + d for d in dir1]
  for dd in path:
    path_all.append(dd)

def mkcamera(path):
  for f in path:
    fr=FrameReader(f)
    fcount = fr.frame_count - 1192  # -1192 produces very small .h5 for debugging
    print(fcount)
    ca=f.replace('fcamera.hevc','camera.h5')
    if not os.path.isfile(ca): 
      with h5py.File(ca,'w') as f2:
        f2.create_dataset('X',(fcount,550,550,3))
        for i in range(fcount): 
          img=fr.get(i,pix_fmt='rgb24')[0]
          #img=img[:650,:,:]
          img=cv2.resize(img,(550,550))  
          f2['X'][i]=img
          plt.imshow(img) # see resized img
          plt.show()
          print(i)      
        f3 = h5py.File(ca, 'r') # read .h5, 'w': write
        print(f3.keys())
        dset = f3['X']
        print(dset.shape)
        print(dset.dtype)
        print(dset[0])
        
if __name__ == '__main__':
  print(path_all)
  mkcamera(path_all)