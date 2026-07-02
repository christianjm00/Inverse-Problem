import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import Tarea_2_2 
import time as tm
def c_A(dim_i,dim_x,dim_y,theta):
    A=np.zeros((dim_x,dim_y))
    pos=np.arange(dim_i)
    print("pos:",pos)
    pos_theta=np.arange(len(theta))
    print("theta",theta)
    for pos_theta_i in pos_theta:
        print("pos_theta_i",pos_theta_i)
        for pos_i in pos:
            print("pos_i",pos_i)
            print("theta[pos_theta_i]:",theta[pos_theta_i])
            print(np.round(pos*np.cos(theta[pos_theta_i])))
            print(np.round(pos*np.sin(theta[pos_theta_i])))
            x=pos_theta_i*dim_i+np.clip(np.round(pos*np.cos(theta[pos_theta_i])),0,dim_i-1)
            y=pos_theta_i*dim_i+np.clip(pos_i+np.round(pos*np.sin(theta[pos_theta_i])),0,dim_i-1)
            x=x.astype(int)
            y=y.astype(int)
            print("x:",x)
            print("x:",x.shape)
            print("y:",y)
            print("y:",y.shape)
            print("A:",A.shape)
            #tm.sleep(1)
            A[y,x]+=1.0
            print("A:",A)

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)
dim=image.shape[0]
SNR=100
div_theta=3
theta=np.linspace(0.0,180.0,div_theta,endpoint=False)
sinogram=radon(image, theta=theta)

image_re=np.zeros_like(image)
#
dim=3

c_A(dim,dim*div_theta,dim,theta)

dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(-dx,180.0+dx,-dy,sinogram.shape[0]+dy),
    aspect='auto',
)
ax3.imshow(image_re, cmap=plt.cm.Greys_r)
fig.tight_layout()
plt.show()