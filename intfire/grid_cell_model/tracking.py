import numpy as np
import pylab
import cv2

x1 = np.linspace(0, np.pi*2.0, 60.0)
y1 = x1.copy()

x2 = x1-np.pi/4.0
y2 = x2.copy()

xy1 = np.meshgrid(x1, y1)[0]
xy2 = np.meshgrid(x2, y2)[0]

img1 = np.sin(xy1) + np.sin(xy1).T + np.random.normal(0, 0.5, xy1.shape)
img2 = np.sin(xy2) + np.sin(xy2).T + np.random.normal(0, 0.5, xy2.shape)

shift = lambda x: np.fft.fftshift(x)
ishift = lambda x: np.fft.ifftshift(x)

imgf1 = shift(cv2.dft(img1-img1.mean(), flags=cv2.DFT_COMPLEX_OUTPUT))
imgf2 = shift(cv2.dft(img2-img2.mean(), flags=cv2.DFT_COMPLEX_OUTPUT))

imgfi1 = cv2.idft(ishift(imgf1), flags=cv2.DFT_COMPLEX_OUTPUT)
imgfi2 = cv2.idft(ishift(imgf2), flags=cv2.DFT_COMPLEX_OUTPUT)

pylab.figure(figsize=(16,6))

pylab.subplot(1,3,1)
pylab.pcolormesh(img1)
pylab.axis('equal')
pylab.colorbar()

pylab.subplot(1,3,2)
pylab.pcolormesh(np.log(imgf1[:,:,0]-imgf1[:,:,0].min()+1.0))
pylab.axis('equal')
pylab.colorbar()

pylab.subplot(1,3,3)
pylab.pcolormesh(np.log(imgf1[:,:,1]-imgf1[:,:,1].min()+1.0))
pylab.axis('equal')
pylab.colorbar()

pylab.figure()
pylab.plot(imgf1[30,:,1])
pylab.plot(imgf1[:,30,1])

pylab.figure(figsize=(16,6))

pylab.subplot(1,3,1)
pylab.pcolormesh(img2)
pylab.axis('equal')
pylab.colorbar()

pylab.subplot(1,3,2)
pylab.pcolormesh(np.log(imgf2[:,:,0]-imgf2[:,:,0].min()+1.0))
pylab.axis('equal')
pylab.colorbar()

pylab.subplot(1,3,3)
pylab.pcolormesh(np.log(imgf2[:,:,1]-imgf2[:,:,1].min()+1.0))
pylab.axis('equal')
pylab.colorbar()

pylab.figure()
pylab.plot(imgf2[30,:,1])
pylab.plot(imgf2[:,30,1])

pylab.figure(figsize=(16,6))

pylab.subplot(1,2,1)
pylab.pcolormesh(imgfi1[:,:,0])
pylab.axis('equal')

pylab.subplot(1,2,2)
pylab.pcolormesh(imgfi1[:,:,1])
pylab.axis('equal')

pylab.figure(figsize=(16,6))

pylab.subplot(1,2,1)
pylab.pcolormesh(imgfi2[:,:,0])
pylab.axis('equal')

pylab.subplot(1,2,2)
pylab.pcolormesh(imgfi2[:,:,1])
pylab.axis('equal')

pylab.show()
