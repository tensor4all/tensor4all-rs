"""Separate bounded complex FFT-product validation, not paper Fig.14 reproduction."""
import numpy as np
import reference as ref
from fixtures import OUT,write_fixture

def prepare():
    n=16;size=2**n;x=np.arange(size)/size
    a=np.exp(-((x-.35)/.12)**2)*np.exp(2j*np.pi*(13*x+37*x*x))
    b=np.exp(-((x-.58)/.17)**2)*np.exp(-2j*np.pi*(7*x+21*x*x))
    spectral=[np.fft.fft(a),np.fft.fft(b)];inputs=[ref.dense_to_chain(v,[2]*n,64) for v in spectral]
    represented=[ref.dense(c) for c in inputs]
    folder=write_fixture('complex-convolution-n16',inputs,dict(kind='convolution',
        scope='separate circular convolution; NumPy FFT outside timer, no Rust QTT Fourier operator claim; original Fig14 parameters unavailable',
        input_preparation='independent bounded TT-SVD cap64, MSB-first bits',
        input_relative_l2=[float(np.linalg.norm(u-v)/np.linalg.norm(v)) for u,v in zip(represented,spectral)]))
    np.save(folder/'expected-dense.npy',represented[0]*represented[1])
    np.save(folder/'physical-dense.npy',spectral[0]*spectral[1])
    np.save(folder/'physical-convolution.npy',np.fft.ifft(spectral[0]*spectral[1]))
    print('CONVOLUTION',folder,flush=True)
if __name__=='__main__':prepare()
