import torch
import torch.fft
import torch.nn.functional as F

class FourierRegularizer(object):
    def sfs(self, image):
        if image.shape[-1] % 2 == 1: 
            # convert to even shaped input
            image = F.pad(image, (0,1,0,1))

        L=torch.arange(-image.shape[-1]/2,image.shape[-1]/2)
        x,y = torch.meshgrid(L, L)
        R = torch.round(torch.sqrt(x**2+y**2))
        
        r = torch.unique(R)
        r = r[1:] # exclude DC-component
        
        if self.radial_indices is None:
            self.radial_indices = {}
            for _ in r:
                self.radial_indices[int(_)] = (R==_)

        f = lambda r : image[:, self.radial_indices[int(r)]].sum(dim=1)

        radial_sums = []

        for _ in r:
            radial_sums.append(f(_))
        
        radial_sums = torch.stack(radial_sums, dim=1)

        radial_normalization_constant = radial_sums.sum(dim=1)

        for _ in range(radial_sums.shape[0]):
            radial_sums[_] /= radial_normalization_constant[_]

        sfs = radial_sums.mean(dim=0)

        return sfs

    def __init__(self, mode, lambda_):
        self.mode = mode
        self.LAMBDA = lambda_
        self.radial_indices = None

        print('==> Initialized Fourier-regularizer: %s, lambda: %.2f' % (mode, lambda_))

    def __call__(self, inp, loss):
        J = torch.autograd.grad(loss, inp, create_graph=True)[0]

        J = J.mean(dim=1)

        J_FFT = torch.fft.fftshift(torch.fft.fftn(J, norm='ortho', dim=(1,2)), dim=(1,2)) # unitary-DFT

        J_FFT_POW = torch.abs(J_FFT)**2
    
        SFS = self.sfs(J_FFT_POW)

        N = inp.shape[-1]

        if self.mode == 'LSF':
            SFS_LOSS = SFS[int(N/6):].sum()

        elif self.mode == 'MSF':
            SFS_LOSS = SFS[:int(N/6)].sum() + SFS[int(N/3):].sum()

        elif self.mode == 'ASF':
            entropy = 0
            
            SFS /= SFS[:int(N/2)].sum() # normalize

            for _ in range(int(N/2)):
                entropy += -SFS[_] * torch.log2(SFS[_])

            SFS_LOSS = -entropy

        else:
            raise Exception('unrecognized Fourier regularizer')

        if torch.isnan(SFS_LOSS):
            return 0

        return SFS_LOSS * self.LAMBDA

