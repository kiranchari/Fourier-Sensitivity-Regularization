import torch
import torch.fft
import torch.nn.functional as F

class FourierSensitivity(object):
    def sfs(self, image):
        '''
        image is a (N, H, W) array of the Jacobian power spectrum
        '''
        # rotavg expects even shaped input; if odd shaped, pad width and height by 1 each (assumes square image)
        if image.shape[-1] % 2 == 1: 
            image = F.pad(image, (0,1,0,1)) # pads last two dims i.e. (N,H,W) -> (N,H+1,W+1)

        L=torch.arange(-image.shape[-1]/2,image.shape[-1]/2) # [-N/2, ... 0, ... N/2-1]
        x,y = torch.meshgrid(L, L) # matrix of co-ordinate positions. for e.g. print(np.meshgrid(4, 4)) 
        R = torch.round(torch.sqrt(x**2+y**2)) # a matrix same size as input, containing distance of each pixel from centre
        
        r = torch.unique(R) # return all distances in matrix
        r = r[1:] # remove DC
        
        if self.radial_indices is None:
            self.radial_indices = {}
            for _ in r:
                self.radial_indices[int(_)] = (R==_)

        f = lambda r : image[:, self.radial_indices[int(r)]].sum(dim=1) # returns (N). sum for radial elements in each sample

        radial_sums = [] # [ (N), (N), ... R times ]

        for _ in r:
            radial_sums.append(f(_))
        
        radial_sums = torch.stack(radial_sums, dim=1) # (N,R); for each sample sum across radii

        radial_normalization_constant = radial_sums.sum(dim=1) # (N); sum across radii per sample

        for _ in range(radial_sums.shape[0]):
            radial_sums[_] /= radial_normalization_constant[_]

        sfs = radial_sums.mean(dim=0) # (R); mean sfs across samples

        return sfs

    def __init__(self, mode, lambda_):
        self.mode = mode
        self.LAMBDA = lambda_
        self.radial_indices = None

        print('*** Initialized regularizer %s, %.2f ***\n' % (mode, lambda_))

    def __call__(self, inp, loss):
        J = torch.autograd.grad(loss, inp, create_graph=True)[0] # (N, C, H, W)

        if self.mode == 'dc':
            J_FFT = torch.fft.fftn(J, dim=(1,2,3)) # (N, C, H, W) -> (N, C, H, W)
            DC_LOSS = J_FFT[:,0,0,0].mean()
            return DC_LOSS *  self.LAMBDA

        # mean of J across channels to compute 2D DFT
        J = J.mean(dim=1) # (N, H, W)

        # NOTE dim params for both fftn as well as fftshift
        J_FFT = torch.fft.fftshift(torch.fft.fftn(J, norm='ortho', dim=(1,2)), dim=(1,2)) # (N, H, W) -> (N, H, W)

        J_FFT_POW = torch.abs(J_FFT)**2 # power
    
        SFS = self.sfs(J_FFT_POW) # (R), sfs averaged across batch

        N = inp.shape[-1] # N is size of image. len(SFS) = N/sqrt(2); N = len(SFS) * sqrt(2)

        if self.mode == 'low': # penalizing low freqs
            SFS_LOSS = SFS[:int(len(SFS)/3)].sum()
        elif self.mode == 'max': # penalizing max element in SFS
            SFS_LOSS = SFS[SFS.argmax()]
        elif self.mode == 'lsf': # penalize mid and high freqs
            SFS_LOSS = SFS[int(N/6):].sum()
        elif self.mode == 'msf': # penalize low and high freqs
            SFS_LOSS = SFS[:int(N/6)].sum() + SFS[int(N/3):].sum()
        elif self.mode == 'asf': # entropy across SFS (up to N/2 only)
            entropy = 0
            
            SFS /= SFS[:int(N/2)].sum() # renormalize SFS

            for _ in range(int(N/2)):
                entropy += -SFS[_] * torch.log2(SFS[_])

            SFS_LOSS = -entropy # loss is negative entropy

        else:
            raise Exception('unrecognized')

        if torch.isnan(SFS_LOSS):
            return 0

        return SFS_LOSS * self.LAMBDA

