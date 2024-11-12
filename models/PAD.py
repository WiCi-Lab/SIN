import numpy as np
from numpy.core.shape_base import vstack, hstack
from numpy.linalg import pinv, svd, norm
from numpy import expand_dims, floor, abs
import matplotlib.pyplot as plt 
import matplotlib
import math
import scipy.io as scio
import pvec

pi = math.pi

def DFT(N):
    row = np.arange(N).reshape(1, N)
    column = np.arange(N).reshape(N, 1)
    W = 1 / np.sqrt(N) * np.exp(-2j * pi / N * np.dot(column, row))
    return W

def PAD(y, p = 10, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    #注意此情境下p最好取偶数
    L = int(floor(p/2) * 2 - 1)
    N = int((L + 1) / 2)
    S = np.kron(DFT(subcarriernum), DFT(Nt))
    gamma = 0.1
    '''
    gu = S.T.conj().dot(y[:, 0, 0, 0])
    guselect = abs(gu)
    gusort = np.sort(guselect)[::-1]
    guidx = np.argsort(guselect)[::-1]
    for Ns in range(subcarriernum):
        if(guselect[Ns] < gamma * guselect[0]):
            break
    gusort = gusort[0:Ns]
    guidx = guidx[0:Ns]
    '''
    #hp1 = np.zeros([subcarriernum, Nr, Nt], dtype=np.complex128)
    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    for Nridx in range(Nr):
        ypad = np.zeros([subcarriernum * Nt, y.shape[1]], dtype=np.complex128)
        for idx in range(subcarriernum):
            ypad[idx*Nt:(idx+1)*Nt, :] = np.squeeze(y[idx, :, Nridx, :]).T
        gu = S.T.conj().dot(ypad[:, startidx-p])
        guselect = abs(gu)
        '''
        matplotlib.use('Agg')
        fig = plt.figure()
        plt.plot(guselect)
        filename = "./fig/2dsparse"
        plt.savefig(filename)
        '''
        gusort = np.sort(guselect)[::-1]
        guidx = np.argsort(guselect)[::-1]
        '''
        for Ns in range(subcarriernum):
            if(guselect[Ns] < gamma * guselect[0]):
                break
        '''
        Ns = subcarriernum
        gusort = gusort[0:Ns]
        guidx = guidx[0:Ns]
        ghat = np.zeros([subcarriernum * Nt, pre_len], dtype=np.complex128)            
        g = S.T.conj().dot(ypad)
        for Nsidx in range(Ns):
            rn = guidx[Nsidx]
            calG = np.zeros([N, N], dtype=np.complex128)
            for idx in range(N):
                calG[idx, :] = g[rn, startidx-p+idx:startidx-p+idx+N]
            boldg = g[rn, startidx-p+N:startidx-p+2*N].T
            phat = np.dot(-pinv(calG), boldg)
            gnew = g[rn, startidx-p+L-N+1:startidx-p+L+1]
            for timeidx in range(pre_len):
                ghat[rn, timeidx] = np.dot(-gnew, phat)    
                gnew = hstack((gnew[1:N], ghat[rn, timeidx]))
                calG = hstack((calG[:, 1:N], vstack((np.expand_dims(calG[1:N, N-1], axis=1), ghat[rn, timeidx]))))
                phat = np.dot(-pinv(calG), gnew.T)
        hhat = S.dot(ghat)
        for idx in range(subcarriernum):
            hp2[idx, :, Nridx, :] = hhat[idx*Nt:(idx+1)*Nt, :].T
    '''
    hp1 = hp2[:, 0, :, :]
    gu = abs(S.T.conj().dot(np.reshape(hp1[:, 0, :], [1666,1])))
    matplotlib.use('Agg')
    fig = plt.figure()
    plt.plot(guselect)
    filename = "./fig/2dsparse1.png"
    plt.savefig(filename)
    '''
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2

def PAD2(y, p = 10, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    #注意此情境下p最好取偶数
    L = int(floor(p/2) * 2 - 1)
    N = int((L + 1) / 2)
    S = np.kron(DFT(subcarriernum), DFT(Nt))
    gamma = 0.1
    '''
    gu = S.T.conj().dot(y[:, 0, 0, 0])
    guselect = abs(gu)
    gusort = np.sort(guselect)[::-1]
    guidx = np.argsort(guselect)[::-1]
    for Ns in range(subcarriernum):
        if(guselect[Ns] < gamma * guselect[0]):
            break
    gusort = gusort[0:Ns]
    guidx = guidx[0:Ns]
    '''
    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    for Nridx in range(Nr):
        ypad = np.zeros([subcarriernum * Nt, y.shape[1]], dtype=np.complex128)
        for idx in range(subcarriernum):
            ypad[idx*Nt:(idx+1)*Nt, :] = np.squeeze(y[idx, :, Nridx, :]).T
        gu = S.T.conj().dot(ypad[:, startidx-p])
        guselect = abs(gu)
        '''
        matplotlib.use('Agg')
        fig = plt.figure()
        plt.plot(guselect)
        filename = "./fig/2dsparse"
        plt.savefig(filename)
        '''
        gusort = np.sort(guselect)[::-1]
        guidx = np.argsort(guselect)[::-1]
        '''
        for Ns in range(subcarriernum):
            if(guselect[Ns] < gamma * guselect[0]):
                break
        '''
        Ns = 128
        gusort = gusort[0:Ns]
        guidx = guidx[0:Ns]
        ghat = np.zeros([subcarriernum * Nt, pre_len], dtype=np.complex128)            
        g = S.T.conj().dot(ypad)
        for Nsidx in range(Ns):
            rn = guidx[Nsidx]
            calG = np.zeros([N, N], dtype=np.complex128)
            for idx in range(N):
                calG[idx, :] = g[rn, startidx-p+idx:startidx-p+idx+N]
            boldg = g[rn, startidx-p+N:startidx-p+2*N].T
            phat = np.dot(-pinv(calG), boldg)
            gnew = g[rn, startidx-p+L-N+1:startidx-p+L+1]
            for timeidx in range(pre_len):
                ghat[rn, timeidx] = np.dot(-gnew, phat)    
                gnew = hstack((gnew[1:N], ghat[rn, timeidx]))
        hhat = S.dot(ghat)
        for idx in range(subcarriernum):
            hp2[idx, :, Nridx, :] = hhat[idx*Nt:(idx+1)*Nt, :].T
    '''
    hp1 = hp2[:, 0, :, :]
    gu = abs(S.T.conj().dot(np.reshape(hp1[:, 0, :], [1666,1])))
    matplotlib.use('Agg')
    fig = plt.figure()
    plt.plot(guselect)
    filename = "./fig/2dsparse1.png"
    plt.savefig(filename)
    ''' 
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2

def PAD3(y, p = 10, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    #注意此情境下p最好取偶数
    S = np.kron(DFT(subcarriernum), DFT(Nt))
    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    for Nridx in range(Nr):
        ypad = np.zeros([subcarriernum * Nt, y.shape[1]], dtype=np.complex128)
        for idx in range(subcarriernum):
            ypad[idx*Nt:(idx+1)*Nt, :] = np.squeeze(y[idx, :, Nridx, :]).T
        gu = S.T.conj().dot(ypad)
        gu = gu.reshape([y.shape[0], y.shape[3], y.shape[1]]).transpose(0, 2, 1)
        gu = np.expand_dims(gu, axis=2)
        ghat = pvec.pronyvec(gu, p, pre_len, startidx, subcarriernum, Nt, 1)
        ghat = np.reshape(ghat.transpose(0, 2, 1), [y.shape[0]*y.shape[3], pre_len, 1]).squeeze()
        hhat = S.dot(ghat)
        for idx in range(subcarriernum):
            hp2[idx, :, Nridx, :] = hhat[idx*Nt:(idx+1)*Nt, :].T
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2


'''
if __name__ == "__main__":
    _, sr, fd, Nt, Nr, tau = utils.loaddata('./SRS/data/CDL_1.mat')
    subcarriernum = 833         # number of subcarriers
    channelnum = 1            # number of channels
    SNR = 14
    p = 78
    predict_num = 20
    rate_1 = np.zeros([predict_num + 1, 1])     # PAD
    rate_2 = np.zeros([predict_num + 1, 1])     # previous
    rate_3 = np.zeros([predict_num + 1, 1])     # perfect CSI
    for channelidx in range(channelnum):
        data = np.load('./SRS/data/channel_%d.npz' % (channelidx + 1))
        y = data['h_hat'][0:0+subcarriernum, :, :, :]
        realchannel = data['h_stack'][0:0+subcarriernum, :, :, :]
        h_predict2 = PAD(y, p, predict_num, subcarriernum, Nt, Nr)
        ''
        a = y[:, p+1, :, :]
        b = y[:, p, :, :]
        error1 = h_predict1 - a
        error2 = a - b
        NMSE1 = np.sqrt(np.sum(np.abs(error1)**2) / np.sum(np.abs(a)**2))
        NMSE2 = np.sqrt(np.sum(np.abs(error2)**2) / np.sum(np.abs(a)**2))
        ''
        ratetemp1 = 0
        ratetemp2 = 0
        ratetemp3 = 0
        for subcarrieridx in range(subcarriernum):
            u1, _, vh1 = svd(y[subcarrieridx, p, :, :].T)
            u1 = u1[:, 0:Nt]
            v1 = vh1[0:Nt, :]
            H1 = u1.T.conj().dot(realchannel[subcarrieridx, p, :, :].T).dot(v1.T.conj())
            ratetemp1 += np.real(np.log2(det(np.eye(Nt) + np.dot(H1, H1.T.conj())*SNR)))
            u2, _, vh2 = svd(realchannel[subcarrieridx, p, :, :].T)
            u2 = u2[:, 0:Nt]
            v2 = vh2[0:Nt, :]
            H2 = u2.T.conj().dot(realchannel[subcarrieridx, p, :, :].T).dot(v2.T.conj())
            ratetemp2 += np.real(np.log2(det(np.eye(Nt) + np.dot(H2, H2.T.conj())*SNR)))
        rate_1[0, 0] += ratetemp1/subcarriernum/channelnum
        rate_2[0, 0] += ratetemp1/subcarriernum/channelnum
        rate_3[0, 0] += ratetemp2/subcarriernum/channelnum
        ratetemp1 = 0
        ratetemp2 = 0  
        for timeidx in range(predict_num): 
            for subcarrieridx in range(subcarriernum):
                u1, _, vh1 = svd(h_predict2[subcarrieridx, timeidx, :, :].T)
                u1 = u1[:, 0:Nt]
                v1 = vh1[0:Nt, :]
                H1 = u1.T.conj().dot(realchannel[subcarrieridx, p + 1 + timeidx, :, :].T).dot(v1.T.conj())
                ratetemp1 += np.real(np.log2(det(np.eye(Nt) + np.dot(H1, H1.T.conj())*SNR)))
                u2, _, vh2 = svd(y[subcarrieridx, p, :, :].T)
                u2 = u2[:, 0:Nt]
                v2 = vh2[0:Nt, :]
                H2 = u2.T.conj().dot(realchannel[subcarrieridx, p + 1 + timeidx, :, :].T).dot(v2.T.conj())
                ratetemp2 += np.real(np.log2(det(np.eye(Nt) + np.dot(H2, H2.T.conj())*SNR)))
                u3, _, vh3 = svd(realchannel[subcarrieridx, p + 1 + timeidx, :, :].T)
                u3 = u3[:, 0:Nt]
                v3 = vh3[0:Nt, :]
                H3 = u3.T.conj().dot(realchannel[subcarrieridx, p + 1 + timeidx, :, :].T).dot(v3.T.conj())
                ratetemp3 += np.real(np.log2(det(np.eye(Nt) + np.dot(H3, H3.T.conj())*SNR)))
            rate_1[timeidx + 1, 0] += ratetemp1/subcarriernum/channelnum
            rate_2[timeidx + 1, 0] += ratetemp2/subcarriernum/channelnum
            rate_3[timeidx + 1, 0] += ratetemp3/subcarriernum/channelnum
            ratetemp1 = 0
            ratetemp2 = 0
            ratetemp3 = 0
        print('channenumber=%d' % (channelidx + 1))

    matplotlib.use('Agg')
    fig = plt.figure()
    # plt.Hold(True)

    plt.plot(rate_3, '--')
    plt.plot(rate_1)
    plt.plot(rate_2)
    plt.legend(['Perfect CSI', 'PAD', 'Previous CSI'])
    plt.ylim([rate_2.min()*0.9, rate_3.max()*1.1])
    plt.xlabel('predict slots')
    plt.ylabel('Average Rate')
    plt.title('predict %d slots using %d previous channel information' % (predict_num, p))
    filename = "./fig/padccnum=%dorder=%dprenum=%d" % (channelnum, p, predict_num)
    plt.savefig(filename)
    print('debug buffer')

'''