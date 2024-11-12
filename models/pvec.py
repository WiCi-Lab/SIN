import numpy as np
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import svd
from numpy.linalg import norm
import matplotlib.pyplot as plt 
import matplotlib

def pronyvec(y, p = 4, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    # 刚增加的startidx表示输入的维度，比如之前输入25，就把它设置为25，现在是10，就设置为10
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    calH = np.zeros([subcarriernum*Nt*Nr, p], dtype=np.complex128)
    pL = np.zeros([subcarriernum*Nt*Nr, 1], dtype=np.complex128)
    for idx1 in range(p):
        for idx2 in range(Nt):
            for idx3 in range(subcarriernum):
                calH[idx2*Nr*subcarriernum + idx3*Nr: idx2*Nr*subcarriernum + (idx3+1)*Nr, idx1] = y[idx3, startidx-1-p+idx1, :, idx2]
    for idx2 in range(Nt):
        for idx3 in range(subcarriernum):
            pL[idx2*Nr*subcarriernum + idx3*Nr: idx2*Nr*subcarriernum + (idx3+1)*Nr, :] = np.expand_dims(y[idx3, startidx-1, :, idx2], axis=1)
    calH = np.matrix(calH)
    phat = -pinv(calH)*pL
    calH = np.hstack((calH[:, 1:p], pL))
    hpredict = -calH*phat
    hp1 = np.zeros([subcarriernum, Nr, Nt], dtype=np.complex128) 
    for idx1 in range(Nt):
        for idx2 in range(subcarriernum):
            hp1[idx2, :, idx1] = np.squeeze(hpredict[idx1*Nr*subcarriernum + idx2*Nr: idx1*Nr*subcarriernum + (idx2+1)*Nr, :])
    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    hp2[:, 0, :, :] = hp1
    for idx1 in range(pre_len - 1):
        calH = np.hstack((calH[:, 1:p], hpredict))
        hpredict = -calH*phat
        for idx2 in range(Nt):
            for idx3 in range(subcarriernum):
                hp2[idx3, idx1+1, :, idx2] = np.squeeze(hpredict[idx2*Nr*subcarriernum + idx3*Nr: idx2*Nr*subcarriernum + (idx3+1)*Nr, :])
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2

'''
if __name__ == "__main__":
    _, sr, fd, Nt, Nr, tau = utils.loaddata('./SRS/data/CDL_1.mat')
    subcarriernum = 833        # number of subcarriers
    channelnum = 1            # number of channels
    SNR = 14
    p = 10
    predict_num = 20
    rate_1 = np.zeros([predict_num + 1, 1])     # prony vec
    rate_2 = np.zeros([predict_num + 1, 1])     # previous
    rate_3 = np.zeros([predict_num + 1, 1])     # perfect CSI
    for channelidx in range(channelnum):
        data = np.load('./SRS/data/channel_%d.npz' % (channelidx + 1))
        y = data['h_hat'][0:0+subcarriernum, :, :, :]
        realchannel = data['h_stack'][0:0+subcarriernum, :, :, :]
        y = y.reshape([y.shape[0], y.shape[1], 8])
        h_predict2 = PAD.PAD(y[:, 0:25, :], p, predict_num, subcarriernum, Nt, Nr)
        #h_predict2 = STAR.STAR(y[:, 0:25, :], p, predict_num, subcarriernum, Nt, Nr)
        #h_predict2 = pronyvec(y[:, 0:25, :], p, predict_num, subcarriernum, Nt, Nr)
        y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
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
            u1, _, vh1 = svd(y[subcarrieridx, 24, :, :].T)
            u1 = u1[:, 0:Nt]
            v1 = vh1[0:Nt, :]
            H1 = u1.T.conj().dot(realchannel[subcarrieridx, 24, :, :].T).dot(v1.T.conj())
            ratetemp1 += np.real(np.log2(det(np.eye(Nt) + np.dot(H1, H1.T.conj())*SNR)))
            u2, _, vh2 = svd(realchannel[subcarrieridx, 24, :, :].T)
            u2 = u2[:, 0:Nt]
            v2 = vh2[0:Nt, :]
            H2 = u2.T.conj().dot(realchannel[subcarrieridx, 24, :, :].T).dot(v2.T.conj())
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
                H1 = u1.T.conj().dot(realchannel[subcarrieridx, 25 + timeidx, :, :].T).dot(v1.T.conj())
                ratetemp1 += np.real(np.log2(det(np.eye(Nt) + np.dot(H1, H1.T.conj())*SNR)))
                u2, _, vh2 = svd(y[subcarrieridx, 24, :, :].T)
                u2 = u2[:, 0:Nt]
                v2 = vh2[0:Nt, :]
                H2 = u2.T.conj().dot(realchannel[subcarrieridx, 25 + timeidx, :, :].T).dot(v2.T.conj())
                ratetemp2 += np.real(np.log2(det(np.eye(Nt) + np.dot(H2, H2.T.conj())*SNR)))
                u3, _, vh3 = svd(realchannel[subcarrieridx, 25 + timeidx, :, :].T)
                u3 = u3[:, 0:Nt]
                v3 = vh3[0:Nt, :]
                H3 = u3.T.conj().dot(realchannel[subcarrieridx, 25 + timeidx, :, :].T).dot(v3.T.conj())
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
    plt.legend(['Perfect CSI', 'prony_vec', 'Previous CSI'])
    plt.ylim([rate_2.min()*0.9, rate_3.max()*1.1])
    plt.xlabel('predict slots')
    plt.ylabel('Average Rate')
    plt.title('predict %d slots using %d previous channel information' % (predict_num, p))
    filename = "./fig/ccnum=%dorder=%dprenum=%d" % (channelnum, p, predict_num)
    plt.savefig(filename)
    print('debug buffer')
'''