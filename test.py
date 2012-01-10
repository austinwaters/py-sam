import numpy as np
from math_util import avk, deriv_avk
from optimize import check_grad

from io.corpus import CorpusReader
from vem.model import VEMModel

CORPUS_FILENAME = 'nips-425D.h5'

reader = CorpusReader(CORPUS_FILENAME, data_series='sam')
model = VEMModel(reader)

while True:
    model.run_one_iteration()
    model.print_topics()



def check_grads(model):
    assert np.isfinite(model.l_alpha())
    assert np.isfinite(model.l_valpha())

    x = model.grad_l_vmu()
    assert np.isfinite(x).all()

    import pdb
    try:
        # Main update rules
        print 'xi update:', check_grad(model, 'xi', model.l_xi, model.grad_l_xi)
        print 'valpha update:', check_grad(model, 'valpha', model.l_valpha, model.grad_l_valpha)
        print 'alpha update:', check_grad(model, 'alpha', model.l_alpha, model.grad_l_alpha)

        print 'vmu update:', check_grad(model, 'vmu', model.l_vmu, model.tangent_grad_l_vmu)


        f = lambda: avk(model.V, model.xi)
        g = lambda: deriv_avk(model.V, model.xi)
        print 'avk_xi', check_grad(model, 'xi', f, g)

        f = lambda: np.sum(model.e_squared_norm_batch())
        g = lambda: np.sum(model.grad_e_squared_norm_xi())
        print 'grad_esn_xi', check_grad(model, 'xi', f, g)

        f = lambda: np.sum(model.rho_batch())
        g = lambda: np.sum(model.deriv_rho_xi())
        print 'deriv_rho_xi', check_grad(model, 'xi', f, g)


    except Exception, e:
        print e
        pdb.post_mortem()


    #print check_grad(model, 'vmu', model.l_vmu, model.grad_l_vmu)