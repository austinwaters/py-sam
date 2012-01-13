import numpy as np
from scipy.special import gammaln, psi, polygamma
import sys

from pickle_file_io import PickleFileIO
from math_util import *
import optimize


class VEMModel(PickleFileIO):
    def __init__(self, reader=None, T=None):
        assert reader is not None
        self.T = T if T is not None else 10 # Number of topics
        self.iteration = 0  # Number of iterations completed

        # Set up a reader to access the corpus.  This gives the data dimensionality (D), the number of documents,
        # the number of data, the number of document classes, and the document labels (as well as whether a
        # document label was observed).
        self.reader = reader
        self.corpus_file = self.reader.filename
        self.V = self.reader.dim  # Vocab size
        self.D = self.reader.num_docs
        self.num_data = self.reader.num_data
        self.num_docs = self.reader.num_docs

        # For efficiency, read the corpus into memory
        self.load_corpus_as_matrix()

        # Variational parameters
        self.alpha = np.ones(self.T)*1.0 + 1.0
        self.m = l2_normalize(np.ones(self.V))  # Parameter to p(mu)
        self.kappa0 = 10.0
        self.kappa1 = 5000.0
        self.xi = 5000.0

        self.vm = l2_normalize(np.random.rand(self.V))
        self.vmu = l2_normalize(np.random.rand(self.V, self.T))

        # Initialize vAlpha
        self.valpha = np.empty((self.T, self.num_docs))
        for d in range(self.num_docs):
            distances_from_topics = np.abs(cosine_similarity(self.v[:, d], self.vmu)) + 0.01
            self.valpha[:, d] = distances_from_topics / sum(distances_from_topics) * 3.0

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reset corpus_file just in case the reader loads the corpus from a different directory
        self.corpus_file = self.reader.filename
        self.load_corpus_as_matrix()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['v']
        return state

    def load_corpus_as_matrix(self):
        self.v = np.empty((self.V, self.num_docs))
        for d in xrange(self.num_docs):
            self.v[:, d] = self.reader.read_doc(d).T

    def l_valpha(self):
        alpha0 = np.sum(self.alpha)
        psi_valpha = psi(self.valpha)

        valpha0s = self.valpha.sum(axis=0)  # Sum of each document's valpha vector
        psi_valpha0s = psi(valpha0s)

        # XXX: direct translation from matlab; probably not necessary with np broadcasting
        alpha_minus_one_matrix = np.tile(ascolvector(self.alpha - 1.0), [1, self.D])

        sum_of_rhos = np.sum(self.rho_batch())

        like = (alpha_minus_one_matrix * psi_valpha).sum() \
            - (alpha0 - self.T)*psi_valpha0s.sum() \
            + self.D*gammaln(alpha0) \
            - self.D*gammaln(self.alpha).sum() \
            + self.kappa1 * sum_of_rhos \
            - np.sum((self.valpha - 1.0) * psi_valpha) \
            + np.sum(psi_valpha0s * (valpha0s - self.T)) \
            - np.sum(gammaln(valpha0s)) \
            + np.sum(gammaln(self.valpha))
        return like

    def l_alpha(self):
        alpha0 = np.sum(self.alpha)

        psi_valpha = psi(self.valpha)
        psi_valpha0s = psi(np.sum(self.valpha, axis=0))

        likelihood = np.sum( ascolvector(self.alpha - 1) * psi_valpha ) \
                     - (alpha0 - self.T)*np.sum(psi_valpha0s) \
                     + self.D*gammaln(alpha0) \
                     - self.D*np.sum(gammaln(self.alpha))
        return likelihood

    def l_vmu(self):
        a_xi = avk(self.V, self.xi)
        a_k0 = avk(self.V, self.kappa0)

        sum_of_rhos = sum(self.rho_batch())

        vm_dot_sum_of_vmu = np.dot(self.vm.T, np.sum(self.vmu, axis=1))
        likelihood = a_xi*a_k0*self.xi*vm_dot_sum_of_vmu + self.kappa1*sum_of_rhos
        return likelihood

    def l_xi(self):
        a_xi = avk(self.V, self.xi)
        a_k0 = avk(self.V, self.kappa0)
        sum_of_rhos = sum(self.rho_batch())

        return a_xi*self.xi * (a_k0*np.dot(self.vm.T, np.sum(self.vmu, axis=1)) - self.T) \
            + self.kappa1*sum_of_rhos

    def grad_l_valpha(self):
        alpha0 = np.sum(self.alpha)
        valpha0s = np.sum(self.valpha, axis=0)
        grad_psi_valpha = polygamma(1, self.valpha)
        grad_psi_valpha0s = polygamma(1, valpha0s)

        grads_of_rho = self.grad_rho_valpha_batch()

        grad = ascolvector(self.alpha - 1.0) * grad_psi_valpha \
            + self.kappa1*grads_of_rho \
            - (self.valpha - 1)*grad_psi_valpha

        addToEachRow = -grad_psi_valpha0s*(alpha0 - self.T) \
            + grad_psi_valpha0s*(valpha0s - self.T)

        grad += asrowvector(addToEachRow)
        return grad

    def grad_l_vmu(self):
        a_xi = avk(self.V, self.xi)
        a_xi_squared = a_xi**2
        a_k0 = avk(self.V, self.kappa0)

        esns = self.e_squared_norm_batch()

        valpha0s = np.sum(self.valpha, axis=0)

        # For single d:  aXi/vAlphaD0 /sqrt(esn) * vd * vAlphaD'
        first_term = np.dot(self.v, (self.valpha * a_xi / asrowvector(valpha0s * np.sqrt(esns))).T)

        # Weights per document for everything that was in GradESN; from second term in GradRhoVMu_BatchT
        # For a single d: aXi/vAlphaD0 / (2*esn^(3/2)) * dot(model.vMu*vAlphaD, vd)
        per_doc_weights = a_xi / valpha0s / (2*esns ** (3./2.)) \
                        * (self.valpha * np.dot(self.vmu.T, self.v)).sum(axis=0).T

        second_term_doc_weights = 2*(1-a_xi_squared) / (valpha0s*(valpha0s+1))  # Same dim as valpha0s
        second_term = np.sum(per_doc_weights * second_term_doc_weights) * self.vmu

        # Last term in GradESN times per-doc factors from GradRhoVMu...
        third_term_doc_weights = per_doc_weights * 2*a_xi_squared / (valpha0s*(valpha0s+1))  # From GradESN
        # Instead of
        #rescaled_valphas = self.valpha * asrowvector(np.sqrt(third_term_doc_weights))
        #third_term = np.dot(self.vmu, np.dot(rescaled_valphas, rescaled_valphas.T))
        third_term = np.dot(self.vmu, np.dot(self.valpha * asrowvector(third_term_doc_weights), self.valpha.T))

        sum_over_documents = first_term - second_term - third_term
        return ascolvector(a_xi*a_k0*self.xi*self.vm) + self.kappa1*sum_over_documents

    def grad_l_alpha(self):
        alpha0 = np.sum(self.alpha)
        valpha0s = np.sum(self.valpha, axis=0)

        return np.sum(psi(self.valpha), axis=1) - np.sum(psi(valpha0s)) \
            + self.D*psi(alpha0) - self.D*psi(self.alpha)

    def grad_l_xi(self):
        a_xi = avk(self.V, self.xi)
        a_prime_xi = deriv_avk(self.V, self.xi)
        a_k0 = avk(self.V, self.kappa0)

        sum_over_documents = sum(self.deriv_rho_xi())
        #                                    (a_k0*np.dot(self.vm.T, np.sum(self.vmu, axis=1)) - self.T)
        return (a_prime_xi*self.xi + a_xi) * (a_k0*np.dot(self.vm.T, np.sum(self.vmu, axis=1)) - self.T) \
            + self.kappa1*sum_over_documents

    def tangent_grad_l_vmu(self):
        """
        The gradient of the likelihood bound with respect to vMu, projected into the tangent space of the hypersphere.
        """
        grad = self.grad_l_vmu()
        # Project the gradients into the tangent space at each topic
        for t in range(self.T):
            vmu_t = self.vmu[:, t]
            grad[:, t] = grad[:, t] - np.dot(vmu_t, np.dot(vmu_t.T, grad[:, t]))
        return grad

    def rho_batch(self):
        esns = self.e_squared_norm_batch()
        valpha0s = self.valpha.sum(axis=0)

        # vmu: V by T
        # v:  V by D
        # vmu' * v: T by D
        vmu_times_v = self.vmu.T.dot(self.v)
        return np.sum(self.valpha * asrowvector(1.0/valpha0s/np.sqrt(esns)) * vmu_times_v, axis=0) \
                   * avk(self.V, self.xi)

    def grad_rho_valpha_batch(self):
        valpha0s = np.sum(self.valpha, axis=0)
        a_xi = avk(self.V, self.xi)

        derivsOfESquaredNorm = self.grad_e_squared_norm_batch()
        esns = self.e_squared_norm_batch()

        vMuDotVd = np.dot(self.vmu.T, self.v)  # T by D
        vMuTimesVAlphaDotVd = np.sum(self.valpha * vMuDotVd, axis=0)

        grad = vMuDotVd / asrowvector(valpha0s)

        # Subtract a constant from each column
        grad -= asrowvector(vMuTimesVAlphaDotVd / (valpha0s**2))

        # Divide each column by sqrt(esns(d))
        grad /= asrowvector(np.sqrt(esns))

        s = vMuTimesVAlphaDotVd / valpha0s / (2*esns**(3./2.))
        grad = a_xi * (grad - derivsOfESquaredNorm * asrowvector(s))
        return grad

    def e_squared_norm_batch(self):
        valpha0s = self.valpha.sum(axis=0)
        valpha_squares = np.sum(self.valpha**2, axis=0)
        a_xi_squared = avk(self.V, self.xi) ** 2

        vMuDotVMu = np.dot(self.vmu.T, self.vmu)  # T by T
        vMuVAlphaVMuVAlpha = np.sum(
            np.dot(self.valpha.T, vMuDotVMu).T * self.valpha,
            axis=0)

        esns = (valpha0s + (1.0-a_xi_squared)*valpha_squares + a_xi_squared*vMuVAlphaVMuVAlpha) / (valpha0s * (valpha0s + 1))
        return esns

    def grad_e_squared_norm_batch(self):
        valpha0s = np.sum(self.valpha, axis=0)
        a_xi_squared = avk(self.V, self.xi) ** 2
        esns = self.e_squared_norm_batch()

        vMuTimesVAlphaTimesVMu = np.dot(self.valpha.T, np.dot(self.vmu.T, self.vmu))  # D by T
        per_doc_weights = 1./(valpha0s * (valpha0s + 1))
        grad = (1 + 2*(1-a_xi_squared)*self.valpha + 2*a_xi_squared*vMuTimesVAlphaTimesVMu.T)
        grad -= esns * asrowvector(2*valpha0s + 1)
        grad *= asrowvector(per_doc_weights)
        return grad

    def deriv_rho_xi(self):
        """ Gradient of each Rho_d with respect to xi. """
        a_xi = avk(self.V, self.xi)
        deriv_a_xi = deriv_avk(self.V, self.xi)
        valpha0s = np.sum(self.valpha, axis=0)
        esns = self.e_squared_norm_batch()
        deriv_e_squared_norm_xis  = self.grad_e_squared_norm_xi()

        vMuTimesVAlphaDotDoc = np.sum(self.valpha * np.dot(self.vmu.T, self.v), axis=0)

        deriv = deriv_a_xi * vMuTimesVAlphaDotDoc / (valpha0s * np.sqrt(esns)) \
            - a_xi/2 * vMuTimesVAlphaDotDoc / (valpha0s * esns**1.5) * deriv_e_squared_norm_xis
        return deriv

    def grad_e_squared_norm_xi(self):
        """ Gradient of the expectation of the squared norms with respect to xi """
        a_xi = avk(self.V, self.xi)
        deriv_a_xi = deriv_avk(self.V, self.xi)

        valpha0s = np.sum(self.valpha, axis=0)
        sum_valphas_squared = np.sum(self.valpha**2, axis=0)
        vMuVAlphaVMuVAlpha = np.sum(np.dot(self.valpha.T, np.dot(self.vmu.T, self.vmu)).T * self.valpha, axis=0)
        deriv = 2*a_xi*deriv_a_xi*(vMuVAlphaVMuVAlpha - sum_valphas_squared) / (valpha0s * (valpha0s + 1))
        return deriv

    def update_vm(self):
        self.vm = l2_normalize(
            self.kappa0*self.m + avk(self.V, self.xi)*self.xi*np.sum(self.vmu, axis=1)
        )

    def update_m(self):
        self.m = l2_normalize(np.sum(self.vmu, axis=1))  # Sum across topics

    def update_valpha(self):
        optimize.optimize_parameter(self, 'valpha', self.l_valpha, self.grad_l_valpha)

    def update_alpha(self):
        optimize.optimize_parameter(self, 'alpha', self.l_alpha, self.grad_l_alpha)

    def update_xi(self):
        optimize.optimize_parameter(self, 'xi', self.l_xi, self.grad_l_xi)

    def update_vmu(self):
        # XXX: The topics (vmus) must lie on the hypersphere, i.e. have unit L2 norm.  I'm not sure if scipy has
        # an optimization method that can accommodate this type of constraint, so instead, I'm encoding
        # it here a Lagrange multiplier.  This should at least push the optimizer towards solutions close to the
        # L2 constraint.

        # Set the strength of the Lagrange multipler to something much larger than the objective
        LAMBDA = 10.0*self.l_vmu()
        def f():
            squared_norms = np.sum(self.vmu ** 2, axis=0)
            return self.l_vmu() - LAMBDA*np.sum((squared_norms - 1.0)**2)

        def g():
            squared_norms = np.sum(self.vmu ** 2, axis=0)
            return self.tangent_grad_l_vmu() - LAMBDA*2.0*(squared_norms - 1.0)*(2.0*self.vmu)

        optimize.optimize_parameter(self, 'vmu', f, g, bounds=(-1.0, 1.0))
        self.vmu = l2_normalize(self.vmu)  # Renormalize

    def run_one_iteration(self):
        print 'Updating vAlpha'
        self.update_valpha()

        print 'Updating vMu'
        self.update_vmu()

        print 'Updating vM'
        self.update_vm()

        print 'Updating M'
        self.update_m()

        print 'Updating xi'
        self.update_xi()

        print 'Updating alpha'
        self.update_alpha()

        self.iteration += 1

    def print_topics(self, num_top_words=10, num_bottom_words=10, f=None):
        if f is None:
            f = sys.stdout

        wordlist = open(self.corpus_file + '.wordlist').readlines()  # TODO: not hardcode this?
        wordlist = np.array([each.strip() for each in wordlist], str)

        for t in range(self.T):
            print >>f, 'Topic %d' % t
            print >>f, '--------'

            sorted_indices = np.argsort(self.vmu[:, t])
            sorted_weights = self.vmu[sorted_indices, t]
            sorted_words = wordlist[sorted_indices]

            print >>f, 'Top weighted words:'
            for word, weight in zip(sorted_words[:-num_top_words:-1], sorted_weights[:-num_top_words:-1]):
                print >>f, '  %.4f %s' % (weight, word)

            print
            print >>f, 'Bottom weighted words:'
            for word, weight in zip(sorted_words[:num_bottom_words], sorted_weights[:num_bottom_words]):
                print >>f, '  %.4f %s' % (weight, word)

            print
            print

        if f is not sys.stdout:
            f.close()

    def write_topic_weights_arff(self, f=None):
        if f is None:
            f = sys.stdout

        mean_topic_weights = self.valpha / asrowvector(np.sum(self.valpha, axis=0))

        print >>f, '@RELATION topicWeights'
        for t in range(self.T):
            print >>f, '@ATTRIBUTE topic%d NUMERIC' % t
        print >>f, '@ATTRIBUTE class {%s}' % ','.join(self.reader.class_names)
        print >>f, '@DATA'

        for d in range(self.num_docs):
            weights_string = ', '.join([str(each) for each in mean_topic_weights[:, d]])
            label = self.reader.raw_labels[d]
            print >>f, '%s, %s' % (weights_string, label)

        if f is not sys.stdout:
            f.close()