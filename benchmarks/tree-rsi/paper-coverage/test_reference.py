import unittest
import numpy as np
import reference as r

class ReferenceTests(unittest.TestCase):
    def test_complex_reference_contracts(self):
        rng=np.random.default_rng(41)
        cores=[rng.normal(size=shape)+1j*rng.normal(size=shape) for shape in [(1,3,2),(2,3,3),(3,3,1)]]
        full=r.dense(cores);prob=np.abs(full)**2
        points=np.indices((3,3,3)).reshape(3,-1).T
        np.testing.assert_allclose(r.evaluate(cores,points),full,rtol=1e-13,atol=1e-13)
        sq=r.product(cores,[a.conj() for a in cores]);np.testing.assert_allclose(r.dense(sq),prob,rtol=1e-13,atol=1e-13)
        sz=np.array([1,0,-1]);energy=np.sum(prob*(sz[points[:,0]]*sz[points[:,1]]+sz[points[:,1]]*sz[points[:,2]]))
        for actual in [r.wavefunction_observables(cores),r.diagonal_observables(sq)]:np.testing.assert_allclose(actual,[prob.sum(),energy],rtol=1e-13,atol=1e-13)
        self.assertAlmostEqual(r.norm(cores)/np.linalg.norm(full),1.0,places=13)
        self.assertGreaterEqual(r.maxabs_upper_bound(cores),float(np.max(np.abs(full)))*(1-1e-13))
        rounded,_=r.round_chain(sq,9);np.testing.assert_allclose(r.dense(rounded),prob,rtol=1e-12,atol=1e-12)
        self.assertLess(r.norm(r.difference(sq,rounded))/r.norm(sq),1e-12)
        rebuilt=r.dense_to_chain(full,[3,3,3],3);np.testing.assert_allclose(r.dense(rebuilt),full,rtol=1e-13,atol=1e-13)
        from calibrate import spectrum
        np.testing.assert_allclose(spectrum(cores,1),np.linalg.svd(full.reshape(3,-1),compute_uv=False)[:2],rtol=1e-13,atol=1e-13)
        from gpe_global import frames
        for cut in (1,2):
            left,right=frames(cores,cut)
            np.testing.assert_allclose((left@right).ravel(),full,rtol=1e-13,atol=1e-13)

    def test_sampling_known_product_distribution(self):
        core=np.sqrt([0.2,0.3,0.5]).reshape(1,3,1)
        samples=r.born_samples([core]*4,20000,42)
        frequencies=np.bincount(samples.ravel(),minlength=3)/samples.size
        np.testing.assert_allclose(frequencies,[0.2,0.3,0.5],atol=.006,rtol=0)
        np.testing.assert_array_equal(r.born_samples([np.array([0,1,0]).reshape(1,3,1).astype(float)]*3,10,2),np.ones((10,3),dtype=int))

    def test_dense_budget(self):
        with self.assertRaises(ValueError):r.dense([np.ones((1,3,1))]*20)
        with self.assertRaises(ValueError):r.product([np.ones((20,3,20))],[np.ones((20,3,20))],100)
        from gpe_global import frames
        with self.assertRaises(ValueError):frames([np.ones((1,2,1))]*52,26)

    def test_prefix_upper_bound(self):
        cores=[np.array([1.,2.]).reshape(1,2,1)]*20
        self.assertAlmostEqual(r.maxabs_upper_bound(cores)/2**20,1.,places=12)

    def test_branching_oracle(self):
        from tree_cases import materialize
        rng=np.random.default_rng(73)
        cores=[rng.normal(size=shape)+1j*rng.normal(size=shape) for shape in [(2,3,2,2),(2,2),(3,2),(2,2)]]
        metadata=[dict(node=i,neighbors=ns) for i,ns in enumerate([[1,2,3],[0],[0],[0]])]
        expected=np.einsum('abcd,ae,bf,cg->defg',*cores)
        np.testing.assert_allclose(materialize(cores,metadata),expected,rtol=1e-13,atol=1e-13)

    def test_analytic_constructor(self):
        from extend_fixtures import polynomial
        cores=polynomial(np.array([2.,-3.,5.]),n=5)
        x=np.arange(32)/32
        np.testing.assert_allclose(r.dense(cores),2-3*x+5*x*x,rtol=1e-13,atol=1e-13)

if __name__=='__main__':unittest.main()
