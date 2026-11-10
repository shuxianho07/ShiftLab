import unittest
import numpy as np
from benchmark import MLP, targets, sample

class Correctness(unittest.TestCase):
    def test_reference_functions(self):
        y=targets(np.array([[0.,0.,0.,0.],[-1.,-1.,-1.,-1.],[1.,1.,1.,1.]]))
        np.testing.assert_allclose(y[:,0], [0,-1,1])
        np.testing.assert_allclose(y[:,1],0,atol=1e-12)
        np.testing.assert_allclose(y[:,2],[0,1-.95**4,0],atol=1e-12)
        np.testing.assert_allclose(y[:,4],[0,0,1])
        np.testing.assert_allclose(y[:,5],[0,1,-1])
    def test_gradients(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(5,3));y=rng.normal(size=(5,2))
        model=MLP(3,4,2,0);_,grads=model.loss_grad(x,y)
        for param,grad in zip(model.p,grads):
            for idx in np.ndindex(param.shape):
                old=param[idx];eps=1e-5
                param[idx]=old+eps;plus=model.loss_grad(x,y)[0]
                param[idx]=old-eps;minus=model.loss_grad(x,y)[0];param[idx]=old
                self.assertAlmostEqual(grad[idx],(plus-minus)/(2*eps),places=6)
    def test_learning_and_reproducibility(self):
        rng=np.random.default_rng(1);x=sample(rng,128,4,'iid');y=x.mean(1,keepdims=True)
        m=MLP(4,8,1,1);before=m.loss_grad(x,y)[0];m.fit(x,y,150,64,1)
        self.assertLess(m.loss_grad(x,y)[0],before*.1)
        twin=MLP(4,8,1,1).fit(x,y,150,64,1)
        np.testing.assert_array_equal(m.predict(x),twin.predict(x))


class Checkpoints(unittest.TestCase):
    def test_validation_checkpoint(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(32,3));y=x.mean(1,keepdims=True)
        m=MLP(3,8,1,0).fit(x,y,120,16,0,(x,y))
        best=min(m.history,key=lambda h:h['validation_mse'])
        self.assertEqual(m.best_step,best['step'])
        self.assertAlmostEqual(m.loss_grad(x,y)[0],best['validation_mse'])

    def test_saved_inference(self):
        from predict import infer
        from tempfile import TemporaryDirectory
        from pathlib import Path
        rng=np.random.default_rng(1);x=rng.normal(size=(10,4))
        model=MLP(4,8,6,0)
        weights=dict(mean=np.arange(6),std=np.arange(1,7),ridge=np.zeros((5,6)),
                     **{f'shared_{j}':p for j,p in enumerate(model.p)})
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.npz';np.savez(path,**weights)
            with np.load(path,allow_pickle=False) as saved:
                np.testing.assert_allclose(infer(saved,x),model.predict(x)*weights['std']+weights['mean'])
                with self.assertRaises(ValueError):infer(saved,[[float('nan')]*4])
                with self.assertRaises(ValueError):infer(saved,[[1,2]])

if __name__=='__main__':unittest.main()
