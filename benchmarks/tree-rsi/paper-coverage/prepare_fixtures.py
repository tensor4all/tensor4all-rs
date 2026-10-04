"""Complete bounded fixture preparation in a fresh artifact directory."""
from fixtures import OUT,dmrg,analytic_input,analytic_product
from extend_fixtures import complete_functions,gpe
from tree_cases import prepare as prepare_trees
from convolution import prepare as prepare_convolution

def main():
    folder=OUT/'fixtures'
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('existing fixtures must not be overwritten; reproduce in a fresh artifact directory')
    folder.mkdir(parents=True,exist_ok=True)
    for n,chis in [(10,[20]),(20,[10,15,20,25,30]),(50,[20,40,60,80,100,150])]:
        for chi in chis:dmrg(n,chi)
    for a,b,sigma in [(.4,.6,.15),(.25,.75,.15),(.1,.9,.15)]:
        operands=[];configs=[];errors=[]
        for mu in (a,b):
            cfg=dict(function='gaussian',mu=mu,sigma=sigma,n=25,cap=10,pivots=[mu,.125,.375,.5,.625,.875])
            op,error=analytic_input(f'gaussian-{mu}-{sigma}',cfg)
            operands.append(op);configs.append(cfg);errors.append(error)
        name=f'gaussian-{a}-{b}-{sigma}';analytic_product(name,operands,configs,errors)
        if a==.4:
            for ids in ([0,1,1],[0,0,1,1]):analytic_product(name+f'-m{len(ids)}',[operands[i] for i in ids],[configs[i] for i in ids],[errors[i] for i in ids])
    complete_functions();gpe();prepare_trees();prepare_convolution()
if __name__=='__main__':main()
