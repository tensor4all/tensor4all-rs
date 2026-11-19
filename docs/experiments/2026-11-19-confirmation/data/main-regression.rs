use anyhow::Result;
use tensor4all_treetci::*;
struct InitialCrossOnly;
impl PivotCandidateProposer for InitialCrossOnly {
    fn candidates_with_rng<T,R:rand::Rng+?Sized>(&self,_:&TreeTCI2<T>,_:TreeTciEdge,_:&mut R)->TreeTciResult<(Vec<Vec<usize>>,Vec<Vec<usize>>)> {
        Ok((vec![vec![0]],vec![vec![0]]))
    }
}
fn main()->Result<()> {
    let mut state=TreeTCI2::<f64>::new(vec![2,2],TreeTciGraph::linear_chain(2)?)?;
    state.add_global_pivots(&[vec![0,0]])?;
    state.set_max_sample_value(1.0);
    let options=TreeTciOptions{tolerance:1e-12,max_iter:4,nsearch:20,seed:Some(7),..Default::default()};
    let result=optimize_with_proposer(&mut state,|b:GlobalIndexBatch<'_>|->Result<Vec<f64>>{
        Ok((0..b.n_points()).map(|p|(1+b.get(0,p).unwrap()+b.get(1,p).unwrap()) as f64).collect())
    },&options,&InitialCrossOnly,&AllEdges,&mut ())?;
    println!("errors={:?} ranks={:?} termination={:?} stats={:?}",result.errors,result.ranks,result.termination,result.evaluation);
    Ok(())
}
