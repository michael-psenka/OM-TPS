import sys

sys.path.append("/home/sanjeevr/om-diffusion/two-for-one-diffusion")
from evaluate.evaluators import TicEvaluator

tic_evaluator = TicEvaluator(
    val_data=None,
    mol_name="delta",
    eval_folder=None,
    saved_ref="saved_references/saved_TICA_DELTA_testset.pickle",
    data_folder="/data/sanjeevr/atlas_final",
    folded_pdb_folder="/data/sanjeevr/atlas_interpolation",
    bins=101,
    lagtime=10,  # ATLAS trajectory spacing is 10 ps, and we want to use 100 ps as lagtime (following MDGen paper)
    evalset="testset",
)

fig = tic_evaluator._plot_tic(
    tic_evaluator.gt_prob,
    file_name="TICA_reference.png",
    title="Reference testset",
    save_plot=True,
)
