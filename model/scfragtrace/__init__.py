"""scFragTrace: cancer detection and tissue of origin from plasma cell-free DNA fragments, using single-cell ATAC-seq
lineage footprints, copy-number templates and a lineage-constrained graph network on the single-cell module graph.

Quick use (model/README.md):
    from scfragtrace import load_model, load_inputs
    model = load_model()                                   # trained bundle model/weights/cristiano_only
    result = model.predict(load_inputs("model/examples/data/snyder_gse71378"))
    result.table.head(); result.save("predictions/")
Command line: ``scfragtrace predict --inputs <dir> --out <dir>`` (see ``scfragtrace --help``).

Modules: priors (label-free footprint and copy-number prior), reference (module graph, node features), gnn (networks
and training loops; needs torch), preprocess (batch referencing), inputs (input folder), model (trained bundle,
prediction), train (training a bundle), features (fragment file -> inputs), fusion and evaluation (weight selection and
metrics used in the paper)."""
__version__ = "1.4.0"
from . import priors, fusion, evaluation, reference, preprocess  # noqa: F401
from .priors import CLASSES  # noqa: F401
from .inputs import load_inputs  # noqa: F401


def load_model(bundle="cristiano_only", **kw):
    """Load a trained weight bundle (name of a folder under model/weights, or a path). Needs torch."""
    from .model import load_model as _lm
    return _lm(bundle, **kw)
