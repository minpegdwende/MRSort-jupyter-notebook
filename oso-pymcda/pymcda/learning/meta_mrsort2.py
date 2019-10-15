from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
import random
from itertools import product

from pymcda.electre_tri import ElectreTri
from pymcda.types import AlternativeAssignment, AlternativesAssignments
from pymcda.types import PerformanceTable
from pymcda.utils import compute_ca
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_random_profiles
from pymcda.generate import generate_alternatives
from lp_mrsort_weights import LpMRSortWeights
from heur_mrsort_profiles4 import MetaMRSortProfiles4

class MetaMRSort2():

    def __init__(self, model, pt_sorted, aa_ori):
        self.model = model
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori
        self.lp = LpMRSortWeights(self.model, pt_sorted.pt, self.aa_ori)
        self.meta = MetaMRSortProfiles4(self.model, pt_sorted,
                                               self.aa_ori)

    def init_profiles(self):
        b = self.model.categories_profiles.get_ordered_profiles()
        worst = self.pt_sorted.pt.get_worst(self.model.criteria)
        best = self.pt_sorted.pt.get_best(self.model.criteria)
        self.model.bpt = generate_random_profiles(b, model.criteria,
                                                  worst = worst,
                                                  best = best)

    def optimize(self, nmeta):
        self.lp.update_linear_program()
        obj = self.lp.solve()

        self.meta.rebuild_tables()
        ca = self.meta.good / self.meta.na

        best_bpt = self.model.bpt.copy()
        best_ca = ca

        for i in range(nmeta):
            ca = self.meta.optimize()
            if ca > best_ca:
                best_ca = ca
                best_bpt = self.model.bpt.copy()

            if ca == 1:
                break

        self.model.bpt = best_bpt
        return best_ca

if __name__ == "__main__":
    import time
    from pymcda.generate import generate_alternatives
    from pymcda.generate import generate_random_performance_table
    from pymcda.utils import compute_winning_and_loosing_coalitions
    from pymcda.types import AlternativePerformances
    from pymcda.electre_tri import ElectreTri
    from pymcda.ui.graphic import display_electre_tri_models

    # Generate a random ELECTRE TRI BM model
    model = generate_random_mrsort_model(10, 3, 123)
    worst = AlternativePerformances("worst",
                                     {c.id: 0 for c in model.criteria})
    best = AlternativePerformances("best",
                                    {c.id: 1 for c in model.criteria})

    # Generate a set of alternatives
    a = generate_alternatives(1000)
    pt = generate_random_performance_table(a, model.criteria)
    aa = model.pessimist(pt)

    nmodels = 1
    nmeta = 20
    nloops = 50

    print('Original model')
    print('==============')
    cids = model.criteria.keys()
    model.bpt.display(criterion_ids = cids)
    model.cv.display(criterion_ids = cids)
    print("lambda\t%.7s" % model.lbda)

    ncriteria = len(model.criteria)
    ncategories = len(model.categories)
    pt_sorted = SortedPerformanceTable(pt)

    metas = []
    for i in range(nmodels):
        model_meta = generate_random_mrsort_model(ncriteria,
                                                  ncategories)

        meta = MetaMRSort2(model_meta, pt_sorted, aa)
        meta.init_profiles()
        metas.append(meta)

    t1 = time.time()

    for i in range(nloops):
        models_ca = {}
        for meta in metas:
            m = meta.model
            ca = meta.optimize(nmeta)
            models_ca[m] = ca
            if ca == 1:
                break

        models_ca = sorted(models_ca.iteritems(),
                                key = lambda (k,v): (v,k),
                                reverse = True)
        print i, models_ca[0][1]

        if models_ca[0][1] == 1:
            break

        for j in range(int((nmodels + 1) / 2), nmodels):
            model_meta = generate_random_mrsort_model(ncriteria,
                                                      ncategories)

            metas[j] = MetaMRSort2(model_meta, pt_sorted, aa)

    t2 = time.time()
    print("Computation time: %g secs" % (t2-t1))

    model2 = models_ca[0][0]
    aa_learned = model2.pessimist(pt)

    print('Learned model')
    print('=============')
    model2.bpt.display(criterion_ids = cids)
    model2.cv.display(criterion_ids = cids)
    print("lambda\t%.7s" % model2.lbda)
    #print(aa_learned)

    total = len(a)
    nok = 0
    anok = []
    for alt in a:
        if aa(alt.id) <> aa_learned(alt.id):
            anok.append(alt)
            nok += 1

    print("Good assignments: %g %%" % (float(total-nok)/total*100))
    print("Bad assignments : %g %%" % (float(nok)/total*100))

    win1, loose1 = compute_winning_and_loosing_coalitions(model.cv,
                                                          model.lbda)
    win2, loose2 = compute_winning_and_loosing_coalitions(model2.cv,
                                                          model2.lbda)
    coali = list(set(win1) & set(win2))
    coal1e = list(set(win1) ^ set(coali))
    coal2e = list(set(win2) ^ set(coali))

    print("Number of coalitions original: %d"
          % len(win1))
    print("Number of coalitions learned: %d"
          % len(win2))
    print("Number of common coalitions: %d"
          % len(coali))
    print("Coallitions in original and not in learned: %s"
          % '; '.join(map(str, coal1e)))
    print("Coallitions in learned and not in original: %s"
          % '; '.join(map(str, coal2e)))

    display_electre_tri_models([model, model2],
                               [worst, worst], [best, best])
