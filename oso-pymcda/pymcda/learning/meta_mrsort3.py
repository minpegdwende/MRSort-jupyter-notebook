from __future__ import division
import errno
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
import random
import math
from itertools import product
from multiprocessing import Pool, Process, Queue
from threading import Thread

from pymcda.electre_tri import MRSort
from pymcda.types import AlternativeAssignment, AlternativesAssignments
from pymcda.types import PerformanceTable
from pymcda.learning.heur_mrsort_init_profiles import HeurMRSortInitProfiles
from pymcda.learning.lp_mrsort_weights import LpMRSortWeights
from pymcda.learning.heur_mrsort_profiles4 import MetaMRSortProfiles4
from pymcda.utils import compute_ca
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_alternatives
from pymcda.generate import generate_categories_profiles

def compute_ca_prime(categories, aa, aa2):
    ca = 0
    cat_order = {cat: i + 1 for i, cat in enumerate(categories)}
    for a in aa:
        aid = a.id
        cat = a.category_id
        if cat < aa2(aid):
            ca += 0.2
        elif cat == aa2(aid):
            ca += 1

    return ca / len(aa)

def queue_get_retry(queue):
    while True:
        try:
            return queue.get()
        except IOError, e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise

class MetaMRSortPop3():

    def __init__(self, nmodels, criteria, categories, pt_sorted, aa_ori,
                 heur_init_profiles = HeurMRSortInitProfiles,
                 lp_weights = LpMRSortWeights,
                 heur_profiles= MetaMRSortProfiles4,
                 seed = 0):
        self.nmodels = nmodels
        self.criteria = criteria
        self.categories = categories
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori

        self.heur_init_profiles = heur_init_profiles
        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles

        self.metas = list()
        for i in range(self.nmodels):
            meta = self.init_one_meta(i + seed)
            self.metas.append(meta)

    def init_one_meta(self, seed):
        cps = generate_categories_profiles(self.categories)
        model = MRSort(self.criteria, None, None, None, cps)
        meta = MetaMRSort3(model, self.pt_sorted, self.aa_ori,
                           self.heur_init_profiles,
                           self.lp_weights,
                           self.heur_profiles)
        random.seed(seed)
        meta.random_state = random.getstate()
        meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)
        return meta

    def sort_models(self):
        metas_sorted = sorted(self.metas, key = lambda (k): k.ca,
                              reverse = True)
        return metas_sorted

    def reinit_worst_models(self):
        metas_sorted = self.sort_models()
        nmeta_to_reinit = int(math.ceil(self.nmodels / 2))
        for meta in metas_sorted[nmeta_to_reinit:]:
            meta.init_profiles()

    def _process_optimize(self, meta, nmeta):
        random.setstate(meta.random_state)
        ca = meta.optimize(nmeta)
        meta.queue.put([ca, meta.model.bpt, meta.model.cv,
                        meta.model.lbda, random.getstate()])

    def optimize(self, nmeta):
        self.reinit_worst_models()

        for meta in self.metas:
            meta.queue = Queue()
            meta.p = Process(target = self._process_optimize,
                             args = (meta, nmeta))
            meta.p.start()

        for meta in self.metas:
            output = queue_get_retry(meta.queue)

            meta.ca = output[0]
            meta.model.bpt = output[1]
            meta.model.cv = output[2]
            meta.model.lbda = output[3]
            meta.random_state = output[4]
            meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)

        self.models = {meta.model: meta.ca for meta in self.metas}

        metas_sorted = self.sort_models()

        return metas_sorted[0].model, metas_sorted[0].ca

class MetaMRSortPop3AUC(MetaMRSortPop3):

    def sort_models(self):
        metas_sorted = sorted(self.metas, key = lambda (k): k.auc,
                              reverse = True)
        return metas_sorted

class MetaMRSort3():

    def __init__(self, model, pt_sorted, aa_ori,
                 heur_init_profiles = HeurMRSortInitProfiles,
                 lp_weights = LpMRSortWeights,
                 heur_profiles = MetaMRSortProfiles4):
        self.model = model
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori

        self.heur_init_profiles = heur_init_profiles
        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles

        self.init_profiles()
        self.lp = self.lp_weights(self.model, pt_sorted.pt, self.aa_ori)

        # Because MetaMRSortProfiles4 needs weights in initialization
        self.lp.solve()

        self.meta = self.heur_profiles(self.model, pt_sorted, self.aa_ori)

        self.ca = self.meta.good / self.meta.na

    def init_profiles(self):
        cats = self.model.categories_profiles.to_categories()
        heur = self.heur_init_profiles(self.model, self.pt_sorted, self.aa_ori)
        heur.solve()

    def optimize(self, nmeta):
        self.lp.update_linear_program()
        obj = self.lp.solve()

        self.meta.rebuild_tables()
#        ca = self.meta.good / self.meta.na
        aa2 = self.model.pessimist(self.pt_sorted.pt)
        ca = compute_ca(self.aa_ori, aa2)

        best_bpt = self.model.bpt.copy()
        best_ca = ca

        for i in range(nmeta):
            cah = self.meta.optimize()
            aa2 = self.model.pessimist(self.pt_sorted.pt)
            ca = compute_ca(self.aa_ori, aa2)
            if ca > best_ca:
                best_ca = ca
                best_bpt = self.model.bpt.copy()

            if cah == 1:
                break

        self.model.bpt = best_bpt
        self.ca = best_ca
        aa2 = self.model.pessimist(self.pt_sorted.pt)
        return compute_ca(self.aa_ori, aa2)

if __name__ == "__main__":
    import time
    from pymcda.generate import generate_alternatives
    from pymcda.generate import generate_random_performance_table
    from pymcda.utils import compute_winning_and_loosing_coalitions
    from pymcda.types import AlternativePerformances
    from pymcda.ui.graphic import display_electre_tri_models

    # Generate a random ELECTRE TRI BM model
    model = generate_random_mrsort_model(10, 3, 1)
    worst = AlternativePerformances("worst",
                                     {c.id: 0 for c in model.criteria})
    best = AlternativePerformances("best",
                                    {c.id: 1 for c in model.criteria})

    # Generate a set of alternatives
    a = generate_alternatives(1000)
    pt = generate_random_performance_table(a, model.criteria)
    aa = model.pessimist(pt)

    nmeta = 20
    nloops = 30

    print('Original model')
    print('==============')
    cids = model.criteria.keys()
    model.bpt.display(criterion_ids = cids)
    model.cv.display(criterion_ids = cids)
    print("lambda\t%.7s" % model.lbda)

    ncriteria = len(model.criteria)
    ncategories = len(model.categories)
    pt_sorted = SortedPerformanceTable(pt)

    model2 = generate_random_mrsort_model(ncriteria, ncategories)

    t1 = time.time()

    meta = MetaMRSortPop3(10, model.criteria,
                          model.categories_profiles.to_categories(),
                          pt_sorted, aa)
    for i in range(nloops):
        model2, ca = meta.optimize(20)
        print("%d: ca: %f" % (i, ca))

    t2 = time.time()
    print("Computation time: %g secs" % (t2-t1))

    print('Learned model')
    print('=============')
    model2.bpt.display(criterion_ids = cids)
    model2.cv.display(criterion_ids = cids)
    print("lambda\t%.7s" % model2.lbda)
    #print(aa_learned)

    aa_learned = model2.pessimist(pt)

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
