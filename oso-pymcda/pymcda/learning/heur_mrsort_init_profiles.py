from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
import random
from pymcda.types import AlternativePerformances
from pymcda.types import PerformanceTable
from collections import OrderedDict

class HeurMRSortInitProfiles():

    def __init__(self, model, pt_sorted, aa):
        self.model = model
        self.pt_sorted = pt_sorted
        self.aa = aa
        self.delta = 0.0000001
        self.b0 = pt_sorted.pt.get_worst(self.model.criteria)
        self.compute_categories_probabilities()

    def compute_categories_probabilities(self):
        self.cat_proba = {}
        total = len(self.aa)
        for cat in self.model.categories:
            aa_cat = self.aa.get_alternatives_in_category(cat)
            self.cat_proba[cat] = total - len(aa_cat)

    def compute_histogram(self, crit, cat_above, cat_below, above):
        h1 = {}
        h2 = {}
        below = self.b0.performances[crit.id]
        aids, perfs = self.pt_sorted.get_middle(crit.id, below, above)

        delta = self.delta * crit.direction

        # From smallest to biggest
        val = 0
        for aid, perf in zip(aids, perfs):
            cat = self.aa[aid].category_id
            if cat == cat_below:
                val += self.cat_proba[cat]
                h1[perf + delta] = val
            elif cat == cat_above:
                h1[perf] = val

        # From biggest to smallest
        val = 0
        aids.reverse()
        perfs.reverse()
        for aid, perf in zip(aids, perfs):
            cat = self.aa[aid].category_id
            if cat == cat_above:
                val += self.cat_proba[cat]
                h2[perf] = val
            elif cat == cat_below:
                h2[perf + delta] = val

        return { key: h1[key] + h2[key] for key in h1 }

    def weighted_choice(self, h):
        total = sum(h.values())
        r = random.uniform(0, total)
        tmp = 0
        for perf, proba in h.items():
            tmp += proba
            if tmp > r:
                break

        return perf

    def init_profile(self, profile_id, cat_above, cat_below, pabove):
        ap = AlternativePerformances(profile_id, OrderedDict({}))
        for c in self.model.criteria:
            perf = pabove.performances[c.id]
            h = self.compute_histogram(c, cat_above, cat_below, perf)
            if h:
                perf = self.weighted_choice(h)
            ap.performances[c.id] = perf

        return ap

    def solve(self):
        cats = self.model.categories[:]
        cats.reverse()

        profiles = self.model.profiles[:]
        profiles.reverse()

        bpt = PerformanceTable()
        pabove = self.pt_sorted.pt.get_best(self.model.criteria)
        for i in range(len(cats) - 1):
            profile_id = profiles[i]
            bp = self.init_profile(profile_id, cats[i], cats[i+1], pabove)
            bpt.append(bp)
            pabove = bp

        self.model.bpt = bpt

if __name__ == "__main__":
    from pymcda.generate import generate_random_mrsort_model
    from pymcda.generate import generate_alternatives
    from pymcda.generate import generate_random_performance_table
    from pymcda.generate import generate_random_profiles
    from pymcda.pt_sorted import SortedPerformanceTable
    from pymcda.utils import compute_ca
    from pymcda.utils import compute_winning_and_loosing_coalitions
    from pymcda.utils import display_coalitions
    from pymcda.learning.lp_mrsort_weights import LpMRSortWeights
    from pymcda.ui.graphic import display_electre_tri_models

    model = generate_random_mrsort_model(10, 3, 17)
    winning, loosing = compute_winning_and_loosing_coalitions(model.cv,
                                                              model.lbda)
    print("Number of coalitions: %d" % len(winning))

    a = generate_alternatives(1000)
    pt = generate_random_performance_table(a, model.criteria)
    sorted_pt = SortedPerformanceTable(pt)

    aa = model.pessimist(pt)

    for cat in model.categories_profiles.get_ordered_categories():
        pc = len(aa.get_alternatives_in_category(cat)) / len(aa) * 100
        print("Percentage of alternatives in %s: %g %%" % (cat, pc))

    # Learn the weights with random generated profiles
    for i in range(10):
        model2 = model.copy()
        b = model.categories_profiles.get_ordered_profiles()
        model2.bpt = generate_random_profiles(b, model2.criteria)

        lp_weights = LpMRSortWeights(model2, pt, aa)
        lp_weights.solve()

        aa2 = model2.pessimist(pt)
        ca2 = compute_ca(aa, aa2)

        win2, loose2 = compute_winning_and_loosing_coalitions(model2.cv,
                                                              model2.lbda)
        coal2_ni = list((set(winning) ^ set(win2)) & set(winning))
        coal2_add = list((set(winning) ^ set(win2)) & set(win2))

        print("Classification accuracy with random profiles: %g" % ca2)
        print("Coalitions: total: %d, common: %d, added: %d" % \
              (len(win2), (len(winning) - len(coal2_ni)), len(coal2_add)))

    # Learn the weights with profiles generated by the heuristic
    model3 = model.copy()
    heur = HeurMRSortInitProfiles(model3, sorted_pt, aa)
    heur.solve()

    lp_weights = LpMRSortWeights(model3, pt, aa)
    lp_weights.solve()

    aa3 = model3.pessimist(pt)
    ca3 = compute_ca(aa, aa3)

    win3, loose3 = compute_winning_and_loosing_coalitions(model3.cv,
                                                          model3.lbda)
    coal3_ni = list((set(winning) ^ set(win3)) & set(winning))
    coal3_add = list((set(winning) ^ set(win3)) & set(win3))

    print("Classification accuracy with heuristic: %g" % ca3)
    print("Coalitions: total: %d, common: %d, added: %d" % \
          (len(win3), (len(winning) - len(coal3_ni)), len(coal3_add)))

    display_electre_tri_models([model], [pt.get_worst(model.criteria)],
                               [pt.get_best(model.criteria)],
                               [[ap for ap in model3.bpt]])
