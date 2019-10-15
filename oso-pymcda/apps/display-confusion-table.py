#!/usr/bin/env python

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import bz2
from xml.etree import ElementTree

from pymcda.electre_tri import MRSort
from pymcda.uta import AVFSort
from pymcda.types import PerformanceTable
from pymcda.types import AlternativesAssignments
from pymcda.types import AlternativePerformances
from pymcda.utils import compute_ca
from pymcda.utils import compute_confusion_matrix
from pymcda.utils import print_confusion_matrix
from pymcda.utils import print_pt_and_assignments
from pymcda.ui.graphic import display_electre_tri_models
from test_utils import is_bz2_file

f = sys.argv[1]
if not os.path.isfile(f):
    print("Invalid file %s" % f)
    sys.exit(1)

if is_bz2_file(f) is True:
    f = bz2.BZ2File(f)

tree = ElementTree.parse(f)
root = tree.getroot()

try:
    pt_learning = PerformanceTable().from_xmcda(root, 'learning_set')
except:
    pt_learning = None

try:
    pt_test = PerformanceTable().from_xmcda(root, 'test_set')
except:
    pt_test = None

aa_learning_m1, aa_learning_m2 = None, None
aa_test_m1, aa_test_m2 = None, None

if root.find("ElectreTri[@id='initial']") is not None:
    m1 = MRSort().from_xmcda(root, 'initial')
    if pt_learning is not None:
        aa_learning_m1 = m1.pessimist(pt_learning)
    if pt_test is not None:
        aa_test_m1 = m1.pessimist(pt_test)
elif root.find("AVFSort[@id='initial']") is not None:
    m1 = AVFSort().from_xmcda(root, 'initial')
    if pt_learning is not None:
        aa_learning_m1 = m1.get_assignments(pt_learning)
    if pt_test is not None:
        aa_test_m1 = m1.get_assignments(pt_test)
else:
    if root.find("alternativesAffectations[@id='learning_set']") is not None:
        aa_learning_m1 = AlternativesAssignments().from_xmcda(root,
                                                              'learning_set')

    if root.find("alternativesAffectations[@id='test_set']") is not None:
        aa_test_m1 = AlternativesAssignments().from_xmcda(root, 'test_set')

if root.find("ElectreTri[@id='learned']") is not None:
    m2 = MRSort().from_xmcda(root, 'learned')
    if pt_learning is not None:
        aa_learning_m2 = m2.pessimist(pt_learning)
    if pt_test is not None:
        aa_test_m2 = m2.pessimist(pt_test)
elif root.find("AVFSort[@id='learned']") is not None:
    m2 = AVFSort().from_xmcda(root, 'learned')
    if pt_learning is not None:
        aa_learning_m2 = m2.get_assignments(pt_learning)
        aids = []
        from pymcda.utils import print_pt_and_assignments
        for aid in aa_learning_m2.keys():
            if aa_learning_m2[aid].category_id != aa_learning_m1[aid].category_id:
                aids.append(aid)
            else:
                aids.append(aid)

        au = m2.global_utilities(pt_learning)
        print_pt_and_assignments(aids, None, [aa_learning_m1, aa_learning_m2], pt_learning, au)
#        for i in range(1, len(pt_learning) + 1):
#            aid = "a%d" % i
#            uti = m2.global_utility(pt_learning["a%d" % i])
#            if aa_learning_m2[aid].category_id != aa_learning_m1[aid].category_id:
#                print("%s %g %s %s" % (aid, uti.value, aa_learning_m2[aid].category_id, aa_learning_m1[aid].category_id))
#        print_pt_and_assignments(anok, c, [aa_learning_m1, aa_learning_m2], pt_learning)
    if pt_test is not None:
        aa_test_m2 = m2.get_assignments(pt_test)

def compute_auc_histo(aa):
    pass

if aa_learning_m1 is not None:
    ca_learning = compute_ca(aa_learning_m1, aa_learning_m2)
    auc_learning = m2.auc(aa_learning_m1, pt_learning)

    print("Learning set")
    print("============")
    print("CA : %g" % ca_learning)
    print("AUC: %g" % auc_learning)
    print("Confusion table:")
    matrix = compute_confusion_matrix(aa_learning_m1, aa_learning_m2,
                                      m2.categories)
    print_confusion_matrix(matrix, m2.categories)
    aids = [a.id for a in aa_learning_m1 \
            if aa_learning_m1[a.id].category_id != aa_learning_m2[a.id].category_id]
    if len(aids) > 0:
        print("List of alternatives wrongly assigned:")
        print_pt_and_assignments(aids, None, [aa_learning_m1, aa_learning_m2],
                                 pt_learning)

if aa_test_m1 is not None and len(aa_test_m1) > 0:
    ca_test = compute_ca(aa_test_m1, aa_test_m2)
    auc_test = m2.auc(aa_test_m1, pt_test)

    print("\n\nTest set")
    print("========")
    print("CA : %g" % ca_test)
    print("AUC: %g" % auc_test)
    print("Confusion table:")
    matrix = compute_confusion_matrix(aa_test_m1, aa_test_m2, m2.categories)
    print_confusion_matrix(matrix, m2.categories)
    aids = [a.id for a in aa_test_m1 \
            if aa_test_m1[a.id].category_id != aa_test_m2[a.id].category_id]
    if len(aids) > 0:
        print("List of alternatives wrongly assigned:")
        print_pt_and_assignments(aids, None, [aa_test_m1, aa_test_m2],
                                 pt_test)

if type(m2) == MRSort:
    worst = AlternativePerformances('worst', {c.id: 0 for c in m2.criteria})
    best = AlternativePerformances('best', {c.id: 1 for c in m2.criteria})

    categories = m2.categories

    a_learning = aa_learning_m1.keys()
    pt_learning_ok = []
    pt_learning_too_low = []
    pt_learning_too_high = []
    for a in a_learning:
        i1 = categories.index(aa_learning_m1[a].category_id)
        i2 = categories.index(aa_learning_m2[a].category_id)
        if i1 == i2:
            pt_learning_ok.append(pt_learning[a])
        elif i1 < i2:
            pt_learning_too_high.append(pt_learning[a])
        elif i1 > i2:
            pt_learning_too_low.append(pt_learning[a])

    a_test = aa_test_m1.keys()
    pt_test_ok = []
    pt_test_too_low = []
    pt_test_too_high = []
    for a in a_test:
        i1 = categories.index(aa_test_m1[a].category_id)
        i2 = categories.index(aa_test_m2[a].category_id)
        if i1 == i2:
            pt_test_ok.append(pt_test[a])
        elif i1 < i2:
            pt_test_too_high.append(pt_test[a])
        elif i1 > i2:
            pt_test_too_low.append(pt_test[a])

    display_electre_tri_models([m2, m2], [worst, worst], [best, best],
                               [m2.vpt, m2.vpt],
                               [pt_learning_too_low, pt_test_too_low],
                               None,
                               [pt_learning_too_high, pt_test_too_high])
