from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import time
import random
from itertools import product

from pymcda.types import AlternativesAssignments, PerformanceTable
from pymcda.types import AlternativePerformances
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.uta import AVFSort
from pymcda.electre_tri import ElectreTri
from pymcda.learning.lp_avfsort import LpAVFSort
from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_alternatives
from pymcda.generate import generate_random_performance_table
from pymcda.utils import compute_ca
from pymcda.utils import add_errors_in_assignments
from test_utils import test_result, test_results

DATADIR = os.getenv('DATADIR', '%s/pymcda-data' % os.path.expanduser('~'))

def test_lp_avfsort(seed, na, nc, ncat, ns, na_gen, pcerrors):
    # Generate a random ELECTRE TRI model and assignment examples
    model = generate_random_mrsort_model(nc, ncat, seed)

    # Generate a first set of alternatives
    a = generate_alternatives(na)
    pt = generate_random_performance_table(a, model.criteria)

    aa = model.pessimist(pt)

    # Add errors in assignment examples
    aa_err = aa.copy()
    aa_erroned = add_errors_in_assignments(aa_err, model.categories,
                                           pcerrors / 100)

    gi_worst = AlternativePerformances('worst', {c.id: 0
                                                  for c in model.criteria})
    gi_best = AlternativePerformances('best', {c.id: 1
                                                for c in model.criteria})

    css = CriteriaValues([])
    for c in model.criteria:
        cs = CriterionValue(c.id, ns)
        css.append(cs)

    # Run linear program
    t1 = time.time()
    lp = LpAVFSort(model.criteria, css,
                   model.categories_profiles.to_categories(),
                   gi_worst, gi_best)
    t2 = time.time()
    obj, cv_l, cfs_l, catv_l = lp.solve(aa_err, pt)
    t3 = time.time()

    model2 = AVFSort(model.criteria, cv_l, cfs_l, catv_l)

    # Compute new assignment and classification accuracy
    aa2 = model2.get_assignments(pt)
    ok = ok_errors = ok2 = ok2_errors = 0
    for alt in a:
        if aa_err(alt.id) == aa2(alt.id):
            ok2 += 1
            if alt.id in aa_erroned:
                ok2_errors += 1

        if aa(alt.id) == aa2(alt.id):
            ok += 1
            if alt.id in aa_erroned:
                ok_errors += 1

    total = len(a)

    ca2 = ok2 / total
    ca2_errors = ok2_errors / total

    ca = ok / total
    ca_errors = ok_errors / total

    # Perform the generalization
    a_gen = generate_alternatives(na_gen)
    pt_gen = generate_random_performance_table(a_gen, model.criteria)
    aa = model.pessimist(pt_gen)
    aa2 = model2.get_assignments(pt_gen)
    ca_gen = compute_ca(aa, aa2)

    # Save all infos in test_result class
    t = test_result("%s-%d-%d-%d-%d-%d-%g" % (seed, na, nc, ncat, ns,
                    na_gen, pcerrors))

    # Input params
    t['seed'] = seed
    t['na'] = na
    t['nc'] = nc
    t['ncat'] = ncat
    t['ns'] = ns
    t['na_gen'] = na_gen
    t['pcerrors'] = pcerrors

    # Output params
    t['obj'] = obj
    t['ca'] = ca
    t['ca_errors'] = ca_errors
    t['ca2'] = ca2
    t['ca2_errors'] = ca2_errors
    t['ca_gen'] = ca_gen
    t['t_total'] = t3 - t1
    t['t_const'] = t2 - t1
    t['t_solve'] = t3 - t2

    return t

def run_tests(na, nc, ncat, ns, na_gen, pcerrors, nseeds, filename):
    # Create the CSV writer
    writer = csv.writer(open(filename, 'wb'))

    # Write the test options
    writer.writerow(['na', na])
    writer.writerow(['nc', nc])
    writer.writerow(['ncat', ncat])
    writer.writerow(['ns', ns])
    writer.writerow(['na_gen', na_gen])
    writer.writerow(['pcerrors', pcerrors])
    writer.writerow(['nseeds', nseeds])
    writer.writerow(['', ''])

    # Create a test results instance
    results = test_results()

    # Initialize the seeds
    seeds = range(nseeds)

    # Run the algorithm
    initialized = False
    for _na, _nc, _ncat, _ns, _na_gen, _pcerrors, seed \
        in product(na, nc, ncat, ns, na_gen, pcerrors, seeds):

        t1 = time.time()
        t = test_lp_avfsort(seed, _na, _nc, _ncat, _ns,  _na_gen, _pcerrors)
        t2 = time.time()

        if initialized is False:
            writer.writerow(t.get_attributes())
            initialized = True

        t.tocsv(writer)
        print("%s (%5f seconds)" % (t, t2 - t1))

        results.append(t)

    # Perform a summary
    writer.writerow(['', ''])
    t = results.summary(['na', 'nc', 'ncat', 'ns', 'na_gen', 'pcerrors'],
                        ['obj', 'ca', 'ca_errors', 'ca2', 'ca2_errors',
                         'ca_gen', 't_total', 't_const', 't_solve'])
    t.tocsv(writer)

if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser(usage = "python %s [options]" % sys.argv[0])
    parser.add_option("-n", "--na", action = "store", type="string",
                      dest = "na",
                      help = "number of assignment examples")
    parser.add_option("-c", "--nc", action = "store", type="string",
                      dest = "nc",
                      help = "number of criteria")
    parser.add_option("-t", "--ncat", action = "store", type="string",
                      dest = "ncat",
                      help = "number of categories")
    parser.add_option("-p", "--ns", action = "store", type="string",
                      dest = "ns",
                      help = "number of segments")
    parser.add_option("-g", "--na_gen", action = "store", type="string",
                      dest = "na_gen",
                      help = "number of generalization alternatives")
    parser.add_option("-e", "--errors", action = "store", type="string",
                      dest = "pcerrors",
                      help = "ratio of errors in the learning set")
    parser.add_option("-s", "--nseeds", action = "store", type="string",
                      dest = "nseeds",
                      help = "number of seeds")
    parser.add_option("-f", "--filename", action = "store", type="string",
                      dest = "filename",
                      help = "filename to save csv output")

    (options, args) = parser.parse_args()

    while not options.na:
        options.na = raw_input("Number of assignment examples ? ")
    options.na = options.na.split(",")
    options.na = [ int(x) for x in options.na ]

    while not options.nc:
        options.nc = raw_input("Number of criteria ? ")
    options.nc = options.nc.split(",")
    options.nc = [ int(x) for x in options.nc ]

    while not options.ncat:
        options.ncat = raw_input("Number of categories ? ")
    options.ncat = options.ncat.split(",")
    options.ncat = [ int(x) for x in options.ncat ]

    while not options.ns:
        options.ns = raw_input("Number of function segments ? ")
    options.ns = options.ns.split(",")
    options.ns = [ int(x) for x in options.ns ]

    while not options.na_gen:
        options.na_gen = raw_input("Number of generalization " \
                                   "alternatives ? ")
    options.na_gen = options.na_gen.split(",")
    options.na_gen = [ int(x) for x in options.na_gen ]

    while not options.pcerrors:
        options.pcerrors = raw_input("Ratio of errors ? ")
    options.pcerrors = options.pcerrors.split(",")
    options.pcerrors = [ float(x) for x in options.pcerrors ]

    while not options.nseeds:
        options.nseeds = raw_input("Number of seeds ? ")
    options.nseeds = int(options.nseeds)

    while not options.filename:
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        default_filename = "%s/test_lp_avfsort_etri-%s.csv" % (DATADIR, dt)
        options.filename = raw_input("File to save CSV data [%s] ? " \
                                     % default_filename)
        if not options.filename:
            options.filename = default_filename

    if options.filename[-4:] != ".csv":
        options.filename += ".csv"

    run_tests(options.na, options.nc, options.ncat, options.ns,
              options.na_gen, options.pcerrors, options.nseeds,
              options.filename)

    print("Results saved in '%s'" % options.filename)
