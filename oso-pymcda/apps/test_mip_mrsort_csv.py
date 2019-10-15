import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import random
import time
from collections import OrderedDict
from itertools import product
from pymcda.learning.mip_mrsort import MipMRSort
from pymcda.learning.mip_mrsort_veto import MipMRSortVC
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import Alternatives, Criteria, PerformanceTable
from pymcda.types import AlternativesAssignments, Categories
from pymcda.electre_tri import MRSort
from pymcda.generate import generate_alternatives
from pymcda.generate import generate_categories_profiles
from pymcda.generate import generate_random_profiles
from pymcda.generate import generate_random_criteria_weights
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.utils import compute_ca
from pymcda.utils import compute_confusion_matrix
from test_utils import test_result, test_results
from test_utils import load_mcda_input_data
from test_utils import save_to_xmcda

DATADIR = os.getenv('DATADIR', '%s/python_workspace/pymcda-master/pymcda-data' % os.path.expanduser('~'))

mip_mrsort = MipMRSort
#mip_mrsort = MipMRSortVC

def run_test(seed, data, pclearning):
    random.seed(seed)

    # Separate learning data and test data
    pt_learning, pt_test = data.pt.split(2, [pclearning, 100 - pclearning])
    aa_learning = data.aa.get_subset(pt_learning.keys())
    aa_test = data.aa.get_subset(pt_test.keys())

    # Initialize ELECTRE-TRI BM model
    cat_profiles = generate_categories_profiles(data.cats)
    worst = data.pt.get_worst(data.c)
    best = data.pt.get_best(data.c)
    b = generate_alternatives(len(data.cats) - 1, 'b')
    bpt = None
    cvs = None
    lbda = None

    model = MRSort(data.c, cvs, bpt, lbda, cat_profiles)

    # Run the linear program
    t1 = time.time()

    mip = mip_mrsort(model, pt_learning, aa_learning)
    obj = mip.solve()

    t_total = time.time() - t1

    # CA learning set
    aa_learning2 = model.pessimist(pt_learning)
    ca_learning = compute_ca(aa_learning, aa_learning2)
    auc_learning = model.auc(aa_learning, pt_learning)
    diff_learning = compute_confusion_matrix(aa_learning, aa_learning2,
                                             model.categories)

    # Compute CA of test setting
    if len(aa_test) > 0:
        aa_test2 = model.pessimist(pt_test)
        ca_test = compute_ca(aa_test, aa_test2)
        auc_test = model.auc(aa_test, pt_test)
        diff_test = compute_confusion_matrix(aa_test, aa_test2,
                                           model.categories)
    else:
        ca_test = 0
        auc_test = 0
        ncat = len(data.cats)
        diff_test = OrderedDict([((a, b), 0) for a in model.categories \
                                             for b in model.categories])

    # Compute CA of whole set
    aa2 = model.pessimist(data.pt)
    ca = compute_ca(data.aa, aa2)
    auc = model.auc(data.aa, data.pt)
    diff_all = compute_confusion_matrix(data.aa, aa2, model.categories)

    t = test_result("%s-%d-%d" % (data.name, seed, pclearning))

    model.id = 'learned'
    aa_learning.id, aa_test.id = 'learning_set', 'test_set'
    pt_learning.id, pt_test.id = 'learning_set', 'test_set'
    save_to_xmcda("%s/%s.bz2" % (directory, t.test_name),
                  model, aa_learning, aa_test, pt_learning, pt_test)

    t['seed'] = seed
    t['na'] = len(data.a)
    t['nc'] = len(data.c)
    t['ncat'] = len(data.cats)
    t['pclearning'] = pclearning
    t['na_learning'] = len(aa_learning)
    t['na_test'] = len(aa_test)
    t['obj'] = obj
    t['ca_learning'] = ca_learning
    t['ca_test'] = ca_test
    t['ca_all'] = ca
    t['auc_learning'] = auc_learning
    t['auc_test'] = auc_test
    t['auc_all'] = auc

    for k, v in diff_learning.items():
        t['learn_%s_%s' % (k[0], k[1])] = v
    for k, v in diff_test.items():
        t['test_%s_%s' % (k[0], k[1])] = v
    for k, v in diff_all.items():
        t['all_%s_%s' % (k[0], k[1])] = v

    t['t_total'] = t_total

    return t

def run_tests(nseeds, data, pclearning, filename):
    # Create the CSV writer
    f = open(filename, 'wb')
    writer = csv.writer(f)

    # Write the test options
    writer.writerow(['data', data.name])
    writer.writerow(['mip_mrsort', mip_mrsort.__name__])
    writer.writerow(['pclearning', pclearning])

    # Create a test_results instance
    results = test_results()

    # Initialize the seeds
    seeds = range(nseeds)

    # Run the algorithm
    initialized = False
    for _pclearning, seed in product(pclearning, seeds):
        t1 = time.time()
        t = run_test(seed, data, _pclearning)
        t2 = time.time()

        if initialized is False:
            fields = t.get_attributes()
            writer.writerow(fields)
            initialized = True

        t.tocsv(writer, fields)
        f.flush()
        print("%s (%5f seconds)" % (t, t2 - t1))

        results.append(t)

    # Perform a summary
    writer.writerow(['', ''])

    t = results.summary(['na', 'nc', 'ncat', 'pclearning', 'na_learning',
                         'na_test'],
                        ['obj', 'ca_learning', 'ca_test', 'ca_all',
                         'auc_learning', 'auc_test', 'auc_all',
                         't_total'])
    t.tocsv(writer)

if __name__ == "__main__":
    from optparse import OptionParser
    from test_utils import read_single_integer, read_multiple_integer
    from test_utils import read_csv_filename

    parser = OptionParser(usage = "python %s [options]" % sys.argv[0])
    parser.add_option("-i", "--csvfile", action = "store", type="string",
                      dest = "csvfile",
                      help = "csv file with data")
    parser.add_option("-p", "--pclearning", action = "store", type="string",
                      dest = "pclearning",
                      help = "Percentage of data to use in learning set")
    parser.add_option("-s", "--nseeds", action = "store", type="string",
                      dest = "nseeds",
                      help = "number of seeds")
    parser.add_option("-f", "--filename", action = "store", type="string",
                      dest = "filename",
                      help = "filename to save csv output")
    (options, args) = parser.parse_args()

    while not options.csvfile:
        options.csvfile = raw_input("CSV file containing data ? ")
    data = load_mcda_input_data(options.csvfile)
    if data is None:
        exit(1)

    options.pclearning = read_multiple_integer(options.pclearning,
                                               "Percentage of data to " \
                                               "use in the learning set")
    options.nseeds = read_single_integer(options.nseeds, "Number of seeds")

    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_filename = "%s/test_mip_mrsort-%s-%s.csv" \
                       % (DATADIR, data.name, dt)
    options.filename = read_csv_filename(options.filename, default_filename)

    directory = options.filename + "-data"
    if not os.path.exists(directory):
        os.makedirs(directory)

    run_tests(options.nseeds, data, options.pclearning, options.filename)

    print("Results saved in '%s'" % options.filename)
